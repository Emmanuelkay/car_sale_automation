import logging
import os
import shutil
from uuid import uuid4
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Response, status, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from openai import AsyncOpenAI
import instructor
import httpx

from config import settings
from worker import process_whatsapp_message, generate_rag_response, embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global clients
qdrant_client: AsyncQdrantClient = None
openai_client: instructor.AsyncInstructor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from database import init_db
    init_db()
    
    global qdrant_client, openai_client
    logger.info("Initializing global clients...")
    
    # Initialize Qdrant Client
    qdrant_client = AsyncQdrantClient(
        path=settings.QDRANT_PATH
    )
    
    # Programmatically verify and create Qdrant collection if it does not exist
    try:
        collections = await qdrant_client.get_collections()
        exist = any(c.name == settings.QDRANT_COLLECTION_NAME for c in collections.collections)
        if not exist:
            from qdrant_client.models import Distance, VectorParams
            await qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,  # BAAI/bge-small-en-v1.5 embedding dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection {settings.QDRANT_COLLECTION_NAME} programmatically.")
    except Exception as e:
        logger.error(f"Error checking/creating Qdrant collection: {e}")
        
    # Initialize Async OpenAI Client with Instructor
    # We use httpx.AsyncClient with verify=False to bypass self-signed cert issues (like your curl -k)
    http_client = httpx.AsyncClient(verify=False)
    client = AsyncOpenAI(
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        http_client=http_client
    )
    openai_client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    
    yield
    
    # Shutdown
    logger.info("Closing global clients...")
    await qdrant_client.close()

app = FastAPI(title="WhatsApp RAG API", lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

@app.get("/webhook/whatsapp")
async def verify_webhook(request: Request):
    """
    Token verification handshake endpoint for Meta webhooks.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == settings.META_VERIFY_TOKEN:
            logger.info("Webhook verified successfully!")
            return Response(content=challenge, status_code=status.HTTP_200_OK)
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Verification failed")
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing parameters")

@app.post("/webhook/whatsapp")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    JSON payload processor. Instantly queues the background task and returns 200 OK.
    """
    try:
        body = await request.json()
        logger.info(f"--- WEBHOOK PAYLOAD ---")
        logger.info(body)
        
        # Parse Meta WhatsApp payload
        # Standard structure: body["entry"][0]["changes"][0]["value"]["messages"][0]
        if body.get("object") == "whatsapp_business_account":
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])
                    
                    if messages:
                        message = messages[0]
                        phone_number = message.get("from")
                        
                        # Meta WhatsApp message deduplication check
                        message_id = message.get("id")
                        if message_id:
                            from database import is_message_processed, mark_message_processed
                            if is_message_processed(message_id):
                                logger.info(f"Duplicate Meta webhook received (ID: {message_id}). Skipping processing.")
                                return Response(status_code=status.HTTP_200_OK)
                            
                            # Mark as processed immediately
                            mark_message_processed(message_id)

                        # Only handle text messages for now
                        if message.get("type") == "text":
                            message_body = message["text"]["body"]
                            
                            # Only queue if the message contains text content
                            if message_body and message_body.strip():
                                # Enqueue processing to background tasks
                                background_tasks.add_task(
                                    process_whatsapp_message, 
                                    phone_number, 
                                    message_body,
                                    qdrant_client,
                                    openai_client
                                )
                            
        # Always return 200 OK immediately
        return Response(status_code=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        # Still return 200 to acknowledge receipt and prevent Meta from retrying unnecessarily
        return Response(status_code=status.HTTP_200_OK)

@app.post("/api/chat")
async def handle_web_chat(request: ChatRequest):
    """
    Endpoint for Web UI to chat with the RAG system directly.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        response = await generate_rag_response(request.message, qdrant_client, openai_client, history=request.history)
        return {
            "response": response.message_body,
            "image_url": response.selected_image_url
        }
    except Exception as e:
        logger.error(f"Error handling web chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/admin/add_car")
async def add_car(
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    price_ksh: int = Form(...),
    mileage_km: int = Form(...),
    listing_details: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        car_id = str(uuid4())
        
        # Save image
        image_ext = os.path.splitext(image.filename)[1]
        image_filename = f"{car_id}{image_ext}"
        image_path = os.path.join("static", "images", image_filename)
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        image_url = f"http://localhost:8000/images/{image_filename}"
        
        # Embed details
        embeddings = list(embedding_model.embed([listing_details]))
        vector = embeddings[0].tolist()
        
        # Upsert to Qdrant
        point = PointStruct(
            id=car_id,
            vector=vector,
            payload={
                "listing_details": listing_details,
                "metadata": {
                    "make": make,
                    "model": model,
                    "year": year,
                    "price_ksh": price_ksh,
                    "mileage_km": mileage_km,
                    "image_url": image_url,
                    "status": "available"
                },
                "status": "available"
            }
        )
        
        await qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=[point]
        )
        
        return {"success": True, "car_id": car_id}
    except Exception as e:
        logger.error(f"Error adding car: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/inventory")
async def list_inventory():
    try:
        # Scroll all records (up to 100 for admin UI)
        records, _ = await qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        inventory = []
        for record in records:
            inventory.append({
                "id": record.id,
                "payload": record.payload
            })
            
        return {"inventory": inventory}
    except Exception as e:
        logger.error(f"Error listing inventory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/delete_car/{car_id}")
async def delete_car(car_id: str):
    try:
        await qdrant_client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=[car_id]
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Error deleting car: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/leads")
async def list_leads():
    try:
        from database import get_all_leads
        leads = get_all_leads()
        return {"leads": leads}
    except Exception as e:
        logger.error(f"Error fetching leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/leads/export/csv")
async def export_leads_csv():
    import csv
    import io
    try:
        from database import get_all_leads
        leads = get_all_leads()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Customer Name", "Contact", "Car of Interest", "Preferred Time", "Status", "Created At"])
        for lead in leads:
            writer.writerow([
                lead["id"], lead["customer_name"], lead["customer_contact"], 
                lead["car_of_interest"], lead["preferred_date_time"], 
                lead["status"], lead["created_at"]
            ])
            
        headers = {
            "Content-Disposition": "attachment; filename=elite_auto_leads.csv"
        }
        return Response(content=output.getvalue(), media_type="text/csv", headers=headers)
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/leads/{lead_id}/ics")
async def generate_ics(lead_id: int):
    try:
        from database import get_all_leads
        leads = get_all_leads()
        lead = next((l for l in leads if l["id"] == lead_id), None)
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
            
        # Create a simple ICS format
        import datetime
        now = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        
        ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Elite Auto//Test Drive Appointments//EN
BEGIN:VEVENT
UID:{lead_id}-{now}@eliteauto.com
DTSTAMP:{now}
SUMMARY:Test Drive: {lead["car_of_interest"]} - {lead["customer_name"]}
DESCRIPTION:Customer: {lead["customer_name"]}\\nContact: {lead["customer_contact"]}\\nPreferred Time: {lead["preferred_date_time"]}
BEGIN:VALARM
TRIGGER:-PT15M
ACTION:DISPLAY
DESCRIPTION:Reminder
END:VALARM
END:VEVENT
END:VCALENDAR"""
        
        headers = {
            "Content-Disposition": f"attachment; filename=Test_Drive_{lead['customer_name'].replace(' ', '_')}.ics"
        }
        return Response(content=ics_content, media_type="text/calendar", headers=headers)
    except Exception as e:
        logger.error(f"Error generating ICS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files at the root
app.mount("/", StaticFiles(directory="static", html=True), name="static")
