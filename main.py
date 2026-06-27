import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Response, status
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI
import instructor
import httpx

from config import settings
from worker import process_whatsapp_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global clients
qdrant_client: AsyncQdrantClient = None
openai_client: instructor.AsyncInstructor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global qdrant_client, openai_client
    logger.info("Initializing global clients...")
    
    # Initialize Qdrant Client
    qdrant_client = AsyncQdrantClient(
        path=settings.QDRANT_PATH
    )
    
    # Initialize Async OpenAI Client with Instructor
    # We use httpx.AsyncClient with verify=False to bypass self-signed cert issues (like your curl -k)
    http_client = httpx.AsyncClient(verify=False)
    client = AsyncOpenAI(
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        http_client=http_client
    )
    openai_client = instructor.from_openai(client)
    
    yield
    
    # Shutdown
    logger.info("Closing global clients...")
    await qdrant_client.close()

app = FastAPI(title="WhatsApp RAG API", lifespan=lifespan)

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
                        
                        # Only handle text messages for now
                        if message.get("type") == "text":
                            message_body = message["text"]["body"]
                            
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
