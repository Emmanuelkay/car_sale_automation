import logging
import httpx
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import instructor
from fastembed import TextEmbedding

from config import settings

# Initialize FastEmbed model globally
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

logger = logging.getLogger(__name__)

# Pydantic Guardrail Schema
class CarSalesResponse(BaseModel):
    message_body: str = Field(description="Markdown text containing clean bolding and structured spacing.")
    selected_image_url: str = Field(description="Must be empty or a direct URL fetched from the matched context payload.")

SYSTEM_INSTRUCTION = """
You are an elite automotive sales advisor for a car dealership. 
Your goal is to assist customers professionally, referencing the available inventory provided in the context.
You must strictly refrain from hallucinating inventory. If a customer asks for a car not in the context, politely inform them we do not have it currently available.
Format your response using clean Markdown with bolding and structured spacing.
If the matched car has an image_url, include it in the selected_image_url field. Otherwise, leave it empty.
"""

async def process_whatsapp_message(
    phone_number: str, 
    message_body: str, 
    qdrant_client: AsyncQdrantClient, 
    openai_client: instructor.AsyncInstructor
):
    """
    Executes the decoupled async workflow: vector search, LLM generation, and outbound Meta gateway delivery.
    """
    try:
        logger.info(f"--- NEW MESSAGE ---")
        logger.info(f"From: {phone_number} | Message: {message_body}")
        
        # Step 1: Embed the incoming message
        embeddings = list(embedding_model.embed([message_body]))
        query_vector = embeddings[0].tolist()

        # Step 2: Vector Search in Qdrant with status filter
        search_results = await qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="status",
                        match=MatchValue(value="available")
                    )
                ]
            ),
            limit=3
        )

        # Step 3: Context Injection
        context_parts = []
        for hit in search_results:
            context_parts.append(
                f"ID: {hit.id}\nDetails: {hit.payload.get('listing_details')}\nMetadata: {hit.payload.get('metadata')}"
            )
        
        context_string = "\n\n".join(context_parts) if context_parts else "No available inventory matches this query."
        
        prompt = f"Context (Available Inventory):\n{context_string}\n\nCustomer Message:\n{message_body}"

        # Step 4: Guardrailed Output using Instructor with OpenAI
        structured_response = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            response_model=CarSalesResponse,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        # Step 5: Outbound Meta Gateway Delivery
        logger.info(f"--- OUTGOING RESPONSE ---")
        logger.info(f"Sending to {phone_number}: {structured_response.message_body}")
        logger.info(f"Image URL: {structured_response.selected_image_url}")
        
        await send_whatsapp_message(
            phone_number=phone_number,
            message_body=structured_response.message_body,
            image_url=structured_response.selected_image_url
        )

    except Exception as e:
        logger.error(f"Error in background worker processing message from {phone_number}: {e}")

async def send_whatsapp_message(phone_number: str, message_body: str, image_url: str = ""):
    """
    Sends an HTTP POST directly to Meta's Graph API.
    """
    url = f"https://graph.facebook.com/v20.0/{settings.META_PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {settings.META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    if image_url:
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "image",
            "image": {
                "link": image_url,
                "caption": message_body
            }
        }
    else:
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "text",
            "text": {
                "body": message_body
            }
        }
        
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully sent message to {phone_number}")
