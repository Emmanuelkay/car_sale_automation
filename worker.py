import logging
import httpx
from pydantic import BaseModel, Field
from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import instructor
from fastembed import TextEmbedding

from config import settings

# Initialize FastEmbed model globally
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

logger = logging.getLogger(__name__)

# Pydantic Schema for frontend API responses
class CarSalesResponse(BaseModel):
    message_body: str
    selected_image_url: str

# Pydantic Schema for Pass 2 Structured Extraction
class LeadExtraction(BaseModel):
    customer_name: Optional[str] = Field(None, description="The name of the customer")
    customer_contact: Optional[str] = Field(None, description="A valid contact phone number or WhatsApp number")
    preferred_vehicle: Optional[str] = Field(None, description="The vehicle they want to test drive")
    preferred_date_time: Optional[str] = Field(None, description="The date and time they want to schedule for a test drive")
    ready_for_booking: bool = Field(False, description="Set to True ONLY if the customer has explicitly agreed to a test drive AND provided their name and contact info. Otherwise False.")

SYSTEM_INSTRUCTION = """
You are an elite, highly persuasive automotive sales advisor for a premium car dealership.
Your goal is to enthusiastically assist customers, build rapport, highlight features, and secure test drives.

CRITICAL ROLES: 
- You are the SALES ADVISOR. 
- The user chatting with you is the CUSTOMER. 
- Do NOT invert these roles. Do NOT write messages pretending to be the customer scheduling a test drive. Your responses must always be from the perspective of the advisor helper.

CONVERSATION FLOW:
1. First, answer the customer's question directly using the available inventory details. Highlight the car's best features, and sell it enthusiastically!
2. If they are checking stock or asking general questions, pitch the car and end by asking if they would like to schedule a test drive.
3. If they agree to or ask for a test drive, warmly ask them to provide their Name, Phone number, and preferred Date and time so you can book it.
4. Do NOT say things like "I've booked it for you" unless they have actually typed and provided their name, contact, and time in the conversation history. If any of those are missing, ask for the missing details.

PRICE BUDGET FILTERING (CRITICAL):
- Pay close attention to the customer's budget (e.g. "under 1 million", "below 3M").
- Examine the `price_ksh` field in the metadata for each car. Perform a strict mathematical comparison: only recommend cars that are strictly within their budget.
- If a customer asks for a car under 1 million, only the Suzuki Jimny (Ksh 250,000) is eligible. Recommending the Honda CR-V (Ksh 4,200,000) or Toyota Camry (Ksh 3,500,000) is a direct failure.
- If no cars in the context fit their budget, state clearly that we have no cars in that budget range, and present our cheapest option as an alternative.

UNAVAILABLE VEHICLES & PIVOTING (CRITICAL):
- If the customer asks for a vehicle NOT present in our inventory (e.g., Mercedes Benz GLE, Lexus, etc.) or asks about services we don't provide (like custom importing):
  1. Warmly and politely acknowledge their request and state that we do not currently have that specific vehicle in stock.
  2. Pivot to proposing the closest match we *do* have in stock based on the vehicle type (e.g. recommend our Honda CR-V EX-L SUV if they asked for a Mercedes GLE SUV, or recommend our Toyota Camry if they asked for a sedan).
  3. Never ignore their request or jump straight to pitching a car without first explaining that we do not have the vehicle they originally asked for.

Respond in plain text. Format your response using clean Markdown with bolding and structured spacing. DO NOT embed markdown image links (e.g. ![image](url)) inside the message body.
"""

EXTRACTION_PROMPT = """
Analyze the recent chat history between the customer and the salesperson.
Extract the customer's contact details and booking intent into the required JSON structure.
Fields:
- customer_name: The customer's name (only if explicitly provided in the chat by the user).
- customer_contact: A phone number, WhatsApp number, or contact info (only if explicitly provided by the user).
- preferred_date_time: The date and time they want to schedule for a test drive (only if explicitly provided).
- preferred_vehicle: The car they want to test drive.
- ready_for_booking: Set to True ONLY if the customer has explicitly agreed to a test drive AND provided their name AND contact info. Otherwise, leave as False.

If a detail has not been provided yet, leave it as null. Do NOT guess, assume, or hallucinate any details.
"""

async def generate_rag_response(
    message_body: str, 
    qdrant_client: AsyncQdrantClient, 
    openai_client: instructor.AsyncInstructor,
    history: list = None
) -> CarSalesResponse:
    """
    Two-Pass Conversational Routing and RAG pipeline.
    """
    # Step 1: Smart RAG - Decide if we need to search Qdrant
    needs_search = True
    query_lower = message_body.lower()
    words = query_lower.split()
    
    # Simple booking responses and patterns that do NOT need database search
    booking_responses = ["yes", "no", "sure", "ok", "yep", "nah", "okay", "correct", "lets do it", "lets schedule", "cancel", "stop", "confirm"]
    
    import re
    is_simple_confirmation = query_lower in booking_responses or len(words) == 1 and query_lower.strip("?.!") in booking_responses
    is_phone = bool(re.search(r'\d{6,}', query_lower))
    
    # We only skip search if it's a simple confirmation/phone and doesn't mention car terms
    car_keywords = ["suzuki", "jimny", "toyota", "camry", "honda", "cr-v", "lexus", "bmw", "car", "suv", "sedan", "stock", "inventory", "price", "ksh", "cost", "under", "below", "cheapest", "expensive", "specs", "specification", "features", "details", "mileage", "color", "year"]
    has_car_keywords = any(kw in query_lower for kw in car_keywords)
    
    if (is_simple_confirmation or is_phone) and not has_car_keywords:
        needs_search = False

    search_results = []
    if needs_search:
        logger.info(f"Performing vector search for query: {message_body}")
        query_text = message_body
        if history and len(history) > 0:
            query_text = f"Previous context: {history[-1].content}\nUser Question: {message_body}"
            
        embeddings = list(embedding_model.embed([query_text]))
        query_vector = embeddings[0].tolist()

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

        context_parts = []
        for hit in search_results:
            context_parts.append(
                f"ID: {hit.id}\nDetails: {hit.payload.get('listing_details')}\nMetadata: {hit.payload.get('metadata')}"
            )
        
        context_string = "\n\n".join(context_parts) if context_parts else "No available inventory matches this query."
        prompt = f"Context (Available Inventory):\n{context_string}\n\nCustomer Message:\n{message_body}"
    else:
        logger.info("Smart RAG: Skipping vector search and context injection.")
        prompt = message_body

    # ==========================================
    # PASS 1: Conversational Voice (Plain Text)
    # ==========================================
    llm_messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    if history:
        for h in history:
            role = "assistant" if h.role == "bot" else "user"
            llm_messages.append({"role": role, "content": h.content})
    llm_messages.append({"role": "user", "content": prompt})

    try:
        # Use raw client for Pass 1 (completely natural voice, no schemas)
        chat_completion = await openai_client.client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            messages=llm_messages,
            temperature=0.2
        )
        ai_response_text = chat_completion.choices[0].message.content
        logger.info(f"Pass 1 Response: {ai_response_text}")
    except Exception as e:
        logger.error(f"Pass 1 generation error: {e}")
        ai_response_text = "I'm sorry, I encountered an error while searching the inventory. Could you please rephrase your request?"

    # DYNAMIC IMAGE RESOLUTION
    selected_image_url = ""
    lower_response = ai_response_text.lower()
    if needs_search and search_results:
        for hit in search_results:
            make = hit.payload.get("metadata", {}).get("make", "").lower()
            model = hit.payload.get("metadata", {}).get("model", "").lower()
            if make in lower_response or model in lower_response:
                selected_image_url = hit.payload.get("metadata", {}).get("image_url", "")
                break

    # ==========================================
    # PASS 2: Silent Lead Parser (Structured Background)
    # ==========================================
    recent_history_msgs = []
    if history:
        for h in history[-2:]:
            role = "assistant" if h.role == "bot" else "user"
            recent_history_msgs.append(f"{role}: {h.content}")
    recent_history_msgs.append(f"user: {message_body}")
    recent_history_msgs.append(f"assistant: {ai_response_text}")
    conversation_log = "\n".join(recent_history_msgs)

    try:
        # Extract silent leads using Pydantic format
        raw_extraction = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            response_model=LeadExtraction,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Recent Conversation Log:\n{conversation_log}"}
            ],
            temperature=0.0
        )
        logger.info(f"Pass 2 Extraction: {raw_extraction.model_dump()}")

        if raw_extraction.ready_for_booking:
            name = raw_extraction.customer_name
            contact = raw_extraction.customer_contact
            time_pref = raw_extraction.preferred_date_time
            
            user_messages = [h.content for h in history if h.role == "user"] if history else []
            user_messages.append(message_body)
            combined_user_input = " ".join(user_messages).lower()
            
            name_present = name and name.lower() in combined_user_input
            contact_present = contact and contact.lower() in combined_user_input

            if not name_present or not contact_present:
                logger.warning(f"Pass 2 verification check failed. Hallucinated Name: {name} (Present: {name_present}), Contact: {contact} (Present: {contact_present})")
            else:
                from database import add_lead
                add_lead(
                    customer_name=name,
                    customer_contact=contact,
                    car_of_interest=raw_extraction.preferred_vehicle or "Unknown",
                    preferred_date_time=time_pref or "Unknown"
                )
                logger.info("Lead successfully saved to CRM in Pass 2.")
                
                # Append beautiful formatting to response text programmatically
                ai_response_text = (
                    f"Perfect, **{name}**! I have successfully registered your test drive request for the **{raw_extraction.preferred_vehicle or 'vehicle'}** on **{time_pref or 'your requested time'}**.\n\n"
                    f"✅ *Our showroom team will contact you shortly at **{contact}** to confirm your appointment. We look forward to showing you the car!*"
                )
    except Exception as e:
        logger.error(f"Pass 2 structured extraction error: {e}")

    return CarSalesResponse(
        message_body=ai_response_text,
        selected_image_url=selected_image_url
    )


async def process_whatsapp_message(
    phone_number: str, 
    message_body: str, 
    qdrant_client: AsyncQdrantClient, 
    openai_client: instructor.AsyncInstructor
):
    """
    Executes the decoupled async workflow for WhatsApp gateway delivery.
    """
    try:
        logger.info(f"--- NEW WHATSAPP MESSAGE ---")
        logger.info(f"From: {phone_number} | Message: {message_body}")
        
        structured_response = await generate_rag_response(message_body, qdrant_client, openai_client)

        # Step 5: Outbound Meta Gateway Delivery
        logger.info(f"--- OUTGOING WHATSAPP RESPONSE ---")
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
