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

from typing import Optional

class CarSalesResponse(BaseModel):
    message_body: str = Field(description="Markdown text containing clean bolding and structured spacing.")
    selected_image_url: str = Field(description="Must be empty or a direct URL fetched from the matched context payload.")
    wants_test_drive: bool = Field(default=False, description="Set to true ONLY if the customer has explicitly agreed to a test drive AND provided their name, contact info, and preferred time.")
    customer_name: Optional[str] = Field(default=None, description="The customer's name if provided.")
    customer_contact: Optional[str] = Field(default=None, description="The customer's phone number or WhatsApp number if provided.")
    preferred_date_time: Optional[str] = Field(default=None, description="The preferred date and time for the test drive.")
    car_of_interest: Optional[str] = Field(default=None, description="The make and model of the car they want to test drive.")

SYSTEM_INSTRUCTION = """
You are an elite, highly persuasive automotive sales advisor for a premium car dealership.
Your goal is to enthusiastically assist customers, build rapport, highlight features, and secure test drives.

CRITICAL ROLES: 
- You are the SALES ADVISOR. 
- The user chatting with you is the CUSTOMER. 
- Do NOT invert these roles. Do NOT write messages pretending to be the customer scheduling a test drive (e.g. "I want to schedule a test drive"). Your responses must always be from the perspective of the advisor helper.

CONVERSATION FLOW:
1. First, answer the customer's question directly using the available inventory details. Highlight the car's best features, and sell it enthusiastically!
2. If they are checking stock or asking general questions, pitch the car and end by asking if they would like to schedule a test drive.
3. If (and ONLY if) the customer explicitly agrees to or asks for a test drive, warmly ask them to provide their Name, Phone number, and preferred Date and time so you can book it.
4. ONLY set the `wants_test_drive` flag to true and populate the fields (customer_name, customer_contact, preferred_date_time) once they have actually typed and provided those details in their message. Do NOT make up or guess these details.

PRICE BUDGET FILTERING (CRITICAL):
- Pay close attention to the customer's budget (e.g. "under 1 million", "below 3M").
- Examine the `price_ksh` field in the metadata for each car. Perform a strict mathematical comparison: only recommend cars that are strictly within their budget.
- If a customer asks for a car under 1 million, only the Suzuki Jimny (Ksh 250,000) is eligible. Recommending the Honda CR-V (Ksh 4,200,000) or Toyota Camry (Ksh 3,500,000) is a direct failure.
- If no cars in the context fit their budget, state clearly that we have no cars in that budget range, and present our cheapest option as an alternative.

You must strictly refrain from hallucinating inventory. If a customer asks for a car not in the context, politely pivot to a similar available model.
Format your response using clean Markdown with bolding and structured spacing. DO NOT embed markdown image links (e.g. ![image](url)) inside the message body.
Be charismatic, warm, and conversational. Do not just list bullet points—sell the car! Always end by asking an engaging closing question (e.g. asking to schedule a test drive).
If the matched car has an image_url, include it in the selected_image_url field. Otherwise, leave it empty.
CRITICAL: You must generate EXACTLY ONE response. Do NOT attempt to output multiple tool calls, even if multiple cars match the query. Synthesize your answer into a single message.
"""

async def generate_rag_response(
    message_body: str, 
    qdrant_client: AsyncQdrantClient, 
    openai_client: instructor.AsyncInstructor,
    history: list = None
) -> CarSalesResponse:
    """
    Core RAG logic independent of the delivery channel.
    """
    # Step 1: Embed the contextualized incoming message
    query_text = message_body
    if history and len(history) > 0:
        query_text = f"Previous context: {history[-1].content}\nUser Question: {message_body}"
        
    embeddings = list(embedding_model.embed([query_text]))
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

    try:
        # Step 4: Guardrailed Output using Instructor with OpenAI
        llm_messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        
        if history:
            for h in history:
                role = "assistant" if h.role == "bot" else "user"
                llm_messages.append({"role": role, "content": h.content})
                
        llm_messages.append({"role": "user", "content": prompt})
        
        logger.info(f"Sending messages to LLM: {llm_messages}")

        structured_response = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            response_model=CarSalesResponse,
            messages=llm_messages,
            temperature=0.2,
            max_retries=2
        )
        
        # Intercept Test Drive request
        logger.info(f"LLM returned structured_response: {structured_response.model_dump()}")
        if structured_response.wants_test_drive:
            name = structured_response.customer_name
            contact = structured_response.customer_contact
            time_pref = structured_response.preferred_date_time
            
            # Programmatic verification: Check if name and contact are actually present in the user's messages
            user_messages = [h.content for h in history if h.role == "user"] if history else []
            user_messages.append(message_body)
            combined_user_input = " ".join(user_messages).lower()
            
            name_present = name and name.lower() in combined_user_input
            contact_present = contact and contact.lower() in combined_user_input
            
            # If the name or contact is missing from what the user actually typed, it's a hallucination
            if not name_present or not contact_present:
                logger.warning(f"LLM triggered wants_test_drive prematurely or hallucinated details. Name present: {name_present}, Contact present: {contact_present}. Overriding to False.")
                structured_response.wants_test_drive = False
                structured_response.customer_name = None
                structured_response.customer_contact = None
                
                # Rewrite message to ask for the details if the LLM prematurely output a success response
                structured_response.message_body = "I'd love to schedule that test drive for you! Could you please share your name, phone number, and preferred date/time so I can get it all set up?"
            else:
                from database import add_lead
                add_lead(
                    customer_name=name,
                    customer_contact=contact,
                    car_of_interest=structured_response.car_of_interest or "Unknown",
                    preferred_date_time=time_pref or "Unknown"
                )
                logger.info("Lead successfully saved to CRM.")
                
                # Override the message body with a clean, professional, dynamic confirmation message
                car = structured_response.car_of_interest or "the vehicle"
                structured_response.message_body = (
                    f"Perfect, **{name}**! I have successfully registered your test drive request for the **{car}** on **{time_pref}**.\n\n"
                    f"✅ *Our showroom team will contact you shortly at **{contact}** to confirm your appointment. We look forward to showing you the car!*"
                )
            
        logger.info(f"LLM returned text: {structured_response.message_body}")
        logger.info(f"LLM returned image URL: {structured_response.selected_image_url}")
        return structured_response
    except Exception as e:
        logger.error(f"LLM Generation Error: {e}")
        return CarSalesResponse(
            message_body="I'm sorry, I encountered an error while searching the inventory. Could you please rephrase your request?",
            selected_image_url=""
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
