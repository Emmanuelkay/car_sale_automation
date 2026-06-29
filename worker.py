import logging
import httpx
from pydantic import BaseModel, Field
from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import instructor
from fastembed import TextEmbedding

from config import settings

import re
from pydantic import BaseModel, Field, field_validator

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

    @field_validator("customer_contact")
    @classmethod
    def validate_phone_number(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        
        # Strip spaces, dashes, and parenthetical symbols
        clean_num = re.sub(r"[\s\-\(\)]", "", v)
        
        # Kenyan context validation (supports 07..., 01..., or +254...)
        # Rejects obvious emergency numbers, shortcodes, or too-short inputs
        if len(clean_num) < 9 or clean_num in ["911", "999", "112"]:
            raise ValueError("Invalid phone number format")
            
        return clean_num

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

OUT-OF-DOMAIN GUARDRAIL (CRITICAL):
- If the customer asks questions completely unrelated to cars, test drives, purchasing, or Elite Auto (e.g. asking for code, cooking recipes, general trivia, trivia questions, etc.):
  1. Politely and humorously pivot back to cars (e.g., "While I'd love to help you cook that, I'm here to help you get behind the wheel of a premium car!").
  2. Do NOT answer their out-of-domain request. Always steer them back to our available inventory or scheduling a test drive.

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

def needs_inventory_search(user_message: str, chat_history: list) -> bool:
    """
    Smart RAG Bypass: Returns True if we should query Qdrant for car data,
    and False if the message is purely conversational/booking-related.
    """
    user_msg_clean = user_message.strip().lower()
    word_count = len(user_msg_clean.split())
    
    # 1. Hard Explicit Keywords: Always trigger search if these are present
    car_keywords = ["honda", "crv", "cr-v", "jimny", "camry", "suzuki", "toyota", "mercedes", "gle", "price", "cost", "under", "below", "cheapest", "expensive", "specs", "specification", "features", "details", "mileage"]
    if any(keyword in user_msg_clean for keyword in car_keywords):
        return True
        
    # 2. Hard Conversational Skips: If it's a short agreement word, skip immediately
    booking_signals = ["yes", "sure", "okay", "ok", "fine", "sounds good", "let's do it", "tomorrow"]
    if user_msg_clean in booking_signals or (word_count < 4 and any(b in user_msg_clean for b in booking_signals)):
        return False

    # 3. CONTEXT WINDOW CHECK (The Fix for "specs?"):
    if word_count < 4:
        # Get the AI's very last message
        last_ai_message = ""
        for message in reversed(chat_history):
            role = message.role if hasattr(message, 'role') else message.get('role', '')
            if role in ["assistant", "bot"]:
                content = message.content if hasattr(message, 'content') else message.get('content', '')
                last_ai_message = content.lower()
                break
        
        # Follow-up specifications keywords
        spec_triggers = ["spec", "specs", "specification", "specifications", "mileage", "color", "colour", "year", "engine", "details", "photo", "photos", "picture", "pictures", "image", "images"]
        if any(trigger in user_msg_clean for trigger in spec_triggers):
            return True
            
        # If the AI was just talking about a specific inventory car, keep the context alive
        if any(car in last_ai_message for car in ["honda", "cr-v", "jimny", "camry"]):
            return True

    # Default fallback
    return word_count >= 4

async def generate_rag_response(
    message_body: str, 
    qdrant_client: AsyncQdrantClient, 
    openai_client: instructor.AsyncInstructor,
    history: list = None
) -> CarSalesResponse:
    """
    Two-Pass Conversational Routing and RAG pipeline.
    """
    # ==========================================
    # PASS 1: Silent Lead Parser (Background Structured Extraction)
    # ==========================================
    # We construct the conversation history log for the parser
    recent_history_msgs = []
    if history:
        for h in history[-2:]:
            role = "assistant" if h.role == "bot" else "user"
            recent_history_msgs.append(f"{role}: {h.content}")
    recent_history_msgs.append(f"user: {message_body}")
    conversation_log = "\n".join(recent_history_msgs)

    nudge_message = ""
    lead_saved = False
    name = None
    contact = None
    time_pref = None
    vehicle = None
    is_providing_lead_details = False

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
        logger.info(f"Pass 1 Extraction Output: {raw_extraction.model_dump()}")

        name = raw_extraction.customer_name
        contact = raw_extraction.customer_contact
        time_pref = raw_extraction.preferred_date_time
        vehicle = raw_extraction.preferred_vehicle

        # If the LLM successfully parsed any lead details, flag it to skip RAG search
        if name or contact or time_pref or vehicle:
            is_providing_lead_details = True

        if raw_extraction.ready_for_booking:
            # Programmatic verification: Check if name and contact are actually present in the user's messages
            user_messages = [h.content for h in history if h.role == "user"] if history else []
            user_messages.append(message_body)
            combined_user_input = " ".join(user_messages).lower()
            
            name_present = name and name.lower() in combined_user_input
            
            # Robust digit-only check for phone presence (e.g. comparing last 9 digits to ignore prefixes)
            contact_present = False
            if contact:
                import re
                contact_digits = re.sub(r"\D", "", contact)
                user_digits = re.sub(r"\D", "", combined_user_input)
                if contact_digits:
                    core_digits = contact_digits[-9:]
                    if core_digits in user_digits:
                        contact_present = True

            if not name_present or not contact_present:
                logger.warning(f"Pass 1 verification check failed. Hallucinated Name: {name} (Present: {name_present}), Contact: {contact} (Present: {contact_present})")
                nudge_message = "\n\nNote: Do NOT confirm booking. The user has not provided their name or contact info yet. Politely ask them for their details."
            else:
                # Programmatically resolve vehicle of interest if LLM hallucinated it (e.g. BMW)
                user_and_bot_history = [h.content.lower() for h in history] if history else []
                # Check both history and the current user message to see if the vehicle was mentioned
                combined_history_text = " ".join(user_and_bot_history)
                full_chat_text = combined_history_text + " " + message_body.lower()
                
                resolved_vehicle = vehicle
                vehicle_lower = vehicle.lower() if vehicle else ""
                
                # Check for standard model keywords to normalize shorthand inputs
                is_partial = vehicle_lower in ["honda", "toyota", "suzuki", "camry", "jimny", "crv", "cr-v"]
                
                # If the extracted vehicle is None, not in the conversation, or is a shorthand, look it up / normalize it
                if not vehicle or vehicle_lower not in full_chat_text or is_partial:
                    for car_key in ["camry", "toyota", "jimny", "suzuki", "cr-v", "crv", "honda"]:
                        if car_key in full_chat_text:
                            if "camry" in car_key or "toyota" in car_key:
                                resolved_vehicle = "Toyota Camry SE"
                                break
                            elif "jimny" in car_key or "suzuki" in car_key:
                                resolved_vehicle = "Suzuki Jimny"
                                break
                            elif "cr-v" in car_key or "crv" in car_key or "honda" in car_key:
                                resolved_vehicle = "2021 Honda CR-V EX-L"
                                break
                vehicle = resolved_vehicle or "Unknown Vehicle"

                from database import add_lead
                add_lead(
                    customer_name=name,
                    customer_contact=contact,
                    car_of_interest=vehicle,
                    preferred_date_time=time_pref or "Unknown"
                )
                logger.info("Lead successfully saved to CRM in Pass 1.")
                lead_saved = True
                nudge_message = f"\n\nNote: The test drive has been successfully booked for {name} on {time_pref}. Warmly confirm the booking."
    except Exception as e:
        logger.error(f"Pass 1 structured extraction error: {e}")
        # Detect Pydantic validation error or ValueError
        err_msg = str(e).lower()
        if "validation" in err_msg or "valueerror" in err_msg or "value_error" in err_msg or "phone number" in err_msg:
            is_providing_lead_details = True  # Still skip RAG search since they tried to book
            nudge_message = "\n\nNote: The customer tried to book but provided an invalid phone number. Politely ask them to provide a valid contact phone number so our team can confirm the booking."
        else:
            nudge_message = "\n\nNote: If the customer is trying to book, make sure to collect their name, phone number, and preferred date/time first."

    needs_search = needs_inventory_search(message_body, history or [])
    # If they are providing booking details and NOT asking a spec/inventory question, skip RAG search to avoid distraction
    if is_providing_lead_details and not needs_search:
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
    # PASS 2: Conversational Voice (Plain Text Response)
    # ==========================================
    # Inject the lead extraction status nudge into the conversational agent's instructions
    custom_system_prompt = SYSTEM_INSTRUCTION + nudge_message

    llm_messages = [{"role": "system", "content": custom_system_prompt}]
    if history:
        # Prevent context drift and token overflow by limiting memory to the last 10 messages
        for h in history[-10:]:
            role = "assistant" if h.role == "bot" else "user"
            llm_messages.append({"role": role, "content": h.content})
    llm_messages.append({"role": "user", "content": prompt})

    try:
        # Use raw client for Pass 2 (natural voice)
        chat_completion = await openai_client.client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            messages=llm_messages,
            temperature=0.3,
            frequency_penalty=0.8,
            presence_penalty=0.6
        )
        ai_response_text = chat_completion.choices[0].message.content
        logger.info(f"Pass 2 Response: {ai_response_text}")
    except Exception as e:
        logger.error(f"Pass 2 generation error: {e}")
        ai_response_text = "I'm sorry, I encountered an error while searching the inventory. Could you please rephrase your request?"

    # DYNAMIC IMAGE RESOLUTION (Context-Aware Multi-Image Selector)
    selected_image_url = ""
    lower_response = ai_response_text.lower()
    lower_user_msg = message_body.lower()
    combined_query = lower_user_msg + " " + lower_response

    if search_results:
        for hit in search_results:
            metadata = hit.payload.get("metadata", {})
            make = metadata.get("make", "").lower()
            model = metadata.get("model", "").lower()
            if make in lower_response or model in lower_response:
                images_dict = metadata.get("images", {})
                
                # Check message context to dynamically pick interior vs dashboard vs exterior
                if images_dict:
                    if any(word in combined_query for word in ["interior", "inside", "seat", "cabin", "backseat"]):
                        selected_image_url = images_dict.get("interior", "")
                    elif any(word in combined_query for word in ["dashboard", "cockpit", "wheel", "steering", "screen", "instrument"]):
                        selected_image_url = images_dict.get("dashboard", "")
                    
                    # Fallback to exterior if no specific context matches
                    if not selected_image_url:
                        selected_image_url = images_dict.get("exterior", "")
                
                # Backwards-compatible fallback to image_url
                if not selected_image_url:
                    selected_image_url = metadata.get("image_url", "")
                break

    # If lead was successfully saved, override with the perfect confirmation message
    if lead_saved:
        ai_response_text = (
            f"Perfect, **{name}**! I have successfully registered your test drive request for the **{vehicle or 'vehicle'}** on **{time_pref or 'your requested time'}**.\n\n"
            f"✅ *Our showroom team will contact you shortly at **{contact}** to confirm your appointment. We look forward to showing you the car!*"
        )
    else:
        # Programmatic check to ensure the LLM never falsely confirms a booking if lead_saved is False
        lower_response = ai_response_text.lower()
        if any(word in lower_response for word in ["confirm", "book", "register", "success", "received"]):
            if "phone number" in nudge_message.lower():
                name_str = f" {name}" if name else ""
                ai_response_text = f"Thanks{name_str}! I see your details, but the phone number provided is invalid. Could you please share a valid phone number so our team can reach out to confirm your test drive?"
            else:
                ai_response_text = "I'd love to schedule that test drive for you! Could you please share your name, phone number, and preferred date/time so I can get it all set up?"

    return CarSalesResponse(
        message_body=ai_response_text,
        selected_image_url=selected_image_url
    )


class ChatMessage(BaseModel):
    role: str
    content: str


async def process_whatsapp_message(
    phone_number: str, 
    message_body: str, 
    qdrant_client: AsyncQdrantClient, 
    openai_client: instructor.AsyncInstructor
):
    """
    Executes the decoupled async workflow for WhatsApp gateway delivery.
    Persists and retrieves conversation history to support multi-turn RAG memory.
    """
    try:
        logger.info(f"--- NEW WHATSAPP MESSAGE ---")
        logger.info(f"From: {phone_number} | Message: {message_body}")
        
        # Save incoming user message to CRM DB
        from database import save_chat_message, get_chat_history
        save_chat_message(phone_number, "user", message_body)
        
        # Load last 10 messages of history (includes the one we just saved)
        raw_history = get_chat_history(phone_number, limit=10)
        # Convert to ChatMessage schema, excluding the last message (which is current message_body)
        history_objs = [
            ChatMessage(role=h["role"], content=h["content"]) 
            for h in raw_history[:-1]
        ]
        
        structured_response = await generate_rag_response(
            message_body=message_body, 
            qdrant_client=qdrant_client, 
            openai_client=openai_client,
            history=history_objs
        )

        # Save outgoing bot response to CRM DB
        save_chat_message(phone_number, "bot", structured_response.message_body)
 
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
