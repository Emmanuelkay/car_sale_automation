import asyncio
import logging
from uuid import uuid4
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ingest_data():
    """
    Creates a Qdrant collection and ingests sample car inventory data.
    """
    qdrant_client = AsyncQdrantClient(
        path=settings.QDRANT_PATH
    )

    collection_name = settings.QDRANT_COLLECTION_NAME

    # BAAI/bge-small-en-v1.5 uses 384 dimensions
    vector_size = 384
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5") 

    # Recreate collection to wipe old fake URL records
    if await qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(f"Deleting old collection {collection_name}...")
        await qdrant_client.delete_collection(collection_name=collection_name)
        
    logger.info(f"Creating collection {collection_name}...")
    await qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Sample inventory
    inventory = [
        {
            "id": str(uuid4()),
            "listing_details": "2023 Toyota Camry SE. Sleek gray exterior, well-maintained, reliable sedan perfect for daily commuting. Features a spacious interior and advanced safety systems.",
            "metadata": {
                "make": "Toyota",
                "model": "Camry",
                "price_ksh": 3500000,
                "year": 2023,
                "mileage_km": 15000,
                "image_url": "http://localhost:8000/images/camry.png",
                "images": {
                    "exterior": "http://localhost:8000/images/camry.png",
                    "interior": "http://localhost:8000/images/camry_interior.png",
                    "dashboard": "http://localhost:8000/images/camry_dashboard.png"
                },
                "status": "available"
            }
        },
        {
            "id": str(uuid4()),
            "listing_details": "2021 Honda CR-V EX-L. Blue SUV with leather seats, sunroof, and all-wheel drive. Great family car with excellent fuel economy and a smooth ride.",
            "metadata": {
                "make": "Honda",
                "model": "CR-V",
                "price_ksh": 4200000,
                "year": 2021,
                "mileage_km": 30000,
                "image_url": "http://localhost:8000/images/crv.png",
                "images": {
                    "exterior": "http://localhost:8000/images/crv.png"
                },
                "status": "available"
            }
        },
        {
            "id": str(uuid4()),
            "listing_details": "2020 BMW 3 Series 330i. Luxurious black sedan, sport package, premium sound system. Thrilling performance and high-end features.",
            "metadata": {
                "make": "BMW",
                "model": "3 Series",
                "price_ksh": 5800000,
                "year": 2020,
                "mileage_km": 45000,
                "image_url": "http://localhost:8000/images/bmw.png",
                "status": "sold" # Should be filtered out by worker
            }
        }
    ]

    points = []
    for item in inventory:
        # Generate embedding
        logger.info(f"Embedding item {item['metadata']['make']} {item['metadata']['model']}...")
        embeddings = list(embedding_model.embed([item["listing_details"]]))
        vector = embeddings[0].tolist()
        
        # Create PointStruct
        point = PointStruct(
            id=item["id"],
            vector=vector,
            payload={
                "listing_details": item["listing_details"],
                "metadata": item["metadata"],
                "status": item["metadata"]["status"]
            }
        )
        points.append(point)

    logger.info("Upserting to Qdrant...")
    await qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    logger.info("Ingestion complete!")
    await qdrant_client.close()

if __name__ == "__main__":
    asyncio.run(ingest_data())
