from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_DETAILS = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = AsyncIOMotorClient(MONGO_DETAILS)
database = client["questionsDB"]  # Create or connect to the database
questions_collection = database.get_collection("questions")  # Create or connect to the collection


async def save_question_to_db(session_id: str, user_input: str, response: str):
    # Create a QuestionRequest object and insert it into MongoDB
    question_document = {
        "session_id": session_id,
        "question": user_input,
        "answer": response,
        "timestamp": datetime.utcnow()  # Optional timestamp
    }
    await questions_collection.insert_one(question_document)