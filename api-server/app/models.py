from pydantic import BaseModel
import uuid

class QuestionRequest(BaseModel):
    session_id: str = str(uuid.uuid4())  # Generate a new UUID if not provided
    question: str
    answer: str
