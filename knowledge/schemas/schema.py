from pydantic import BaseModel


class UploadResponse(BaseModel):
    status: str
    message: str
    file_name: str
    chunks_added: int


class QueryResponse(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str
