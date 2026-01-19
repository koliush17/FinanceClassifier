from pydantic import BaseModel, Field

class TransactionType(BaseModel):
    purpose_text: str = Field(description="Input text to classify")
