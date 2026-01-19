from fastapi import APIRouter, Request
from mlproject.schemas.transaction_schema import TransactionType

from typing import Dict

router = APIRouter(tags=["Classification"])

@router.post("/classify")
async def classify_transation(data: TransactionType, request: Request) -> Dict:
    model = request.app.state.model
    preds = model.predict(data.purpose_text)

    return {"transaction_type": preds[0]}
