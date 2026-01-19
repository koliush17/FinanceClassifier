from contextlib import asynccontextmanager
from fastapi import FastAPI

from mlproject.utils.load_model import load_model
from mlproject.api import classify, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield 

app = FastAPI(lifespan=lifespan)
app.include_router(classify.router)
app.include_router(health.router)





