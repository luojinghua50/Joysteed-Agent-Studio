import uvicorn
from src.api import create_app
from src.config import RAGSettings

settings = RAGSettings()
app = create_app(settings)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8010, reload=True)
