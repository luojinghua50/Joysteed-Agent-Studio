import uvicorn

from src.api.routes import create_app
from src.config import Settings

settings = Settings()
app = create_app(settings)

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
