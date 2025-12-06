"""Run the FastAPI application."""

import uvicorn
from api.config import api_config

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=api_config.API_HOST,
        port=api_config.API_PORT,
        reload=api_config.API_RELOAD,
        workers=api_config.API_WORKERS if not api_config.API_RELOAD else 1,
        log_level="info",
        access_log=True,
    )
