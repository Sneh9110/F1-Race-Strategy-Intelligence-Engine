"""Run the FastAPI application."""

import uvicorn
from api.config import api_config
from app.utils.logger import setup_logging
from config.settings import settings

# Setup logging before starting the server
setup_logging(
    level=settings.logging.level,
    format_type=settings.logging.format,
    log_file=settings.logging.file
)

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=api_config.API_HOST,
        port=api_config.API_PORT,
        reload=api_config.API_RELOAD,
        workers=api_config.API_WORKERS if not api_config.API_RELOAD else 1,
        log_level=settings.logging.level.lower(),
        access_log=True,
    )
