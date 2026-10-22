import os

import uvicorn

from .api import create_app

uvicorn.run(create_app(), host=os.environ.get("OPENJEV_HOST", "127.0.0.1"), port=int(os.environ.get("OPENJEV_PORT", "8080")),
            log_level=os.environ.get("OPENJEV_LOG_LEVEL", "info"))
