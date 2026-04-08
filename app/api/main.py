from contextlib import asynccontextmanager

import vendor_bootstrap  # noqa: F401
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import router
from graph.builder import reset_chat_graph
from graph.memory import memory_service
from infrastructure.logging.logger import logger
from infrastructure.tools.mcp.mcp_manager import mcp_cleanup, mcp_connect


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing memory backend...")
    try:
        await memory_service.initialize()
        logger.info("Memory backend ready: %s", memory_service.backend)
    except Exception as exc:
        logger.error("Memory backend initialization failed: %s", exc)

    logger.info("Connecting MCP services...")
    try:
        await mcp_connect()
        logger.info("MCP services connected")
    except Exception as exc:
        logger.error("MCP connection failed: %s", exc)

    yield

    logger.info("Cleaning up MCP services...")
    try:
        await mcp_cleanup()
        logger.info("MCP cleanup complete")
    except Exception as exc:
        logger.error("MCP cleanup failed: %s", exc)

    logger.info("Closing memory backend...")
    try:
        await memory_service.close()
        reset_chat_graph()
        logger.info("Memory backend closed")
    except Exception as exc:
        logger.error("Memory backend shutdown failed: %s", exc)


def create_fast_api() -> FastAPI:
    app = FastAPI(title="ITS API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router=router)
    return app


if __name__ == "__main__":
    try:
        uvicorn.run(app=create_fast_api(), host="127.0.0.1", port=8000)
    except KeyboardInterrupt as exc:
        logger.error("Web service stopped: %s", exc)
