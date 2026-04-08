from __future__ import annotations

import asyncio
from typing import Any

import vendor_bootstrap  # noqa: F401
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from graph.models import build_sub_chat_model
from infrastructure.logging.logger import logger

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # type: ignore

from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

try:  # pragma: no cover - exercised by integration/runtime use
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from psycopg_pool import AsyncConnectionPool

    HAS_POSTGRES_MEMORY = True
except ImportError:  # pragma: no cover
    AsyncPostgresSaver = None  # type: ignore[assignment]
    AsyncPostgresStore = None  # type: ignore[assignment]
    AsyncConnectionPool = None  # type: ignore[assignment]
    HAS_POSTGRES_MEMORY = False


class MemoryService:
    """Owns LangGraph short-term memory and LangMem long-term memory."""

    PROFILE_NAMESPACE = ("users", "{user_id}", "profile")
    SEMANTIC_NAMESPACE = ("users", "{user_id}", "semantic")

    def __init__(self) -> None:
        self.checkpointer: Any | None = None
        self.store: Any | None = None
        self.profile_manager: Any | None = None
        self.semantic_manager: Any | None = None

        self._pool: AsyncConnectionPool | None = None
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._backend = "uninitialized"

    @property
    def backend(self) -> str:
        return self._backend

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            if self._should_use_postgres():
                try:
                    await self._initialize_postgres_backend()
                    self._backend = "postgres"
                    logger.info("Memory backend initialized with PostgreSQL.")
                except Exception as exc:
                    logger.warning(
                        "Initializing PostgreSQL memory backend failed, falling back to in-memory: %s",
                        exc,
                    )
                    await self._close_pool()
                    self._initialize_in_memory_backend()
            else:
                self._initialize_in_memory_backend()

            self._create_memory_managers()
            self._initialized = True

    async def close(self) -> None:
        async with self._init_lock:
            await self._close_pool()
            self.checkpointer = None
            self.store = None
            self.profile_manager = None
            self.semantic_manager = None
            self._initialized = False
            self._backend = "uninitialized"

    async def recall_memories(self, user_id: str, query: str) -> str:
        await self.initialize()

        config = {"configurable": {"user_id": user_id}}
        profile_items = await self._search_manager(
            self.profile_manager,
            query=query,
            limit=3,
            config=config,
        )
        semantic_items = await self._search_manager(
            self.semantic_manager,
            query=query,
            limit=5,
            config=config,
        )

        sections: list[str] = []
        if profile_items:
            sections.append("User profile:\n" + "\n".join(f"- {item}" for item in profile_items))
        if semantic_items:
            sections.append("Long-term memory:\n" + "\n".join(f"- {item}" for item in semantic_items))
        return "\n\n".join(sections)

    async def _search_manager(
        self,
        manager: Any,
        *,
        query: str,
        limit: int,
        config: dict[str, Any],
    ) -> list[str]:
        if manager is None:
            return []

        try:
            results = await manager.search(
                query=query,
                limit=limit,
                config=config,
            )
        except Exception as exc:
            logger.warning("Search memory manager failed: %s", exc)
            return []

        serialized: list[str] = []
        for item in results:
            value = getattr(item, "value", None)
            if value is None and isinstance(item, dict):
                value = item.get("value")
            if isinstance(value, dict):
                text = value.get("content") or value.get("text") or str(value)
            else:
                text = str(value)
            if text:
                serialized.append(text)
        return serialized

    def schedule_memory_write(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[BaseMessage],
    ) -> None:
        payload = {"messages": messages}
        config = {"configurable": {"user_id": user_id, "session_id": session_id}}

        async def runner() -> None:
            await self.initialize()
            try:
                if self.profile_manager is not None:
                    await self.profile_manager.ainvoke(payload, config=config)
                if self.semantic_manager is not None:
                    await self.semantic_manager.ainvoke(payload, config=config)
            except Exception as exc:
                logger.warning("Persisting long-term memory failed: %s", exc)

        task = asyncio.create_task(runner())
        task.add_done_callback(lambda fut: fut.exception())

    def _should_use_postgres(self) -> bool:
        return bool(
            settings.USE_POSTGRES_MEMORY
            and settings.postgres_conn_string
            and HAS_POSTGRES_MEMORY
        )

    async def _initialize_postgres_backend(self) -> None:
        if AsyncConnectionPool is None or AsyncPostgresSaver is None or AsyncPostgresStore is None:
            raise RuntimeError("PostgreSQL memory dependencies are not installed.")

        pool = AsyncConnectionPool(
            conninfo=settings.postgres_conn_string,
            min_size=settings.POSTGRES_POOL_MIN_SIZE,
            max_size=settings.POSTGRES_POOL_MAX_SIZE,
            timeout=settings.POSTGRES_POOL_TIMEOUT,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            },
        )
        await pool.open()

        checkpointer = AsyncPostgresSaver(pool)
        store = AsyncPostgresStore(pool, index=self._build_store_index())

        try:
            await checkpointer.setup()
            await store.setup()
        except Exception:
            await pool.close()
            raise

        self._pool = pool
        self.checkpointer = checkpointer
        self.store = store

    def _initialize_in_memory_backend(self) -> None:
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore(index=self._build_store_index())
        self._backend = "memory"
        logger.info("Memory backend initialized with in-memory fallback.")

    async def _close_pool(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _create_memory_managers(self) -> None:
        if self.store is None:
            raise RuntimeError("Memory store is not initialized.")

        manager_model = build_sub_chat_model(streaming=False)
        self.profile_manager = create_memory_store_manager(
            manager_model,
            store=self.store,
            namespace=self.PROFILE_NAMESPACE,
            instructions=(
                "Maintain a concise user profile. Keep only stable preferences, "
                "device information, location preferences, and response style hints."
            ),
            enable_inserts=False,
        )
        self.semantic_manager = create_memory_store_manager(
            manager_model,
            store=self.store,
            namespace=self.SEMANTIC_NAMESPACE,
            instructions=(
                "Store durable user facts and ongoing issues that should persist "
                "across sessions, such as active troubleshooting topics or long-lived goals."
            ),
            enable_inserts=True,
        )

    def _build_store_index(self) -> dict[str, Any]:
        return {
            "dims": settings.MEMORY_EMBEDDING_DIMS,
            "embed": self._build_embeddings(),
            "fields": ["$"],
        }

    def _build_embeddings(self) -> OpenAIEmbeddings:
        base_url = settings.MEMORY_EMBEDDING_BASE_URL or settings.SF_BASE_URL or settings.AL_BAILIAN_BASE_URL
        api_key = settings.MEMORY_EMBEDDING_API_KEY or settings.SF_API_KEY or settings.AL_BAILIAN_API_KEY
        return OpenAIEmbeddings(
            model=settings.MEMORY_EMBEDDING_MODEL,
            base_url=base_url,
            api_key=api_key,
        )

    @staticmethod
    def build_summary_candidate(
        messages: list[BaseMessage],
        max_messages: int = 12,
    ) -> list[BaseMessage]:
        visible_messages = [
            message
            for message in messages
            if isinstance(message, (HumanMessage, AIMessage))
        ]
        return visible_messages[:-max_messages] if len(visible_messages) > max_messages else []


memory_service = MemoryService()
