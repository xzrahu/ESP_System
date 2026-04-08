from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import vendor_bootstrap  # noqa: F401

from config.settings import settings
from graph.memory import MemoryService


class _SettingsMixin:
    SETTING_NAMES = (
        "USE_POSTGRES_MEMORY",
        "POSTGRES_URI",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
        "POSTGRES_SSLMODE",
    )

    def setUp(self) -> None:
        super().setUp()
        self._settings_backup = {
            name: getattr(settings, name)
            for name in self.SETTING_NAMES
        }

    def tearDown(self) -> None:
        for name, value in self._settings_backup.items():
            setattr(settings, name, value)
        super().tearDown()


class SettingsTest(_SettingsMixin, unittest.TestCase):
    def test_postgres_conn_string_prefers_uri(self) -> None:
        settings.POSTGRES_URI = "postgresql://user:pass@host:5432/db"
        settings.POSTGRES_HOST = "should-not-be-used"
        self.assertEqual(settings.postgres_conn_string, "postgresql://user:pass@host:5432/db")

    def test_postgres_conn_string_builds_from_fields(self) -> None:
        settings.POSTGRES_URI = None
        settings.POSTGRES_HOST = "localhost"
        settings.POSTGRES_PORT = 5432
        settings.POSTGRES_USER = "postgres"
        settings.POSTGRES_PASSWORD = "secret"
        settings.POSTGRES_DATABASE = "its_memory"
        settings.POSTGRES_SSLMODE = "require"

        self.assertEqual(
            settings.postgres_conn_string,
            "postgresql://postgres:secret@localhost:5432/its_memory?sslmode=require",
        )


class MemoryServiceInitializationTest(_SettingsMixin, unittest.IsolatedAsyncioTestCase):
    async def test_initialize_uses_postgres_backend_when_configured(self) -> None:
        settings.USE_POSTGRES_MEMORY = True
        settings.POSTGRES_URI = "postgresql://postgres:secret@localhost:5432/its_memory"

        service = MemoryService()
        with (
            patch.object(service, "_initialize_postgres_backend", new=AsyncMock()) as init_pg,
            patch.object(service, "_create_memory_managers") as create_managers,
        ):
            await service.initialize()

        init_pg.assert_awaited_once()
        create_managers.assert_called_once()
        self.assertEqual(service.backend, "postgres")

    async def test_initialize_falls_back_to_in_memory_when_postgres_unconfigured(self) -> None:
        settings.USE_POSTGRES_MEMORY = True
        settings.POSTGRES_URI = None
        settings.POSTGRES_HOST = None
        settings.POSTGRES_USER = None
        settings.POSTGRES_DATABASE = None

        service = MemoryService()
        with patch.object(service, "_create_memory_managers") as create_managers:
            await service.initialize()

        create_managers.assert_called_once()
        self.assertEqual(service.backend, "memory")
        self.assertIsNotNone(service.checkpointer)
        self.assertIsNotNone(service.store)

    async def test_initialize_falls_back_when_postgres_setup_fails(self) -> None:
        settings.USE_POSTGRES_MEMORY = True
        settings.POSTGRES_URI = "postgresql://postgres:secret@localhost:5432/its_memory"

        service = MemoryService()
        with (
            patch.object(
                service,
                "_initialize_postgres_backend",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ) as init_pg,
            patch.object(service, "_build_store_index", return_value=None),
            patch.object(service, "_create_memory_managers") as create_managers,
        ):
            await service.initialize()

        init_pg.assert_awaited_once()
        create_managers.assert_called_once()
        self.assertEqual(service.backend, "memory")
        self.assertIsNotNone(service.checkpointer)
        self.assertIsNotNone(service.store)


if __name__ == "__main__":
    unittest.main()
