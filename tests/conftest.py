"""Shared pytest fixtures and marker registration."""
import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test needs a live QuantX Dataloader / Massive API key"
    )


@pytest.fixture(scope="session")
def has_market_data_keys() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))
