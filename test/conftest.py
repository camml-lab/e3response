from jax import random
import pytest


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)
