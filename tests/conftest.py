"""Shared test configuration."""

import jax

# The reference cases compare against float64 anchors; set precision once here,
# never at module import, so collection order cannot decide it.
JAX_ENABLE_X64 = True


def pytest_configure(config):
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)
