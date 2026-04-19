"""Canonical environment package for DSL search control.

Keep this module intentionally minimal to avoid circular imports between
dataset/task-loading utilities and the environment runtime surface.
Import concrete symbols from submodules such as:

- ``env.dsl_env``
- ``env.dsl_schema``
- ``env.state_encoder``
- ``env.rewards``
- ``env.verifier``
"""
