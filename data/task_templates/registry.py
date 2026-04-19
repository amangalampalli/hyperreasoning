"""Central registry for available task template families."""

from __future__ import annotations

from .base import TaskTemplate
from .families import (
    AsyncRetryContractTemplate,
    ASTTransformScopeBugTemplate,
    CacheInvalidationDependencyTemplate,
    ConcurrencySafeMemoizationTemplate,
    DescriptorPropertyMROTemplate,
    IncrementalBuildGraphBugTemplate,
    MultiFileInterfaceDriftTemplate,
    SerializerRoundtripEscapeTemplate,
    StatefulIteratorResumeBugTemplate,
    StreamingParserReentrancyTemplate,
)


registry: dict[str, TaskTemplate] = {
    template.family: template
    for template in [
        StreamingParserReentrancyTemplate(),
        AsyncRetryContractTemplate(),
        ASTTransformScopeBugTemplate(),
        CacheInvalidationDependencyTemplate(),
        DescriptorPropertyMROTemplate(),
        IncrementalBuildGraphBugTemplate(),
        SerializerRoundtripEscapeTemplate(),
        StatefulIteratorResumeBugTemplate(),
        MultiFileInterfaceDriftTemplate(),
        ConcurrencySafeMemoizationTemplate(),
    ]
}


def get_template(family: str) -> TaskTemplate:
    """Look up a registered task template by family name."""

    try:
        return registry[family]
    except KeyError as exc:
        raise KeyError(f"Unknown task family: {family}") from exc


def list_families() -> list[str]:
    """List registered family names in stable order."""

    return sorted(registry)
