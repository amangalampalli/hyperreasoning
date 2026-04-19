"""Concrete task template families."""

from .async_retry_contract import AsyncRetryContractTemplate
from .ast_transform_scope_bug import ASTTransformScopeBugTemplate
from .cache_invalidation_dependency import CacheInvalidationDependencyTemplate
from .concurrency_safe_memoization import ConcurrencySafeMemoizationTemplate
from .descriptor_property_mro import DescriptorPropertyMROTemplate
from .incremental_build_graph_bug import IncrementalBuildGraphBugTemplate
from .multi_file_interface_drift import MultiFileInterfaceDriftTemplate
from .serializer_roundtrip_escape import SerializerRoundtripEscapeTemplate
from .stateful_iterator_resume_bug import StatefulIteratorResumeBugTemplate
from .streaming_parser_reentrancy import StreamingParserReentrancyTemplate

__all__ = [
    "AsyncRetryContractTemplate",
    "ASTTransformScopeBugTemplate",
    "CacheInvalidationDependencyTemplate",
    "ConcurrencySafeMemoizationTemplate",
    "DescriptorPropertyMROTemplate",
    "IncrementalBuildGraphBugTemplate",
    "MultiFileInterfaceDriftTemplate",
    "SerializerRoundtripEscapeTemplate",
    "StatefulIteratorResumeBugTemplate",
    "StreamingParserReentrancyTemplate",
]
