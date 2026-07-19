"""Attribution methods: which passages did the answer depend on?

Importing this package registers all built-in methods, controls, and any optional
adapters whose third-party dependency is installed.
"""

from rag_evidence.attribution import (  # noqa: F401
    adapters,
    citation,
    controls,
    embedding,
    leave_one_out,
)
