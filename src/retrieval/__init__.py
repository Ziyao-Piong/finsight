"""Retrieval layer: metadata-filtered semantic search over the filing corpus.

This package is the single owner of vector-store access (``get_vectorstore``) and
turns a question + explicit filters into structured, citable results. Phase 4's
agent will decide *which* filters to pass; Phase 3 just retrieves and cites.
"""
