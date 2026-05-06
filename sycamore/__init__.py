"""Sycamore: a three-condition probe of synthetic personas evaluating Geranium.

The persona-generator engine (step1..step8 + llm_provider) lives at the project
root and is used unmodified. This package wraps it with:

  - a Geranium HTTP client (search/gallery against the real retrieval server)
  - a three-part user-study protocol runner (workflow -> tool -> exploration)
  - condition builders (ungrounded vs grounded) that share the runner
  - cross-condition analysis (modality ranking aggregates + theme-alignment scaffold)
  - a Sycamore FastAPI interface for single-evaluator session walkthroughs
"""
__version__ = "0.1.0"
