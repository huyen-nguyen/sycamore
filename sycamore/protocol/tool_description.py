"""Textual description of Geranium, used in lieu of a live demonstration.

The Sycamore manuscript (Section 3.2) states:
  "We retain this protocol for the synthetic conditions with one
   adjustment: the tool demonstration is replaced by a textual
   description of Geranium's features and capabilities, since synthetic
   personas cannot observe a live demonstration."
"""

GERANIUM_TOOL_DESCRIPTION = """\
Geranium is a multimodal retrieval system for genomic data visualizations.
It indexes ~3,200 visualizations across ~50 categories. Each visualization is
stored as a triplet:

  - image      : a rendered PNG of the chart
  - text       : a natural-language description (AltGosling + LLM)
  - spec       : a Gosling specification (JSON) that can be edited and
                 re-rendered as a starting template for new charts

You can query Geranium in three modalities:

  1. TEXT modality: type a natural-language description of what you want to
     visualize (e.g., "circular ideogram with copy number variants").

  2. IMAGE modality: upload an example chart (PNG); Geranium retrieves the
     most visually similar templates from the index using BiomedCLIP image
     embeddings.

  3. SPEC modality: paste a Gosling specification (JSON) you already have;
     Geranium retrieves the most structurally similar specifications using
     a frequency-count embedding over Gosling spec rules.

Results: for each query, Geranium returns the top-k matching triplets
{image, text, spec}, ranked by cosine similarity. You can:

  - Inspect any result's image, text, or spec directly.
  - Browse a curated gallery of representative charts to orient yourself
    before starting a query.
  - Pick a result whose spec is closest to your goal and modify it as a
    template, rather than authoring a Gosling spec from scratch.

The intended workflow: search -> pick a near-match template -> adapt it.
The system does not author visualizations from scratch; it retrieves the
nearest existing exemplars and lets you start from one.
"""
