# Persona Generator

A RAG-grounded chat interface for interviewing synthetic research personas derived from qualitative study data, built using the [PersonaCite](https://doi.org/10.1145/3772363.3798543) methodology and grounded in an [interview study](https://doi.org/10.1109/TVCG.2024.3456298) of genomics researchers. Each persona responds only when relevant evidence exists.

## Overview

The system constructs 7 synthetic evaluator personas across four groups — Biologists, Computational Biologists, Bioinformaticians, and Software Engineers — based on participant data from a qualitative interview study. When asked a question, each persona retrieves semantically similar quotes from its evidence pool and generates a response grounded in that evidence.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API key
```

Configure your LLM provider in `.env`:

```
OPENAI_API_KEY=your-key-here
# or
ANTHROPIC_API_KEY=your-key-here
LLM_PROVIDER=openai  # or anthropic
```

## Running the chat interface

```bash
python server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser. API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

## Pipeline

The pipeline scripts build the persona data and evidence store from source files:

| Script | Description |
|--------|-------------|
| `step1_parse_personas.py` | Parse `personas.xlsx` → `data/personas.json` |
| `step2_parse_evidence.py` | Extract quotes and codes from interview transcripts |
| `step3_retrieve.py` | Embed evidence and build the retrieval index |
| `step4_build_prompts.py` | Construct system prompts and evidence blocks |
| `step4b_validate.py` | Filter retrieved evidence by relevance to the question |
| `step5_chat.py` | CLI chat interface |
| `step6_test.py` | Batch question testing |
| `step7_interview.py` | Structured interview runner |
| `step8_evaluators.py` | Define the 7 evaluator personas |

## Persona groups

- **Biologist (1)** — biology focus, low automation, self audience
- **Computational Biologist (2)** — moderate vs. high programming/visualization skills
- **Bioinformatician (2)** — expert scientist+engineer vs. tool-builder engineer
- **Software Engineer (2)** — moderate genomics vs. domain-aware high visualization

## Architecture

- **Retrieval**: sentence-transformers embeddings with cosine similarity; abstains if no evidence clears the similarity threshold
- **Validation**: a fast LLM call filters retrieved quotes to those that actually support the question before generation
- **Generation**: OpenAI or Anthropic, switchable at request time via `provider` field
- **Frontend**: single-page app served from `static/`, streaming responses via SSE
