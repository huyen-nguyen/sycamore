# Sycamore

**Synthetic Characterization for Evaluating Genomics Visualization Retrieval.**

A three-condition probe of how synthetic personas evaluate the [Geranium](https://github.com/gosling-lang/geranium) retrieval system, built on top of the [persona-generator](./PERSONA_GENERATOR.md) (PersonaCite-grounded synthetic personas of genomics researchers).

```
┌────────────────────────────────────────────────────────────────────────┐
│  Sycamore (this layer)                                                 │
│   • Geranium HTTP client (search, gallery)                             │
│   • 3-part user-study protocol runner                                  │
│   • condition builders: ungrounded vs grounded                         │
│   • cross-condition analysis (ranking aggregates, theme alignment)     │
│   • Sycamore FastAPI session interface                                 │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │  uses
┌──────────────────────────▼─────────────────────────────────────────────┐
│  persona-generator (engine, used unmodified)                           │
│   step1_parse_personas.py     personas.xlsx -> data/personas.json      │
│   step2_parse_evidence.py     ATLAS.ti     -> data/evidence.json       │
│   step3_retrieve.py           sentence-transformers + cosine retrieval │
│   step4_build_prompts.py      profile + grounding-rules prompts        │
│   step4b_validate.py          relevance filter (structured output)     │
│   step5..7                    chat / batch-test / structured interview │
│   step8_evaluators.py         the 7 EVALUATORS (Bio, CB1/2, BIF1/2…)   │
│   llm_provider.py             Anthropic / OpenAI provider abstraction  │
│   server.py                   reference chat UI (one persona at a time)│
└────────────────────────────────────────────────────────────────────────┘
```

The persona-generator is the engine. Sycamore is the protocol layer that drives one of its `EVALUATORS` (or, in the ungrounded condition, no evaluator at all) through the same three-part protocol that the published Geranium user study used with human participants.

## Quick start

```bash
# 1. Install deps (engine + Sycamore)
pip install -r requirements.txt

# 2. Configure your provider (engine uses python-dotenv)
cp .env.example .env 2>/dev/null || true
echo "ANTHROPIC_API_KEY=sk-..."           >> .env
echo "LLM_PROVIDER=anthropic"             >> .env

# 3. One-time: build the persona / evidence / embedding artefacts
python step1_parse_personas.py        # -> data/personas.json
python step2_parse_evidence.py        # -> data/evidence.json
python step3_retrieve.py --rebuild    # -> data/embeddings.npy   (~30 s on CPU)

# 4. Sanity-check the engine itself
python step6_test.py                  # 3 quick checks: known-answer, abstain, cross-persona
python step8_evaluators.py            # prints all 7 evaluator system prompts

# 5. Make sure Geranium is running (real or mock).
#    Real:  follow the Geranium repo, run `cd server && python app.py` on :5001.
#    Mock:  python scripts/mock_geranium.py    (returns canned templates)

# 6. Run Sycamore
python -m sycamore.cli run --condition both
```

## What the protocol does, per evaluator

For each evaluator (whether ungrounded or grounded), the runner executes the published Geranium study's three-part protocol (Section 3.2 of the manuscript), with one adjustment: the live tool demonstration is replaced by a textual description.

| Phase | What happens | Synthetic-evaluator equivalent |
| --- | --- | --- |
| Workflow description (10 min) | Participant describes their typical workflow. | One open-ended evaluator turn. **Grounded:** answer drawn from retrieved interview excerpts of the persona's own pool, with citations. **Ungrounded:** answer from generic LLM priors. |
| Tool description (10 min) | Live demo of Geranium. | The evaluator reads `sycamore/protocol/tool_description.py` and reacts. |
| Hands-on exploration (30 min) | Participant uses the tool in three modalities. | For each of `Text`, `Image`, `Spec`: the evaluator generates `--queries-per-modality` queries (default 3), the runner sends each to the **real Geranium server** via `GeraniumClient.search(...)`, and the evaluator reacts to the actual returned top-k triplets. |
| Closing | Modality preference (1-3 ranking + rationale) and overall assessment. | Two final turns; ranking returned as JSON. |

The full transcript and all retrieved evidence are saved per evaluator under `data/sycamore_outputs/records/<evaluator_id>.json`. A run-level summary with modality aggregates and a heuristic theme-alignment scaffold is written to `data/sycamore_outputs/summary.{json,md}`.

## Conditions

**Ungrounded** (`condition: ungrounded`): N=7 evaluators (configurable), instantiated from a generic genomics-researcher system prompt with no retrieval, no profile block, no validation. Same `LLMProvider` is used so generation behaviour is otherwise comparable.

**Grounded** (`condition: grounded`): one evaluator per entry in `step8_evaluators.EVALUATORS` — the 7 instances defined in the persona-generator with the manuscript's distribution: 1 Biologist + 2 Computational Biologists + 2 Bioinformaticians + 2 Software Engineers. Each evaluator uses the engine's full pipeline:

1. `get_evidence_with_scores(persona, query, k)` retrieves quotes from the persona's pool. If the top similarity is below `MIN_SIMILARITY=0.25`, the evaluator hard-abstains for that turn.
2. `validate_evidence(fast_provider, query, quotes)` (a fast LLM call) drops irrelevant retrievals.
3. `make_evaluator_prompt(ev)` + `make_evaluator_profile_block(ev)` + `make_evidence_block(quotes)` assemble the prompt.
4. `provider.complete(...)` generates the response in first person with `Sources: (P_, P_)` citations.

Each grounded evaluator preserves its conversation history across turns, so the hands-on exploration phase reads as one continuous session.

## CLI flags

```
python -m sycamore.cli run \
    --condition {ungrounded|grounded|both} \
    [--n 7]                          # ungrounded evaluator count
    [--queries-per-modality 3]
    [--k 5]                          # top-k results per Geranium query
    [--geranium-url http://localhost:5001]
    [--only CB1,SE2]                 # restrict grounded condition to specific abbrs
    [--out data/sycamore_outputs]
```

Provider selection follows the engine: `LLM_PROVIDER=anthropic` (default) or `LLM_PROVIDER=openai`.

## Sycamore session interface

`server.py` from the engine remains the per-persona chat UI. Sycamore's interface is a **session driver** that runs one evaluator through the full protocol:

```bash
uvicorn sycamore.interface.app:app --reload --port 8001
```

```bash
# Start a session (background thread runs the full protocol)
curl -X POST http://localhost:8001/api/sessions \
     -H 'content-type: application/json' \
     -d '{"condition":"grounded","abbr":"CB2"}'

# Poll it
curl http://localhost:8001/api/sessions/<sid>

# Persist the EvaluationRecord to disk
curl -X POST http://localhost:8001/api/sessions/<sid>/save
```

## Layout

```
.
├── personas.xlsx, coding_info/         # source artefacts (engine inputs)
├── llm_provider.py, step1..step8       # persona-generator engine (unchanged)
├── server.py, static/                  # engine reference chat UI
├── data/                               # engine outputs (personas.json, evidence.json, …)
├── sycamore/                           # this layer
│   ├── geranium_client.py              # Geranium HTTP client
│   ├── evaluator/                      # base + grounded + ungrounded
│   ├── conditions/                     # build_evaluators(condition, …)
│   ├── protocol/                       # 3-part runner + tool description
│   ├── analysis/                       # rankings + theme alignment
│   ├── interface/                      # FastAPI session driver
│   └── cli.py                          # `python -m sycamore.cli run …`
├── scripts/mock_geranium.py            # canned-response Geranium for plumbing tests
└── tests/                              # smoke tests + StubProvider
```

## Notes & limitations

- Image and Spec query content is materialised by picking the gallery item whose description has the most token overlap with the evaluator's described query. A more faithful implementation would let the evaluator pick a gallery index by ID.
- Per-persona run-to-run variance is left to the researcher (re-run with different seeds / temperature; the engine respects whatever the provider's default sampling is).
- The expert reference condition (Geranium user study findings) is loaded by the researcher as a JSON artefact for cross-condition comparison; it is not generated here.