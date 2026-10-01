# ToneSearch reuse review

Reviewed the indexed `tonesearch` project against `hybrid-nam-builder`'s built-in AI discovery and TONE3000 search paths. This is a source-level review, not a code change or runtime evaluation.

## Findings

1. **Persistent search cache is the clearest immediate reuse.** `tonesearch/tonesearch/cache.py` provides expiring JSON cache entries, key-prefix listing, and topic invalidation; the TONE3000 search caches public catalogue responses by canonicalized request parameters. `hybrid-nam-builder/hybrid/services/research.py` already ranks TONE3000 results against the AI's richer request, but its current search path does not appear to persist those catalogue responses. Reuse the cache policy and invalidation concepts for TONE3000 and expensive provider search calls, keeping per-provider TTLs and treating storage failure as a cache miss. This should reduce repeat API/AI work without replacing the current query planning or ranking.

2. **Curated, reviewed research can make repeated AI searches more consistent.** ToneSearch stores fresh research by normalized topic, preserves approved entries, leaves flagged entries flagged, and has admin search/feedback paths (`tonesearch/knowledge.py`, `tonesearch/feedback.py`). The current app builds amp discovery context from web research and local model planning, but the inspected search path has no comparable reviewed knowledge library. A small library of verified artist/amp/capture aliases and confirmed search outcomes could help all AI discovery modes. Keep web/provider evidence and user feedback separate from approved facts; require review before promotion.

3. **The MCP server exposes useful product capabilities beyond the browser UI.** `tonesearch/mcp_server.py` implements a local stdio JSON-RPC MCP server with tools for web research, pack search and download links, plus a prompt. It asks for the caller's own TONE3000 key and validates inputs. The builder's search operations are currently exposed through Flask routes such as `/api/tone3000/search`; an MCP adapter could expose those same core services to external agents and desktop MCP clients. Implement it as a thin adapter over `hybrid.services`, not by copying ToneSearch's standalone request code. Recheck current MCP protocol/library support before adopting its hand-written transport.

4. **Provider-key handling is a useful pattern to preserve.** ToneSearch passes the MCP caller's API key through the tool call and tests that the server's configured key is never substituted. This is relevant if builder search tools are exposed to external agents; it avoids silently using the app owner's credentials for a caller's action. The builder's existing provider-scoped settings should remain the UI path.

## Existing overlap and limits

- Both projects already support local and hosted LLM providers and TONE3000 search. ToneSearch is not a wholesale replacement for `hybrid.services.local_llm` or its amp-query planner.
- The builder already has relevant safety and quality work in `hybrid/services/research.py`: bounded page fetches, public-host checks, redirect refusal, metadata scoring, and whole-page ranking before truncation. Keep these protections when adding caching or MCP access.
- ToneSearch's SQLite schema, website UI, admin library and API-key settings are coupled to its own app. Reuse the behavior and concepts, not the whole package.
- MCP is an interface expansion, not by itself an improvement to retrieval relevance. The reusable retrieval improvements are the cache, reviewed research library, feedback loop, and search tools over the existing service layer.

## Prompt and research review: low-effort opportunities

ToneSearch's most useful prompt-engineering work is its **explicit evidence policy**, not a uniquely complex prompt framework. Its planning prompt separates artist-era-specific confirmed gear from career-wide artist gear and suggestions, tells the model how to resolve source conflicts, constrains output to concrete product names, and gives task-specific search-query rules. It then validates and normalizes the model's output in code (gear kind correction, generic-name removal, practice-amp filtering, confidence normalization and query deduplication). The builder already has a rich recipe schema and prompt constraints, but can borrow the pattern: make evidence status and query purpose explicit in the AI's structured result, and keep deterministic post-validation outside the prompt. This is a small targeted change to the existing discovery planner, not a prompt rewrite.

ToneSearch's research flow makes several cheap quality improvements that appear only partly present in the builder:

- It strips conversational filler before composing web queries. The builder already has an extensive `_FILLER_WORDS` list and `_topic` helper in `hybrid/services/research.py`, so this is already adopted; no work needed there.
- It searches two distinct query formulations in parallel (artist/bassist gear and interview/gear), retries other engines, and only falls back to a topic-only query if both focused searches return nothing. This can improve coverage over a single query strategy. The builder's `web_notes` currently searches two queries, but they are equipment/equipboard-oriented. A low-effort change is to replace the weaker generic/equipboard formulation with an interview/rig-rundown formulation, or add it only when the first pass has no useful evidence. Preserve the builder's existing host filtering and bounded concurrency.
- It limits evidence to one page per host, preventing a single gear-catalogue domain from occupying the whole prompt context. The builder caps notes and sources, but should also deduplicate by host if it does not already do so in the current implementation.
- It rejects forum sentences describing the poster's own rig and avoids treating sales copy/questions as evidence. This is valuable but should be a second small step because it needs forum/source classification and tests; don't blindly transplant heuristics without confirming source shapes.
- It labels research in the prompt as potentially partial/noisy and asks the model to prefer direct player interviews or rig rundowns over tone-site recommendations. The builder can adopt this as a short prompt instruction immediately, with no architecture change.

### Recommended simple sequence

1. Inspect the exact AI discovery prompt and its structured output schema. Add concise confidence/source rules (recording/era-specific, artist-general, inference) and direct-source precedence; retain deterministic output validation.
2. Add a rig-rundown/interview query as a fallback to current web queries, and deduplicate sources by hostname. Keep current safe-host and redirect protections.
3. Add one focused regression fixture for conflicting sources and one for a noisy/community page before broad prompt iteration.

The source-level review identifies plausible improvements; it does not establish measured accuracy gains. Confirm them with representative artist requests and before/after evidence traces before adopting more invasive prompt or research changes.

## Suggested order

1. Add a small best-effort TTL cache around TONE3000 results and repeated AI/web search work, with explicit invalidation and tests for expiry/failure.
2. Prototype a reviewed alias/evidence library for amp and artist discovery, with clear source and approval state.
3. If external agent access is desired, add MCP tools over the service layer and use caller-provided credentials where the action needs them.

## Source locations

- ToneSearch: `tonesearch/cache.py`, `tonesearch/knowledge.py`, `tonesearch/feedback.py`, `tonesearch/mcp_server.py`, `tonesearch/research.py`.
- Builder: `hybrid/services/research.py`, `hybrid/services/local_llm.py`, `app.py` (`/api/tone3000/search`).
