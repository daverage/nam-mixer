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

## Suggested order

1. Add a small best-effort TTL cache around TONE3000 results and repeated AI/web search work, with explicit invalidation and tests for expiry/failure.
2. Prototype a reviewed alias/evidence library for amp and artist discovery, with clear source and approval state.
3. If external agent access is desired, add MCP tools over the service layer and use caller-provided credentials where the action needs them.

## Source locations

- ToneSearch: `tonesearch/cache.py`, `tonesearch/knowledge.py`, `tonesearch/feedback.py`, `tonesearch/mcp_server.py`, `tonesearch/research.py`.
- Builder: `hybrid/services/research.py`, `hybrid/services/local_llm.py`, `app.py` (`/api/tone3000/search`).
