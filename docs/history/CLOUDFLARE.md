# Cloudflare Workers AI / OpenAI-Compatible Provider Support

## Objective

Extend NAM Mixer’s existing AI Assistant so it can use either:

* a local OpenAI-compatible endpoint such as Ollama, LM Studio, or llama.cpp
* Cloudflare Workers AI
* a generic remote OpenAI-compatible `/v1` endpoint

The goal is to keep the existing AI Assistant behaviour unchanged while making cloud-hosted inference available to users who do not want to run a local model. “Unchanged” applies to prompts, recipe schemas, validation, conversation handling, and fallback behaviour; the Settings labels and provider fields may change as described below.

The current `hybrid/local_llm.py` already uses the OpenAI Chat Completions API shape, so this should be treated as a provider/configuration refactor rather than a new AI subsystem.

## Current State

NAM Mixer currently expects:

* a model name
* a local OpenAI-compatible base URL
* `POST /chat/completions`
* `response_format: {"type":"json_object"}`
* no authentication header

The current implementation deliberately restricts the AI base URL to localhost and plain HTTP.

Example:

```text
NAM_MIXER_LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
NAM_MIXER_LOCAL_LLM_MODEL=gemma4:e4b
```

This works well for Ollama but prevents remote providers such as Cloudflare Workers AI.

## Target Architecture

Rename/generalise the local LLM layer so it represents an AI provider rather than a local-only service.

Conceptually:

```text
AI Assistant
    |
    +-- Local OpenAI-compatible
    |      Ollama
    |      LM Studio
    |      llama.cpp
    |
    +-- Cloudflare Workers AI
    |
    +-- Custom OpenAI-compatible endpoint
```

All providers should continue through the same internal `converse()` path and the same existing recipe validation.

Do not create separate prompt logic for Cloudflare.

## Cloudflare Support

Cloudflare Workers AI exposes an OpenAI-compatible endpoint for chat completions. The implementation must verify the endpoint/model contract against the current Cloudflare documentation when this feature is released; provider model availability can change independently of NAM Mixer.

NAM Mixer should support configuration equivalent to:

```text
Base URL:
https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1

Model:
@cf/openai/gpt-oss-20b

Authorization:
Bearer <API_TOKEN>
```

NAM Mixer should construct the Cloudflare URL from the Account ID rather than requiring users to understand or paste the complete endpoint.

The token must have the account permission required to invoke Workers AI. The UI must tell users that Cloudflare account limits, quota, and billing may apply before they test or use the provider. A connection test is a real remote inference request unless the provider offers a non-consuming validation endpoint.

## Suggested Settings

Replace the current local-only terminology with something like:

```text
AI Assistant

Provider
[ Local / Cloudflare Workers AI / Custom OpenAI-compatible ]

Model
[____________________________]

API key
[••••••••••••••••••••••••••]

[Test connection]
```

### Local provider

Fields:

```text
Base URL
Model
```

Defaults:

```text
http://127.0.0.1:11434/v1
gemma4:e4b
```

No API key required.

### Cloudflare Workers AI

Fields:

```text
Account ID
API Token
Model
```

NAM Mixer internally builds:

```text
https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1
```

Recommended starting model can be configured separately and changed later based on app-specific testing.

### Custom OpenAI-Compatible

Fields:

```text
Base URL
Model
API Key
```

API key should be optional because some local/private servers do not require authentication.

## Configuration

Prefer general provider-neutral names rather than adding more `LOCAL_LLM_*` settings.

For example:

```text
NAM_MIXER_AI_PROVIDER
NAM_MIXER_AI_BASE_URL
NAM_MIXER_AI_MODEL
NAM_MIXER_AI_API_KEY
NAM_MIXER_AI_TIMEOUT_SECONDS
NAM_MIXER_AI_TEMPERATURE
NAM_MIXER_AI_MAX_TOKENS
```

Backward compatibility should be considered.

Existing users with:

```text
NAM_MIXER_LOCAL_LLM_*
```

should either:

1. continue to work through compatibility aliases, or
2. be migrated automatically.

Migration precedence must be deterministic:

1. An explicitly configured new `NAM_MIXER_AI_*` value wins.
2. If no new value exists, the corresponding legacy `NAM_MIXER_LOCAL_LLM_*` value is used.
3. Saving from Settings writes only the new provider-neutral names; legacy values remain readable for one compatibility cycle but are not silently overwritten.

Avoid requiring existing users to manually rewrite their `.env`.

## Security Requirements

### API keys

API tokens must:

* remain server-side
* never be returned through `/api/settings`
* never be sent to browser JavaScript
* never be logged
* use the existing secret-setting behaviour already used for `TONE3000_API_KEY`

The Settings API must return only `has_value`, never the token itself. Because a blank secret currently means “leave unchanged,” the provider settings UI must provide an explicit **Clear API token** action. Clearing must remove the stored value from the configured `.env` file and the current process environment. Switching providers must not implicitly reuse a token saved for another provider.

### Remote URL validation

Do not simply remove the existing localhost restriction.

Instead:

```text
localhost endpoint
    HTTP allowed

remote endpoint
    HTTPS required
```

Local hosts allowed should remain:

```text
127.0.0.1
localhost
::1
```

Remote custom endpoints should require HTTPS. HTTPS alone is not sufficient SSRF protection: before connecting, resolve the hostname and reject loopback, link-local, private, multicast, benchmark, and other reserved address ranges for every resolved address. Disable redirects or revalidate every redirect target. Enforce short connect/read timeouts and a bounded response size.

Preserve SSRF protection. Do not allow arbitrary browser-provided URLs to override the configured server-side endpoint. The browser may choose a configured provider, but it must never supply the request URL, headers, model endpoint, or API token.

### Cloudflare

For the Cloudflare provider, do not accept an arbitrary base URL.

Generate the URL internally from the supplied Account ID.

## Request Handling

The existing Chat Completions request should remain essentially unchanged. The provider transport owns URL construction, authentication headers, timeout handling, and response normalization; prompt construction and recipe validation remain shared.

Current shape:

```json
{
  "model": "...",
  "messages": [...],
  "temperature": 0.2,
  "max_tokens": 1400,
  "response_format": {
    "type": "json_object"
  }
}
```

Add an Authorization header only when an API key is configured:

```text
Authorization: Bearer <token>
```

No authentication header should be sent for normal Ollama usage.

## Status / Connection Test

The current status check assumes an OpenAI-compatible `/models` endpoint.

Retain this where available, but do not assume every provider implements `/models` identically. `/models` is an optional liveness probe, not the authoritative success condition.

Add a provider-aware connection test.

Cloudflare connection validation should ideally verify:

* Account ID appears valid
* API token works
* requested model can be invoked
* JSON-mode request succeeds

Define the test sequence and error mapping explicitly:

1. Validate the Account ID format locally and construct the fixed Cloudflare base URL.
2. Make a minimal, bounded chat-completions request using the selected model and the same JSON-mode shape used by the assistant.
3. Do not send the user's conversation, research notes, or stored prompt history as part of the test.
4. Map responses to user-facing states: `401` invalid token, `403` insufficient account permission, `404` account/model/endpoint unavailable, `429` quota or rate limit, timeout, and malformed JSON. Never display raw response bodies when they may contain secrets.
5. Keep the test request small and clearly warn that it is a real provider request that may count against quota or billing.

Return user-friendly errors rather than raw HTTP failures.

Examples:

```text
Connected
Invalid API token
Account not found
Model unavailable
Endpoint did not return valid JSON
Connection timed out
```

## Terminology Changes

Rename user-facing references from:

```text
Local AI Assistant
Local LLM
```

to:

```text
AI Assistant
AI Provider
```

Internal code should ideally follow the same direction.

Possible rename:

```text
hybrid/local_llm.py
```

to:

```text
hybrid/ai_provider.py
```

or:

```text
hybrid/assistant.py
```

Keep `hybrid/local_llm.py` for the first implementation. A module rename is optional cleanup and should not be part of the provider-support change unless it is needed to avoid a misleading public API.

## Preserve Existing Behaviour

The following must not change as part of this work:

* prompt construction
* recipe schema
* blend/hybrid/character selection
* source-plan handling
* TONE3000 integration
* research-note handling
* conversation-history limits
* JSON validation
* recipe field range validation
* fallback behaviour
* user-visible AI explanations

This task is provider transport/configuration work, not a prompt redesign.

## Suggested Provider Model

Use a small provider abstraction, for example conceptually:

```python
AiConfig(
    provider="local",
    base_url="http://127.0.0.1:11434/v1",
    model="gemma4:e4b",
    api_key=None,
)
```

Cloudflare:

```python
AiConfig(
    provider="cloudflare",
    base_url="https://api.cloudflare.com/client/v4/accounts/.../ai/v1",
    model="@cf/openai/gpt-oss-20b",
    api_key="...",
)
```

Custom:

```python
AiConfig(
    provider="custom",
    base_url="https://example.com/v1",
    model="...",
    api_key="...",
)
```

The rest of the application should not need to care which provider is being used.

## UI Behaviour

The Settings page should dynamically show only fields relevant to the chosen provider. Provider switching must preserve unrelated settings, must not leak or silently reuse secrets, and must make the token-clearing action explicit.

Example:

```text
AI Assistant

Provider
Cloudflare Workers AI

Account ID
[_____________________]

API Token
[••••••••••••••••••••]

Model
[@cf/openai/gpt-oss-20b]

[Test connection]

✓ Connected
```

For Local:

```text
Provider
Local

Base URL
http://127.0.0.1:11434/v1

Model
gemma4:e4b

✓ Ollama reachable
```

## Failure Behaviour

Cloud AI must remain optional.

If the configured provider is unavailable:

* the rest of NAM Mixer must continue working
* no startup failure
* no repeated background retries
* AI Assistant should show a clear unavailable state
* users should still be able to disable AI completely

## Tests

Add tests covering at minimum:

### Configuration

* local provider accepts localhost HTTP
* remote provider rejects HTTP
* remote HTTPS is accepted
* Cloudflare URL is constructed correctly
* API secrets never appear in settings responses
* legacy `NAM_MIXER_LOCAL_LLM_*` settings still work if compatibility is retained

### Requests

* no Authorization header for Ollama
* Bearer header added for Cloudflare
* Bearer header added for authenticated custom providers
* existing JSON request payload remains unchanged

### Responses

* valid recipe JSON behaves exactly as before
* invalid JSON still raises the existing safe error
* HTTP auth failure becomes a useful error
* forbidden account permission becomes a useful error
* rate limiting becomes a useful error
* timeout becomes a useful error
* unavailable model becomes a useful error
* redirects to disallowed or private addresses are rejected
* oversized responses are rejected

## Non-Goals

Do not add:

* Cloudflare AI Gateway
* central NAM Mixer accounts
* shared NAM Mixer API keys
* billing/subscription logic
* automatic provider fallback
* telemetry
* remote prompt storage
* user authentication

Those can be considered separately later.

## Acceptance Criteria

The work is complete when:

1. Existing Ollama users continue working without changing their workflow.
2. A user can choose Cloudflare Workers AI in Settings.
3. They only need to supply Account ID, API Token, and model.
4. NAM Mixer successfully uses Cloudflare through `/v1/chat/completions`.
5. Existing recipe generation and conversation behaviour are unchanged.
6. API tokens never reach browser code, logs, error messages, or response payloads.
7. Remote providers require HTTPS and pass hostname-resolution, private-address, redirect, timeout, and response-size checks.
8. AI failure never prevents the rest of NAM Mixer from working.
9. A generic OpenAI-compatible HTTPS provider can also be configured.
10. Automated tests cover provider selection, authentication, permission/rate-limit errors, URL normalization, secret clearing, SSRF protections, and backward compatibility.

## Recommended Scope

Implement this as a small provider abstraction around the current `converse()` transport layer.

Do not rewrite the AI system.

The desired end state is:

```text
same prompts
same validation
same UI behaviour
same recipes

different inference provider
```

That keeps the change low-risk while opening NAM Mixer to local and cloud AI without locking the project to any single vendor.
