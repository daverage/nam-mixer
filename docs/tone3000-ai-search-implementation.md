# TONE3000 AI search: how it works

The TONE3000 tab has two modes: **Search by name** (the original catalogue search) and
**Describe a tone (AI)**. The AI mode gives generic tone advice and finds real captures; it never
builds a blend recipe or a preset (that is the AI Assistant tab's job).

## Flow

1. **Tone brief** (`hybrid/services/tone_search_ai.py: plan_tone`). The player's description, plus
   optional DuckDuckGo research (`research.web_notes`), becomes a summary, practical advice, the
   specific gear involved (amps, effects, guitars, pickups, cabs; effects only when the tone really
   uses them) and 1-3 short catalogue searches naming an amp make and model.
2. **Catalogue search** (`research.tone3000_search`). Each search returns real A2 packs with display
   metadata (`_tone3000_card_fields`: image, tags, gear type, A2 count, downloads, pack URL). Links
   are kept only when they point at tone3000.com over https.
3. **Ranking** (`rank_packs`). A catalogue-ranked shortlist of 12 goes to the AI, which scores fit and
   gives a one-line reason. Ids the model invents are dropped. Small local models may rate only
   some packs; the UI lists unrated ones separately as "Other catalogue matches".
4. **Refine**. Follow-up messages ("more gain for solos") resend the conversation history, so the
   brief and search evolve instead of starting over. "Start over" clears it.
5. **Files & questions** (`ask_about_pack`, `POST /api/tone3000/ai_pack_chat`). Opening a pack lists
   every NAM file with a download button and a per-pack chat. The server fetches the file list itself,
   so the AI only sees real file names; recommended files are matched back to real names (a short
   name like "EDGY" counts only if it identifies exactly one file) and highlighted as AI picks.

## Why a separate AI module

`local_llm.converse` is built for Amp A/B blend recipes and validates recipe output, which made the
first version of this search give off-target answers. `tone_search_ai` reuses its provider
plumbing (`_config`, `_post_chat_completion` with an optional `schema`, `_decode_json_content`), so
every provider in Settings works, but asks with its own prompts and JSON schemas.

## Files

- Backend: `app.py` (`/api/tone3000/ai_search`, `/api/tone3000/ai_pack_chat`),
  `hybrid/services/tone_search_ai.py`, `hybrid/services/research.py`
- Frontend: `templates/index.html` (TONE3000 panel), `static/tone3000-ai.js`, `static/style.css` (`t3ai-*`)
- Tests: `tests/test_tone_search_ai.py`, `tests/test_template_ids.py`

## Known limits

- With web research on, a search takes about 40-60 s on a small local model (research plus two AI calls).
- Gear and advice quality depends on the configured model; larger models name gear more reliably.
