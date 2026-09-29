# TONE3000 AI Search Enhancement Plan

## Overview

Enhance the TONE3000 tab with AI-powered tone discovery, inspired by ToneFinder but tailored for generic tone guidance without GP-50 rig building. Users can search for tones by natural language description and get AI-ranked results with tone characteristics, effects suggestions, and pack downloads.

## Key Differences from ToneFinder

| Aspect | ToneFinder | tone3000 AI Search |
|--------|-----------|-------------------|
| Focus | GP-50 rig building | Generic tone discovery |
| Output | GP-50 preset (.prst) | Downloaded NAM + tone insights |
| Effects | GP-50 specific | Generic effects reference |
| Scope | Full rig proposal | Tone characteristics & amp packs |
| Research | Integrated web search | Optional (may reuse AI Assistant) |

## Architecture

### 1. Frontend Changes

#### New UI Components (tone3000 tab)

```
TONE3000 Tab
├── Standard Search (existing)
│   ├── Text query
│   ├── Rig scope (heads/anything)
│   └── Author filter
│
└── AI Search (new)
    ├── "Use AI to find tones" toggle
    ├── Natural language prompt
    ├── Research options checkbox
    ├── Results display with:
    │   ├── Tone characteristics
    │   ├── Match score & reason
    │   ├── Effects summary (if applicable)
    │   ├── Tone color/visual indicator
    │   ├── Pack preview/info
    │   └── Download buttons
    └── Conversation continuation (optional)
```

#### Visual Enhancements

- Tone color chips (warm/bright/dark/etc.)
- Small tone images or icons (not required for MVP)
- Effect badges (distortion, reverb, delay, etc.)
- Pack metadata (creator, captures count, style tags)

### 2. Backend Changes

#### New API Endpoint

`POST /api/tone3000/ai_search`

Request:
```json
{
  "prompt": "Warm, vintage tube tone with natural breakup",
  "use_research": true,
  "rig_scope": "heads",
  "author_filter": null
}
```

Response:
```json
{
  "characteristic": {
    "warmth": 0.8,
    "brightness": 0.3,
    "dynamics": 0.6,
    "compression": 0.4,
    "saturation": 0.5
  },
  "search_plan": ["vintage tube tone", "warm breakup", "natural saturation"],
  "research_notes": "Optional web research snippets if enabled (mentions reverb, natural compression)",
  "results": [
    {
      "tone_id": "...",
      "name": "...",
      "pack_name": "...",
      "creator": "...",
      "match_score": 0.92,
      "match_reason": "Matches warm, vintage characteristics with natural compression mentioned in tone description"
    }
  ]
}
```

#### Integration Points

1. **LLM Integration** (reuse existing AI Assistant setup)
   - Use `hybrid/services/llm_service.py` or equivalent
   - Parse tone description → characteristics
   - Rerank TONE3000 results based on characteristics

2. **TONE3000 API**
   - Use existing `/api/tone3000/search` endpoint
   - Filter results through AI ranking

3. **Effects Database** (optional for MVP)
   - Reference common effects for certain tones
   - Could be a simple hardcoded mapping or separate DB

### 3. Implementation Phases

#### Phase 1: MVP (minimal)
1. Add AI search toggle to tone3000 tab
2. Implement prompt → characteristics parsing via LLM
3. Optional web research (research checkbox, like AI Assistant)
4. Search TONE3000 and rerank results
5. Display results with match score & reason
6. Show any effects mentioned in research/reasoning
7. Keep download flow same as standard search

#### Phase 2: Enhanced Results
1. Add pack metadata (creator, style tags, capture count)
2. Better visual result cards (improved typography/spacing)

#### Phase 3: Conversation (optional)
1. Allow follow-up questions about results
2. Refine search with conversational feedback
3. Ask about specific effects in selected packs

#### Phase 4: Images & Polish (future)
1. Add tone descriptor images/icons
2. Enhanced pack previews
3. Tone characteristic sliders
4. Visual effects selector

## Reuse Opportunities

### From AI Assistant
- LLM prompt engineering for tone analysis
- Conversation history & continuation logic
- Research checkbox implementation
- Error handling & status messages

### From ToneFinder
- Prompt structures for tone description
- Characteristic extraction patterns
- LLM ranking logic for search results
- Effects mapping (if applicable)

## Data Structures

### Tone Characteristics (output from LLM)
```python
{
  "warmth": float,           # 0-1, warm vs bright
  "dynamics": float,         # 0-1, dynamic vs compressed
  "saturation": float,       # 0-1, clean vs distorted
  "sustain": float,          # 0-1, short vs long
  "natural_breakup": float,  # 0-1, amount of organic overdrive
  "perceived_gain": float,   # 0-1, low to high output level
}
```

### Result Card (MVP)
```python
{
  "tone_id": str,
  "tone_name": str,
  "pack_name": str,
  "creator": str,
  "match_score": float,      # 0-1
  "match_reason": str,       # Why this matches, mentions effects if relevant
}
```

### Result Card (Phase 2+)
```python
{
  # All MVP fields above, plus:
  "style_tags": list,        # ["vintage", "rock", etc.]
  "capture_count": int,
}
```

## Decisions Made

1. **Web Research**: ✓ Include optional web research in Phase 1 (like AI Assistant)
2. **Visual Tone Representation**: ✗ Not in MVP (can add later if valuable)
3. **Effects Suggestions**: Organic only - show only if mentioned during research/ranking, not systematic
4. **Conversation**: Still open - Phase 3 candidate if desired

## Questions for User (Remaining)

1. **Conversation**: Should users be able to refine searches conversationally?
2. **Pack Info**: What metadata is most useful for amp pack downloads?
3. **Mobile**: Any mobile-specific UX considerations?

## Success Criteria

- [ ] AI can interpret tone descriptions accurately
- [ ] Results ranked meaningfully by tone characteristics
- [ ] Users can download from results easily
- [ ] Search is faster than manual browsing
- [ ] UI distinguishes AI search from standard search clearly
- [ ] Fallback to standard search if LLM unavailable

## Files to Create/Modify

### New Files
- `routes/tone3000_ai.py` - Backend AI search logic
- `hybrid/services/tone_analysis.py` - Tone characteristic extraction
- `static/tone3000-ai.js` - Frontend AI search UI (optional, can be in app.js)
- `docs/tone3000-ai-search-plan.md` - This document

### Modified Files
- `app.py` - Register new route
- `static/app.js` - UI logic for AI search toggle/UI
- `templates/index.html` - HTML for AI search section
- `static/styles.css` - Styling for new components

## Testing Strategy

1. **Unit Tests**
   - Tone characteristic extraction
   - Result reranking logic
   - API response validation

2. **Integration Tests**
   - End-to-end AI search flow
   - TONE3000 API integration
   - LLM availability fallback

3. **UI Tests**
   - Toggle between standard/AI search
   - Result rendering
   - Download flow

## Timeline

- **Phase 1**: 2 days (API + basic UI + web research)
- **Phase 2**: 1 day (enhanced result cards)
- **Phase 3**: 1-2 days (conversation refinement, if desired)
- **Phase 4**: Future (visual enhancements)
