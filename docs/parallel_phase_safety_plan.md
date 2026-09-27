# AI Codebase Review & Debugging: A Prompt Guide

Sep 26, 2026 · Andrzej

## How to use this guide

This is a step-by-step process for reviewing and improving a large codebase with AI agents while keeping token use low. Work through the steps in order. The core rule: **tools scan the whole repository; the agent reads only the specific symbols, ranges and findings a question needs.**

Every step starts with a **Who** line:

| Who | Meaning |
| --- | --- |
| You | A manual action. No agent involved. |
| Strong model | Use the best agent you have (for example Claude Code with a frontier model). |
| Local model OK | A local model (for example OpenCode with Qwen) can do this. A strong model also can. |
| Independent reviewer | A different model from the one that did the work, or a person. |

Every block is labelled with where it goes:

| Label | What you do with the block |
| --- | --- |
| **Paste into the agent** | Copy it into the agent's chat as your message. Replace [bracketed] parts first. |
| **Add to file** | Copy it into the named file in your repository. |
| **Run in your terminal** | Run it yourself in a shell. |

Pasted prompts refer only to `AGENTS.md` and files in `review/`, never to this guide, because the agent cannot see the guide. Terms such as *consequential* and *fresh context* are defined in `AGENTS.md` (Step 2), so the agent knows them too.

## Part 1: One-time setup

### Step 1: Create a review branch

**Who:** You

**Run in your terminal:**

```bash
git checkout -b review/[short-name]
mkdir -p review/tools
```

Every session commits `review/` to this branch. Never gitignore `review/`: handoffs between sessions and agents depend on it. If the project already has a review tracker, you will link to it from `review/HANDOFF.md` in Step 17 rather than keeping two formats.

### Step 2: Add the project rules

**Who:** You

**Add to file:** `AGENTS.md` (create it at the repository root if it doesn't exist; if it exists, add these sections)

```markdown
# Project constraints
- [Public contracts, safety or real-time constraints, protected paths]

# Commands
- Build: [command]
- Targeted test: [command]
- Full required checks: [commands]
- Baseline comparison: [filled in at Step 4]

# Definitions
- Consequential: affects a project constraint, a public contract, security,
  data integrity, or output people see or hear.
- Fresh context: a new session or subagent that has not seen the work.
- Independent reviewer: a different model from the one that did the work,
  or a person.

# Retrieval
- Index: [existing code graph or index and how to query it, or review/INDEX.md]
- Exclude vendor, generated and build output, media, model weights and large data.
- Save raw tool output to review/tools/; read summaries and relevant excerpts.
- Search symbols before opening source. Read line ranges before whole files.

# Retrieval budget
- Routine question: at most 3 files, 300 source lines, 5 short tool excerpts.
- Cross-module question: at most 8 files, 800 source lines.
- Before exceeding a budget, record the open question and why the next
  read answers it in review/PROGRESS.md.

# Output limits
- Chat replies: at most 400 words and 10 findings. Full detail goes in review/.

# Review sessions
- Write only to review/. Never edit production source during a review session.
- Commit review/ at the end of every session.

# Stop and hand off when
- the same check fails twice after your changes
- the next step needs more than the cross-module retrieval budget
- you cannot state what would confirm or falsify a finding
- a change would touch a project constraint or protected path
Before stopping: commit work in progress to the current branch with a
message starting "WIP:" (or use git stash if committing is not allowed).
Set review/HANDOFF.md status to "blocked" with the reason, the last check
result and the commit hash or stash reference. Never discard uncommitted work.
```

The budgets are working heuristics, not measured thresholds; adjust them to your codebase.

If you use Claude Code, also **add to file:** `CLAUDE.md` (at the repository root)

```markdown
@AGENTS.md
```

### Step 3: Enforce the review-only rule

**Who:** You

`AGENTS.md` is advisory; an agent can still ignore it. In your agent's permission settings, deny edits outside `review/` during review sessions, or add a pre-edit hook that blocks writes to source paths. Enforce required checks with CI or build scripts where possible.

### Step 4: Set up a baseline

**Who:** Strong model (Local model OK if good tests already exist)

Skip this step if existing tests already exercise the main behaviour end to end quickly.

**Paste into the agent:**

```text
Set up a baseline using existing tests and tooling where possible. Run
[main entry point] on the inputs in [fixture directory], save outputs and key
statistics under [baseline directory], and label them with the current commit
hash. Add a comparison command with explicit tolerances for volatile fields
and numeric noise. Add it as "Baseline comparison" under Commands in
AGENTS.md. Do not change production source.
```

**Who:** You. Run the comparison command once to confirm it passes, then commit.

A baseline shows agreement with previous behaviour; it can preserve an old bug. Useful baselines and extra checks by project type:

| Project | Baseline | Also check |
| --- | --- | --- |
| API or service | Recorded request and response comparison | Contract and error-case tests |
| CLI or pipeline | Fixture output comparison | Invariants and known-correct examples |
| Audio or DSP | Rendered WAV comparison with stated tolerance | Signal properties; you listen for consequential changes |
| UI | Scripted interaction and visual comparison | Accessibility; you look for consequential changes |
| Library | Existing regression tests | Unit and property tests |

## Part 2: Map the codebase

### Step 5: Run the tools

**Who:** Local model OK

**Paste into the agent:**

```text
Inspect [source roots] with existing deterministic tools: compiler or type
checker, lint, static analysis, complexity, duplication, dependency audit and
secret scanning where available. Save full outputs under review/tools/ and a
short inventory in review/TOOLS.md. Include build and test status,
language/module manifests, entry points, public symbols, dependency edges,
test locations, file size and recent change frequency. Record tool versions,
scope, commands and failures. If a tool is missing, run it ephemerally or
outside the project; do not change manifests or lockfiles. Do not edit
production source. Distinguish missing tool coverage from a clean result.
Reply with the inventory summary and file paths only.
```

See Appendix A and B for which tools to name if the project has none configured.

### Step 6: Set up the index

**If the project already has a code graph or index** (for example a code-graph MCP server or an LSP-backed index):

**Who:** You. Fill in the `Index:` line under Retrieval in `AGENTS.md` with its name and how to query it. Skip the prompt below.

**Otherwise:**

**Who:** Local model OK

**Paste into the agent:**

```text
Write a script that builds review/INDEX.md (or review/INDEX.csv if large)
from tool output: [ctags or LSP] for public symbols, [dependency tool] for
dependencies and dependants, git log for recent commits, and [complexity and
duplication tools] for metrics, one row per file with its module, path and
LOC. Save the script under review/tools/ and run it. Do not write index rows
yourself and do not paste the index into the reply. Reply with the script
path, row count and any tool failures.
```

**Who:** You. Set the `Index:` line in `AGENTS.md` to `review/INDEX.md`.

### Step 7: Rank hotspots

**Who:** Strong model (Local model OK for a first draft)

**Paste into the agent:**

```text
From review/TOOLS.md, review/tools/ and the index named under Retrieval in
AGENTS.md, write review/HOTSPOTS.md. List candidates with their separate
signals: change frequency, complexity, duplication, coverage, criticality and
tool findings. Rank with a stated reason or normalised scores, not a product
of unrelated raw metrics. Note blind spots and low-change areas with high
consequence. Do not open source files. Reply with the top 10 only.
```

A hotspot is a review priority, not proof of a defect.

### Step 8: Write the architecture map

**Who:** Strong model

**Paste into the agent:**

```text
Build review/MAP.md from the index named under Retrieval in AGENTS.md,
manifests, dependency graphs, entry points and tests. Do not read modules file
by file. Open source only where metadata cannot answer a specific architectural
question, within the retrieval budget in AGENTS.md. Record major
responsibilities, boundaries, state/concurrency model, likely hot paths and
dominant conventions with cited examples or counts. Mark uncertainty rather
than guessing. Keep MAP.md to two pages; do not critique or edit source.
```

**Who:** You. Check the map against two or three entry points you know, correct any errors in `review/MAP.md`, and commit. Later reviews judge conventions against this map, so errors here spread.

## Part 3: Review in batches

### Step 9: Choose a track and a batch

**Who:** You

Pick one track: correctness (contracts, edge cases, error handling, state, tests), security (trust boundaries, authorisation, secrets, unsafe operations), architecture (coupling, ownership), performance (measured hot paths) or maintainability (duplication, dead code, needless complexity). Pick the five highest-ranked candidates for that track from `review/HOTSPOTS.md`. Don't run every track on every module.

### Step 10: Review the batch

**Who:** Strong model (Local model OK for a single symbol or range)

**Paste into the agent:**

```text
Review [specific symbols/ranges] for [one track and concrete concern], using
review/MAP.md, the relevant review/tools/ output and named callers. Start with
symbol search and stay within the retrieval budget in AGENTS.md. For each
finding, append one row to review/FINDINGS.md:
ID | track | path | symbol | line range | claim | evidence (max 5 lines)
| related symbols/callers | tool reference | impact | consequential (yes/no)
| confidence | effort.
Separate observation from inference. Record only actionable findings and
state what would falsify uncertain ones. No source edits. Max 10 findings.
Reply with finding IDs and one line each; full detail stays in FINDINGS.md.
```

### Step 11: Record progress and decide

**Who:** Local model OK

**Paste into the agent:**

```text
Update review/PROGRESS.md for the batch just finished: findings that look
worthwhile, false-positive rate so far, approximate context and tool cost,
remaining high-risk questions, and the expected value of another batch.
Recommend continue or stop, with one reason.
```

**Who:** You. Decide. Go back to Step 9 if a meaningful finding emerged or a high-risk area is unresolved; otherwise go to Step 12. A mandated audit or incident can override this.

## Part 4: Verify the findings

### Step 12: Triage in a fresh context

**Who:** Strong model, in a new session

**Paste into the agent:**

```text
For each finding in review/FINDINGS.md, inspect its exact symbol/range and the
minimum related callers needed. Classify it CONFIRMED, THEORETICAL, WRONG or
NOT WORTH IT. Give one evidence-based reason and any remaining uncertainty,
and confirm or correct its consequential (yes/no) value using the definition
in AGENTS.md. Write review/TRIAGED.md sorted by practical impact, confidence
and effort. Do not edit source or rediscover repository structure.
```

### Step 13: Independent check of consequential findings

**Who:** Independent reviewer

Skip if no CONFIRMED finding is marked consequential.

**Paste into the agent:**

```text
Check findings [IDs] in review/TRIAGED.md. For each, inspect the exact
symbol/range and minimum related callers, and state whether you agree with
the classification, with one evidence-based reason. Record disagreements in
review/TRIAGED.md. Do not edit source.
```

**Who:** You. Spot-check a few CONFIRMED findings yourself. If more than one is wrong, tighten the Step 12 prompt or use a different reviewer before fixing anything.

## Part 5: Fix one finding at a time

### Step 14: Implement

**Who:** Strong model (Local model OK for non-consequential findings)

This is not a review session, so allow source edits for it (Step 3 settings).

**Paste into the agent:**

```text
Implement CONFIRMED finding [ID] from review/TRIAGED.md only. State the
intended behaviour and files before editing. Open only the named files and
directly referenced definitions; if the finding looks false or scope must
expand, record that in review/TRIAGED.md and stop. For a bug, first add a
failing regression test where feasible. For a refactor, preserve behaviour
against the baseline and relevant invariants. Run the targeted test after each
edit, affected tests before commit, then the full required checks and
baseline comparison from AGENTS.md. Review the diff for unrelated changes. Do
not weaken tests to make them pass. Commit once, then record commands, results
and the commit hash in review/HANDOFF.md. If the finding is consequential, set
the next action to "independent review of [commit]".
```

For duplication, first make the copies behave identically, then extract shared code, with the baseline comparison at each step. For performance findings, profile a representative workload first, change one thing, and compare mean and worst-case times under the same conditions; there is no universal minimum gain.

### Step 15: Independent review of consequential changes

**Who:** Independent reviewer, in a new session

Skip for non-consequential findings.

**Paste into the agent:**

```text
Review commit [hash] against finding [ID] in review/TRIAGED.md and the
Project constraints in AGENTS.md. Report only: behaviour changes, constraint
violations and edits outside the finding's scope. If the change affects
output people see or hear, list what a human should check. No style comments.
```

**Who:** You. Do the listed human checks (listen, look) before accepting the change.

### Step 16: Repeat

**Who:** You

Commit `review/`, then go back to Step 14 for the next CONFIRMED finding in `review/TRIAGED.md`.

## Part 6: Ending and resuming sessions

Keep one narrow task per session. Start a new session after two corrections on the same point or when the conversation drifts. Carry work forward through `review/` files and Git, not pasted logs or agent checkpoints.

### Step 17: Create the handoff file (once)

**Who:** You

**Add to file:** `review/HANDOFF.md`

```markdown
# Handoff
- Phase and status: done / in progress / blocked (with reason)
- Branch and last commit (or stash reference):
- Completed: finding IDs, paths and commits
- Checks: exact commands and results, with links to logs in review/tools/
- Existing review tracker: [link, if the project has one]
- Open questions and blocked items:
- Next action: one bounded task and the files, symbols or ranges it needs
```

### Step 18: End every session

**Who:** Same agent that did the work

**Paste into the agent:**

```text
Update review/HANDOFF.md for this session using its existing headings. Keep
it short and link to files instead of pasting logs. Then commit review/.
```

### Step 19: Resume in a new session or a different agent

**Who:** Any agent the next step suits (see the Who line of that step)

**Paste into the agent:**

```text
Read review/HANDOFF.md and AGENTS.md. Open only the index entries, findings
and source ranges needed for the next action. Do not replay completed
analysis. If the branch or environment changed, rerun the last recorded
check first. Do the next action, then update review/HANDOFF.md with evidence
and one bounded next action, and commit review/.
```

If a local agent hits a condition in the Stop and hand off section of `AGENTS.md`, it commits its work in progress and marks the handoff blocked; resume with Step 19 in a strong model.

## Separate path: debugging a known bug

Use this instead of Parts 2 to 5 when you are chasing one specific problem.

**D1. Investigate.** **Who:** Strong model

**Paste into the agent:**

```text
Symptom: [exact symptom, when it happens, what "fixed" looks like].
Do not make fixes yet. Reproduce it if possible. Trace the narrow data and
control path from [entry point] to [where it goes wrong], within the
retrieval budget in AGENTS.md. List the assumptions the code makes about
inputs, state, concurrency and lifetimes. Give 3 ranked hypotheses, each with
the cheapest experiment that would confirm or rule it out.
```

**D2. Instrument, if needed.** **Who:** Local model OK

**Paste into the agent:**

```text
Add temporary logging to test hypothesis [n] only, kept cheap on hot paths.
Mark every added line with DEBUG-TEMP. After I paste the log back, remove all
DEBUG-TEMP lines and confirm with a search.
```

**Who:** You. Reproduce the bug and paste the log into the agent.

**D3. Regressions only: bisect.** **Who:** Local model OK

**Paste into the agent:**

```text
This worked at [commit or tag] and fails at HEAD. Write a bisect script that
builds, runs [check], and exits 0 for good, 1 for bad, 125 if the build
fails. Test it on both endpoints first, then run git bisect run. Reply with
the first bad commit and its diff summary.
```

**Who:** You. Look at the reported commit yourself before trusting it.

**D4. Fix.** Use the Step 14 prompt, replacing its first line with: "Fix the confirmed cause: [cause]. First add a failing test for it."

During an outage, exploit or active data corruption, contain first (rollback, feature switch) and find the root cause afterwards. Repeated failed fixes mean the assumptions or design need re-examining.

## Using a local model

**Who:** You, once

- Set the context length explicitly in your model server (for example Ollama or LM Studio) and check the effective context and memory use. A 64K context can help when hardware permits; don't fill it.
- Enable tool calling in your agent's model configuration, and use a model that supports it.
- Prefer a coder model with a mixture-of-experts design (for example Qwen3-Coder-30B-A3B) over a small dense model if your hardware allows.
- Follow the **Who** line of each step: local models suit tool runs, scripts, progress notes and small non-consequential fixes. Mapping, triage and consequential work go to a strong model.

Defaults and model capabilities change; verify them in your own setup.

## Reference

### Evidence and limits

The studies support verification, not a fixed estimate of agent performance. METR's 2025 randomised study found a 19% slowdown for experienced open-source developers in familiar repositories using early-2025 AI tools. Its 2026 update reports new estimates with wide confidence intervals and serious selection effects, so the current direction and size of the effect are uncertain. CodeScene's 2024 JavaScript/TypeScript benchmark found at best about 37% behaviour-preserving refactorings among the tested models; its fact-checking layer raised correctness among accepted candidates to roughly 96 to 99%. Neither number describes all 2026 agents. Measure your own review yield, verification cost and regression rate.

### Appendix A: Code quality tools

| Language | Static analysis | Duplication | Complexity |
| --- | --- | --- | --- |
| C / C++ | clang-tidy, cppcheck | jscpd or PMD CPD | lizard |
| Python | ruff, mypy | jscpd | radon or lizard |
| JavaScript / TypeScript | eslint, tsc --noEmit | jscpd | eslint complexity rule or lizard |
| Java / Kotlin | PMD, SpotBugs or detekt | PMD CPD | PMD |
| Go | go vet, staticcheck | jscpd | gocyclo |
| Rust | clippy | jscpd | clippy |

lizard and jscpd cover most languages. For symbol indexes, use universal-ctags or your editor's language server.

### Appendix B: Security tools

| Language or area | Code scanning | Dependency audit |
| --- | --- | --- |
| Python | bandit | pip-audit |
| JavaScript / TypeScript | eslint security plugins or semgrep | npm audit |
| Rust | clippy | cargo audit |
| Go | gosec | govulncheck |
| C / C++ | flawfinder, clang-tidy security checks | Depends on package manager |
| Any language | semgrep | Secrets: gitleaks or trufflehog |

### Sources

- [Anthropic, Claude Code best practices](https://code.claude.com/docs/en/best-practices): scoped exploration, verification, context and fresh review.
- [Google Engineering Practices, what to look for in code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
- [OWASP Secure Code Review Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secure_Code_Review_Cheat_Sheet.html)
- [CodeScene, Refactoring vs Refuctoring (2024)](https://codescene.com/hubfs/whitepapers/Refactoring%20vs%20Refuctoring%20Advancing%20the%20state%20of%20AI%20automated%20code%20improvements.pdf) ([summary](https://www.janeasystems.com/blog/ai-and-refactoring-part-2))
- [METR 2025 developer productivity study](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) and [2026 update](https://metr.org/blog/2026-02-24-uplift-update/)
- [Ollama context length documentation](https://docs.ollama.com/context-length)
- [OpenCode documentation](https://opencode.ai/docs) and [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder)
