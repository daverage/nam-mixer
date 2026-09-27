# AI Codebase Review & Debugging: A Prompt Guide

Sep 26, 2026 · Andrzej

## Purpose

Review and improve a large codebase while controlling token use. Let deterministic tools examine the whole repository. Give the agent progressively narrower evidence, then source ranges only when a specific question warrants them. Record decisions in files so a new session can continue without replaying the investigation.

**The governing rule:** whole-repository analysis is for tools; LLM reasoning is for selected paths, symbols and findings. Do not read a repository file by file to understand it.

This is a prioritised improvement workflow, not a claim that every defect has been found. Choose a review track to match the task: correctness, security, architecture, performance or maintainability. The prompts assume a coding agent and Git; replace bracketed fields with project-specific commands and paths.

## Operating rules

1. **Evidence before edits.** Review writes only to `review/`. Implement one confirmed finding at a time.
2. **Retrieval ladder.** Use symbol/reference search, exact search, dependency or call graph, targeted line range, surrounding function, then whole file only if structurally necessary. Follow additional files along a named dependency. Never open a directory of source indiscriminately.
3. **Input and output budgets.** A routine investigation starts with at most 3 files, 300 relevant source lines and 5 short tool excerpts. A cross-module question starts with at most 8 files and 800 lines. These are working heuristics, not measured thresholds, and not permission to omit necessary evidence. Before exceeding them, record the missing question and why the next read answers it. Limit each response to 10 findings and 400 words; put detail in a file.
4. **Verification is independent.** Test a claim against actual callers and constraints. Review a change with fresh context when the risk justifies it.
5. **Behavioural baselines are limited.** Golden tests establish agreement with previous behaviour; they may preserve an old bug. Add known-correct cases, invariants or property tests when available.
6. **Use subagents selectively.** Delegate bounded, independent investigations that would otherwise fill the main context. Search and inspect directly for small questions.
7. **Match the model to the job.** Use the strongest available model for mapping, triage and verification. Use a cheaper or local model for mechanical work: running tools, summarising output, bounded single-finding edits.
8. **Stop when marginal value falls.** Review hotspots in batches; report confirmed value, false positives, remaining risk and cost before continuing.

## Setup: rules and verification

Put shared advisory rules in `AGENTS.md` (or the project's equivalent). Import it from `CLAUDE.md` if using Claude Code. Enforce truly mandatory restrictions with permissions, hooks, CI or build checks where possible; prose instructions alone are not a guarantee.

```markdown
# Project constraints
- [Public contracts, safety or real-time constraints, protected paths]
- Build: [command]
- Targeted test: [command]
- Full required checks: [commands]
- Baseline comparison: [command, if applicable]

# Review retrieval
- Exclude vendor, generated and build output, media, model weights and large data.
- Save raw tool output to review/tools/; read summaries and relevant excerpts.
- Search symbols before opening source. Read line ranges first.
- Review tasks may write review/ files but not production source.
```

Use existing tests and project tooling first. If the main behaviour lacks a fast end-to-end check, build a small harness using representative fixture inputs. Capture current output and key statistics. Keep volatile fields and numeric tolerances explicit. Baseline outputs must be labelled with the source commit. Do not change production source in this step.

| Project | Useful baseline | Complementary correctness check |
| --- | --- | --- |
| API or service | Recorded request and response comparison | Contract and error-case tests |
| CLI or pipeline | Fixture output comparison | Invariants and known-correct examples |
| Audio or DSP | Rendered WAV comparison with stated tolerance | Signal properties and listening checks |
| UI | Scripted interaction and visual comparison | Accessibility and behavioural assertions |
| Library | Existing regression tests | Unit and property tests |

## Phase 1: Broad, cheap repository intelligence

Run configured compiler/type checker, lint, static analysis, complexity and duplication checks as relevant (see Appendix A for common tools). Add dependency graph, test coverage, git change frequency and secret/dependency scanning when appropriate. Exclude generated, vendored and irrelevant files. Use existing tooling first; if extra tools are needed, execute them ephemerally or outside the project so the review does not change manifests or lockfiles. Record versions, scope, commands and failures.

```text
Inspect [source roots] with existing deterministic tools. Save full outputs
under review/tools/ and a short inventory in review/TOOLS.md. Include build
and test status, language/module manifests, entry points, public symbols,
dependency edges, test locations, file size and recent change frequency.
Do not edit production source or add project dependencies. Distinguish missing
tool coverage from a clean result. Return only the inventory summary and paths.
```

Create a machine-oriented `review/INDEX.md` (or CSV/JSON if large) with module, path, LOC, public symbols, dependencies and dependants, recent commits, complexity, duplication and test references. **Generate it with tools, not the model:** ctags or an LSP for symbols, a dependency-graph tool for edges, `git log` for change counts, and lizard/duplication CSV output for metrics, merged by a short script. The agent writes the script and a summary, never the index rows. The index can be long on disk; don't paste it into context.

Create `review/HOTSPOTS.md` with candidates and their separate signals: change frequency, complexity, duplication, coverage, criticality and tool findings. Rank with a documented judgement or normalised scores, not a product of unrelated raw metrics. Note blind spots and low-change areas with high consequence. A hotspot is a review priority, not proof of a defect.

## Phase 2: Small architecture map

```text
Build review/MAP.md from review/INDEX.md, manifests, dependency graphs,
entry points and tests. Do not read modules file by file. Open source only
where metadata cannot answer a specific architectural question. Record major
responsibilities, boundaries, state/concurrency model, likely hot paths and
dominant conventions with cited examples or counts. Mark uncertainty rather
than guessing. Keep MAP.md to two pages; do not critique or edit source.
```

Check the map against a few representative entry points and correct errors before using it to judge conventions. The map describes the system; the index keeps detailed retrieval coordinates.

## Phase 3: Narrow review by question

Choose a track and start with the five highest-value candidates. Possible tracks are correctness (contracts, edge cases, error handling, state and tests), security (trust boundaries, authorisation, secrets and unsafe operations), architecture (coupling and ownership), performance (measured hot paths) and maintainability (duplication, dead code and needless complexity). Do not run every checklist on every module.

```text
Review [specific symbols/ranges] for [one track and concrete concern], using
review/MAP.md, the relevant review/tools/ summary and named callers. Start
with symbol search; obey the retrieval budget in this guide. For each candidate
write to review/FINDINGS.md:
ID | category | path | symbol | line range | claim | evidence (max 5 lines)
| related symbols/callers | tool reference | impact | confidence | effort.
Separate observation from inference. Report only actionable findings and
state what would falsify uncertain ones. No source edits. Max 10 findings.
```

After each batch of five, update `review/PROGRESS.md`: confirmed worthwhile findings, false-positive rate, approximate context/tool cost, remaining high-risk questions and expected value of another batch. Continue when a meaningful finding emerged or an unresolved high-risk area remains. Otherwise stop. This is a heuristic; mandated assurance or incident scope can override it.

## Phase 4: Verify and prioritise

```text
In a fresh context, inspect each finding's exact symbol/range and the minimum
related callers needed. Classify CONFIRMED, THEORETICAL, WRONG, or NOT WORTH IT.
State one evidence-based reason and any remaining uncertainty. Write
review/TRIAGED.md sorted by practical impact, confidence and effort. Do not
edit source or rediscover repository structure.
```

Spot-check consequential confirmed findings. If verification is unreliable, revise the prompt or use an independent reviewer. Do not promote tool warnings to confirmed defects without contextual checks.

## Phase 5: Implement one finding

`TRIAGED.md` is the starting record for why the change is needed. The implementer opens only the named files and directly referenced definitions. If the finding appears false or scope must expand, record that and return to triage.

```text
Implement confirmed finding [ID] only. State the intended behaviour and files
before editing. For a bug, first add a failing regression test where feasible.
For a refactor, preserve behaviour against the baseline and relevant invariants.
Run fast targeted checks after each edit, affected/module tests before commit,
then the required full checks and baseline comparison before completion.
Review the diff for unrelated changes. Record commands, results and commit in
review/HANDOFF.md. Do not weaken tests to make them pass.
```

For duplication, first reconcile behavioural differences, then extract shared code with a check at each step. A fresh reviewer should inspect high-risk diffs for behaviour, constraints and scope. A commit per coherent finding is useful; don't force unrelated mechanical steps into separate commits merely to satisfy a count.

### Performance track

Profile a representative workload and record workload sizes, environment, mean and tail or worst-case latency as relevant. Select targets from observed cost, implement one change and compare under the same conditions. A small measured gain may still matter at scale; a fixed 2% threshold is not universal. Check correctness and variance before claiming improvement.

## Debugging a known issue

For normal bugs, reproduce, trace the narrow data/control path, rank hypotheses and choose the cheapest discriminating experiment before changing production logic. Add targeted temporary instrumentation when needed and remove it afterwards. For regressions, test known-good and known-bad endpoints before automating `git bisect`; independently inspect the resulting commit. Write a failing test for the confirmed cause, fix, then run affected and baseline checks.

During an outage, exploit or active data corruption, containment, rollback or feature disablement can precede full root-cause confirmation. Preserve evidence and investigate after stabilisation. Repeated failed fixes are a signal to re-examine assumptions and design, not a universal numeric rule.

## Context and handoff

Use one narrow task per session. Clear context after repeated correction or when the investigation drifts; carry forward findings through files, not pasted logs. Git is the durable record for edits, including shell-made changes. Do not rely on agent checkpoints as a substitute.

```markdown
# review/HANDOFF.md
- Phase and status: done / in progress / blocked (with reason)
- Completed: finding IDs, paths and commits
- Checks: exact commands and results, with links to logs
- Open questions and blocked items:
- Next action: one concrete bounded task and its retrieval coordinates
```

Receiving prompt:

```text
Read review/HANDOFF.md and the project rules. Open only the named index,
finding and source ranges needed for the next action. Do not replay completed
analysis. Verify the last relevant check if the working tree or environment
changed. Update the handoff with evidence and a bounded next step.
```

## Local models and escalation

Local models can handle tool runs and bounded edits when tool calling is reliable. Pin context explicitly, check the effective context and memory use, and narrow the task if a larger window causes offload or poor retrieval. A 64K context can help agentic coding when hardware permits; it is not a reason to fill the window. Tool versions, defaults and model capabilities change; verify them in the actual environment.

| Work | Small local model | Larger local model | Stronger independent model or human |
| --- | --- | --- | --- |
| Phase 1 tool runs and summaries | Yes | Yes | Not needed |
| INDEX script, baseline harness | With a clear spec | Yes | If design decisions are needed |
| Phase 2 map | No | Draft only | Yes |
| Phase 3 review | Single symbol or range | Bounded batches | Cross-module findings |
| Phase 4 verification | No | No | Always a different model or person |
| Phase 5 single finding | Low-risk edits | Yes | Consequential changes |

Add these stop rules to `AGENTS.md` so a smaller agent hands off instead of thrashing:

```markdown
# Stop and hand off when
- the same check fails twice after your changes
- the next step needs more than the cross-module retrieval budget
- you cannot state what would confirm or falsify a finding
- a change would touch a project constraint or protected path
Before stopping: revert uncommitted changes, set HANDOFF.md status to
"blocked" with the reason and the last check result, then end the session.
```

Handoff works in both directions: once a stronger model writes MAP.md or TRIAGED.md, the local agent can take the mechanical work back.

## Evidence and limits

The historical studies support verification, not a fixed estimate of current agent performance. METR's 2025 randomised study found a 19% slowdown for experienced open-source developers in familiar repositories using early-2025 AI tools. Its 2026 update reports new estimates with wide confidence intervals and serious selection effects, so the current direction and size of the effect are uncertain. CodeScene's 2024 JavaScript/TypeScript code-smell benchmark found at best about 37% behaviour-preserving refactorings among the tested models; its fact-checking layer raised correctness among accepted candidates to roughly 96–99%. Neither number estimates all 2026 coding agents. Measure your own review yield, verification cost and regression rate.

## Appendix A: Common deterministic tools

| Language | Static analysis | Duplication | Complexity |
| --- | --- | --- | --- |
| C / C++ | clang-tidy, cppcheck | jscpd or PMD CPD | lizard |
| Python | ruff, mypy | jscpd | radon or lizard |
| JavaScript / TypeScript | eslint, tsc --noEmit | jscpd | eslint complexity rule or lizard |
| Java / Kotlin | PMD, SpotBugs or detekt | PMD CPD | PMD |
| Go | go vet, staticcheck | jscpd | gocyclo |
| Rust | clippy | jscpd | clippy |

lizard and jscpd cover most languages, so they are a safe default for mixed codebases. For symbol indexes, use universal-ctags or the language server your editor already runs.

## Sources

- [Anthropic, Claude Code best practices](https://code.claude.com/docs/en/best-practices): scoped exploration, verification, context and fresh review.
- [Google Engineering Practices, what to look for in code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html): design, functionality, complexity and tests.
- [OWASP Secure Code Review Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secure_Code_Review_Cheat_Sheet.html): contextual security review.
- [CodeScene, Refactoring vs Refuctoring (2024)](https://codescene.com/hubfs/whitepapers/Refactoring%20vs%20Refuctoring%20Advancing%20the%20state%20of%20AI%20automated%20code%20improvements.pdf): benchmark and verification figures ([summary](https://www.janeasystems.com/blog/ai-and-refactoring-part-2)).
- [METR 2025 developer productivity study](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) and [2026 update](https://metr.org/blog/2026-02-24-uplift-update/): changing productivity evidence.
- [Ollama context length documentation](https://docs.ollama.com/context-length): verify current defaults and hardware trade-offs.
- [OpenCode with local models](https://yuv.ai/learn/opencode-cli) and [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder): local agent setup and coder models.
