# model-atlas Context

## What You Are Doing

MCP server exposing a navigable semantic network of ML models. Your job is to write correct code; LintGate's job is to catch the discipline failures that waste your intelligence budget.

## Know Your Epistemic State

Before acting on a file you haven't read, before trying a third approach, before ignoring a behavioral finding — ask yourself:

- **Do I have theory context?** If not → `build_theory_pack`. The theory profile tells you what this project values, how it solves problems, and what it considers anti-patterns. Without it you are guessing at alignment.
- **Am I session-ready?** The session gate checks three things: theory profile has core_theory + problem_solving + alignment facets with claims, at least one enforceable rule exists, and no missing required facets. If the gate fires an advisory, run `bootstrap_context_files` before continuing.
- **What is my prediction accuracy?** If `constraint_check` shows accuracy below 50% after 5+ predictions, your mental model of the project's constraints is wrong. Stop and re-orient.
- **What is the coherence state?** If `controlplane_run` shows cross-channel disagreement (lint + tests + git pointing at the same files), that convergence IS the diagnosis. Read it.

## Dispositions

**When starting a session** — orient first. Call `build_theory_pack` or `controlplane_status`. Read theory claims before writing code. The 200 tokens you spend orienting prevent the 2,000 tokens you spend on a doomed approach.

**When acting** — register predictions. Before any Bash command, call `constraint_check` with a structured prediction (`prediction_type`: exit_code/error_signature/stdout_contains, `prediction_value`: what you expect). The system checks your prediction on the next tool event. This is not bureaucracy — it builds the accuracy signal that modulates behavioral finding confidence.

**When stuck** — DO NOT try variant #4. If you have cycled through 3 approaches, your problem is not execution — it is understanding. Run `constraint_check` to see your constraint coverage. Read the theory claims attached to behavioral findings. The answer is almost always in the constraints you have not yet verified, not in the approaches you have not yet tried.

**When the system speaks** — findings are weather reports, not commands. "3 approaches in 20min, all failed" is an observation. The theory coda attached to it connects the observation to the project's values. You decide what to do. But if you ignore a hard signal and try the same pattern again, the system will escalate — and it will be right.

**When context evolves** — review and apply patches explicitly. When the living_context system generates a context patch (from accepted constraints, confirmed predictions, or recurring behavioral signals), use `context_patch_review` to see the diff. Use `context_patch_apply` to write it. Patches are never auto-applied. The cumulative rebasing ensures multiple patches to the same section compose correctly — each apply re-reads the current on-disk state.

**When you change the system** — update the docs immediately, in the same action. If you add an MCP tool, add it to AGENTS.md and README.md tool tables and increment the count. Source of truth for tool count: `grep -Rho "@mcp.tool()" mcp_server.py mcp_tools | wc -l`. Documentation precision has compounding returns — one stale count becomes a chain of wrong assumptions across every session that reads it.

## Mission

- Keep feedback loops tight between generated code and validated code quality.
- Prefer deterministic checks and explicit diagnostics over ambiguous heuristics.
- Preserve graceful degradation when optional tooling is unavailable.
- Offload discipline to the deterministic layer so the agent spends its intelligence budget on novel reasoning.

## Guardrails

- DO NOT disable lint channels globally to hide regressions.
- DO NOT auto-apply generated repairs without explicit acceptance.
- DO NOT try a 4th approach without running `constraint_check` first.
- DO NOT ignore theory codas on behavioral findings — they exist to connect observations to project values.
- MUST keep hook and MCP outputs machine-readable and stable for downstream consumers.
- MUST preserve backward-compatible MCP tool contracts unless versioned intentionally.
- MUST update AGENTS.md, README.md, and docs/design.md when adding, removing, or changing MCP tools. Verify with `grep -Rho "@mcp.tool()" mcp_server.py mcp_tools | wc -l`.
- MUST update docs/design.md YAML examples when adding config options.

<!-- LINTGATE:BEGIN theory_alignment v1 -->
## Theory-Aligned Development
- Core theory: Understand before acting — orient on the constraint space before writing code, because the cost of a wrong approach compounds while the cost of reading is fixed.
- Preferred approach: A structured semantic network of ML models, exposed as an MCP tool, so the LLM you're already talking to can see the model landscape instead of guessing at it.
- Alignment criteria: Why signed hierarchies instead of flat categories: A categorical "size" field with values {small, medium, large} can't express proximity.
- Architecture intent: Four indexed queries instead of 18K individual get_model() calls.
<!-- LINTGATE:END theory_alignment -->

<!-- LINTGATE:BEGIN do_dont v1 -->
- DO: A structured semantic network of ML models, exposed as an MCP tool, so the LLM you're already talking to can see the model landscape instead of guessing at it.
- DO: Why signed hierarchies instead of flat categories: A categorical "size" field with values {small, medium, large} can't express proximity.
- DO NOT: Do not try a 4th approach without first enumerating all known constraints and verifying which ones the new approach actually addresses.
- DO NOT: Do not discover constraints one-at-a-time through failure — enumerate the full constraint space upfront by reading before acting.
- DO NOT: Do not re-attempt an approach that already failed unless the conditions that caused the failure have changed.
- DO NOT: Do not use O(n²) algorithms when O(n) alternatives exist — quadratic membership checks on lists, re.compile inside loops, and sorted()[0] instead of min() are structural mistakes, not style issues.
<!-- LINTGATE:END do_dont -->

<!-- LINTGATE:BEGIN machine_rules v1 -->
# Add project-specific constraints as they become stable:
# LINTGATE_FORBID_REGEX: <regex>
# LINTGATE_REQUIRE_REGEX: <regex>
<!-- LINTGATE:END machine_rules -->

<!-- LINTGATE:BEGIN context_map v1 -->
## Context Map
- `.claude/rules/theory.md` - extracted theory summaries and anti-patterns.
- `.claude/lintgate.yaml` - lint and ControlPlane configuration.
<!-- LINTGATE:END context_map -->

## Debt Tracking Policy

- Known structural debt should be tracked in .claude/lintgate.yaml exemptions with ticket references.
- Exemptions should target specific files and codes instead of global severity downgrades.
- New exemptions require a concrete rationale and a remediation ticket.

## Deep Reference

- Architecture of Inquiry protocol: `.claude/rules/inquiry.md`
- Tool reference by cognitive mode: `AGENTS.md`
- Design deep dive: `docs/design.md`
