# Mastra Observational Memory: Research Document

**Date:** 2026-03-16
**Sources:** mastra.ai/research, mastra.ai/blog, mastra.ai/docs, GitHub, VentureBeat, techbuddies.io

---

## 1. Executive Summary

Mastra's Observational Memory (OM) is a text-based, long-context memory system for AI agents that achieves **94.87% on LongMemEval** (gpt-5-mini) — the highest score ever recorded on this benchmark by any system with any model. It uses two background agents (Observer and Reflector) to compress conversation history into a dated observation log, replacing raw message history as it grows. No vector or graph database required.

Key claims:
- 94.87% LongMemEval (gpt-5-mini), beating previous SOTA by 3+ points
- 84.23% with gpt-4o, beating the gpt-4o oracle baseline (82.4%)
- 5-40x compression ratios on tool-heavy workloads
- 4-10x token cost reduction via prompt caching
- Fully open source: `packages/memory/src/processors/observational-memory/`

---

## 2. Architecture

### 2.1 Two-Block Context Window

The context window is split into two blocks:

1. **Observation Block** (prefix): Compressed, dated log of prior interactions. Append-only, stable, and prompt-cacheable.
2. **Raw History Block** (suffix): Current session messages preserved verbatim until compression threshold is hit.

This structure enables full cache hits on every turn (messages append at the end). When observation runs, messages are replaced with new observations (partial cache hit). Reflection (infrequent) invalidates entire cache.

### 2.2 Three-Tier Memory Hierarchy

1. **Recent messages**: Exact conversation history (short-term)
2. **Observations**: Dense notes created by the Observer about what occurred
3. **Reflections**: Condensed observations created by the Reflector when memory grows too large

### 2.3 Background Agents

**Observer Agent:**
- Triggers when unobserved message history reaches **30,000 tokens** (default, configurable)
- Compresses raw messages into structured, dated observations
- Appends new observations to the observation block
- Removes original raw messages after compression
- Achieves **3-6x compression** on text content
- Achieves **5-40x compression** on tool-heavy workloads (large tool outputs)
- Tracks current task and suggested response

**Reflector Agent:**
- Triggers when total observation size reaches **40,000 tokens** (default, configurable)
- Performs "garbage collection" on observations
- Consolidates related items, identifies patterns
- Removes superseded/irrelevant observations
- Maintains stable, cacheable context window

### 2.4 Three-Date Temporal Model

Each observation carries up to three temporal anchors for enhanced temporal reasoning:

1. **Observation date**: When the observation was created
2. **Referenced date**: Dates explicitly mentioned in the content
3. **Relative date**: Computed offsets (e.g., "2 days from today")

This is critical for temporal reasoning questions in benchmarks.

---

## 3. Observation Format

Text-based structured format using:
- Two-level bulleted lists (tasks/events with details)
- Emoji-based priority signals
- Date-grouped titled sections

### Priority Emoji System
- `🔴` = High priority / Critical
- `🟡` = Medium priority / Possibly important
- `🟢` = Low priority / Informational

### Example

```
Date: 2026-01-15
- 🔴 12:10 User building Next.js app with Supabase auth, due January 22nd
  - 🔴 12:10 Server components with client-side hydration
  - 🟡 12:12 Middleware configuration question for protected routes
  - 🔴 12:15 App name: "Acme Dashboard"
```

---

## 4. Configuration & API

### 4.1 Minimal Setup

```typescript
import { Memory } from '@mastra/memory'
import { Agent } from '@mastra/core/agent'

export const agent = new Agent({
  name: 'my-agent',
  instructions: 'You are a helpful assistant.',
  model: 'openai/gpt-5-mini',
  memory: new Memory({
    options: {
      observationalMemory: true,  // uses default google/gemini-2.5-flash
    },
  }),
})
```

### 4.2 Custom Model

```typescript
const memory = new Memory({
  options: {
    observationalMemory: {
      model: 'deepseek/deepseek-reasoner',
    },
  },
})
```

### 4.3 Token Budget Configuration

```typescript
const memory = new Memory({
  options: {
    observationalMemory: {
      observation: {
        messageTokens: 30_000,        // Observer triggers at this threshold
      },
      reflection: {
        observationTokens: 40_000,    // Reflector triggers at this threshold
      },
      shareTokenBudget: false,
    },
  },
})
```

### 4.4 Async Buffering Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `observation.bufferTokens` | `0.2` | Buffer frequency (20% of messageTokens threshold) |
| `observation.bufferActivation` | `0.8` | Aggressiveness clearing message window |
| `observation.blockAfter` | `1.2` | Safety threshold multiplier |
| `reflection.bufferActivation` | `0.5` | Reflection start point |
| `reflection.blockAfter` | `1.2` | Reflection safety threshold |

Disable async buffering:
```typescript
observation: {
  bufferTokens: false,
}
```

### 4.5 Observer Context Optimization

```typescript
observation: {
  previousObserverTokens: 10_000, // ~10k tokens of recent observations sent to Observer
}
```

Options: `2000` (default), `0` (omit previous observations), `false` (disable truncation)

### 4.6 Scopes

- **Thread scope** (default, tested): Each thread maintains separate observations. Requires valid `threadId`.
- **Resource scope** (experimental): Observations shared across threads for a resource. Enables cross-conversation memory.

```typescript
observationalMemory: {
  scope: 'thread', // or 'resource'
}
```

### 4.7 Recommended Models

- `google/gemini-2.5-flash` (default)
- `openai/gpt-5-mini`
- `anthropic/claude-haiku-4-5`
- `deepseek/deepseek-reasoner`
- Context window requirement: **128K+ tokens** recommended

### 4.8 Storage Adapters

Only supports: `@mastra/pg`, `@mastra/libsql`, `@mastra/mongodb`

---

## 5. Benchmark Results: LongMemEval

### 5.1 Dataset

- **LongMemEval_s**: 500 questions across ~57M tokens of conversation data
- Each question has ~50 sessions attached
- Sequential chronological data processing
- gpt-4o serves as judge model (binary correct/incorrect)
- Unweighted averaging across 6 categories

### 5.2 Overall Scores

| System | Model | Score |
|--------|-------|-------|
| **Mastra OM** | **gpt-5-mini** | **94.87%** |
| **Mastra OM** | **gemini-3-pro-preview** | **93.27%** |
| Hindsight | gemini-3-pro-preview | 91.40% |
| Mastra OM | gemini-3-flash-preview | 89.20% |
| **Mastra OM** | **gpt-4o** | **84.23%** |
| Oracle baseline | gpt-4o | 82.40% |
| Supermemory | gpt-4o | 81.60% |
| Supermemory | gemini-3-pro | 85.20% |
| Mastra RAG | gpt-4o | 80.05% |
| Full context | gpt-4o | 60.20% |

### 5.3 Per-Category Breakdown (gpt-5-mini, 94.87% overall)

| Category | Score |
|----------|-------|
| Knowledge-update | 96.2% |
| Temporal-reasoning | 95.5% |
| Single-session-preference | 100% |
| Single-session-assistant | 94.6% |
| Multi-session | 87.2% |
| Single-session-user | 95.7% |

### 5.4 Key Observations

- **Beats oracle**: OM with gpt-4o (84.23%) exceeds oracle baseline (82.4%) despite ingesting ALL ~50 conversations vs oracle's filtered 1-3 sessions
- **Model scaling advantage**: OM gains 9 points going from gpt-4o to gemini-3-pro, vs Supermemory gaining only 3.6 points — better architecture amplifies model improvements
- **Average context window**: ~30k tokens (despite processing ~57M tokens of input)
- **6x compression** achieved in LongMemEval runs

---

## 6. Comparison: OM vs RAG vs Full Context

| Factor | Observational Memory | Traditional RAG | Full Context |
|--------|---------------------|-----------------|--------------|
| Context stability | Fixed prefix (cacheable) | Dynamic retrieval breaks caching | Everything included |
| Focus | Agent's own history/decisions | External corpus search | Raw everything |
| Output format | Event-based decision log | Narrative summaries | Raw messages |
| Compression | Frequent, small-batch (~30K) | Infrequent, large-batch | None |
| Database required | Text-only (no vector DB) | Vector DB + embeddings | N/A |
| Cost efficiency | 4-10x via caching | No caching benefit | Highest cost |
| LongMemEval (gpt-4o) | 84.23% | 80.05% | 60.20% |

---

## 7. Implementation: Open Source Code Structure

Located at: `github.com/mastra-ai/mastra/tree/main/packages/memory/src/processors/observational-memory/`

### Core Files
- `observational-memory.ts` — Main implementation module
- `observer-agent.ts` — Observer agent (compression)
- `reflector-agent.ts` — Reflector agent (garbage collection)
- `operation-registry.ts` — Registry managing operations
- `index.ts` — Public exports

### Utility Files
- `date-utils.ts` — Date handling (three-date model)
- `markers.ts` — Marker management
- `token-counter.ts` — Token counting (via `tokenx` library)
- `thresholds.ts` — Threshold configuration
- `types.ts` — TypeScript type definitions

### Testing
- `__tests__/` — Test suite
- `__fixtures__/repro-captures/` — Reproducible test fixtures

### LongMemEval Benchmark Runner
- `github.com/mastra-ai/mastra/tree/main/explorations/longmemeval`

---

## 8. Mastra Code: OM in a Coding Agent

Mastra Code is an open-source TUI coding agent that uses observational memory for context compression. Key details:

- **Problem solved**: Context window exhaustion in long coding sessions
- **Storage**: LibSQL (local SQLite) for threads, messages, and observations
- **Result**: Sessions lasting months maintain useful knowledge without degradation
- **Features**: Plan mode, Build mode, AST-aware editing (ast-grep), LSP integration, 1,800+ models
- **Memory becomes invisible**: "Memory stops being something you manage and fades into the background"

Also available as `@mastra/opencode` plugin (PR #12925) that brings OM into OpenCode sessions via standalone `observe()` API.

---

## 9. Mastra's Memory System (Complete)

Four complementary memory types:

1. **Message History**: Recent messages for UI rendering and short-term continuity
2. **Observational Memory**: Background Observer/Reflector compression for long-term continuity
3. **Working Memory**: Persistent structured user info (names, preferences, goals) — essentially a user profile
4. **Semantic Recall**: Vector-based retrieval of past messages by meaning (requires vector DB + embeddings)

---

## 10. "Patterns for Building AI Agents" Book

**Title:** Patterns for Building AI Agents (1st Edition)
**Author:** Sam Bhagwat (CEO of Mastra, co-creator of Gatsby)
**Announced:** November 12, 2025
**Series:** Volume 2 of planned trilogy (after "Principles of Building AI Agents")
**Distribution:** 32,500 books printed across both books, distributed globally
**Available at:** mastra.ai/book-2 and Amazon (ISBN: 9798270198107)

### Chapter Structure

**Chapter 1: From Wishlist to Working Agent**
- Whiteboard Agent Capabilities
- Evolve Your Agent Architecture
- Dynamic Agents
- Human-in-the-Loop

**Chapter 2: Intro to Context Engineering**
- Parallelize Carefully
- Share Context Between Subagents
- Avoid Context Failure Modes
- Compress Context
- Feed Errors Into Context

**Chapter 3: From MVP to Production**
- List Failure Modes
- List Critical Business Metrics
- Cross-Reference Failure Modes and Success Metrics
- Iterate Against Your Evals
- Create an Eval Test Suite
- Have SMEs Label Data
- Create Datasets from Production Data
- Evaluate Production Data

**Chapter 4: Autonomy Is a Two-Edged Sword**
- Prevent the Lethal Trifecta
- Sandbox Code Execution
- Granular Agent Access Control
- Agent Guardrails

**Chapter 5: What's Next(ish)**

### Key Themes
- "Principles" (book 1) covers **what to build**; "Patterns" (book 2) covers **how to build**
- Four pillars: Agent Design/Architecture, Context Engineering, Evals, Agent Security
- Described as "a work in progress and a living document"

---

## 11. Design Decisions & Known Limitations

### Why Text-Based (No Vector/Graph DB)
- Observations are plain text — no separate retrieval phase at inference time
- Context window is completely stable, predictable, reproducible, prompt-cacheable
- Simpler infrastructure requirements

### Best Suited For
- In-app agents for CMS/SaaS (Sanity, Contentful)
- AI SRE and operations agents requiring investigation audit trails
- Document-processing workflows needing durable decision history
- Long-running coding sessions (Mastra Code)

### NOT Suited For
- Open-ended knowledge discovery over large, evolving datasets
- Compliance/audit scenarios requiring full-corpus recall
- Workloads where external information retrieval is primary

### Migration
- No manual migration required
- Existing threads lazily observe backlog when thresholds exceeded

---

## 12. Key Takeaways for Trix

1. **Compression over retrieval**: OM proves that progressive compression (Observer/Reflector) can beat RAG on long-context benchmarks without vector databases
2. **Three-date temporal model**: Critical for temporal reasoning — Trix's enrichment pipeline could adopt similar multi-anchor temporal tagging
3. **Prompt caching**: The stable, append-only observation prefix enables massive cost savings via provider prompt caching. Trix's RAG approach inherently breaks caching.
4. **Emoji priority signals**: Simple but effective way to encode importance in compressed text
5. **Token budget thresholds**: Two-threshold system (30k observe, 40k reflect) is elegant — configurable but with sensible defaults
6. **Model scaling amplification**: Better architecture amplifies model improvements (9pp gain for OM vs 3.6pp for Supermemory when upgrading models)
7. **Average 30k token context**: Despite processing 57M tokens, the effective context window stays at ~30k — very efficient
8. **94.87% vs Trix's 52.8%**: Massive gap, though benchmarks differ (LongMemEval vs LoCoMo) and architectures are fundamentally different (OM = sequential session compression, Trix = RAG over stored memories)

---

## Sources

- [Mastra Research: Observational Memory](https://mastra.ai/research/observational-memory)
- [Mastra Blog: Announcing Observational Memory](https://mastra.ai/blog/observational-memory)
- [Mastra Docs: Observational Memory](https://mastra.ai/docs/memory/observational-memory)
- [Mastra Docs: Memory Overview](https://mastra.ai/docs/memory/overview)
- [GitHub: Observational Memory Source](https://github.com/mastra-ai/mastra/tree/main/packages/memory/src/processors/observational-memory)
- [GitHub: LongMemEval Benchmark Runner](https://github.com/mastra-ai/mastra/tree/main/explorations/longmemeval)
- [GitHub PR: @mastra/opencode Plugin](https://github.com/mastra-ai/mastra/pull/12925)
- [VentureBeat: Observational Memory Cuts AI Agent Costs 10x](https://venturebeat.com/data/observational-memory-cuts-ai-agent-costs-10x-and-outscores-rag-on-long)
- [TechBuddies: How Mastra's OM Beats RAG](https://www.techbuddies.io/2026/02/12/how-mastras-observational-memory-beats-rag-for-long-running-ai-agents/)
- [Mastra Blog: Patterns Book Announcement](https://mastra.ai/blog/patterns-book)
- [Patterns Book on Amazon](https://www.amazon.com/Patterns-Building-Agents-Sam-Bhagwat/dp/B0FXF5T79G)
- [Mastra Blog: Announcing Mastra Code](https://mastra.ai/blog/announcing-mastra-code)
- [Mastra Blog Index](https://mastra.ai/blog)
