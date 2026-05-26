# Memory Systems Implementation Research

Deep-dive into the actual source code of 5 top-performing memory systems, extracting patterns applicable to a TypeScript RAG system.

---

## 1. Graphiti (Zep) — Temporal Knowledge Graph

**Repo**: github.com/getzep/graphiti | **Language**: Python | **Backend**: Neo4j/FalkorDB/Kuzu

### Data Model

Three node types, three edge types:

```
EntityNode {
  uuid, name, group_id, labels, created_at,
  name_embedding: float[],     // vector for name similarity
  summary: string,             // evolving summary of entity
  attributes: dict             // custom structured properties
}

EpisodicNode {
  uuid, name, group_id, created_at,
  source: EpisodeType,         // message | json | text
  source_description: string,
  content: string,             // raw ingested data (provenance)
  valid_at: datetime,          // when original doc was created
  entity_edges: string[]       // edges referenced in this episode
}

CommunityNode {
  uuid, name, group_id, created_at,
  name_embedding: float[],
  summary: string              // community-level summary
}

EntityEdge {
  uuid, group_id, source_node_uuid, target_node_uuid, created_at,
  name: string,                // relationship name (WORKS_AT)
  fact: string,                // human-readable fact description
  fact_embedding: float[],     // vector for semantic search
  valid_at: datetime,          // WHEN this fact became true
  invalid_at: datetime,        // WHEN this fact was superseded
  expired_at: datetime,        // WHEN system detected it stale
  episodes: string[],          // provenance chain
  attributes: dict
}

EpisodicEdge  { MENTIONS relationship, minimal fields }
CommunityEdge { HAS_MEMBER relationship, minimal fields }
```

**Key insight**: The bi-temporal model uses `valid_at` (when fact became true in the real world) and `invalid_at` (when it was superseded) separately from `expired_at` (when the system detected the change). Old facts are never deleted, only marked invalid.

### Ingestion Pipeline (per episode)

Each `add_episode()` call runs this pipeline:

```
1. retrieve_episodes()         — fetch N previous episodes as context
2. extract_nodes() [LLM #1]   — identify entities from text
3. resolve_extracted_nodes()   — deduplicate against existing graph
4. extract_edges() [LLM #2]   — identify relationships between entities
5. resolve_extracted_edges()   — validate + deduplicate edges
6. extract_attributes() [LLM] — enrich entities with properties
7. generate embeddings         — vectorize facts and names
8. persist to graph DB         — save nodes, edges, episodic links
9. update_community() [LLM]   — optional community re-clustering
```

**LLM calls per message**: Minimum 3 (extract nodes, extract edges, resolve edges), up to 5+ with attributes and communities.

### Entity Extraction Prompt Pattern

```
Role: "expert entity extractor"
Input: previous messages (context) + current message + reference timestamp
Rules:
  - Always extract the speaker as first entity
  - Disambiguate pronoun references using context
  - Use full explicit names, not abbreviations
  - Exclude temporal information from entity names
  - Exclude relationship-type nodes
Output: structured list of {name, entity_type}
```

### Edge Extraction Prompt Pattern

```
Role: "expert fact extractor that extracts fact triples"
Input: previous context + current message + entities list + timestamp
Rules:
  - Relationships must connect TWO DISTINCT entities from the provided list
  - Entity names must EXACTLY match the provided list
  - Relation types in SCREAMING_SNAKE_CASE
  - Include temporal validity (valid_at in ISO 8601)
  - Paraphrase, don't quote verbatim
Output: list of {source, relation, target, fact, valid_at, invalid_at}
```

### Edge Invalidation (Bi-Temporal)

When a new edge arrives, the system:

1. Finds existing edges between the same entity pair
2. Calls LLM with `dedupe_edges.resolve_edge()` prompt:
   - Input: existing edges + new edge fact
   - LLM returns: `{duplicate_facts: [indices], contradicted_facts: [indices]}`
3. For contradicted edges:
   - If new edge has later `valid_at` → set existing `invalid_at = new.valid_at`, `expired_at = now()`
   - If new edge has earlier `valid_at` → set new edge's `invalid_at = existing.valid_at`
4. Duplicates are merged (not created twice)

**Key pattern**: The LLM decides what contradicts; the code handles temporal ordering deterministically.

### Hybrid Search

15 pre-configured search recipes combining:

| Method | Implementation |
|--------|---------------|
| **Semantic** | Cosine similarity on `fact_embedding` with `sim_min_score` threshold |
| **BM25** | Full-text keyword search via graph DB native indexes |
| **Graph traversal** | BFS from origin nodes up to `bfs_max_depth` |

**Reranking options**:
- **RRF (Reciprocal Rank Fusion)**: `score += 1/(rank + k)` across lists, k=1
- **Node distance**: Direct graph relationship distance to center node
- **Episode mentions**: Frequency of episode references
- **MMR**: `lambda * query_sim + (lambda-1) * max_diversity_penalty`
- **Cross-encoder**: Neural reranking model

The default `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` runs all 3 methods (BM25 + cosine + BFS) with cross-encoder reranking, across edges, nodes, and communities in parallel.

---

## 2. Hindsight (Vectorize) — Biomimetic Memory

**Repo**: github.com/vectorize-io/hindsight | **Language**: Python (PostgreSQL) | **SOTA on LongMemEval**

### Architecture: Retain / Recall / Reflect

Three operations, four retrieval strategies, three memory pathways (World Knowledge, Experiences, Mental Models).

### Retain Pipeline (10 stages)

```
1. extract_facts_from_contents() [LLM]
   - Chunks content at sentence/turn boundaries (120k char chunks)
   - LLM extracts structured facts: {what, when, where, who, why}
   - Each fact gets temporal fields: occurred_start, occurred_end
   - Entities extracted inline with facts
   - Causal relations extracted: {caused_by, strength: 0-1}
   - Language detection: output matches input language

2. augment_texts_with_dates()
   - Prepends temporal context to fact text for embedding

3. generate_embeddings_batch()
   - Vectorizes augmented fact texts

4. handle_document_tracking()
   - Creates/links document provenance records

5. store_chunks_batch()
   - Persists raw content chunks with doc references

6. insert_facts_batch()
   - Stores facts with deduplication

7. process_entities_batch()
   - Entity resolution via multi-factor scoring:
     - Name similarity (SequenceMatcher, weight 0-0.5)
     - Co-occurrence context (weight 0-0.3)
     - Temporal proximity (7-day window, weight 0-0.2)
   - DB constraint: ON CONFLICT (bank_id, LOWER(canonical_name)) DO NOTHING

8. create_temporal_links_batch()
   - Links facts occurring close in time

9. create_semantic_links_batch()
   - Links facts with similar embeddings

10. create_causal_links_batch()
    - Links facts with causal relationships (from LLM extraction)
```

**Key insight**: Facts get three types of inter-memory links (temporal, semantic, causal) that form a traversable graph, separate from the entity graph.

### Fact Extraction Prompts

Two modes:
- **Concise**: Extract only significant facts (personal info, preferences, events, plans, expertise). "Reduces 90% of unnecessary output."
- **Verbose**: Maximum detail capture across all 5 dimensions (what/when/where/who/why). "Never summarize or omit."

Critical rule: Output must match input language exactly. No translation allowed.

### Entity Resolution Algorithm

```python
def score_candidate(input_entity, candidate):
    score = 0.0
    # Name similarity (0-0.5)
    score += SequenceMatcher(input.text, candidate.canonical_name).ratio() * 0.5
    # Co-occurrence (0-0.3) - entities that appear together frequently
    score += overlap_with_nearby_entities(input, candidate) * 0.3
    # Temporal proximity (0-0.2) - 7-day decay window
    days_diff = abs(input.date - candidate.last_seen).days
    score += max(0, 1.0 - (days_diff / 7)) * 0.2
    return score
```

### Recall (4-Way Parallel Search)

```
Semantic:  vector similarity on fact embeddings
BM25:      full-text keyword matching (PostgreSQL)
Graph:     entity + temporal + causal link traversal
Temporal:  time-range filtering

→ Results merged via RRF + cross-encoder reranking
→ Budget parameter (LOW/MID/HIGH) controls graph depth
→ Token budget limits result size
```

Graph retrieval supports two strategies:
- **BFS**: Breadth-first search from anchor nodes
- **MPFP (Personalized PageRank)**: Spreading activation through the memory graph

### Consolidation (Background Job)

After retain, a background job runs:

```
1. Fetch unconsolidated memories (grouped by tags for tenant isolation)
2. For each memory batch, recall related existing observations
3. Single LLM call with all facts + related observations
4. LLM outputs: {creates: [...], updates: [...], deletes: [...]}
5. Execute actions sequentially
6. Trigger mental model refreshes if needed
```

Consolidation prompt rules:
- **Redundancy**: If same info stated differently → UPDATE to consolidate
- **Temporal contradictions**: Preserve both with time markers ("used to X, now Y")
- **Reference resolution**: When new facts clarify vague placeholders → UPDATE with concrete values
- **Entity separation**: Never merge observations about different people/topics

### Reflect (Agentic Loop)

Reflect runs an iterative tool-calling agent:

```
Forced tool sequence:
1. search_mental_models → user-curated summaries (highest quality)
2. search_observations  → auto-consolidated knowledge
3. recall              → raw facts (ground truth)
4. expand              → deepen context from specific memory IDs
5. done                → signal completion with validated citations

Safety:
- Anti-hallucination: "ONLY use information from retrieved tool results"
- Citation validation: only IDs actually retrieved are included
- Context budgeting: auto-triggers synthesis if approaching token limits
- Evidence requirement: blocks premature completion without evidence
```

---

## 3. A-Mem — Zettelkasten-Inspired Memory

**Repo**: github.com/WujiangXu/A-mem | **Language**: Python | **NeurIPS 2025**

### Memory Note Structure

```python
RobustMemoryNote {
    id: UUID,
    content: string,          # original experience text
    note: string,             # comprehensive LLM analysis
    keywords: list[string],   # extracted via LLM
    tags: list[string],       # semantic tags
    description: string,      # contextual summary
    links: dict,              # Zettelkasten neighbor indices
    context: string,          # semantic category
    importance_score: float,
    retrieval_count: int,
    evolution_history: list,  # change log
    embedding: float[]        # all-MiniLM-L6-v2
}
```

### Zettelkasten Linking Mechanism

Links are neighbor indices stored in `note.links`. The embedding index combines: `"content: {memory} context: {context} keywords: {keywords} tags: {tags}"`.

Retrieval uses k-NN on embeddings, then expands via `find_related_memories_raw()` which traverses linked neighbors beyond initial results — a 2-hop graph expansion.

### 3-Stage Memory Evolution Pipeline

Each new memory triggers:

**Stage 1: Evolution Decision [LLM]**
```
Input: new note + nearest neighbor memories
Output: one of:
  - NO_EVOLUTION → store as-is
  - STRENGTHEN → enhance details/connections
  - UPDATE_NEIGHBOR → propagate changes to linked memories
  - STRENGTHEN_AND_UPDATE → both
```

**Stage 2: Strengthen Details [LLM, conditional]**
```
Input: new note + context
Output: new connections to existing memories + updated tags
Parser: section-marker based (not JSON)
```

**Stage 3: Update Neighbors [LLM, conditional]**
```
Input: neighbor memories + new note
Output: tag updates + context refinements per neighbor
```

**Key insight**: Uses section-marker parsing instead of JSON, with heuristic fallbacks. This makes it robust against LLM formatting failures.

### Robustness Pattern

```python
@retry(max_retries=3, delays=[1, 2, 4])  # exponential backoff
def llm_call(prompt):
    try:
        response = llm.generate(prompt)
        return parse_sections(response)
    except ParseError:
        return heuristic_extract(prompt.content)  # graceful degradation
```

---

## 4. TiMem — Temporal Memory Tree

**Repo**: github.com/TiMEM-AI/timem | **Language**: Python | **75.3% on LoCoMo, 76.9% on LongMemEval**

### 5-Level Memory Hierarchy

```
L1 (Facts/Fragments):
  - Every N dialogue turns → one fragment
  - Extracts: keywords, named entities, urgency indicators
  - Importance scoring: weighted across 5 dimensions
    (content length, keyword density, entity count, question freq, urgency)
  - Two summarization strategies:
    * Independent: summarize current window alone
    * Progressive: M_l = Summarize(λ·F_l ⊕ (1-λ)·M_{l-1})
      where λ is merge weight with previous summary

L2 (Session Summaries):
  - Aggregates L1 fragments per session
  - LLM prompt: "l2_session_summary"

L3 (Daily Patterns):
  - Synthesizes L2 sessions by date
  - Identifies short-term behavioral patterns
  - LLM prompt: "l3_daily_summary"

L4 (Weekly Trends):
  - Integrates L3 daily patterns
  - Tracks recurring behaviors
  - LLM prompt: "l4_weekly_summary"

L5 (Stable Profile):
  - Extracts long-term themes from L4
  - Establishes persistent identity traits
  - LLM prompt: "l5_high_level_summary"
```

Each level follows the same generation pattern:
```
1. Fetch templated prompt with context variables
2. Concatenate child contents + historical memory
3. LLM generation with exponential backoff retry
4. Validate non-empty + minimum length
5. Persist to database
```

### Adaptive Recall

Query complexity determines which levels to traverse:
- **Simple factual**: L1-L2 only (e.g., "What's Alice's email?")
- **Moderate**: L1-L3 (e.g., "What happened this week?")
- **Complex**: L1-L5 (e.g., "What kind of person is Alice?")

### Retry Strategy (notable)

```python
# Error classification determines retry behavior:
- Connection errors:  fast retry (50ms-3s delays)
- Rate limits:        extended wait (30s+)
- Server errors 5xx: progressive backoff with fast initial
- Timeouts:          similar to server errors

# Layer-specific timeouts:
L1: 60s, L2: 120s, L3: 300s, L4: 600s, L5: 900s
# Jitter applied to prevent thundering herd
```

---

## 5. mem0 — Graph Memory with Conflict Resolution

**Repo**: github.com/mem0ai/mem0 | **Language**: Python | **+26% over OpenAI Memory on LoCoMo**

### Two Memory Stores (Parallel)

mem0 runs vector store and graph store concurrently:

```
add(messages):
  1. Validate session identifier (user_id, agent_id, or run_id)
  2. Normalize messages to [{role, content}]
  3. PARALLEL:
     a. Vector store pipeline → flat fact memories
     b. Graph store pipeline → entity-relationship triples
```

### Vector Store Pipeline: Fact Extraction + Conflict Resolution

**Step 1: Extract facts [LLM]**

Prompt (FACT_RETRIEVAL_PROMPT):
```
"You are a Personal Information Organizer, specialized in accurately
storing facts, user memories, and preferences."

Seven categories:
1. Personal Preferences (likes, dislikes, favorites)
2. Important Personal Details (name, relationships, milestones)
3. Plans and Intentions (future actions, goals)
4. Activity Preferences (hobbies, exercise, learning)
5. Health Information (allergies, medications, conditions)
6. Professional Details (job, skills, achievements)
7. Miscellaneous (any other relevant info)

Input-output examples:
- "Hi, I'm John. I'm a software engineer."
  → {"facts": ["Name is John", "Is a Software engineer"]}

Rules: extract from user+assistant messages only, not system messages.
```

**Step 2: Search existing memories**

For each extracted fact, embed and search vector store for similar existing memories.

**Step 3: Conflict resolution [LLM]**

Prompt (DEFAULT_UPDATE_MEMORY_PROMPT):
```
"You are a smart memory manager which controls the memory of a system."

Four operations with examples:

ADD: When new fact has no matching existing memory
  → {id: "new", text: "...", event: "ADD"}

UPDATE: When new fact modifies/contradicts existing memory
  → {id: "existing-uuid", text: "updated text", old_memory: "...", event: "UPDATE"}

DELETE: When new fact makes existing memory obsolete
  → {id: "existing-uuid", event: "DELETE"}

NONE: When fact already exists unchanged
  → {id: "existing-uuid", event: "NONE"}
```

**Step 4: Execute actions**
- ADD: embed + store new memory
- UPDATE: update text, preserve creation timestamp, log history
- DELETE: remove from vector store
- NONE: update session identifiers only

### Graph Store Pipeline

**Step 1: Entity extraction [LLM + tool calling]**

System prompt: "You are a smart assistant who understands entities and their types."
Tool: `extract_entities({entities: [{entity, entity_type}]})`

Special rule: Self-references ('I', 'me', 'my') → replace with `user_id`.

**Step 2: Relationship extraction [LLM + tool calling]**

Prompt: EXTRACT_RELATIONS_PROMPT (establishes relationships among entities)
Tool: `establish_relationships({entities: [{source, relationship, destination}]})`

Relationship names are sanitized for Cypher compatibility.

**Step 3: Similarity search against existing graph**

For each extracted entity, search graph nodes with cosine similarity (threshold 0.7).

**Step 4: Delete evaluation [LLM]**

Prompt: DELETE_RELATIONS_SYSTEM_PROMPT — identifies stale relationships.
Critical rule: "DO NOT DELETE if there is a possibility of same type of relationship but different destination nodes."
Tool: `delete_graph_memory({source, relationship, destination})`

**Step 5: Upsert**

Uses Cypher MERGE with:
- ON CREATE SET: initialize node/edge
- ON MATCH SET: increment mention counter

---

## Cross-Cutting Patterns for TypeScript Adaptation

### Pattern 1: Fact Extraction as Structured Output

All systems extract facts from raw text using LLM calls. The most robust approach:

```typescript
interface ExtractedFact {
  content: string;         // the fact text
  entities: string[];      // mentioned entities
  valid_at?: Date;         // when this became true
  invalid_at?: Date;       // when this was superseded (if known)
  confidence?: number;
}

// Extraction: single LLM call per message/chunk
// mem0 style: category-guided extraction (7 categories)
// Hindsight style: what/when/where/who/why structured output
// Graphiti style: entity-first, then relationship extraction (2 calls)
```

### Pattern 2: Conflict Resolution (ADD/UPDATE/DELETE/NOOP)

mem0's approach is the simplest and most directly applicable:

```typescript
type MemoryAction =
  | { event: 'ADD'; text: string }
  | { event: 'UPDATE'; id: string; text: string; old_memory: string }
  | { event: 'DELETE'; id: string }
  | { event: 'NOOP'; id: string };

// Pipeline:
// 1. Extract facts from new message
// 2. For each fact, search existing memories by embedding similarity
// 3. Send {existing_memories, new_facts} to LLM → MemoryAction[]
// 4. Execute actions atomically
```

### Pattern 3: Bi-Temporal Fact Management

Graphiti's approach is the gold standard for temporal tracking:

```typescript
interface TemporalFact {
  fact: string;
  valid_at: Date;      // when true in real world
  invalid_at?: Date;   // when superseded in real world
  expired_at?: Date;   // when system detected staleness
  created_at: Date;    // when stored
}

// Invalidation logic:
// 1. Find existing facts between same entity pair
// 2. LLM determines: duplicate? contradicted?
// 3. If contradicted + new is newer → mark old invalid_at = new.valid_at
// 4. If contradicted + old is newer → mark new invalid_at = old.valid_at
// 5. Old facts preserved for temporal queries
```

### Pattern 4: Multi-Strategy Search with RRF

Both Graphiti and Hindsight use the same pattern:

```typescript
async function hybridSearch(query: string): Promise<SearchResult[]> {
  const [semantic, keyword, graph] = await Promise.all([
    semanticSearch(query),    // cosine similarity on embeddings
    bm25Search(query),        // full-text keyword matching
    graphTraversal(query),    // BFS/PageRank from anchor entities
  ]);

  return rrf([semantic, keyword, graph]);
}

function rrf(resultLists: string[][], k = 1): RankedResult[] {
  const scores = new Map<string, number>();
  for (const list of resultLists) {
    for (let i = 0; i < list.length; i++) {
      scores.set(list[i], (scores.get(list[i]) ?? 0) + 1 / (i + k));
    }
  }
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id, score]) => ({ id, score }));
}
```

### Pattern 5: Entity Resolution (Fuzzy Matching)

Hindsight's multi-factor scoring is the most practical:

```typescript
function scoreEntityCandidate(input: Entity, candidate: Entity): number {
  let score = 0;
  // Name similarity (50% weight) - edit distance or sequence matching
  score += stringSimilarity(input.name, candidate.name) * 0.5;
  // Co-occurrence (30% weight) - entities that appear together
  score += cooccurrenceOverlap(input, candidate) * 0.3;
  // Temporal proximity (20% weight) - 7-day decay
  const daysDiff = Math.abs(diffDays(input.date, candidate.lastSeen));
  score += Math.max(0, 1.0 - daysDiff / 7) * 0.2;
  return score;
}
```

### Pattern 6: Memory Consolidation (Background)

Hindsight's observation system and TiMem's hierarchy both consolidate:

```typescript
// Hindsight style: flat consolidation
// Input: batch of unconsolidated facts + related existing observations
// LLM output: { creates: [...], updates: [...], deletes: [...] }
// Rules:
//   - Redundant info → UPDATE to consolidate
//   - Contradictions → preserve both with temporal markers
//   - Vague→specific → UPDATE with concrete values
//   - Different entities → NEVER merge

// TiMem style: hierarchical consolidation
// L1: fragments (per N turns) → L2: session → L3: daily → L4: weekly → L5: profile
// Each level summarizes its children + historical context
// Adaptive recall queries appropriate levels based on complexity
```

### Pattern 7: Memory Evolution (A-Mem)

```typescript
enum EvolutionDecision {
  NO_EVOLUTION,           // store as-is
  STRENGTHEN,             // enhance the new memory's connections
  UPDATE_NEIGHBOR,        // propagate changes to linked memories
  STRENGTHEN_AND_UPDATE,  // both
}

// Pipeline:
// 1. Store new memory with embedding
// 2. Find k nearest neighbors
// 3. LLM decides: evolve or not?
// 4. If STRENGTHEN: add links + update tags on new memory
// 5. If UPDATE_NEIGHBOR: update tags/context on neighbor memories
// 6. Rebuild embedding index
```

### Pattern 8: Three Link Types (Hindsight)

Beyond entity relationships, Hindsight creates three inter-memory link types:

```typescript
// Temporal links: facts occurring close in time
// → enables "what else happened around that time?" queries

// Semantic links: facts with similar embeddings
// → enables "what else is related to this topic?" queries

// Causal links: facts with cause-effect relationships (LLM-extracted)
// → enables "what led to this?" and "what resulted from this?" queries
```

### Pattern 9: Reflect as Agentic RAG

Hindsight's reflect pattern — a tool-calling agent that iteratively retrieves and synthesizes:

```typescript
// Tool sequence (forced ordering):
// 1. search_mental_models → curated summaries (if available)
// 2. search_observations  → auto-consolidated knowledge
// 3. recall              → raw facts
// 4. expand              → deepen specific memories
// 5. done                → final answer with citations

// Safeguards:
// - Anti-hallucination: "ONLY use information from retrieved results"
// - Citation validation: only actually-retrieved IDs included
// - Context budgeting: auto-synthesize if approaching token limit
// - Evidence gate: can't call done() without evidence
```

---

## LLM Call Budgets (per ingested message)

| System | Min LLM Calls | Max LLM Calls | Notes |
|--------|--------------|--------------|-------|
| Graphiti | 3 | 5+ | extract nodes + edges + resolve + attributes + community |
| Hindsight | 1 | 2 | fact extraction + consolidation (async background) |
| A-Mem | 1 | 4 | analyze + decide + strengthen + update neighbors |
| TiMem | 1 | 5 | L1 fragment + L2-L5 consolidation (async) |
| mem0 | 2 | 4 | fact extraction + conflict resolution + graph extraction + delete eval |

---

## Recommended Priority for Trix Adaptation

1. **mem0 conflict resolution** (ADD/UPDATE/DELETE/NOOP) — simplest, most directly applicable to existing vector store
2. **Graphiti bi-temporal edges** — critical for "Alice used to work at X, now works at Y" queries
3. **Hindsight's 4-way parallel search + RRF** — proven at scale, straightforward to implement
4. **Hindsight's entity resolution** (multi-factor scoring) — handles "Bob" vs "Robert" without graph DB
5. **Hindsight's 3 link types** (temporal/semantic/causal) — enriches retrieval graph without Neo4j
6. **TiMem's hierarchical consolidation** — good for long-horizon users but complex to implement
7. **A-Mem's evolution triggers** — interesting but high LLM cost per memory
