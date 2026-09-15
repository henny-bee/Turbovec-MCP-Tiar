# Custom Instructions

Without a rule telling it to, an assistant will not search its memory before
answering or write anything back. Paste the block below into your client's
**Custom Instructions** / **System Rules** / `.clinerules`.

---

## The rules

```markdown
You have a local hybrid graph-vector memory server (Turbovec MCP). Use it
without being asked.

**Before answering anything about this project**
1. `search_memory(query="<topic>")` for existing context.
2. `get_bottles()` for notes left by previous sessions.

**When the user states something durable** — an architectural decision, a
convention, a constraint, a preference — store it immediately:
- `create_entity()` for the thing itself (person, project, technology,
  decision, event, concept, file).
- `add_observation()` for facts about it.
- `create_relationship()` for how it connects, e.g. `api` depends_on `database`.
- `add_knowledge()` or `add_file_knowledge()` for long documents and specs.

**Around a unit of work**
- `start_session()` when starting, `record_breakthrough()` for insights worth
  keeping, `end_session(summary=...)` when finishing.
- `create_bottle()` for anything the next session must know, especially
  unfinished work.

**Housekeeping** — occasionally, not every turn:
- `create_relationship()` to connect entities you notice are related.
- `run_librarian_cycle()` to cluster and deduplicate.
- `search_stats()` to check retrieval health.

Store durable facts, not conversation. Prefer one entity with several
observations over many near-duplicate entities.
```

---

## Notes

- **Relationship types are validated** against `ontology.json`; entity types are
  not. Register new relationship types there before using them.
- **Trim the rules if your assistant over-writes.** The housekeeping section is
  the first thing to cut — the background daemon
  (`BACKGROUND_DISCOVERY=true`) does that work anyway.
- **Session tools are optional.** If you only want recall, keep the search and
  store rules and drop the rest.
