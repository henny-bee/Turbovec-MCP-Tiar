# Turbovec MCP: Recommended Custom Instructions & System Rules

To ensure your AI coding assistant makes proactive and flawless use of this advanced hybrid memory database without asking you for permission first, add the following guidelines to your editor's **Custom Instructions**, **System Rules**, or `.clinerules` file.

---

## Custom Instructions Template

Copy and paste the markdown block below directly into your AI Client (Cursor, Claude Desktop, Cline / Roo Code, Windsurf, Zoo Code, etc.):

```markdown
# Long-Term Memory & Knowledge Graph System Instructions
You have access to an enterprise-grade local Hybrid Graph-Vector Memory Server (Turbovec MCP). Use it proactively to read and write context across different chat sessions:

1. At the beginning of a task, check for existing context by running `search_memory(query="[relevant topic]")`.
2. Review active messages left by previous sessions using `get_bottles()`. If you accomplish a milestone that is valuable for future sessions, leave a message with `create_bottle()`.
3. When the user explains an architectural decision, code pattern, custom requirement, or preferences, store it immediately:
   - Use `create_entity()` to define core systems, components, or requirements. Node/Edge types must comply with ontology rules (e.g. entities: person, project, technology, decision, event, concept, file).
   - Use `add_observation()` to attach notes, facts, and code guidelines to those entities.
   - Use `create_relationship()` to model dependencies (e.g., `ComponentA` -> `depends_on` -> `ComponentB`).
4. If documenting large files or external APIs, use `add_knowledge()` or `add_file_knowledge()`.
5. When starting a development phase, initiate a session with `start_session()`. Record any core insights or breakthroughs with `record_breakthrough()`, and summarize accomplishments with `end_session()`.
6. Proactively maintain memory structure. If you find related but disconnected entities, run `create_relationship()` to bridge them.
7. Periodically monitor retrieval performance and database health with `search_stats()` or run on-demand reorganization using `run_librarian_cycle()`.
```
