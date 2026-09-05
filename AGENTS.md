# AGENTS.md — BiedaMT

Guide for coding agents (Claude Code, Codex, opencode, ...) working in this codebase.

## What this is

Single-file desktop translator app (Windows, Python 3.10+): llama-server subprocess + TMX memory RAG (ChromaDB) + global hotkey clipboard translation + pystray tray icon + FastAPI side API.

## Commands

- Run the full app: `python main.py` (needs `config.yaml`, llama-server and the model; see Testing for running without the full stack)
- Preview the Settings window standalone: `python settings_gui.py` (fake save/restart callbacks, never touches `config.yaml`)
- Import/syntax check without the full stack: `python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read())"`
- There is no test suite; "Testing without the full stack" below lists what can be exercised where.

## Layout

- `main.py` — everything: config load, replacements, translation pipeline, hotkey action, tray menu, llama-server lifecycle, FastAPI app, `main()`
- `settings_gui.py` — Settings window (tkinter in its own thread): field spec (`FIELD_SECTIONS`, the GUI's view of every config knob incl. restart badges), opened from the `Settings` tray item
- `rag_engine.py` — `TMXTranslatorRAG` (one collection per TMX) + `RAGManager` (load/reload/delete/search_all)
- `config.yaml` — **single source of truth**. No hardcoded paths, ports, model names or tunables in code. If you need a new knob, add it here and read via `CONFIG[...]`. Note: `config.yaml` is git-ignored (owner's private local paths); `config.example.yaml` is the tracked template — when adding a new knob, add it to both.
- `replacements.py` — user-editable regex replacements applied before ("source") and after ("target") translation
- `memories/*.tmx` — TMX memories; filename (sans extension) = memory name
- `chroma/` — ChromaDB persistent store (created at runtime, never commit/edit)
- `assets/` — tray icons

## Architecture invariants — do not break

1. **pystray must run on the main thread.** `icon.run()` is the last call of `main()`; uvicorn runs in a daemon thread. Windows message pump requirement — inverting this produces a silent app (uvicorn up, no tray).
2. **pystray action signatures are validated.** Menu actions must be callables accepting `(icon, item)`. Use `functools.partial(fn, arg)` where `fn` takes the bound arg first plus `*args`, or plain `fn(icon, item)`. A lambda with a third default param raises `ValueError` from `pystray._base._assert_action` at menu build time.
3. **Rebuild the menu after any state change** (memory selection, hotkey toggle, language, memory list changes) via `rebuild_menu()`.
4. **Hotkey pipeline order** (`action()`): guard `hotkey_active` → busy icon → snapshot clipboard → `keyboard.send('ctrl+c')` → poll until clipboard content changes (25 ms interval, 0.5 s cap; replaced a fixed sleep that was too short on slow apps) → `pyperclip.paste()` → source replacements → translate → target replacements → `pyperclip.copy` → `delete` → `ctrl+v` → active icon.
5. **Translation pipeline** (`translate_with_rag`): exact TM lookup (sqlite `where` on metadata — direct return, no LLM) → RAG search (selected memory or `search_all`) → `filter_references` (distance threshold + sort + dedupe + truncation, knobs in `rag.*`) → fuzzy-TM shortcut (best hit within `rag.fuzzy_match_distance` returns directly, no LLM) → glossary lookup (`find_terms`: short TMX segments act as a termbase; hits are injected by `build_rag_prompt` as a mandatory "Terminy" block that overrides the references) → `build_rag_prompt` (plain-text, direction-aware prompt with EN/PL references; wording = `prompt.*` templates in `config.yaml`, live-reloadable via the `Reload prompt` tray item — `DEFAULT_PROMPT` in main.py is only a fallback) → llama-server `/v1/chat/completions`. It does NOT use `build_rag_chatmessage` / the `[USER_PROMPT]`/`[REFERENCES]` protocol — that protocol is only for the external-facing `/v1/chat/completions` endpoint, whose clients own their system prompt (the external endpoint also gets no glossary — protocol stays stable).
6. **RAG search context**: `translate_with_rag` uses `current_memory` (tray selection); `None` means search all memories (`search_all`, few results each). Selected memory = many results from one.
7. **llama-server lifecycle**: started in `start_llama()` (falls back to `llama-server` on PATH if configured path missing; readiness = polling `/health` up to `llama.startup_timeout`, then a background warmup request), stopped via `atexit` + `stop_llama()` on Quit. Config: `llama.server_path`, `llama.model_path`, `llama.args` (port lives in both `llama.port` and `llama.args` — the Settings GUI auto-syncs them on save; args are stored as one string — the GUI format — or a YAML list; `start_llama` flattens both to argv tokens via `split_args_string`), `llama.request_timeout`.
8. **No filesystem watcher for memories — by design.** The owner updates TMX files by copying large exports into `memories/`; polling could catch a half-written file. Loading is explicit: `Refresh memories` (pick up NEW tmx files; existing ones re-attach from the Chroma cache) and `Reload memory → <name>` (force re-import of an updated TMX into Chroma). Do not reintroduce automatic polling.
9. **RAG embeddings are computed manually** (sentence-transformers directly, NOT chromadb's embedding functions): documents embedded with the `passage: ` prefix, queries with `query: ` (e5 model requirement), `normalize_embeddings=True`, collections created with `hnsw:space: cosine`. Each collection carries an `embedding` marker in its metadata (`EMBEDDING_MARKER` + model name via `embedding_marker()` in rag_engine.py) — when embedding logic or the model changes, bump the marker: old collections are detected and re-imported from TMX on next start (collections marked with the bare legacy marker are still accepted for `LEGACY_E5_MODEL`). `search_all` embeds the query once and reuses it across all memories. Lazy RAG costs (first `model.encode`, per-collection HNSW index load, glossary metadata scan) are preheated on purpose: `RAGManager.warmup()` runs in the background thread at startup (`warmup_llama`) and after `Refresh memories`, and `reload_memory` warms the reloaded engine via `_warmup_engine` — new engines have empty `_terms_cache`, so cover any new lazy cost in those spots.
10. **Settings GUI: thread split + restart policy.** The window runs Tk in a daemon thread spawned by `settings_gui.open_settings()` — never touch Tk objects from other threads; cross-thread traffic goes through `SettingsWindow._ui_queue` (polled in the Tk loop) and the module-level `_focus_request` event. Restart-required metadata lives in `settings_gui` (`Field.restart` = `""`/`"llama"`/`"app"`, exposed as `RESTART_LLAMA_KEYS`/`RESTART_APP_KEYS`); the live-apply policy (merge into `CONFIG`, hotkey rebind, prompt reload, tray icon refresh, `rebuild_menu()`) lives in `main.save_settings()` — keep that split (spec/labels in settings_gui, behavior in main). When adding a config knob: add it to `FIELD_SECTIONS` *and* to both `config.yaml` and `config.example.yaml`. `start_llama()` raises `LlamaStartError` instead of `sys.exit` so the GUI's "Restart LLM" button can fail gracefully — only `main()` converts it to process exit at startup. GUI saves go through `save_config_values()` (ruamel round-trip preserves comments; creates `config.yaml.bak`). `restart_llama_async()` invokes its callbacks on a worker thread — the receiver must marshal them to its own UI thread (the Tk window does this via `_ui_queue`).

## Platform gotchas (Windows)

- `keyboard` lib: global hotkeys may require running as admin depending on the target window; `` ` `` (backtick) is fine as a hotkey string.
- `windll.shcore.SetProcessDpiAwareness(1)` and AppUserModelID are wrapped in try/except — keep them non-fatal.
- Source files use CRLF. `git`/editors may normalize; Python doesn't care, but keep diffs clean.
- Paths in `config.yaml` use Windows-style strings (`d:\\Tools\\...`) — in YAML double-quoted strings backslash escapes apply; when adding paths prefer forward slashes or single quotes.

## Conventions

- User-facing strings, comments, log prints: Polish (project owner's language). Code identifiers: English. README.md is owner-facing too — keep it in Polish.
- Exception: tray menu labels are English-only (owner's decision) — e.g. `Memory`, `All`, `Reload memory`, `(no memories)`. Keep new menu items consistent with that.
- Emoji log prefixes are the established debug style (`🔥 hotkey`, `📋 clipboard`, `📖 RAG`, `🤖 llama`, `❌ error`, `✅ done`). Keep new logs consistent.
- Debug prints in `action()`/`translate_with_rag` are intentional (active development). Don't remove wholesale; gate behind a `debug` config flag if noise becomes a problem.

## Testing without the full stack

- If llama-server/GPU is not available in the environment (e.g. a Linux dev box): import-test with
  `python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read())"`
  and unit-test pure functions (`build_rag_prompt`, `make_replacements`, `filter_references`, config load).
- Importing `main` executes the module-level config load, so `config.yaml` must exist even just to import (copy from `config.example.yaml` if missing). Module-level load is the ONLY module-level side effect — everything else (tray, hotkey, llama, uvicorn) lives inside `main()`. Keep it that way.
- RAG requires `chromadb` + `sentence-transformers` (downloads `intfloat/multilingual-e5-small` on first run, ~470 MB).
- The FastAPI app can be exercised without tray/keyboard by importing `app` from `main`.
- The Settings window can be previewed without the app: `python settings_gui.py` (fake save/restart callbacks). Pure logic worth unit-testing lives in module-level functions: `sync_args_port`, field validation in `SettingsWindow._collect`, `_merge_config_values`/`save_config_values`/`_changed_keys` (main), `_check_prompt_template` (main).
- Real end-to-end (hotkey, tray, llama) is validated by the owner on Windows. Ask for console logs when debugging — the emoji prints trace the full chain.

## API surface (external tools)

- `POST /v1/chat/completions` — OpenAI-compatible. If a user message starts with `\n[USER_PROMPT]\n`, RAG references are injected as a separate user message `\n[REFERENCES]\n{json}` before it. Clients own their system prompt. Optional `rag_memory` field restricts the search to one memory.
- `GET /v1/memories`, `POST /v1/memories/{name}/reload`, `DELETE /v1/memories/{name}` — memory management (also used by the GUI at `GET /`).
- `GET /health` — llama status, loaded memories, current memory, hotkey state.
- No authentication and permissive CORS; default `server.host` is `0.0.0.0`, so treat every endpoint as reachable from the LAN — don't add anything here you wouldn't expose to the local network.
- There is no internal translation HTTP endpoint — the hotkey path calls `translate_with_rag()` directly. Do not reintroduce an internal-HTTP hop.
