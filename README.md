# BiedaMT

Desktopowy tłumacz EN↔PL z pamięcią TMX (RAG) — jedna aplikacja: globalny hotkey, tray, llama-server i API.

## Co robi

- Uruchamia w tle **llama-server** (llama.cpp) z podanym modelem GGUF i komunikuje się z nim po HTTP
- Ładuje pamięci TMX z folderu `memories/` do **ChromaDB** (embeddingi `intfloat/multilingual-e5-small` z prefiksami `query:`/`passage:`)
- Rejestruje **hotkey** — po naciśnięciu: `ctrl+c` → tłumaczenie → `delete` → `ctrl+v`, czyli tekst podmieniany jest w miejscu kursora
- **Exact/fuzzy match** — dokładnie lub niemal identyczne zdanie wraca prosto z pamięci TMX (natychmiast, bez LLM); reszta: referencje RAG filtrowane po podobieństwie
- **Glosariusz z TMX** — krótkie segmenty pamięci (≤ `rag.terms_max_words` słów) działają jak termbase: trafione w tekście terminy trafiają do promptu jako obowiązkowe tłumaczenia (spójność terminologiczna niezależnie od fuzzy)
- Ikona w **tray** (pystray): wybór pamięci, kierunek tłumaczenia, włączenie/wyłączenie hotkeya; tooltip pokazuje licznik przetłumaczonych słów i znaków
- **Okno ustawień** (`Settings` w menu tray) — cała konfiguracja z GUI: wybór plików, nagrywanie hotkeya z klawiatury, walidacja; jasno oznacza, które zmiany działają od razu, a które wymagają restartu llama-servera (przycisk „Restart LLM”) albo całej aplikacji
- **Ładowanie pamięci na żądanie** — `Refresh memories` dopisuje nowe TMX, `Reload memory → nazwa` wymusza re-import zaktualizowanego TMX (apka nie obserwuje folderu na bieżąco)
- GUI w przeglądarce pod `http://localhost:12345/` (port = `server.port`) — lista pamięci, reload, delete

## Wymagania

- Windows
- Python 3.10+
- llama-server (llama.cpp) + model GGUF (np. Bielik)
- Przy pierwszym starcie: pobranie modelu embeddingów (~470 MB) z HuggingFace

## Instalacja

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy config.example.yaml config.yaml
```

Aplikacja wymaga `config.yaml` przy starcie — bez niego nie uruchomi się.

## Konfiguracja

Wszystko w `config.yaml` — żadnych hardkodów w kodzie. W repo leży szablon `config.example.yaml`; `config.yaml` tworzysz z niego lokalnie i **nie commitujesz** (zawiera prywatne ścieżki do modelu i llama-servera — jest w `.gitignore`):

- `llama` — ścieżka do llama-server, model, argumenty CLI, port, timeouty
- `server` — port FastAPI (domyślnie 12345) i host; domyślny `0.0.0.0` = API dostępne z sieci lokalnej, `127.0.0.1` = tylko ten komputer
- `rag` — model embeddingów, device (cpu/cuda), liczba wyników, progi dopasowań (fuzzy/reference), glosariusz (`terms_*`)
- `translation` — pamięć startowa, gdy nie ma zapisanego stanu sesji (`default_memory`; null = żadna — dalej wybór checkboxami w tray), temperatura, max_tokens
- `prompt` — szablony promptu tłumaczenia; zmiany wchodzą po `Reload prompt` w menu tray, bez restartu
- `memories` — folder TMX, ścieżka ChromaDB
- `ui` — hotkey, ikony, nazwa apki, ścieżka `replacements.py` i pliku stanu (`state_file` — zapamiętany wybór pamięci z tray, JSON tworzony w runtime)

Uwaga na port llama-servera: figuruje w dwóch miejscach (`llama.port` i `--port` w `llama.args`). Okno ustawień synchronizuje je przy zapisie, ale przy ręcznej edycji pliku trzeba zmienić oba.

`config.yaml` możesz edytować ręcznie albo w **oknie ustawień** (`Settings` w menu tray). Zapis z okna zachowuje komentarze w pliku (ruamel.yaml) i tworzy kopię zapasową `config.yaml.bak`. Większość ustawień działa od razu (hotkey, progi RAG, temperatura, szablony promptu, timeouty); zmiany ścieżek/argumentów/portu llama wymagają kliknięcia „Restart LLM” w oknie, a port/host API, ścieżki pamięci, model embeddingów i limity terminów — restartu całej aplikacji (okno mówi, które pola).

## Uruchomienie

```
python main.py
```

## Użycie

1. Skopiuj tekst (`ctrl+c` — normalnie, do schowka)
2. Naciśnij hotkey (domyślnie backtick `` ` ``)
3. Tekst w miejscu kursora zostanie podmieniony na tłumaczenie

Menu tray:
- **Hotkey active/inactive** — włącza/wyłącza hotkey
- **Memory** — checkboxy: które pamięci mają być używane w tłumaczeniu (żadna zaznaczona = tłumaczenie bez pamięci, wszystkie = szukanie we wszystkich; etykieta pokazuje aktualny wybór). Wybór jest zapamiętywany (`ui.state_file`) i wraca po restarcie; bez zapisanego stanu startuje z `translation.default_memory` (domyślnie: bez pamięci)
- **EN -> PL / PL -> EN** — kierunek tłumaczenia
- **Reload memory** — wymusza re-import zaktualizowanego TMX do ChromaDB (submenu: wybór pamięci)
- **Refresh memories** — przeskanowanie folderu `memories/` (dopisuje nowe)
- **Reload replacements** — przeładowanie reguł zamian z `replacements.py`
- **Reload prompt** — przeładowanie szablonów promptu z `config.yaml`
- **Settings** — okno ustawień (edycja `config.yaml` z GUI, z podpowiedzią czego wymaga restart)
- **Quit** — zamyka apkę i llama-server

## Pamięci TMX

Wrzuć pliki `.tmx` do folderu `memories/`. Nazwa pliku (bez rozszerzenia) to nazwa pamięci. Przy pierwszym załadowaniu plik jest parsowany i indeksowany do ChromaDB — kolejne starty ładują się z cache. Format indeksu jest wersjonowany (metadane kolekcji): po zmianie logiki embeddingów pierwszy start automatycznie wykona re-import wszystkich pamięci z plików TMX. Folder nie jest obserwowany automatycznie — nowe pliki: **Refresh memories** (dopisuje nowe), zaktualizowane: **Reload memory → nazwa** (re-import TMX do ChromaDB) z menu tray.

## API

Dla narzędzi zewnętrznych — klient ustawia własny system prompt. API nie ma żadnego uwierzytelniania, więc przy domyślnym `host: 0.0.0.0` każdy komputer w sieci lokalnej może z niego korzystać i zarządzać pamięciami; jeśli to problem, ustaw `server.host: 127.0.0.1`.

- `POST /v1/chat/completions` — OpenAI-compatible; jeśli wiadomość usera zaczyna się od `\n[USER_PROMPT]\n`, tuż przed nią wstawiana jest osobna wiadomość `\n[REFERENCES]\n{json}` z trafieniami RAG. Bez tego znacznika zapytanie idzie do llama-servera bez RAG. Opcjonalne pole `rag_memory` ogranicza wyszukiwanie do jednej pamięci (brak = wszystkie)
- `GET /v1/memories` — lista pamięci
- `POST /v1/memories/{name}/reload` — przeładowanie pamięci z TMX
- `DELETE /v1/memories/{name}` — usunięcie pamięci z ChromaDB
- `GET /health` — status (llama, pamięci, hotkey, wybór pamięci do tłumaczenia: `selected_memories`, `null` = wszystkie)
- `GET /` — GUI

## Struktura

```
BiedaMT/
├── main.py               # cała apka: FastAPI + pystray + hotkey
├── settings_gui.py       # okno ustawień (tkinter, własny wątek)
├── rag_engine.py         # TMX → ChromaDB, wyszukiwanie
├── config.example.yaml   # szablon konfiguracji (skopiuj do config.yaml)
├── requirements.txt
├── replacements.py       # reguły zamian tekstowych (przed/po tłumaczeniu)
├── memories/             # pamięci TMX (lokalne, nie w repo)
├── chroma/               # baza ChromaDB (tworzona automatycznie, nie w repo)
├── assets/               # ikony
└── LICENSE               # MIT
```

## Licencja

MIT — patrz [LICENSE](LICENSE). Pamięci TMX i pliki modeli pozostają własnością ich właścicieli i nie są częścią tego repozytorium.

## Uwagi

- Hotkey wymaga czasem uruchomienia jako administrator (zależnie od okna, w którym pracujesz)
- Debug printy w konsoli (`🔥`, `📋`, `📖`) — można wyciszyć w razie potrzeby
