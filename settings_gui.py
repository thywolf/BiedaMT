"""Okno ustawień BiedaMT — tkinter w osobnym wątku (pystray zostaje na głównym).

Spec pól konfiguracji w GUI jest tu (FIELD_SECTIONS) i służy też za źródło
polityki restartów w main.save_settings(): pole z restart="llama" wymaga
restartu llama-servera, restart="app" — restartu całej aplikacji, a bez
badge'a jest stosowane live (CONFIG czytany przy każdym tłumaczeniu).

Podgląd bez pełnej apki (zapis jest atrapą, nic nie rusza):
    python settings_gui.py
"""
import copy
import ctypes
import os
import queue
import shlex
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import keyboard

try:
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, ttk
    TK_OK = True
except Exception:  # pragma: no cover — środowiska bez tkinter
    TK_OK = False

try:
    import ruamel.yaml  # noqa: F401 — tylko do ostrzeżenia w nagłówku okna
    _RUAMEL_OK = True
except ImportError:
    _RUAMEL_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Spec pól konfiguracji (etykiety PL, badge'y restart)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str = "str"          # str|int|float|bool|combo|path|dir|args|text|hotkey|memory
    hint: str = ""             # podpowiedź pod polem
    restart: str = ""          # "" = live | "llama" = restart llama-servera | "app" = restart apki
    required: bool = False
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    choices: Tuple[str, ...] = ()          # dla combo
    text_height: int = 4                   # dla text
    file_patterns: Tuple[Tuple[str, str], ...] = ()  # dla path (filedialog filetypes)


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    icon: str
    note: str                    # banner na górze sekcji (legenda badge'ów / wpływ zmian)
    fields: Tuple[Field, ...]


EXE_PATTERNS = (("llama-server", "*.exe"), ("Wszystkie pliki", "*.*"))
GGUF_PATTERNS = (("Model GGUF", "*.gguf"), ("Wszystkie pliki", "*.*"))
PNG_PATTERNS = (("Obrazki", "*.png"), ("Wszystkie pliki", "*.*"))
ICO_PATTERNS = (("Ikony", "*.ico"), ("Wszystkie pliki", "*.*"))

FIELD_SECTIONS: Tuple[Section, ...] = (
    Section(
        key="llama", title="Serwer LLM", icon="🤖",
        note="Pola „↻ LLM” wymagają restartu llama-servera — po zapisaniu pojawi się przycisk "
             "„Restart LLM”. „● live” działa od razu po zapisaniu.",
        fields=(
            Field("server_path", "Ścieżka do llama-server", kind="path", restart="llama", required=True,
                  hint='Sama nazwa (np. "llama-server") = szukanie w PATH', file_patterns=EXE_PATTERNS),
            Field("model_path", "Ścieżka do modelu (.gguf)", kind="path", restart="llama", required=True,
                  hint="Plik GGUF ładowany do llama-servera", file_patterns=GGUF_PATTERNS),
            Field("args", "Argumenty llama-server", kind="args", restart="llama",
                  hint="Jeden ciąg, np.: -c 8192 -ngl 99 --flash-attn on. Wartości ze spacjami "
                       "ujmij w cudzysłów; --port synchronizuje się z polem „Port llama-server”."),
            Field("port", "Port llama-server", kind="int", restart="llama", vmin=1, vmax=65535,
                  hint="Przy zapisie zsynchronizowany z --port w argumentach"),
            Field("startup_timeout", "Limit czasu startu (s)", kind="int", restart="llama", vmin=1,
                  hint="Ile sekund pollować /health zanim start zostanie uznany za nieudany"),
            Field("request_timeout", "Timeout zapytań (s)", kind="int", vmin=1,
                  hint="Maks. czas tłumaczenia jednego zapytania — działa od razu"),
        ),
    ),
    Section(
        key="translation", title="Tłumaczenie", icon="🌐",
        note="Wszystko w tej sekcji działa od razu po zapisaniu.",
        fields=(
            Field("default_memory", "Domyślna pamięć", kind="memory",
                  hint="„(wszystkie)” = szukaj w każdej pamięci; wybór w tray nadpisuje to na czas sesji"),
            Field("temperature", "Temperatura", kind="float", vmin=0.0, vmax=2.0,
                  hint="0.0–2.0; niższa = bardziej przewidywalne tłumaczenie"),
            Field("max_tokens", "Maks. tokenów odpowiedzi", kind="int", vmin=1),
        ),
    ),
    Section(
        key="rag", title="Pamięć RAG", icon="🧠",
        note="Wyszukiwanie i progi („● live”) działają od razu. Model embeddingów i limity terminów "
             "(„↻ restart”) wymagają restartu aplikacji — zmiana modelu wymusi dodatkowo ponowne "
             "indeksowanie pamięci z TMX przy starcie.",
        fields=(
            Field("search_results", "Wyników z wybranej pamięci", kind="int", vmin=1,
                  hint="Liczba referencji RAG, gdy w tray wybrano konkretną pamięć"),
            Field("search_all_per_memory", "Wyników na pamięć (tryb All)", kind="int", vmin=1),
            Field("fuzzy_match_distance", "Próg fuzzy match", kind="float", vmin=0.0, vmax=1.0,
                  hint="Odległość cosine (0 = identyczne); trafienie poniżej progu zwraca tłumaczenie "
                       "prosto z pamięci, bez LLM"),
            Field("max_reference_distance", "Maks. odległość referencji", kind="float", vmin=0.0, vmax=1.0,
                  hint="Dalsze wyniki pomijane jako szum"),
            Field("max_reference_chars", "Maks. długość referencji (znaki)", kind="int", vmin=1,
                  hint="Dłuższe segmenty ucinane w prompcie"),
            Field("terms_enabled", "Glosariusz z TMX", kind="bool",
                  hint="Krótkie segmenty pamięci wymuszane w promptie jako terminy"),
            Field("terms_max_words", "Termin: maks. słów", kind="int", vmin=1, restart="app"),
            Field("terms_min_chars", "Termin: min. znaków", kind="int", vmin=1, restart="app"),
            Field("embedding_model", "Model embeddingów", kind="str", restart="app",
                  hint="np. intfloat/multilingual-e5-small; zmiana = re-import pamięci przy starcie"),
            Field("embedding_device", "Device embeddingów", kind="combo", choices=("cpu", "cuda"), restart="app"),
        ),
    ),
    Section(
        key="memories", title="Pamięci TMX", icon="📖",
        note="Zmiana ścieżek wymaga restartu aplikacji.",
        fields=(
            Field("dir", "Folder z plikami TMX", kind="dir", restart="app",
                  hint="Nazwa pliku (bez .tmx) staje się nazwą pamięci"),
            Field("chroma_path", "Ścieżka ChromaDB", kind="dir", restart="app",
                  hint="Cache embeddingów — tworzony automatycznie"),
        ),
    ),
    Section(
        key="prompt", title="Prompt", icon="📝",
        note="Szablony działają od razu po zapisaniu. Placeholdery wypełnia str.format(), więc "
             "dosłowny nawias klamrowy zapisz jako {{ i }}. Pusty lub uszkodzony szablon wraca do domyślnego.",
        fields=(
            Field("template", "Szablon główny", kind="text", text_height=9,
                  hint="{src}, {tgt} — języki; {glossary} i {references} — bloki (puste przy braku trafień); "
                       "{text} — tekst do tłumaczenia"),
            Field("glossary_block", "Nagłówek bloku glosariusza", kind="text", text_height=3,
                  hint="{items} — wyrenderowane pozycje"),
            Field("glossary_item", "Pozycja glosariusza", kind="text", text_height=3,
                  hint="{i}, {src}, {tgt}, {term}, {translation}"),
            Field("reference_block", "Nagłówek bloku referencji", kind="text", text_height=3,
                  hint="{items} — wyrenderowane pozycje"),
            Field("reference_item", "Pozycja referencji", kind="text", text_height=3,
                  hint="{i}, {src}, {tgt}, {source_text}, {target_text}"),
        ),
    ),
    Section(
        key="server", title="API", icon="🔌",
        note="Zmiany wymagają restartu aplikacji (serwer API startuje raz).",
        fields=(
            Field("port", "Port API", kind="int", restart="app", vmin=1, vmax=65535,
                  hint="GUI pamięci pod http://localhost:{port}/"),
            Field("host", "Host", kind="str", restart="app",
                  hint='127.0.0.1 = tylko ten komputer; 0.0.0.0 = dostępne z sieci lokalnej'),
        ),
    ),
    Section(
        key="ui", title="Interfejs", icon="🔧",
        note="Hotkey przepina się od razu po zapisaniu; ikony i tytuł tray odświeżają się przy zapisie.",
        fields=(
            Field("hotkey", "Hotkey tłumaczenia", kind="hotkey", required=True,
                  hint='Np. "`", "ctrl+shift+t" — „Nagraj” przechwytuje kombinację z klawiatury'),
            Field("replacements_file", "Plik reguł zamian", kind="path",
                  hint="Reguły regex stosowane przed i po tłumaczeniu"),
            Field("icon_active", "Ikona — aktywny", kind="path", file_patterns=PNG_PATTERNS),
            Field("icon_busy", "Ikona — przetwarzanie", kind="path", file_patterns=PNG_PATTERNS),
            Field("icon_inactive", "Ikona — nieaktywny", kind="path", file_patterns=PNG_PATTERNS),
            Field("icon_ico", "Ikona — .ico (okno)", kind="path", file_patterns=ICO_PATTERNS),
            Field("app_name", "Nazwa aplikacji", kind="str"),
            Field("app_title", "Tooltip tray", kind="str"),
        ),
    ),
)

# Klucze "sekcja.pole" pogrupowane wg wymaganych restartów (zużywa main.save_settings)
RESTART_LLAMA_KEYS = frozenset(
    f"{s.key}.{f.key}" for s in FIELD_SECTIONS for f in s.fields if f.restart == "llama")
RESTART_APP_KEYS = frozenset(
    f"{s.key}.{f.key}" for s in FIELD_SECTIONS for f in s.fields if f.restart == "app")


def sync_args_port(args: List[str], port: int) -> Tuple[List[str], bool]:
    """Ustaw wartość --port w liście argumentów llama-servera (jeśli występuje).
    Zwraca (argumenty, czy coś zmieniono)."""
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            if args[i + 1] == str(port):
                return args, False
            fixed = list(args)
            fixed[i + 1] = str(port)
            return fixed, True
    return args, False


def _argv_split_windows(text: str) -> List[str]:
    """Tokenizacja ciągu wg natywnych reguł Windows (CommandLineToArgvW):
    backslashe w ścieżkach są dosłowne, cudzysłów grupuje tokeny."""
    cmd = ctypes.windll.shell32.CommandLineToArgvW
    cmd.restype = ctypes.POINTER(ctypes.c_wchar_p)
    cmd.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    count = ctypes.c_int(0)
    argv = cmd(text, ctypes.byref(count))
    if not argv or not count.value:
        return []
    try:
        return [argv[i] for i in range(count.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv)


def split_args_string(text: str) -> List[str]:
    """Rozbij ciąg argumentów na tokeny argv (reguły Windows; poza Windows shlex).
    Nowe linie i wielokrotne spacje liczą się jak jeden separator."""
    text = " ".join(text.split())
    if not text:
        return []
    if sys.platform == "win32":
        try:
            return _argv_split_windows(text)
        except Exception:
            pass  # awaryjnie shlex
    return shlex.split(text, posix=True)


def join_args(tokens: List[str]) -> str:
    """Połącz tokeny argv w jeden ciąg — tokeny ze spacjami w cudzysłowie."""
    parts = []
    for token in tokens:
        token = str(token)
        if token and not any(c.isspace() for c in token):
            parts.append(token)
        else:
            parts.append(f'"{token}"')
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Kolory / fonty
# ──────────────────────────────────────────────────────────────────────────────

C_BG = "#f3f5f9"
C_SIDEBAR = "#e9edf4"
C_SIDEBAR_HOVER = "#dde4ef"
C_CARD = "#ffffff"
C_HEADER = "#0f172a"
C_HEADER_SUB = "#94a3b8"
C_TEXT = "#0f172a"
C_MUTED = "#64748b"
C_ACCENT = "#2563eb"
C_ACCENT_HOVER = "#1d4ed8"
C_BORDER = "#dbe2ec"
C_NOTE_BG = "#e8f0fe"
C_NOTE_FG = "#1e40af"
C_BADGE_RESTART_BG = "#fef3c7"
C_BADGE_RESTART_FG = "#92400e"
C_BADGE_LIVE_BG = "#dcfce7"
C_BADGE_LIVE_FG = "#15803d"
C_DANGER = "#ea580c"

ALL_MEMORY = "(wszystkie)"
BADGE_TEXT = {"llama": "↻ LLM", "app": "↻ restart", "": "● live"}
BADGE_COLORS = {
    "llama": (C_BADGE_RESTART_BG, C_BADGE_RESTART_FG),
    "app": (C_BADGE_RESTART_BG, C_BADGE_RESTART_FG),
    "": (C_BADGE_LIVE_BG, C_BADGE_LIVE_FG),
}

# ──────────────────────────────────────────────────────────────────────────────
# Otwieranie okna (jeden egzemplarz, własny wątek)
# ──────────────────────────────────────────────────────────────────────────────

_SPAWNING = object()
_spawn_lock = threading.Lock()
_window = None                          # ustawiane tylko w wątku Tk
_focus_request = threading.Event()


def open_settings(current: dict, save_fn: Callable[[dict], dict],
                  validate_prompt_fn: Optional[Callable[[dict], List[str]]] = None,
                  restart_llama_fn: Optional[Callable[[Callable, Callable], None]] = None,
                  hotkey_pause_fn: Optional[Callable[[], None]] = None,
                  hotkey_resume_fn: Optional[Callable[[], None]] = None,
                  config_path: Optional[Path] = None) -> None:
    """Otwórz okno ustawień (nowy wątek Tk). Jeśli już otwarte — wyrzuć na wierzch."""
    global _window
    if not TK_OK:
        print("❌ tkinter niedostępny — edytuj config.yaml ręcznie")
        return
    with _spawn_lock:
        if _window is not None:
            _focus_request.set()
            return
        _window = _SPAWNING  # rezerwacja miejsca, żeby nie otworzyć dwóch okien
    threading.Thread(
        target=_run_window, daemon=True,
        args=(current, save_fn, validate_prompt_fn, restart_llama_fn,
              hotkey_pause_fn, hotkey_resume_fn, config_path),
    ).start()


def _run_window(current, save_fn, validate_prompt_fn, restart_llama_fn,
                hotkey_pause_fn, hotkey_resume_fn, config_path) -> None:
    global _window
    try:
        root = tk.Tk()
    except Exception as e:
        print(f"❌ Nie mogę otworzyć okna ustawień: {e}")
        _window = None
        return
    try:
        _window = SettingsWindow(root, current, save_fn, validate_prompt_fn,
                                 restart_llama_fn, hotkey_pause_fn, hotkey_resume_fn,
                                 config_path)
        root.mainloop()
    except Exception as e:
        print(f"❌ Błąd okna ustawień: {type(e).__name__}: {e}")
    finally:
        _window = None


# ──────────────────────────────────────────────────────────────────────────────
# Okno
# ──────────────────────────────────────────────────────────────────────────────

_MOD_ORDER = ("ctrl", "alt", "altgr", "shift", "windows")


def _normalize_key(name: str) -> str:
    """Ujednolic nazwę klawisza z biblioteki keyboard (left ctrl -> ctrl itd.)."""
    n = (name or "").lower().strip()
    if n in ("esc", "escape"):
        return "esc"
    if "alt gr" in n or n == "altgr":
        return "altgr"
    for mod in _MOD_ORDER:
        if mod in n:
            return mod
    return n


class SettingsWindow:
    def __init__(self, root: "tk.Tk", current: dict, save_fn, validate_prompt_fn,
                 restart_llama_fn, hotkey_pause_fn, hotkey_resume_fn, config_path):
        self.root = root
        self.save_fn = save_fn
        self.validate_prompt_fn = validate_prompt_fn
        self.restart_llama_fn = restart_llama_fn
        self.hotkey_pause_fn = hotkey_pause_fn
        self.hotkey_resume_fn = hotkey_resume_fn
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self.values: Dict[str, dict] = copy.deepcopy(current)
        self.original: Dict[str, dict] = copy.deepcopy(current)

        self._vars: Dict[Tuple[str, str], tk.Variable] = {}
        self._texts: Dict[Tuple[str, str], "tk.Text"] = {}
        self._selected: Optional[int] = None
        self._cat_items: List[Tuple["tk.Frame", "tk.Frame", "tk.Label"]] = []
        self._capturing = False
        self._cap_hook = None
        self._cap_mods = set()
        self._cap_result = None
        self._cap_finished = False
        self._cap_lock = threading.Lock()
        self._ui_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        self._setup_window()
        self._build_layout()
        self.show_section(0)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.bind_all("<MouseWheel>", self._on_wheel)
        root.after(120, self._poll)

    # ── szkielet okna ────────────────────────────────────────────────────────

    def _setup_window(self):
        root = self.root
        root.title("BiedaMT — Ustawienia")
        root.configure(bg=C_BG)
        root.minsize(860, 580)
        width, height = 940, 700
        x = max(0, (root.winfo_screenwidth() - width) // 2)
        y = max(0, (root.winfo_screenheight() - height) // 3)
        root.geometry(f"{width}x{height}+{x}+{y}")
        try:
            root.iconbitmap(str(self.config_path.parent / "assets" / "corona.ico"))
        except Exception:
            pass
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=9)

    def _build_layout(self):
        root = self.root
        # Nagłówek
        header = tk.Frame(root, bg=C_HEADER, padx=20, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="⚙  Ustawienia", bg=C_HEADER, fg="white",
                 font=("Segoe UI", 15, "bold")).pack(anchor="w")
        subtitle = f"BiedaMT · zmiany zapisywane do {self.config_path}"
        if not _RUAMEL_OK:
            subtitle += "  (⚠ brak ruamel.yaml — komentarze w pliku zostaną utracone)"
        tk.Label(header, text=subtitle, bg=C_HEADER, fg=C_HEADER_SUB,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))
        tk.Frame(root, height=2, bg=C_ACCENT).pack(fill="x")

        body = tk.Frame(root, bg=C_BG)
        body.pack(fill="both", expand=True)

        # Boczna lista sekcji
        sidebar = tk.Frame(body, bg=C_SIDEBAR, width=230, padx=6, pady=12)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        for i, sec in enumerate(FIELD_SECTIONS):
            row = tk.Frame(sidebar, bg=C_SIDEBAR)
            row.pack(fill="x", pady=1)
            strip = tk.Frame(row, width=3, bg=C_SIDEBAR)
            strip.pack(side="left", fill="y")
            label = tk.Label(row, text=f"{sec.icon}  {sec.title}", anchor="w",
                             bg=C_SIDEBAR, fg=C_TEXT, font=("Segoe UI", 10),
                             padx=8, pady=7)
            label.pack(side="left", fill="x", expand=True)
            for widget in (row, strip, label):
                widget.bind("<Button-1>", lambda e, idx=i: self.show_section(idx))
                widget.bind("<Enter>", lambda e, idx=i: self._hover_category(idx, True))
                widget.bind("<Leave>", lambda e, idx=i: self._hover_category(idx, False))
            self._cat_items.append((row, strip, label))

        # Formularz (przewijany)
        form_wrap = tk.Frame(body, bg=C_CARD)
        form_wrap.pack(side="left", fill="both", expand=True, padx=(1, 0), pady=0)
        self.canvas = tk.Canvas(form_wrap, bg=C_CARD, highlightthickness=0)
        self.vsb = ttk.Scrollbar(form_wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.form_inner = tk.Frame(self.canvas, bg=C_CARD)
        self._form_window = self.canvas.create_window((0, 0), window=self.form_inner, anchor="nw")
        self.form_inner.bind("<Configure>",
                             lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._form_window, width=e.width))
        self.form_inner.columnconfigure(0, weight=1)

        # Stopka: status na górze (rozszerza się o komunikaty po zapisie), przyciski pod nim
        tk.Frame(root, height=1, bg=C_BORDER).pack(fill="x")
        footer = tk.Frame(root, bg=C_BG, padx=16, pady=10)
        footer.pack(fill="x")
        footer.columnconfigure(0, weight=1)

        self.status_label = tk.Label(footer, text="", bg=C_BG, fg=C_MUTED,
                                     font=("Segoe UI", 9), wraplength=600, justify="left")
        self.status_label.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(0, 6))
        footer.bind("<Configure>", lambda e: self.status_label.configure(
            wraplength=max(300, e.width - 40)))

        self.restart_btn = tk.Button(footer, text="🔄  Restart LLM", command=self._restart_llama,
                                     bg=C_DANGER, fg="white", activebackground="#c2410c",
                                     activeforeground="white", relief="flat", bd=0,
                                     padx=14, pady=6, cursor="hand2", font=("Segoe UI", 9, "bold"))
        self.restart_btn.grid(row=1, column=1, sticky="e", padx=6)
        self.restart_btn.grid_remove()

        tk.Button(footer, text="Otwórz config.yaml", command=self._open_config_file,
                  bg="white", fg=C_TEXT, activebackground="#eef2f7", activeforeground=C_TEXT,
                  relief="flat", bd=0, highlightthickness=1, highlightbackground=C_BORDER,
                  padx=12, pady=6, cursor="hand2").grid(row=1, column=2, sticky="e", padx=3)
        tk.Button(footer, text="Anuluj", command=self._on_close,
                  bg="white", fg=C_TEXT, activebackground="#eef2f7", activeforeground=C_TEXT,
                  relief="flat", bd=0, highlightthickness=1, highlightbackground=C_BORDER,
                  padx=14, pady=6, cursor="hand2").grid(row=1, column=3, sticky="e", padx=3)
        tk.Button(footer, text="💾  Zapisz zmiany", command=self._on_save,
                  bg=C_ACCENT, fg="white", activebackground=C_ACCENT_HOVER,
                  activeforeground="white", relief="flat", bd=0,
                  padx=16, pady=6, cursor="hand2", font=("Segoe UI", 9, "bold")).grid(row=1, column=4, sticky="e")

    # ── sekcje / budowa pól ──────────────────────────────────────────────────

    def show_section(self, idx: int):
        if self._selected == idx:
            return
        if self._capturing:  # przycisk nagrywania zniknie razem z sekcją
            self._end_capture(commit=False)
        self._harvest_visible()
        self._selected = idx
        sec = FIELD_SECTIONS[idx]
        for i, (row, strip, label) in enumerate(self._cat_items):
            selected = (i == idx)
            bg = C_CARD if selected else C_SIDEBAR
            row.configure(bg=bg)
            strip.configure(bg=C_ACCENT if selected else bg)
            label.configure(bg=bg, fg=C_ACCENT if selected else C_TEXT,
                            font=("Segoe UI", 10, "bold" if selected else "normal"))
        for child in self.form_inner.winfo_children():
            child.destroy()
        self._vars.clear()
        self._texts.clear()

        pad = {"padx": 26}
        tk.Label(self.form_inner, text=f"{sec.icon}  {sec.title}", bg=C_CARD, fg=C_TEXT,
                 font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(20, 6), **pad)
        if sec.note:
            note = tk.Label(self.form_inner, text=sec.note, bg=C_NOTE_BG, fg=C_NOTE_FG,
                            font=("Segoe UI", 8), wraplength=560, justify="left", padx=12, pady=8)
            note.grid(row=1, column=0, sticky="ew", pady=(0, 14), **pad)
        row_idx = 2
        for f in sec.fields:
            row_idx = self._build_field(row_idx, sec, f)
        self.form_inner.grid_rowconfigure(row_idx, weight=1)
        self.canvas.yview_moveto(0)

    def _hover_category(self, idx: int, entered: bool):
        if idx == self._selected:
            return
        row, strip, label = self._cat_items[idx]
        bg = C_SIDEBAR_HOVER if entered else C_SIDEBAR
        row.configure(bg=bg)
        strip.configure(bg=bg)
        label.configure(bg=bg)

    def _build_field(self, row_idx: int, sec: Section, f: Field) -> int:
        value = self.values.get(sec.key, {}).get(f.key)
        parent = self.form_inner
        padx = 26

        head = tk.Frame(parent, bg=C_CARD)
        head.grid(row=row_idx, column=0, sticky="ew", pady=(10, 2), padx=padx)
        tk.Label(head, text=f.label, bg=C_CARD, fg=C_TEXT,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        badge_bg, badge_fg = BADGE_COLORS[f.restart]
        tk.Label(head, text=BADGE_TEXT[f.restart], bg=badge_bg, fg=badge_fg,
                 font=("Segoe UI", 7, "bold"), padx=6, pady=1).pack(side="right")
        row_idx += 1

        widget_row = tk.Frame(parent, bg=C_CARD)
        widget_row.grid(row=row_idx, column=0, sticky="ew", padx=padx)
        widget_row.columnconfigure(0, weight=1)
        row_idx += 1
        key = (sec.key, f.key)

        if f.kind in ("int", "float", "str", "combo", "memory"):
            var = tk.StringVar(value="" if value is None else str(value))
            if f.kind == "combo":
                choices = list(f.choices)
                if value and value not in choices:
                    choices.append(str(value))
                widget = ttk.Combobox(widget_row, textvariable=var, values=choices, state="readonly")
            elif f.kind == "memory":
                choices = self._memory_choices(value)
                widget = ttk.Combobox(widget_row, textvariable=var, values=choices, state="readonly")
                var.set(ALL_MEMORY if value is None else str(value))
            else:
                widget = ttk.Entry(widget_row, textvariable=var, width=52)
            widget.grid(row=0, column=0, sticky="ew")
            self._vars[key] = var
        elif f.kind == "bool":
            var = tk.BooleanVar(value=bool(value))
            widget = tk.Checkbutton(widget_row, variable=var, bg=C_CARD, activebackground=C_CARD,
                                    highlightthickness=0)
            widget.grid(row=0, column=0, sticky="w")
            self._vars[key] = var
        elif f.kind in ("path", "dir"):
            var = tk.StringVar(value="" if value is None else str(value))
            widget = ttk.Entry(widget_row, textvariable=var)
            widget.grid(row=0, column=0, sticky="ew")
            tk.Button(widget_row, text="Wybierz…", command=lambda: self._pick_path(f, var),
                      bg="white", fg=C_TEXT, activebackground="#eef2f7", relief="flat", bd=0,
                      highlightthickness=1, highlightbackground=C_BORDER,
                      padx=10, cursor="hand2").grid(row=0, column=1, sticky="e", padx=(8, 0))
            self._vars[key] = var
        elif f.kind == "hotkey":
            var = tk.StringVar(value="" if value is None else str(value))
            widget = ttk.Entry(widget_row, textvariable=var, width=24)
            widget.grid(row=0, column=0, sticky="w")
            self._record_btn = tk.Button(widget_row, text="⏺  Nagraj", command=lambda: self._toggle_capture(var),
                                         bg=C_CARD, fg=C_ACCENT, activebackground=C_NOTE_BG,
                                         activeforeground=C_ACCENT_HOVER, relief="flat", bd=0,
                                         highlightthickness=1, highlightbackground=C_ACCENT,
                                         padx=10, cursor="hand2", font=("Segoe UI", 9, "bold"))
            self._record_btn.grid(row=0, column=1, sticky="e", padx=(8, 0))
            self._vars[key] = var
        elif f.kind == "args":
            holder = tk.Frame(widget_row, bg=C_BORDER)
            holder.grid(row=0, column=0, sticky="ew")
            text = tk.Text(holder, height=5, font=("Consolas", 9), wrap="word", relief="flat",
                           bg=C_CARD, fg=C_TEXT, highlightthickness=0, padx=6, pady=5,
                           insertbackground=C_TEXT)
            text.pack(side="left", fill="both", expand=True)
            if isinstance(value, list):
                value = join_args(value)
            text.insert("1.0", " ".join(str(value or "").split()))
            self._texts[key] = text
        elif f.kind == "text":
            holder = tk.Frame(widget_row, bg=C_BORDER)
            holder.grid(row=0, column=0, sticky="ew")
            text = tk.Text(holder, height=f.text_height, font=("Segoe UI", 9), wrap="word",
                           relief="flat", bg=C_CARD, fg=C_TEXT, highlightthickness=0,
                           padx=6, pady=5, insertbackground=C_TEXT)
            if f.text_height >= 6:
                scroll = ttk.Scrollbar(holder, orient="vertical", command=text.yview)
                text.configure(yscrollcommand=scroll.set)
                scroll.pack(side="right", fill="y")
            text.pack(side="left", fill="both", expand=True)
            text.insert("1.0", "" if value is None else str(value))
            self._texts[key] = text

        if f.hint:
            tk.Label(parent, text=f.hint, bg=C_CARD, fg=C_MUTED, font=("Segoe UI", 8),
                     wraplength=560, justify="left").grid(row=row_idx, column=0, sticky="w",
                                                          pady=(2, 4), padx=padx)
            row_idx += 1
        return row_idx

    def _memory_choices(self, current) -> List[str]:
        names: List[str] = []
        try:
            mem_dir = Path(str(self.values.get("memories", {}).get("dir", "./memories")))
            if not mem_dir.is_absolute():
                mem_dir = self.config_path.parent / mem_dir
            names = sorted(p.stem for p in mem_dir.glob("*.tmx"))
        except Exception:
            pass
        if current and current not in names:
            names.append(str(current))
        return [ALL_MEMORY] + names

    def _pick_path(self, f: Field, var: tk.StringVar):
        base = self.config_path.parent
        current = var.get().strip()
        if f.kind == "dir":
            initial = base
            if current:
                candidate = Path(current)
                if not candidate.is_absolute():
                    candidate = base / candidate
                if candidate.exists():
                    initial = str(candidate)
            chosen = filedialog.askdirectory(parent=self.root, title=f.label, initialdir=initial)
        else:
            initialfile = None
            initialdir = base
            if current:
                candidate = Path(current)
                if not candidate.is_absolute():
                    candidate = base / candidate
                if candidate.parent != Path("."):
                    initialdir = str(candidate.parent)
                initialfile = candidate.name
            chosen = filedialog.askopenfilename(
                parent=self.root, title=f.label, initialdir=initialdir, initialfile=initialfile,
                filetypes=list(f.file_patterns) or [("Wszystkie pliki", "*.*")])
        if chosen:
            var.set(chosen)

    # ── odczyt wartości z widgetów ───────────────────────────────────────────

    def _harvest_visible(self):
        """Przepisz wartości widocznej sekcji z widgetów do self.values (miękkie typy:
        błędny int/float zostaje napisem — walidacja przy zapisie to wyłapie)."""
        if self._selected is None:
            return
        sec = FIELD_SECTIONS[self._selected]
        target = self.values.setdefault(sec.key, {})
        for f in sec.fields:
            key = (sec.key, f.key)
            if f.kind in ("args", "text"):
                widget = self._texts.get(key)
                if widget is None:
                    continue
                raw = widget.get("1.0", "end-1c")
                target[f.key] = (" ".join(raw.split()) if f.kind == "args" else raw)
            elif f.kind == "bool":
                var = self._vars.get(key)
                if var is not None:
                    target[f.key] = bool(var.get())
            elif f.kind == "memory":
                var = self._vars.get(key)
                if var is not None:
                    raw = var.get()
                    target[f.key] = None if raw == ALL_MEMORY else raw
            elif f.kind in ("int", "float"):
                var = self._vars.get(key)
                if var is not None:
                    raw = var.get().strip().replace(",", ".")
                    try:
                        target[f.key] = float(raw) if f.kind == "float" else int(raw)
                    except ValueError:
                        target[f.key] = raw
            else:
                var = self._vars.get(key)
                if var is not None:
                    target[f.key] = var.get().strip()

    def _collect(self) -> Tuple[Dict[str, dict], List[str]]:
        """Zbierz wartości wszystkich sekcji + walidacja. (wartości, błędy)"""
        self._harvest_visible()
        values: Dict[str, dict] = {}
        errors: List[str] = []
        for sec in FIELD_SECTIONS:
            out = values.setdefault(sec.key, {})
            for f in sec.fields:
                value = self.values.get(sec.key, {}).get(f.key)
                where = f"{sec.title} → {f.label}"
                if f.kind in ("int", "float") and isinstance(value, str):
                    errors.append(f"{where}: podaj liczbę")
                    continue
                if f.kind == "args":
                    # W configu args trzymamy jako jeden ciąg (start_llama rozbija
                    # na tokeny przy starcie); lista z YAML-u też jest akceptowana.
                    if value is None:
                        value = ""
                    elif isinstance(value, list):
                        value = join_args(value)
                    try:
                        split_args_string(value)  # walidacja (shlex może rzucić poza Windows)
                    except ValueError as e:
                        errors.append(f"{where}: niespójne cudzysłowy w argumentach ({e})")
                        continue
                if f.kind == "int" and value is not None and not isinstance(value, bool):
                    if f.vmin is not None and value < f.vmin or f.vmax is not None and value > f.vmax:
                        scope = f"{f.vmin}–{f.vmax}" if f.vmax is not None else f"≥ {f.vmin}"
                        errors.append(f"{where}: dozwolone {scope}")
                        continue
                if f.kind == "float" and value is not None:
                    if f.vmin is not None and value < f.vmin or f.vmax is not None and value > f.vmax:
                        errors.append(f"{where}: dozwolone {f.vmin}–{f.vmax}")
                        continue
                if f.required and (value is None or value == "" or value == []):
                    errors.append(f"{where}: podaj wartość")
                    continue
                out[f.key] = value
        return values, errors

    # ── zapis ────────────────────────────────────────────────────────────────

    def _on_save(self):
        values, errors = self._collect()
        if errors:
            messagebox.showerror("Błędy w formularzu", "• " + "\n• ".join(errors), parent=self.root)
            return

        llama = values["llama"]
        status = ""
        tokens = (split_args_string(llama["args"]) if isinstance(llama["args"], str)
                  else list(llama["args"] or []))
        new_args, synced = sync_args_port(tokens, llama["port"])
        if synced:
            llama["args"] = join_args(new_args)
            status += "Zsynchronizowano --port w argumentach. "
            args_widget = self._texts.get(("llama", "args"))
            if args_widget is not None:  # odśwież widoczne pole po synchronizacji
                args_widget.delete("1.0", "end")
                args_widget.insert("1.0", llama["args"])
        if not any(a == "--port" for a in new_args[:-1]):
            status += "⚠ W argumentach nie ma --port — llama-server użyje portu domyślnego. "

        if self.validate_prompt_fn:
            prompt_errors = self.validate_prompt_fn(values["prompt"])
            if prompt_errors:
                messagebox.showerror("Błędne szablony promptu", "\n".join(prompt_errors), parent=self.root)
                return

        missing = self._missing_paths(values)
        if missing and not messagebox.askyesno(
                "Pliki nie istnieją",
                "Te pliki nie istnieją:\n\n• " + "\n• ".join(missing) +
                "\n\nZapisać konfigurację mimo to?", parent=self.root):
            return

        try:
            result = self.save_fn(copy.deepcopy(values)) or {}
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać config.yaml:\n{e}", parent=self.root)
            return

        self.values = copy.deepcopy(values)
        self.original = copy.deepcopy(values)

        status += "✅ Zapisano config.yaml."
        if result.get("llama_restart"):
            status += ("\n↻ Zastosuje się po restarcie llama-servera: "
                       + ", ".join(result.get("llama_keys", [])) + " — użyj przycisku „Restart LLM”.")
            self.restart_btn.grid()
        if result.get("app_restart"):
            status += ("\n↻ Zastosuje się po restarcie aplikacji (Quit i uruchom ponownie): "
                       + ", ".join(result.get("app_keys", [])) + ".")
        for note in result.get("notes", []):
            status += f"\nℹ {note}"
        self._set_status(status, ok=True)

    def _missing_paths(self, values: Dict[str, dict]) -> List[str]:
        """Pliki (kind=path) zmienione względem oryginału, które nie istnieją na dysku."""
        missing = []
        base = self.config_path.parent
        for sec in FIELD_SECTIONS:
            for f in sec.fields:
                if f.kind != "path":
                    continue
                value = values.get(sec.key, {}).get(f.key)
                if not value or value == self.original.get(sec.key, {}).get(f.key):
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = base / candidate
                if not candidate.parent.name:  # sama nazwa -> komenda z PATH (np. "llama-server")
                    continue
                if not candidate.exists():
                    missing.append(f"{f.label}: {value}")
        return missing

    # ── restart llama-servera ────────────────────────────────────────────────

    def _restart_llama(self):
        if not self.restart_llama_fn:
            self._set_status("Restart llama-servera niedostępny w trybie podglądu.")
            return
        self.restart_btn.configure(state="disabled", text="🔄 Restartuję…")
        self._set_status("🔄 Restartuję llama-servera — czekam na /health…")
        self.restart_llama_fn(
            lambda: self._ui_queue.put(("restart_done", None)),
            lambda msg: self._ui_queue.put(("restart_error", msg)),
        )

    # ── nagrywanie hotkeya ───────────────────────────────────────────────────

    def _toggle_capture(self, var: tk.StringVar):
        if self._capturing:
            self._end_capture(commit=False)
        else:
            self._start_capture(var)

    def _start_capture(self, var: tk.StringVar):
        self._capture_var = var
        self._capturing = True
        with self._cap_lock:
            self._cap_mods = set()
            self._cap_result = None
            self._cap_finished = False
        if self.hotkey_pause_fn:
            self.hotkey_pause_fn()
        self._cap_hook = keyboard.hook(self._on_capture_key, suppress=True)
        self._record_btn.configure(text="⏺  Naciśnij kombinację…", bg=C_NOTE_BG)
        self._set_status("⏺ Naciśnij kombinację klawiszy (Esc = anuluj, limit 10 s).")
        self.root.after(10000, self._capture_timeout)

    def _on_capture_key(self, event):
        """Callback z wątku biblioteki keyboard — tylko stan + pobudka UI."""
        if not self._capturing or event.event_type != "down":
            return
        key = _normalize_key(event.name)
        with self._cap_lock:
            if key == "esc":
                self._cap_result = None
                self._cap_finished = True
            elif key in _MOD_ORDER:
                self._cap_mods.add(key)
            else:
                mods = [m for m in _MOD_ORDER if m in self._cap_mods]
                self._cap_result = "+".join(mods + [key])
                self._cap_finished = True
        self._ui_queue.put(("cap_update", None))

    def _capture_timeout(self):
        if self._capturing:
            self._end_capture(commit=False, timeout=True)

    def _handle_cap_update(self):
        if not self._capturing:
            return
        with self._cap_lock:
            finished, result = self._cap_finished, self._cap_result
            mods = set(self._cap_mods)
        if not finished:
            held = " + ".join(m for m in _MOD_ORDER if m in mods)
            self._set_status(f"⏺ Nagrywam: {held + ' + …' if held else '…'}")
            return
        self._end_capture(commit=result is not None)

    def _end_capture(self, commit: bool, timeout: bool = False):
        if not self._capturing:
            return
        self._capturing = False
        if self._cap_hook is not None:
            try:
                keyboard.unhook(self._cap_hook)
            except Exception:
                pass
            self._cap_hook = None
        if self.hotkey_resume_fn:
            self.hotkey_resume_fn()
        self._record_btn.configure(text="⏺  Nagraj", bg=C_CARD)
        if commit and self._cap_result:
            self._capture_var.set(self._cap_result)
            self._set_status(f"⏺ Ustawiono hotkey: {self._cap_result} (zapamiętaj przyciskiem „Zapisz zmiany”).")
        else:
            self._set_status("⏺ Anulowano nagrywanie." if not timeout
                             else "⏺ Minął limit nagrywania — spróbuj ponownie.")

    # ── zdarzenia UI / pomocnicze ────────────────────────────────────────────

    def _poll(self):
        """Pętla zdarzeń między wątkami: prośby o focus, restart LLM, nagrywanie."""
        if _focus_request.is_set():
            _focus_request.clear()
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.attributes("-topmost", True)
            self.root.after(250, lambda: self.root.attributes("-topmost", False))
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "cap_update":
                    self._handle_cap_update()
                elif kind == "restart_done":
                    self.restart_btn.configure(state="normal", text="🔄  Restart LLM")
                    self._set_status("✅ llama-server zrestartowany na nowych ustawieniach.", ok=True)
                elif kind == "restart_error":
                    self.restart_btn.configure(state="normal", text="🔄  Restart LLM")
                    self._set_status(f"❌ Restart llama-servera nie powiódł się: {payload}")
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _set_status(self, text: str, ok: bool = False):
        self.status_label.configure(text=text, fg=C_ACCENT_HOVER if ok else C_MUTED)

    def _open_config_file(self):
        try:
            if sys.platform == "win32":
                os.startfile(self.config_path)  # noqa: S606
            else:
                os.system(f'xdg-open "{self.config_path}" &')  # noqa: S607
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie mogę otworzyć pliku: {e}", parent=self.root)

    def _on_close(self):
        if self._capturing:
            self._end_capture(commit=False)
        try:
            self.root.unbind_all("<MouseWheel>")
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Podgląd okna bez pełnej apki: zapis i restart są atrapami, config.yaml nietknięty.
    import time
    import yaml

    if sys.platform == "win32":
        from ctypes import windll
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    cfg_path = Path(__file__).parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def fake_save(values: dict) -> dict:
        print("[preview] llama.args (tokeny):", values["llama"]["args"])
        print("[preview] save_fn wywołany (nic nie zapisuje)")
        return {"llama_restart": True, "llama_keys": ["llama.port"],
                "app_restart": True, "app_keys": ["server.port"], "notes": []}

    def fake_restart(on_done, on_error):
        threading.Timer(2.0, on_done).start()

    open_settings(cfg, fake_save, lambda prompt: [], fake_restart,
                  lambda: print("[preview] hotkey wstrzymany"),
                  lambda: print("[preview] hotkey przywrócony"),
                  cfg_path)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
