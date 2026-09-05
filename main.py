"""BiedaMT — all-in-one translator z pamięcią RAG."""
import atexit
import copy
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from ctypes import windll
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import keyboard
import pyperclip
import requests
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from pystray import Menu, MenuItem, Icon

from rag_engine import RAGManager
from settings_gui import RESTART_APP_KEYS, RESTART_LLAMA_KEYS, split_args_string

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Wczytaj config.yaml z katalogu aplikacji."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def _merge_config_values(data: dict, new_values: dict) -> None:
    """Wpisz nowe wartości do wczytanej mapy YAML. Niezmienione klucze zostają
    bez dotyku (ruamel zachowa wtedy oryginalne formatowanie)."""
    try:
        from ruamel.yaml.scalarstring import LiteralScalarString
    except ImportError:
        LiteralScalarString = None
    for section, values in new_values.items():
        target = data.get(section)
        if not isinstance(target, dict):
            target = data[section] = {}
        for key, value in values.items():
            if key in target and target[key] == value:
                continue
            if LiteralScalarString is not None and isinstance(value, str) and "\n" in value:
                value = LiteralScalarString(value)  # wielolinijkowe teksty jako blok | w YAML
            target[key] = value

def _write_config(path: Path, new_values: dict) -> None:
    """Zapisz new_values do podanego pliku YAML, scalając z resztą zawartości.
    Z ruamel.yaml (round-trip) komentarze i kolejność przetrwają; bez niego
    plik jest przepisywany przez PyYAML bez komentarzy."""
    try:
        from ruamel.yaml import YAML
    except ImportError:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        _merge_config_values(data or {}, new_values)
        path.write_text(
            yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print("⚠️ Brak ruamel.yaml — config.yaml przepisany bez komentarzy (pip install ruamel.yaml)")
        return
    if path.exists():  # kopia zapasowa przed pierwszą modyfikacją z GUI
        backup = path.with_suffix(".yaml.bak")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    yml = YAML()
    yml.preserve_quotes = True
    yml.width = 4096
    with open(path, "r", encoding="utf-8") as f:
        data = yml.load(f) or {}
    _merge_config_values(data, new_values)
    with open(path, "w", encoding="utf-8") as f:
        yml.dump(data, f)

def save_config_values(new_values: dict) -> None:
    """Zapisz wartości do config.yaml, zostawiając resztę pliku w spokoju."""
    _write_config(Path(__file__).parent / "config.yaml", new_values)

def _changed_keys(old: dict, new: dict) -> List[str]:
    """Lista kluczy 'sekcja.pole' o zmienionej wartości (stary config vs nowy)."""
    changed = []
    for section, values in new.items():
        old_section = old.get(section) or {}
        for key, value in values.items():
            if old_section.get(key) != value:
                changed.append(f"{section}.{key}")
    return changed

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

rag_manager: Optional[RAGManager] = None
llama_process: Optional[subprocess.Popen] = None
session = requests.Session()

# UI state
hotkey_active = True
current_memory: Optional[str] = CONFIG["translation"]["default_memory"]
state_en = True  # True = EN->PL, False = PL->EN
wordcount = 0
charcount = 0
icon: Optional[Icon] = None

# Replacements
replacements: Dict = {"source": {"en": {}, "pl": {}}, "target": {"en": {}, "pl": {}}}

# ──────────────────────────────────────────────────────────────────────────────
# Replacements
# ──────────────────────────────────────────────────────────────────────────────

def load_replacements():
    """Wczytaj reguły zamian z pliku replacements.py (opcjonalny)."""
    global replacements
    repl_path = Path(CONFIG["ui"]["replacements_file"])
    if not repl_path.exists():
        return
    try:
        spec = importlib.util.spec_from_file_location("replacements_mod", str(repl_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        replacements = getattr(mod, "replacements", replacements)
        print(f"✅ Załadowano replacements z {repl_path}")
    except Exception as e:
        print(f"⚠️ Błąd ładowania replacements: {e}")

def reload_replacements():
    """Przeładuj replacements bez restartu."""
    load_replacements()
    if icon:
        icon.title = f'{CONFIG["ui"]["app_title"]}\n{wordcount} words\n{charcount} characters'

def make_replacements(side: str, language: str, input_string: str) -> str:
    """Zastosuj reguły zamiany."""
    replacement_dict = replacements.get(side, {}).get(language, {})
    for source, target in replacement_dict.items():
        input_string = re.sub(source, target, input_string)
    return input_string

# ──────────────────────────────────────────────────────────────────────────────
# RAG helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_rag_chatmessage(similar_translations: List[Dict]) -> Optional[str]:
    if not similar_translations:
        return None
    for element in similar_translations:
        element.pop("memory", None)
        element.pop("distance", None)
    references_json = json.dumps(similar_translations, ensure_ascii=False)
    return f"\n[REFERENCES]\n{references_json}"

def _normalize_case(source: str, target: str) -> str:
    """
    Zwraca target z dostosowaną wielkością liter, jeśli source i target są
    jednolite (wszystkie litery małe lub wszystkie wielkie) i przeciwne.
    W przeciwnym razie zwraca target bez zmian.
    """
    # Sprawdzamy tylko litery, pomijając cyfry, znaki interpunkcyjne itp.
    src_lower = all(c.islower() for c in source if c.isalpha())
    src_upper = all(c.isupper() for c in source if c.isalpha())
    tgt_lower = all(c.islower() for c in target if c.isalpha())
    tgt_upper = all(c.isupper() for c in target if c.isalpha())

    if src_lower and tgt_upper:
        return target.lower()
    elif src_upper and tgt_lower:
        return target.upper()
    return target

def find_terms(text: str, source_lang: str, memory: Optional[str]) -> List[Tuple[str, str]]:
    """Znajdź terminy glosariusza (z TMX) w tekście do przetłumaczenia."""
    if not CONFIG["rag"].get("terms_enabled", True) or not rag_manager:
        return []
    matcher = rag_manager.get_term_matcher(source_lang, memory)
    if not matcher:
        return []
    raw_terms = matcher.find(text)
    # Poprawiamy wielkość liter w targetach
    return [(src, _normalize_case(src, tgt)) for src, tgt in raw_terms]

# Szablony promptu tłumaczenia (sekcja prompt: w config.yaml).
# Poniższe są tylko fallbackiem, gdy sekcji nie ma albo szablon jest zepsuty —
# źródłem prawdy jest config.yaml.
DEFAULT_PROMPT: Dict[str, str] = {
    "template": (
        "Jesteś profesjonalnym tłumaczem technicznym {src}->{tgt} specjalizującym się w technicznych tłumaczeniach.\n"
        "{glossary}{references}"
        "W odpowiedzi podajesz tylko tłumaczenie poniższego tekstu, bez dodatkowego tekstu, wyjaśnień, notatek lub uwag. Teraz przetłumacz poniższy tekst:\n"
        "{text}\n"
    ),
    "glossary_block": "Terminologia - dla zachowania spójności terminologicznej:\n{items}\n",
    "glossary_item": "Termin {i}:\n{src}: {term}\n{tgt}: {translation}\n\n",
    "reference_block": "Referencje - dla zachowania spójności gramatycznej:\n{items}\n",
    "reference_item": "Referencja {i}:\n{src}: {source_text}\n{tgt}: {target_text}\n\n",
}

# Kontrolne argumenty do przetestowania szablonu z configu przy ładowaniu
# (wyłapuje literówki w placeholderach i niesparowane nawiasy klamrowe)
_PROMPT_DRY_ARGS: Dict[str, dict] = {
    "template": {"src": "EN", "tgt": "PL", "glossary": "", "references": "", "text": "test"},
    "glossary_block": {"items": ""},
    "glossary_item": {"i": 1, "src": "EN", "tgt": "PL", "term": "term", "translation": "termin"},
    "reference_block": {"items": ""},
    "reference_item": {"i": 1, "src": "EN", "tgt": "PL", "source_text": "source", "target_text": "target"},
}

# Placeholdery obowiązkowe — bez nich szablon gubi treść
_PROMPT_REQUIRED: Dict[str, Tuple[str, ...]] = {
    "template": ("{text}",),
    "glossary_block": ("{items}",),
    "reference_block": ("{items}",),
}

def _check_prompt_template(key: str, value: str) -> Optional[str]:
    """Sprawdź pojedynczy szablon promptu (None = OK, str = opis błędu)."""
    dry_args = _PROMPT_DRY_ARGS.get(key)
    if dry_args is None:
        return None
    try:
        value.format(**dry_args)
    except Exception as e:
        return f"błędny szablon ({e})"
    missing = [ph for ph in _PROMPT_REQUIRED.get(key, ()) if ph not in value]
    if missing:
        return f"brak placeholdera {'/'.join(missing)}"
    return None

def _load_prompt_templates(cfg_prompt: Optional[dict] = None) -> Dict[str, str]:
    """Połącz szablony promptu z config.yaml (prompt.*) z domyślnymi.
    Zepsuty szablon (błąd .format / brak obowiązkowego placeholdera) = zostaje domyślny."""
    templates = dict(DEFAULT_PROMPT)
    if cfg_prompt is None:
        cfg_prompt = CONFIG.get("prompt") or {}
    if not isinstance(cfg_prompt, dict):
        return templates
    for key in _PROMPT_DRY_ARGS:
        value = cfg_prompt.get(key)
        if not isinstance(value, str) or not value:
            continue
        error = _check_prompt_template(key, value)
        if error:
            print(f"⚠️ prompt.{key}: {error} — zostaje domyślny")
            continue
        templates[key] = value
    return templates

def validate_prompt_templates(cfg_prompt: dict) -> List[str]:
    """Błędy w szablonach promptu — walidacja dla okna ustawień (pusta lista = OK).
    Puste szablony są dozwolone (liczą się jak 'użyj domyślnego')."""
    if not isinstance(cfg_prompt, dict):
        return ["Prompt: sekcja musi być mapą klucz → szablon"]
    errors = []
    for key in _PROMPT_DRY_ARGS:
        value = cfg_prompt.get(key)
        if not isinstance(value, str) or not value:
            continue
        error = _check_prompt_template(key, value)
        if error:
            errors.append(f"prompt.{key}: {error}")
    return errors

def reload_prompt(icon=None, item=None):
    """Przeładuj szablony promptu z config.yaml (menu: Reload prompt)."""
    global PROMPT
    try:
        cfg = load_config()
    except Exception as e:
        print(f"❌ Błąd wczytywania config.yaml: {e}")
        return
    PROMPT = _load_prompt_templates(cfg.get("prompt") or {})
    print("✅ Przeładowano szablony promptu")

PROMPT: Dict[str, str] = _load_prompt_templates()

def build_rag_prompt(user_message: str, similar_translations: List[Dict],
                     source_lang: str = "en", target_lang: str = "pl",
                     glossary: Optional[List[Tuple[str, str]]] = None) -> str:
    """Prompt tłumaczenia z referencjami — kierunek zależny od wyboru w menu,
    treść z szablonów config.yaml (prompt.*, przeładowywane z menu)."""
    src, tgt = source_lang.upper(), target_lang.upper()

    glossary_items = ""
    if glossary:
        glossary_items = "".join(
            PROMPT["glossary_item"].format(i=i, src=src, tgt=tgt, term=term, translation=term_translation)
            for i, (term, term_translation) in enumerate(glossary, 1)
        )
    reference_items = ""
    if similar_translations:
        parts = []
        for i, ex in enumerate(similar_translations, 1):
            # W bazie 'source' to zawsze EN, 'target' zawsze PL — kolejność zależy od kierunku
            source_text = ex["source"] if source_lang == "en" else ex["target"]
            target_text = ex["target"] if source_lang == "en" else ex["source"]
            parts.append(PROMPT["reference_item"].format(
                i=i, src=src, tgt=tgt, source_text=source_text, target_text=target_text))
        reference_items = "".join(parts)

    return PROMPT["template"].format(
        src=src,
        tgt=tgt,
        glossary=PROMPT["glossary_block"].format(items=glossary_items) if glossary else "",
        references=PROMPT["reference_block"].format(items=reference_items) if similar_translations else "",
        text=user_message,
    )

def filter_references(refs: List[Dict]) -> List[Dict]:
    """Odsiej słabe referencje, posortuj po podobieństwie, deduplikuj, skróć długie segmenty."""
    max_distance = CONFIG["rag"]["max_reference_distance"]
    max_chars = CONFIG["rag"]["max_reference_chars"]
    refs = [r for r in refs if r.get("distance", 0.0) <= max_distance]
    refs.sort(key=lambda r: r.get("distance", 1.0))
    seen = set()
    out = []
    for ref in refs:
        key = (ref["source"], ref["target"])
        if key in seen:
            continue
        seen.add(key)
        if len(ref["source"]) > max_chars:
            ref["source"] = ref["source"][:max_chars] + "…"
        if len(ref["target"]) > max_chars:
            ref["target"] = ref["target"][:max_chars] + "…"
        out.append(ref)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Translation
# ──────────────────────────────────────────────────────────────────────────────

def translate_with_rag(text: str, memory: Optional[str] = None,
                       source_lang: str = "en", target_lang: str = "pl") -> str:
    """Tłumaczenie z RAG — exact/fuzzy match z pamięci -> referencje -> llama-server."""
    llama_url = f"http://127.0.0.1:{CONFIG['llama']['port']}/v1/chat/completions"
    print(f"  🔗 URL: {llama_url}")

    # 1) Exact match w pamięci — gotowe tłumaczenie bez LLM
    if rag_manager:
        exact = rag_manager.exact_lookup(text, source_lang, memory)
        if exact is not None:
            print(f"  🎯 Exact match (pamięć) — bez LLM: {exact[:60]}...")
            return exact

    # 2) Wyszukiwanie podobnych (referencje z odległościami cosine)
    similar = []
    if rag_manager:
        if memory and memory in rag_manager.engines:
            similar = rag_manager.engines[memory].search(text, n_results=CONFIG["rag"]["search_results"])
        else:
            similar = rag_manager.search_all(text, n_results_per_memory=CONFIG["rag"]["search_all_per_memory"])
    similar = filter_references(similar)

    # 3) Fuzzy match — trafienie niemal identyczne, bierzemy tłumaczenie z pamięci
    if similar and similar[0].get("distance", 1.0) <= CONFIG["rag"]["fuzzy_match_distance"]:
        ref = similar[0]
        hit = ref["target"] if source_lang == "en" else ref["source"]
        print(f"  🎯 Fuzzy match (d={ref['distance']:.3f}) — bez LLM: {hit[:60]}...")
        return hit

    # 4) Terminy z glosariusza (krótkie segmenty TMX) — trafiają do promptu jako obowiązkowe
    glossary = find_terms(text, source_lang, memory)
    if glossary:
        print(f"  📖 Terminy: {', '.join(f'{s} -> {t}' for s, t in glossary)}")

    print(f"  📖 RAG: {len(similar)} referencji")
    for i, ref in enumerate(similar, 1):
        print(f"    {i}. EN: {ref['source'][:20]}... | PL: {ref['target'][:20]}... | d={ref['distance']:.3f}")

    # Pełny prompt z terminami i referencjami (kierunek wg wyboru w menu)
    prompt = build_rag_prompt(text, similar, source_lang, target_lang, glossary)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CONFIG["translation"]["temperature"],
        "max_tokens": CONFIG["translation"]["max_tokens"],
        "cache_prompt": True,
        "stream": False,
    }

    try:
        response = session.post(llama_url, json=payload, timeout=CONFIG["llama"]["request_timeout"])
        print(f"  📥 Status: {response.status_code}")
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        print(f"  📥 Odpowiedź: {result[:60]}...")
        return result
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Llama server error: {e}")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# Hotkey action
# ──────────────────────────────────────────────────────────────────────────────

def action():
    """Główna akcja hotkeya — tłumaczenie schowka."""
    global icon, wordcount, charcount
    if not hotkey_active:
        print("⏭️ Hotkey nieaktywny, pomijam")
        return
    
    icon.icon = Image.open(CONFIG["ui"]["icon_busy"])
    clipboard_before = pyperclip.paste()
    keyboard.send('ctrl+c')
    # Czekamy aż schowek realnie zmieni zawartość (poll 25 ms, max 0.5 s) —
    # krótko na szybkich apkach, bezpiecznie dłużej na wolnych niż sztywny sleep.
    # Brak zmiany po limicie = przetłumacz to, co jest (np. ten sam tekst co wcześniej).
    source = clipboard_before
    for _ in range(20):
        time.sleep(0.025)
        source = pyperclip.paste()
        if source != clipboard_before:
            break
    print(f"📋 Schowek: {repr(source[:80] if source else None)}")
    if not source or not source.strip():
        print("⚠️ Schowek pusty, przerywam")
        icon.icon = Image.open(CONFIG["ui"]["icon_active"])
        return
    
    source_lang = 'en' if state_en else 'pl'
    target_lang = 'pl' if state_en else 'en'
    source = make_replacements("source", source_lang, source)
    charcount += len(source)
    wordcount += len(source.split())
    icon.title = f'{CONFIG["ui"]["app_title"]}\n{wordcount} words\n{charcount} characters'
    print(f"📊 Stats: {wordcount} words, {charcount} chars")
    
    try:
        # Specjalny wzorzec (np. "Cena12.34" -> "12,34 Cena")
        result = re.match(r'^([a-zA-Z$€čłвł£]+)([0-9]+\.[0-9]+)$', source)
        if result:
            target = f"{result[2].replace('.', ',')} {result[1]}"
            print(f"🔢 Pattern match: {target}")
        else:
            print(f"🤖 Tłumaczenie przez RAG (memory={current_memory})...")
            target = translate_with_rag(source, current_memory, source_lang, target_lang)
            print(f"🤖 Wynik: {target[:80]}...")
    except Exception as err:
        print(f"❌ BŁĄD: {type(err).__name__}: {err}")
        icon.title = f"Błąd: {err}"
        icon.icon = Image.open(CONFIG["ui"]["icon_active"])
        return
    
    target = make_replacements("target", target_lang, target)
    print(f"📋 Wynik po replacements: {target[:80]}...")
    pyperclip.copy(target)
    keyboard.send('delete')
    keyboard.send('ctrl+v')
    icon.icon = Image.open(CONFIG["ui"]["icon_active"])

# ──────────────────────────────────────────────────────────────────────────────
# Tray menu
# ──────────────────────────────────────────────────────────────────────────────

def set_memory(name: Optional[str], *args):
    """Ustaw aktualną pamięć (*args — pystray dodaje icon, item)."""
    global current_memory
    current_memory = name
    rebuild_menu()

def rebind_hotkey():
    """Przepnij hotkey wg configu (toggle w menu, zmiana w ustawieniach, nagrywanie w GUI)."""
    try:
        keyboard.remove_all_hotkeys()
    except AttributeError:
        pass  # listener biblioteki keyboard jeszcze nie wystartował = nic do usuwania
    if hotkey_active:
        keyboard.add_hotkey(CONFIG["ui"]["hotkey"], action, suppress=True, trigger_on_release=True)

def refresh_tray_icon():
    """Odśwież ikonę i tooltip tray wg bieżących ustawień (po zmianie w GUI)."""
    if not icon:
        return
    icon.icon = Image.open(CONFIG["ui"]["icon_active" if hotkey_active else "icon_inactive"])
    icon.title = f'{CONFIG["ui"]["app_title"]}\n{wordcount} words\n{charcount} characters'

def toggle_hotkey(icon, item):
    global hotkey_active
    hotkey_active = not hotkey_active
    rebind_hotkey()
    refresh_tray_icon()
    rebuild_menu()

def toggle_language_en(icon, item):
    global state_en
    state_en = True
    rebuild_menu()

def toggle_language_pl(icon, item):
    global state_en
    state_en = False
    rebuild_menu()

def refresh_memories(icon=None, item=None):
    """Odśwież pamięci — przeskanuj folder (dopisz nowe)."""
    if rag_manager:
        rag_manager.load_all_memories()
        rag_manager.warmup()  # świeże obiekty pamięci = puste cache glosariuszu
    rebuild_menu()

def reload_memory_menu(memory_name: str, *args):
    """Reload jednej pamięci z menu — re-import TMX do Chromy."""
    if rag_manager:
        ok = rag_manager.reload_memory(memory_name)
        print(f"🔄 Reload pamięci '{memory_name}': {'OK' if ok else 'BŁĄD'}")
    rebuild_menu()

def get_memory_menu() -> Menu:
    """Dynamiczne menu wyboru pamięci."""
    items = []
    if rag_manager:
        for name in sorted(rag_manager.engines.keys()):
            items.append(
                MenuItem(
                    name,
                    partial(set_memory, name),
                    checked=lambda item, n=name: current_memory == n,
                    radio=True,
                )
            )
    items.append(
        MenuItem(
            "All",
            partial(set_memory, None),
            checked=lambda item: current_memory is None,
            radio=True,
        )
    )
    return Menu(*items)

def get_reload_menu() -> Menu:
    """Submenu reloadu pojedynczych pamięci (re-import TMX po aktualizacji)."""
    items = []
    if rag_manager and rag_manager.engines:
        for name in sorted(rag_manager.engines.keys()):
            items.append(MenuItem(name, partial(reload_memory_menu, name)))
    else:
        items.append(MenuItem("(no memories)", None, enabled=False))
    return Menu(*items)

def rebuild_menu():
    """Prbuduj menu tray."""
    if not icon:
        return
    icon.menu = Menu(
        MenuItem(
            lambda text: "Hotkey active" if hotkey_active else "Hotkey inactive",
            toggle_hotkey,
            checked=lambda item: hotkey_active,
            default=True,
        ),
        MenuItem(
            lambda text: "Memory: " + (current_memory or "All"),
            get_memory_menu(),
        ),
        MenuItem(
            lambda text: "EN -> PL" if state_en else "PL -> EN",
            Menu(
                MenuItem("EN -> PL", toggle_language_en, checked=lambda item: state_en, radio=True),
                MenuItem("PL -> EN", toggle_language_pl, checked=lambda item: not state_en, radio=True),
            ),
        ),
        Menu.SEPARATOR,
        MenuItem(
            "Reload memory",
            get_reload_menu(),
        ),
        MenuItem("Refresh memories", refresh_memories),
        MenuItem("Reload replacements", reload_replacements),
        MenuItem("Reload prompt", reload_prompt),
        Menu.SEPARATOR,
        MenuItem("Settings", open_settings_menu),
        Menu.SEPARATOR,
        MenuItem("Quit", exit_prg),
    )

def exit_prg(icon=None, item=None):
    """Zamknij aplikację."""
    keyboard.remove_all_hotkeys()
    stop_llama()
    if icon:
        icon.stop()
    os._exit(0)

# ──────────────────────────────────────────────────────────────────────────────
# Settings GUI (okno ustawień)
# ──────────────────────────────────────────────────────────────────────────────

def open_settings_menu(icon=None, item=None):
    """Akcja menu tray — otwórz okno ustawień (własny wątek Tk, pystray zostaje na głównym)."""
    from settings_gui import open_settings
    open_settings(
        current=copy.deepcopy(CONFIG),
        save_fn=save_settings,
        validate_prompt_fn=validate_prompt_templates,
        restart_llama_fn=restart_llama_async,
        hotkey_pause_fn=keyboard.remove_all_hotkeys,
        hotkey_resume_fn=rebind_hotkey,
        config_path=Path(__file__).parent / "config.yaml",
    )

def save_settings(new_values: dict) -> dict:
    """Zapisz ustawienia z okna GUI: config.yaml + live-apply w pamięci procesu.
    Zwraca informację, które zmiany wymagają restartu (llama-servera / aplikacji)."""
    global current_memory
    old = copy.deepcopy(CONFIG)
    changed = _changed_keys(old, new_values)

    save_config_values(new_values)

    # Wartości czytane przy każdym tłumaczeniu (progi, temperature, porty, timeouty)
    # działają od razu po podmianie CONFIG; resztę przepinamy poniżej.
    for section, values in new_values.items():
        CONFIG.setdefault(section, {}).update(values)

    if "ui.hotkey" in changed:
        rebind_hotkey()
    if any(k.startswith("prompt.") for k in changed):
        reload_prompt()
    if "ui.replacements_file" in changed:
        load_replacements()
    if "translation.default_memory" in changed:
        current_memory = CONFIG["translation"]["default_memory"]
    if any(k.startswith("ui.") for k in changed):
        refresh_tray_icon()
    rebuild_menu()

    llama_keys = sorted(k for k in changed if k in RESTART_LLAMA_KEYS)
    app_keys = sorted(k for k in changed if k in RESTART_APP_KEYS)
    notes = []
    if "rag.embedding_model" in changed:
        notes.append("Zmiana modelu embeddingów wymusi przy restarcie ponowne indeksowanie "
                     "pamięci z TMX (i pobranie modelu, jeśli go nie ma lokalnie).")
    return {
        "llama_restart": bool(llama_keys),
        "app_restart": bool(app_keys),
        "llama_keys": llama_keys,
        "app_keys": app_keys,
        "notes": notes,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Llama server
# ──────────────────────────────────────────────────────────────────────────────

class LlamaStartError(RuntimeError):
    """llama-server nie wystartował (zła ścieżka, crash, brak odpowiedzi /health)."""

def start_llama():
    """Uruchom llama-server jako subprocess. Porażka podnosi LlamaStartError
    (zamiast ubijać proces) — main() przy starcie i tak kończy bieg, a restart
    z okna ustawień może wyświetlić błąd i działać dalej."""
    global llama_process
    llama_cfg = CONFIG["llama"]
    server_path = llama_cfg["server_path"]

    if not os.path.exists(server_path):
        # Spróbuj z PATH
        server_path = "llama-server"
        print(f"⚠️ Nie znaleziono pod podaną ścieżką, próbuję 'llama-server' z PATH")

    # Argumenty: jeden ciąg (format z GUI) albo lista YAML — oba spłaszczamy do
    # tokenów argv wg reguł Windows (elementy listy ze spacjami też rozbijamy,
    # więc żaden mix formatów nie trafi do llama-servera jako zepsuty token)
    raw_args = llama_cfg["args"]
    if isinstance(raw_args, str):
        llama_args = split_args_string(raw_args)
    else:
        llama_args = []
        for element in (raw_args or []):
            llama_args.extend(split_args_string(str(element)))
    args = [server_path, "-m", llama_cfg["model_path"]] + llama_args

    def filter_output(pipe, prefix="[llama]"):
        for line in iter(pipe.readline, ''):
            line = line.strip()
            if not line:
                continue
            if any(kw in line.lower() for kw in ['error', 'warning', 'eval time']):
                print(f"{prefix} {line}")

    print("🚀 Uruchamianie llama-server...")
    print(f"📂 Ścieżka: {server_path}")

    try:
        llama_process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        threading.Thread(target=filter_output, args=(llama_process.stdout, "🤖"), daemon=True).start()
        threading.Thread(target=filter_output, args=(llama_process.stderr, "🤖"), daemon=True).start()

        # Polling /health zamiast sztywnego sleepu — gotowe szybciej na ciepłym starcie,
        # bezpiecznie dłużej na zimnym (limit w llama.startup_timeout)
        health_url = f"http://127.0.0.1:{llama_cfg['port']}/health"
        deadline = time.time() + llama_cfg["startup_timeout"]
        ready = False
        print(f"⏳ Czekam na llama-server (polling /health, limit {llama_cfg['startup_timeout']}s)...")
        while time.time() < deadline:
            if llama_process.poll() is not None:
                raise LlamaStartError("llama-server zakończył się podczas startu (patrz log llama)")
            try:
                if requests.get(health_url, timeout=1).status_code == 200:
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.25)

        if not ready:
            raise LlamaStartError(f"llama-server nie odpowiedział w {llama_cfg['startup_timeout']}s")

        # Warmup w tle — pierwsze tłumaczenie bez opłaty startowej
        threading.Thread(target=warmup_llama, daemon=True).start()

        print(f"✅ llama-server uruchomiony na porcie {llama_cfg['port']}")
    except LlamaStartError:
        raise
    except FileNotFoundError:
        raise LlamaStartError(f"Nie znaleziono llama-server: {server_path}")
    except Exception as e:
        raise LlamaStartError(f"Błąd uruchamiania llama-server: {e}")

def warmup_llama():
    """Rozgrzej model jednym drobnym zapytaniem (w tle, nie blokuje startu)."""
    try:
        requests.post(
            f"http://127.0.0.1:{CONFIG['llama']['port']}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "OK"}], "max_tokens": 4, "stream": False},
            timeout=CONFIG["llama"]["request_timeout"],
        )
        print("🔥 llama-server rozgrzany")
    except Exception as e:
        print(f"⚠️ Warmup pominięty: {e}")
    # Rozgrzej też RAG: pierwszy encode modelu, skan glosariuszu obu kierunków
    # i leniwe ładowanie indeksów HNSW — inaczej te koszty trafiają w pierwsze tłumaczenie
    if rag_manager:
        rag_manager.warmup()

def stop_llama():
    """Zatrzymaj llama-server."""
    global llama_process
    if llama_process and llama_process.poll() is None:
        print("\n🛑 Zatrzymywanie llama-server...")
        llama_process.terminate()
        try:
            llama_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            llama_process.kill()
        print("✅ llama-server zatrzymany")

_llama_restart_lock = threading.Lock()

def restart_llama_async(on_done: Callable[[], None], on_error: Callable[[str], None]) -> None:
    """Zrestartuj llama-servera w tle (np. przycisk w oknie ustawień).
    Callbacki wywoływane z wątku roboczego — odbiorca musi marshalingować do swojego UI."""
    def worker():
        if not _llama_restart_lock.acquire(blocking=False):
            on_error("Restart llama-servera już trwa")
            return
        try:
            print("🔄 Restart llama-servera (ustawienia)...")
            stop_llama()
            start_llama()
            on_done()
        except LlamaStartError as e:
            on_error(str(e))
        except Exception as e:
            on_error(f"{type(e).__name__}: {e}")
        finally:
            _llama_restart_lock.release()
    threading.Thread(target=worker, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="BiedaMT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "bielik-11b-rag"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    rag_memory: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """GUI do zarządzania pamięciami."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BiedaMT — Memories</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
            h1 { color: #333; }
            .memory { display: flex; align-items: center; gap: 10px; padding: 10px; border-bottom: 1px solid #eee; }
            .memory b { flex: 1; }
            .memory span { color: #666; font-size: 0.9em; }
            button { padding: 5px 10px; cursor: pointer; }
            .empty { color: #999; font-style: italic; padding: 20px; }
        </style>
    </head>
    <body>
        <h1>📚 Memories</h1>
        <div id="list"></div>
        <script>
            async function load() {
                const r = await fetch('/v1/memories');
                const data = await r.json();
                const div = document.getElementById('list');
                div.innerHTML = '';
                if (data.memories.length === 0) {
                    div.innerHTML = '<div class="empty">Brak pamięci. Dodaj plik TMX do folderu memories.</div>';
                    return;
                }
                for (const m of data.memories) {
                    div.innerHTML += `
                        <div class="memory">
                            <b>${m.name}</b>
                            <span>${m.segments || 0} seg</span>
                            <button onclick="reloadMem('${m.name}')">Reload</button>
                            <button onclick="delMem('${m.name}')">Delete</button>
                        </div>
                    `;
                }
            }
            async function reloadMem(name) {
                await fetch(`/v1/memories/${name}/reload`, {method: 'POST'});
                load();
            }
            async function delMem(name) {
                if (confirm('Delete ' + name + '?')) {
                    await fetch(`/v1/memories/${name}`, {method: 'DELETE'});
                    load();
                }
            }
            load();
        </script>
    </body>
    </html>
    """

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint kompatybilny z OpenAI (z RAG)."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    source_text = None
    for msg in request.messages:
        if msg.role == 'user' and msg.content.startswith('\n[USER_PROMPT]\n'):
            source_text = msg.content.split('\n[USER_PROMPT]\n', 1)[1]
            break
    
    similar = []
    if source_text and rag_manager:
        if request.rag_memory and request.rag_memory in rag_manager.engines:
            similar = rag_manager.engines[request.rag_memory].search(source_text, n_results=CONFIG["rag"]["search_results"])
        else:
            similar = rag_manager.search_all(source_text, n_results_per_memory=CONFIG["rag"]["search_all_per_memory"])
    similar = filter_references(similar)
    
    rag_prompt = build_rag_chatmessage(similar)
    if rag_prompt:
        request.messages.insert(
            next(i for i, msg in enumerate(request.messages) if msg.role == 'user' and '[USER_PROMPT]' in msg.content),
            ChatMessage(role='user', content=rag_prompt),
        )
    
    llama_url = f"http://127.0.0.1:{CONFIG['llama']['port']}/v1/chat/completions"
    lm_payload = {
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "cache_prompt": True,
        "stream": False,
    }
    
    try:
        response = session.post(llama_url, json=lm_payload, timeout=CONFIG["llama"]["request_timeout"])
        response.raise_for_status()
        translation = response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Llama server error: {str(e)}")
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": translation}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

@app.get("/favicon.ico")
async def favicon():
    """Favicon z assetów."""
    icon_path = Path(CONFIG["ui"]["icon_ico"])
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/x-icon")
    raise HTTPException(status_code=404)

@app.get("/v1/memories")
async def list_memories():
    """Lista dostępnych pamięci."""
    if not rag_manager:
        return {"memories": []}
    return {
        "memories": [
            {
                "name": name,
                "segments": engine.collection.count() if engine.collection else 0,
                "tmx_exists": os.path.exists(f"{CONFIG['memories']['dir']}/{name}.tmx"),
            }
            for name, engine in rag_manager.engines.items()
        ]
    }

@app.post("/v1/memories/{memory_name}/reload")
async def reload_memory(memory_name: str):
    """Przeładuj pamięć."""
    if not rag_manager or not rag_manager.reload_memory(memory_name):
        raise HTTPException(status_code=404, detail=f"Memory {memory_name} not found")
    return {"status": "reloaded", "memory": memory_name}

@app.delete("/v1/memories/{memory_name}")
async def delete_memory(memory_name: str):
    """Usuń pamięć."""
    if not rag_manager or not rag_manager.delete_memory(memory_name):
        raise HTTPException(status_code=404, detail=f"Memory {memory_name} not found")
    return {"status": "deleted", "memory": memory_name}

@app.get("/health")
async def health():
    """Sprawdzenie stanu."""
    lm_status = "ok"
    try:
        requests.get(f"http://127.0.0.1:{CONFIG['llama']['port']}/health", timeout=2)
    except Exception:
        lm_status = "disconnected"
    return {
        "status": "running",
        "llama_server": lm_status,
        "memories": list(rag_manager.engines.keys()) if rag_manager else [],
        "current_memory": current_memory,
        "hotkey_active": hotkey_active,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global rag_manager, icon
    
    # DPI awareness
    try:
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    
    myappid = 'krzyk.biedamt.v1'
    try:
        windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass
    
    # Load replacements
    load_replacements()
    
    # Init RAG
    print("📚 Inicjalizacja RAG...")
    rag_manager = RAGManager(
        memories_dir=CONFIG["memories"]["dir"],
        chroma_path=CONFIG["memories"]["chroma_path"],
        embedding_model=CONFIG["rag"]["embedding_model"],
        embedding_device=CONFIG["rag"]["embedding_device"],
        terms_min_chars=CONFIG["rag"]["terms_min_chars"],
        terms_max_words=CONFIG["rag"]["terms_max_words"],
    )
    
    # Start llama
    try:
        start_llama()
    except LlamaStartError as e:
        print(f"❌ {e}")
        sys.exit(1)
    atexit.register(stop_llama)
    
    # Setup hotkey
    keyboard.add_hotkey(CONFIG["ui"]["hotkey"], action, suppress=True, trigger_on_release=True)
    
    # Setup tray
    icon = Icon(
        CONFIG["ui"]["app_name"],
        Image.open(CONFIG["ui"]["icon_active"]),
        CONFIG["ui"]["app_title"],
    )
    rebuild_menu()
    
    # Run FastAPI w tle (daemon) — pystray potrzebuje głównego wątku
    import uvicorn
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": CONFIG["server"]["host"], "port": CONFIG["server"]["port"]},
        daemon=True,
    )
    api_thread.start()
    
    print("\n" + "="*50)
    print(f"🚀 BiedaMT gotowy!")
    print(f"📁 Pamięci: {CONFIG['memories']['dir']}")
    print(f"🔗 API: http://localhost:{CONFIG['server']['port']}/v1/chat/completions")
    print(f"📚 Załadowane: {list(rag_manager.engines.keys())}")
    print(f"🤖 Llama: http://localhost:{CONFIG['llama']['port']}")
    print(f"⌨️  Hotkey: {CONFIG['ui']['hotkey']}")
    print("="*50 + "\n")
    
    # pystray MUSI być na głównym wątku (Windows message pump)
    icon.run()


if __name__ == "__main__":
    main()
