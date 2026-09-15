"""RAG engine — ładowanie TMX do ChromaDB i wyszukiwanie podobnych tłumaczeń."""
import os
import re
import hashlib
import threading
from typing import List, Dict, Iterable, Optional, Tuple
from lxml import etree

import chromadb
from sentence_transformers import SentenceTransformer

# Model e5 wymaga prefiksów: "query: " dla zapytań, "passage: " dla indeksowanych dokumentów
QUERY_PREFIX = "query"
PASSAGE_PREFIX = "passage"
# Znacznik wersji embeddingów w metadanych kolekcji — zmiana wymusza re-import z TMX.
# Wersjonowany nazwą modelu: inny model embeddingów = niekompatybilne indeksy.
EMBEDDING_MARKER = "e5-prefixed-v1"
LEGACY_E5_MODEL = "intfloat/multilingual-e5-small"  # model sprzed wersjonowania nazwą

def embedding_marker(embedding_model: str) -> str:
    """Znacznik kolekcji dla danego modelu embeddingów."""
    return f"{EMBEDDING_MARKER}:{embedding_model}"
# Partia pobierania metadanych do glosariusza — pełny get() na dużej kolekcji
# wywala SQLite ("too many SQL variables")
TERM_FETCH_BATCH = 5000


def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    """Embedding zapytania z prefiksem e5 (jeden embedding może obsłużyć wszystkie pamięci)."""
    return model.encode(
        [f"{QUERY_PREFIX}: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()


# Funkcyjne słowa (EN+PL) — pojedyncze słowa-nie-terminy są za częste w każdym zdaniu,
# wymuszanie ich tłumaczenia psuje wynik ("To" -> "do" w każdym zdaniu)
SINGLE_WORD_STOPWORDS = {
    # EN
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done",
    "and", "or", "but", "if", "then", "when", "while", "not", "no", "yes", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "them",
    "his", "her", "their", "our", "your", "my", "me", "him", "us", "what", "which",
    "who", "how", "why", "where", "all", "any", "can", "may", "will", "would", "shall",
    "should", "must", "have", "has", "had", "up", "down", "out", "off", "over", "under",
    "into", "onto", "than", "so", "such", "only", "also", "very", "more", "most",
    "some", "each", "per", "via", "there", "here", "other", "new", "use", "used",
    "using", "set", "get", "one", "two", "both", "same", "own",
    # PL
    "a", "i", "o", "w", "z", "do", "na", "od", "po", "za", "we", "ze", "dla", "jak",
    "ale", "lub", "albo", "oraz", "że", "to", "ten", "ta", "te", "tym", "tej", "tam",
    "nie", "tak", "czy", "gdy", "kiedy", "gdzie", "co", "kto", "się", "jest", "są",
    "być", "był", "była", "było", "były", "jeśli", "jeżeli", "aby", "przez", "przy",
    "bez", "pod", "nad", "między", "który", "która", "które", "jego", "jej", "ich",
    "nas", "was", "mnie", "ciebie", "siebie", "taki", "taka", "wszystkie", "oraz",
}


def is_term_candidate(text: str, min_chars: int, max_words: int) -> bool:
    """Czy segment nadaje się na termin w glosariuszu (krótki, nie sam numer)."""
    text = text.strip()
    if len(text) < min_chars or len(text.split()) > max_words:
        return False
    # odrzuć segmenty bez liter (liczby, ceny, znaki interpunkcyjne)
    if not any(c.isalpha() for c in text):
        return False
    words = text.lower().split()
    if len(words) == 1 and words[0] in SINGLE_WORD_STOPWORDS:
        return False
    return True


class TermMatcher:
    """Finds glossary terms in text. Returns canonical term form from dictionary."""

    def __init__(self, terms: Dict[str, str]):
        self._map = {term.lower(): (term, target) for term, target in terms.items()}
        pattern = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
        self._regex = re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)

    def find(self, text: str) -> List[Tuple[str, str]]:
        """Returns (canonical_term, translation) in occurrence order, no duplicates."""
        found, seen = [], set()
        for match in self._regex.finditer(text):
            key = match.group(0).lower()
            if key in seen:
                continue
            seen.add(key)
            orig, trans = self._map[key]
            found.append((orig, trans))
        return found


class TMXTranslatorRAG:
    """RAG engine dla pojedynczej pamięci TMX."""

    def __init__(self, client, model: SentenceTransformer, memory_name: str, tmx_path: str,
                 embedding_marker: str = EMBEDDING_MARKER):
        self.client = client
        self.model = model
        self.memory_name = memory_name
        self.tmx_path = tmx_path
        self.collection_name = f"tmx_{memory_name}"
        self.embedding_marker = embedding_marker
        self.collection = None
        self._terms_cache: Dict[str, Dict[str, str]] = {}  # src_key -> {term: tłumaczenie}
        self._terms_lock = threading.Lock()
        self._load_or_create_collection()

    def _marker_matches(self, marker: Optional[str]) -> bool:
        """Czy kolekcja jest w formacie embeddingów zgodnym z bieżącym modelem."""
        if marker == self.embedding_marker:
            return True
        # Kolekcje sprzed wersjonowania nazwą modelu: ważne tylko, gdy model
        # pozostał starym domyślnym e5 (wtedy embeddingi są te same)
        return marker == EMBEDDING_MARKER and self.embedding_marker == embedding_marker(LEGACY_E5_MODEL)

    def _load_or_create_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            # Kolekcje sprzed przejścia na prefiksy e5 / cosine / z innego modelu —
            # wymuś jednorazowy re-import
            if not self._marker_matches((self.collection.metadata or {}).get("embedding")):
                print(f"🔄 Kolekcja w starym formacie embeddingów, re-import: {self.memory_name}...")
                self.client.delete_collection(self.collection_name)
                self.collection = None
        except Exception:
            self.collection = None
        if self.collection is not None:
            print(f"📚 Załadowano pamięć: {self.memory_name} ({self.collection.count()} fragmentów)")
        else:
            print(f"🔄 Tworzenie nowej pamięci z {self.tmx_path}...")
            self._create_collection_from_tmx()

    def _create_collection_from_tmx(self):
        # cosine — odległość (0 = identyczne) łatwo porównywać z progami z configu
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding": self.embedding_marker},
        )
        try:
            with open(self.tmx_path, "rb") as f:
                content = f.read()
            parser = etree.XMLParser(recover=True, remove_blank_text=True)
            root = etree.fromstring(content, parser=parser)
            tu_elements = root.xpath("//tu")
            print(f"📄 Znaleziono {len(tu_elements)} elementów <tu>")

            documents, metadatas, ids = [], [], []
            seen_pairs = set()
            duplicate_count = 0

            for i, tu in enumerate(tu_elements):
                en_seg = pl_seg = None
                for tuv in tu.xpath(".//tuv"):
                    lang = (
                        tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
                        or tuv.get("xml:lang")
                        or tuv.get("lang")
                    )
                    seg_elem = tuv.find(".//seg")
                    if seg_elem is not None and seg_elem.text:
                        text = seg_elem.text.strip()
                        if lang and lang.lower() in ("en", "en-us", "en-gb", "english"):
                            en_seg = text
                        elif lang and lang.lower() in ("pl", "pl-pl", "polish", "polski"):
                            pl_seg = text

                if en_seg and pl_seg:
                    pair_key = f"{en_seg}|{pl_seg}"
                    if pair_key in seen_pairs:
                        duplicate_count += 1
                        continue
                    seen_pairs.add(pair_key)
                    documents.append(en_seg)
                    metadatas.append({"source": en_seg, "target": pl_seg, "memory": self.memory_name})
                    ids.append(hashlib.md5(pair_key.encode()).hexdigest())
                else:
                    if not en_seg:
                        print(f"⚠️ Brak EN w tu #{i+1}")
                    if not pl_seg:
                        print(f"⚠️ Brak PL w tu #{i+1}")

            if not documents:
                print("⚠️ Brak tłumaczeń do załadowania")
                return

            for i in range(0, len(documents), 100):
                batch = documents[i:i+100]
                batch_embeddings = self.model.encode(
                    [f"{PASSAGE_PREFIX}: {d}" for d in batch],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
                self.collection.add(
                    documents=batch,
                    embeddings=batch_embeddings,
                    metadatas=metadatas[i:i+100],
                    ids=ids[i:i+100],
                )
                if i and i % 2000 == 0:
                    print(f"   ⏳ Zaindeksowano {i}/{len(documents)}...")
            print(f"✅ Załadowano {len(documents)} tłumaczeń do '{self.memory_name}'")
            if duplicate_count:
                print(f"🔁 Pominięto {duplicate_count} duplikatów")
        except Exception as e:
            print(f"❌ Błąd parsowania TMX: {e}")
            raise

    def exact_lookup(self, text: str, src_key: str, tgt_key: str) -> Optional[str]:
        """Dokładne dopasowanie segmentu po metadanych (bez embeddingów i LLM)."""
        if not self.collection or self.collection.count() == 0:
            return None
        try:
            res = self.collection.get(where={src_key: {"$eq": text}}, include=["metadatas"], limit=1)
            metadatas = res.get("metadatas") or []
            if metadatas:
                return metadatas[0][tgt_key]
        except Exception as e:
            print(f"⚠️ Błąd exact match: {e}")
        return None

    def get_terms(self, src_key: str, tgt_key: str,
                  min_chars: int, max_words: int) -> Dict[str, str]:
        """Glosariusz z tej pamięci {termin: tłumaczenie} — leniwie z metadanych Chroma."""
        with self._terms_lock:
            if src_key in self._terms_cache:
                return self._terms_cache[src_key]
            terms: Dict[str, str] = {}
            if self.collection and self.collection.count() > 0:
                try:
                    # Paginacja — patrz TERM_FETCH_BATCH
                    offset = 0
                    while True:
                        res = self.collection.get(
                            include=["metadatas"], limit=TERM_FETCH_BATCH, offset=offset)
                        metadatas = res.get("metadatas") or []
                        for md in metadatas:
                            candidate = md.get(src_key)
                            target = md.get(tgt_key, "")
                            if not candidate or not is_term_candidate(candidate, min_chars, max_words):
                                continue
                            # Sanity: tłumaczenie terminu nie powinno być wielokrotnie
                            # dłuższe niż termin (brudne TU w TMX)
                            if len(target.split()) > max_words * 3:
                                continue
                            terms[candidate] = target
                        if len(metadatas) < TERM_FETCH_BATCH:
                            break
                        offset += len(metadatas)
                except Exception as e:
                    print(f"⚠️ Błąd budowania glosariusza: {e}")
            self._terms_cache[src_key] = terms
            return terms

    def search(self, query: str, n_results: int = 3,
               query_embedding: Optional[List[float]] = None) -> List[Dict]:
        if not self.collection or self.collection.count() == 0:
            return []
        if query_embedding is None:
            query_embedding = embed_query(self.model, query)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
            translations = []
            if results.get("metadatas") and results["metadatas"][0]:
                for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                    translations.append({
                        "source": metadata["source"],
                        "target": metadata["target"],
                        "memory": metadata["memory"],
                        "distance": distance,
                    })
            return translations
        except Exception as e:
            print(f"⚠️ Błąd wyszukiwania: {e}")
            return []


class RAGManager:
    """Manager wielu pamięci TMX."""

    def __init__(self, memories_dir: str = "./memories", chroma_path: str = "./chroma",
                 embedding_model: str = "intfloat/multilingual-e5-small",
                 embedding_device: str = "cpu",
                 terms_min_chars: int = 2, terms_max_words: int = 4):
        self.memories_dir = memories_dir
        self.terms_min_chars = terms_min_chars
        self.terms_max_words = terms_max_words
        self.embedding_marker = embedding_marker(embedding_model)
        self.client = chromadb.PersistentClient(path=chroma_path)
        print(f"🧠 Model embeddingów: {embedding_model} ({embedding_device})")
        self.model = SentenceTransformer(embedding_model, device=embedding_device)
        self.engines: Dict[str, TMXTranslatorRAG] = {}
        self.ensure_memories_dir()
        self.load_all_memories()

    def ensure_memories_dir(self):
        if not os.path.exists(self.memories_dir):
            os.makedirs(self.memories_dir)
            print(f"📁 Utworzono folder {self.memories_dir}")

    def load_all_memories(self):
        tmx_files = [f for f in os.listdir(self.memories_dir) if f.endswith(".tmx")]
        if not tmx_files:
            print("⚠️ Brak plików TMX")
            return
        for tmx_file in tmx_files:
            memory_name = tmx_file.replace(".tmx", "")
            tmx_path = os.path.join(self.memories_dir, tmx_file)
            try:
                engine = TMXTranslatorRAG(
                    client=self.client,
                    model=self.model,
                    memory_name=memory_name,
                    tmx_path=tmx_path,
                    embedding_marker=self.embedding_marker,
                )
                self.engines[memory_name] = engine
            except Exception as e:
                print(f"❌ Błąd ładowania {memory_name}: {e}")

    def warmup(self) -> None:
        """Rozgrzej leniwe ścieżki RAG (w tle po starcie): skan glosariuszu obu
        kierunków, zbudowanie matcherów terminów i po jednym zapytaniu HNSW na
        pamięć (jeden wspólny embedding — pierwszy encode modelu i leniwe
        ładowanie indeksów inaczej trafiają w pierwsze tłumaczenie)."""
        if not self.engines:
            return
        try:
            query_embedding = embed_query(self.model, "warmup")
            for engine in self.engines.values():
                engine.get_terms("source", "target", self.terms_min_chars, self.terms_max_words)
                engine.get_terms("target", "source", self.terms_min_chars, self.terms_max_words)
                engine.search("warmup", n_results=1, query_embedding=query_embedding)
            self.get_term_matcher("en")
            self.get_term_matcher("pl")
            print(f"🔥 RAG rozgrzany ({len(self.engines)} pamięci)")
        except Exception as e:
            print(f"⚠️ Warmup RAG pominięty: {e}")

    def _warmup_engine(self, engine: TMXTranslatorRAG) -> None:
        """Rozgrzej pojedynczą pamięć po re-imporcie: skan glosariuszu obu
        kierunków (świeży obiekt = puste cache) + leniwe ładowanie indeksu HNSW."""
        try:
            engine.get_terms("source", "target", self.terms_min_chars, self.terms_max_words)
            engine.get_terms("target", "source", self.terms_min_chars, self.terms_max_words)
            engine.search("warmup", n_results=1, query_embedding=embed_query(self.model, "warmup"))
        except Exception as e:
            print(f"⚠️ Warmup pamięci {engine.memory_name} pominięty: {e}")

    def resolve_engines(self, memories: Optional[Iterable[str]] = None) -> List[TMXTranslatorRAG]:
        """Silniki dla bieżącego wyboru pamięci: None = wszystkie, zbiór nazw = te pamięci.
        Nazwy bez silnika (usunięta pamięć) są pomijane."""
        if memories is None:
            return list(self.engines.values())
        selected = set(memories)
        return [engine for name, engine in self.engines.items() if name in selected]

    def search_selected(self, query: str, memories: Optional[Iterable[str]] = None,
                        n_results: int = 10, n_results_per_memory: int = 2) -> List[Dict]:
        """Wyszukiwanie w wybranych pamięciach. Jedna pamięć = głębiej (n_results),
        kilka = po n_results_per_memory z każdej (jedno embedowanie zapytania)."""
        engines = self.resolve_engines(memories)
        if not engines:
            return []
        if len(engines) == 1:
            return engines[0].search(query, n_results=n_results)
        query_embedding = embed_query(self.model, query)
        results: List[Dict] = []
        for engine in engines:
            results.extend(engine.search(query, n_results=n_results_per_memory,
                                         query_embedding=query_embedding))
        return results

    def exact_lookup(self, text: str, source_lang: str,
                     memories: Optional[Iterable[str]] = None) -> Optional[str]:
        """Dokładne trafienie w wybranych pamięciach — gotowe tłumaczenie bez LLM (lub None)."""
        if source_lang == "en":
            src_key, tgt_key = "source", "target"
        else:
            src_key, tgt_key = "target", "source"
        for engine in self.resolve_engines(memories):
            hit = engine.exact_lookup(text, src_key, tgt_key)
            if hit is not None:
                return hit
        return None

    def get_term_matcher(self, source_lang: str,
                         memories: Optional[Iterable[str]] = None) -> Optional[TermMatcher]:
        """Matcher termów dla kierunku (wybrane pamięci = terminy tylko z nich)."""
        if source_lang == "en":
            src_key, tgt_key = "source", "target"
        else:
            src_key, tgt_key = "target", "source"
        combined: Dict[str, str] = {}
        for engine in self.resolve_engines(memories):
            terms = engine.get_terms(src_key, tgt_key, self.terms_min_chars, self.terms_max_words)
            combined.update(terms)
        combined = {t: tgt for t, tgt in combined.items() if tgt}
        if not combined:
            return None
        return TermMatcher(combined)

    def reload_memory(self, memory_name: str) -> bool:
        tmx_path = os.path.join(self.memories_dir, f"{memory_name}.tmx")
        if not os.path.exists(tmx_path):
            return False
        try:
            if memory_name in self.engines:
                try:
                    self.client.delete_collection(f"tmx_{memory_name}")
                except Exception:
                    pass
            engine = TMXTranslatorRAG(
                client=self.client,
                model=self.model,
                memory_name=memory_name,
                tmx_path=tmx_path,
                embedding_marker=self.embedding_marker,
            )
            self.engines[memory_name] = engine
            self._warmup_engine(engine)
            return True
        except Exception as e:
            print(f"❌ Błąd reloadu {memory_name}: {e}")
            return False

    def delete_memory(self, memory_name: str) -> bool:
        try:
            if memory_name in self.engines:
                del self.engines[memory_name]
            self.client.delete_collection(name=f"tmx_{memory_name}")
            print(f"🗑️ Usunięto kolekcję: {memory_name}")
            return True
        except Exception as e:
            print(f"❌ Błąd usuwania {memory_name}: {e}")
            return False
