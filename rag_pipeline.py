import os
import time
import json
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

print("=" * 60)
print("  STEP 1 — Data Preparation")
print("=" * 60)

from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_FILE = Path(__file__).parent / "data" / "sports_science_corpus.json"

with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_corpus = json.load(f)

raw_texts = [doc["text"] for doc in raw_corpus]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.create_documents(raw_texts)
text_lines = [chunk.page_content for chunk in chunks]

print(f"  Documents loaded  : {len(raw_corpus)}")
print(f"  Chunks after split: {len(text_lines)}")
print(f"  Sample chunk      : {text_lines[0][:120]}...")


print()
print("=" * 60)
print("  STEP 2 — Embedding Generation (HuggingFace Sentence Transformers)")
print("=" * 60)

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def emb_text(text: str):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

sample = text_lines[0]
vec = emb_text(sample)
print(f"  Model             : {EMBEDDING_MODEL}")
print(f"  Vector dimension  : {len(vec)}")
print(f"  Architecture      : BERT fine-tuned Transformer (sentence-transformers)")
print(f"  Sample norm       : {sum(v**2 for v in vec):.4f} (should be ~1.0 — normalized)")


print()
print("=" * 60)
print("  STEP 3 — Vector Database Setup (ChromaDB)")
print("=" * 60)

from chromadb import PersistentClient

DB_PATH         = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "sports_science_rag"

client = PersistentClient(path=DB_PATH)

try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

print(f"  Building embeddings for {len(text_lines)} chunks…")
embeddings = []
ids        = []

for i, line in enumerate(tqdm(text_lines, desc="  Creating embeddings")):
    embeddings.append(emb_text(line))
    ids.append(str(i))

collection.add(
    documents=text_lines,
    embeddings=embeddings,
    ids=ids,
)

print(f"  ✓ {collection.count()} chunks indexed into ChromaDB at {DB_PATH}")


print()
print("=" * 60)
print("  STEP 4 — Retrieval (semantic search)")
print("=" * 60)

def retrieve(question: str, n_results: int = 3):
    query_embedding = emb_text(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    context = "\n\n".join(results["documents"][0])
    return context, results["documents"][0]

PROMPT = """Use the information enclosed in <context> tags to provide an answer to \
the question enclosed in <question> tags.

<context>
{context}
</context>

<question>
{question}
</question>
"""

PROMPT_STRUCTURED = """You are a sports science expert. \
Use the evidence to answer the question accurately and specifically.

Evidence:
{context}

Question: {question}

Answer (3-5 sentences, practical and evidence-based):"""

PROMPT_MINIMAL = "Answer: {question}\nContext: {context}"

test_question = "How does ankle dorsiflexion range affect sprint acceleration?"
context, docs = retrieve(test_question, n_results=3)

print(f"  Query   : {test_question}")
print(f"  Top doc : {docs[0][:120]}...")

prompt_xml        = PROMPT.format(context=context, question=test_question)
prompt_structured = PROMPT_STRUCTURED.format(context=context, question=test_question)
prompt_minimal    = PROMPT_MINIMAL.format(context=context, question=test_question)
print(f"  Templates tested : XML-style | Structured | Minimal")


print()
print("=" * 60)
print("  STEP 5 — Generation (HuggingFace LLM)")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("  Loading LLM: google/flan-t5-base (250M params, local)…")
t0     = time.time()
_tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def llm(prompt, **_):
    enc = _tok(prompt[:900], return_tensors="pt", truncation=True, max_length=512)
    out = _model.generate(**enc, max_new_tokens=150, no_repeat_ngram_size=4)
    return [{"generated_text": _tok.decode(out[0], skip_special_tokens=True)}]

print(f"  ✓ LLM loaded in {time.time()-t0:.1f}s")


def answer_question(question: str, prompt_template: str = PROMPT, n_results: int = 3):
    t0 = time.time()
    context, _ = retrieve(question, n_results)
    prompt = prompt_template.format(context=context, question=question)
    out    = llm(prompt[:900])
    answer = out[0]["generated_text"].strip()
    latency = time.time() - t0
    return answer, latency


print()
print("=" * 60)
print("  END-TO-END DEMO")
print("=" * 60)

demo_questions = [
    "How does ankle dorsiflexion affect sprint acceleration?",
    "Who is the fastest footballer: Kylian Mbappe or Lamine Yamal?",
    "What exercises improve reactive strength for sprinters?",
    "How does sleep affect athletic recovery?",
]

for q in demo_questions:
    answer, latency = answer_question(q, PROMPT)
    print(f"\n  Q: {q}")
    print(f"  A: {answer}")
    print(f"  ⏱ {latency:.2f}s")


print()
print("=" * 60)
print("  PROMPT TEMPLATE COMPARISON")
print("=" * 60)

test_q = "What muscles are used in sprint acceleration?"
for name, tmpl in [("XML-style", PROMPT), ("Structured", PROMPT_STRUCTURED), ("Minimal", PROMPT_MINIMAL)]:
    ans, lat = answer_question(test_q, tmpl)
    print(f"\n  [{name}] ({lat:.2f}s)")
    print(f"  {ans[:200]}")


print()
print("=" * 60)
print("  STEP 6 — Evaluation (Precision@3 + Latency)")
print("=" * 60)

EVAL_SET = [
    ("ankle dorsiflexion sprint performance",    ["ankle", "dorsiflexion"]),
    ("horizontal force acceleration",            ["force", "horizontal", "acceleration"]),
    ("Mbappe Yamal speed comparison",            ["mbappe", "yamal"]),
    ("reactive strength plyometrics",            ["reactive", "strength"]),
    ("sleep recovery athlete",                   ["sleep", "recovery"]),
    ("protein muscle adaptation",                ["protein", "muscle"]),
    ("Noah Lyles 100m sprint",                   ["lyles", "sprint", "100m"]),
    ("ankle tendon sprint elasticity",           ["tendon", "ankle"]),
]

def precision_at_k(docs, keywords, k=3):
    hits = sum(1 for d in docs[:k] if any(kw.lower() in d.lower() for kw in keywords))
    return hits / k

precisions = []
latencies  = []

print(f"  {'Query':<45} {'P@3':>5} {'Latency':>9}")
print(f"  {'-'*45} {'-'*5} {'-'*9}")

for query, keywords in EVAL_SET:
    t0 = time.time()
    _, docs = retrieve(query, n_results=3)
    lat = (time.time() - t0) * 1000
    p   = precision_at_k(docs, keywords)
    precisions.append(p)
    latencies.append(lat)
    mark = "✓" if p > 0 else "✗"
    print(f"  {mark} {query:<44} {p:>5.2f} {lat:>7.1f}ms")

print(f"\n  Avg Precision@3 : {sum(precisions)/len(precisions):.3f}")
print(f"  Avg Latency     : {sum(latencies)/len(latencies):.2f}ms")
print(f"  Index size      : {collection.count()} chunks")


print()
print("=" * 60)
print("  EMBEDDING MODEL COMPARISON")
print("=" * 60)

MODELS = {
    "all-MiniLM-L6-v2":                       "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2":  "paraphrase-multilingual-MiniLM-L12-v2",
}

for model_name, model_id in MODELS.items():
    m    = SentenceTransformer(model_id)
    t0   = time.time()
    embs = [m.encode([t], normalize_embeddings=True).tolist()[0] for t in text_lines[:20]]
    idx_t = time.time() - t0

    col_name = f"eval_{model_name[:20].replace('-','_')}"
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
    col2 = client.create_collection(name=col_name, metadata={"hnsw:space": "cosine"})
    all_embs = m.encode(text_lines, normalize_embeddings=True).tolist()
    col2.add(documents=text_lines, embeddings=all_embs, ids=[str(i) for i in range(len(text_lines))])

    ps = []
    for query, kws in EVAL_SET[:4]:
        qv  = m.encode([query], normalize_embeddings=True).tolist()[0]
        res = col2.query(query_embeddings=[qv], n_results=3)
        ps.append(precision_at_k(res["documents"][0], kws))

    print(f"\n  Model   : {model_name}")
    print(f"  P@3     : {sum(ps)/len(ps):.3f}")
    print(f"  Dim     : {len(all_embs[0])}")
    print(f"  Idx time: {idx_t:.2f}s (first 20 docs)")


print()
print("=" * 60)
print("  ✓ RAG pipeline complete — all steps demonstrated")
print("  ✓ Retrieval: ChromaDB cosine similarity (HNSW)")
print("  ✓ Embedding: HuggingFace Sentence Transformers")
print("  ✓ Generation: HuggingFace Transformers (flan-t5-base)")
print("  ✓ 2 embedding models compared")
print("  ✓ 3 prompt templates compared")
print("  ✓ Evaluation: Precision@3 + latency")
print("=" * 60)
