import os
import sys
import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

parser = argparse.ArgumentParser()
parser.add_argument("--hf-token",  default=os.environ.get("HF_TOKEN", ""))
parser.add_argument("--model",     default="flan-t5-base",
                    choices=["flan-t5-base", "flan-t5-large", "mistral"])
parser.add_argument("--n-results", default=3, type=int)
args = parser.parse_args()

HF_TOKEN = args.hf_token
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

DATA_FILE = Path(__file__).parent / "data" / "sports_science_corpus.json"
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_corpus = json.load(f)

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks   = text_splitter.create_documents([doc["text"] for doc in raw_corpus])
text_lines    = [c.page_content for c in text_chunks]

from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

def emb_text(text: str):
    return emb_model.encode([text], normalize_embeddings=True).tolist()[0]

from chromadb import PersistentClient

client     = PersistentClient(path=str(Path(__file__).parent / "chroma_db"))
COL_NAME   = "sports_science_rag"

try:
    client.delete_collection(COL_NAME)
except Exception:
    pass

collection = client.create_collection(name=COL_NAME, metadata={"hnsw:space": "cosine"})

embeddings = []
ids        = []
for i, line in enumerate(tqdm(text_lines, desc="Embedding", leave=False)):
    embeddings.append(emb_text(line))
    ids.append(str(i))

collection.add(documents=text_lines, embeddings=embeddings, ids=ids)

llm        = None
llm_name   = ""
use_hf_api = False

if args.model == "mistral" and HF_TOKEN:
    try:
        from huggingface_hub import InferenceClient
        _client    = InferenceClient(token=HF_TOKEN)
        llm        = _client
        llm_name   = "mistralai/Mistral-7B-Instruct-v0.3"
        use_hf_api = True
    except Exception as e:
        args.model = "flan-t5-base"

if args.model in ("flan-t5-base", "flan-t5-large") or llm is None:
    model_id = f"google/flan-t5-{args.model.split('-')[-1]}" if "flan" in args.model else "google/flan-t5-base"
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _tok   = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    def llm(prompt, **_):
        ids = _tok(prompt[:900], return_tensors="pt", truncation=True, max_length=512)
        out = _model.generate(**ids, max_new_tokens=150, no_repeat_ngram_size=4)
        return [{"generated_text": _tok.decode(out[0], skip_special_tokens=True)}]
    llm_name = model_id

PROMPT_XML = """Use the information enclosed in <context> tags to provide an answer to \
the question enclosed in <question> tags.

<context>
{context}
</context>

<question>
{question}
</question>"""

PROMPT_STRUCTURED = """Based on the following excerpts from sports science research, answer the question below.

{context}

{question}"""

ACTIVE_PROMPT = PROMPT_STRUCTURED if use_hf_api else PROMPT_XML

def retrieve_context(question: str, n: int = 3):
    qv  = emb_text(question)
    res = collection.query(query_embeddings=[qv], n_results=n)
    return "\n\n".join(res["documents"][0]), res["documents"][0]

def ask(question: str) -> str:
    context, docs = retrieve_context(question, args.n_results)
    prompt        = ACTIVE_PROMPT.format(context=context[:700], question=question[:300])

    if use_hf_api:
        resp = llm.chat_completion(
            messages=[
                {"role": "system", "content": "You are a sports science expert."},
                {"role": "user",   "content": prompt},
            ],
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_tokens=300,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    else:
        out = llm(prompt[:900])
        return out[0]["generated_text"].strip()

SUGGESTIONS = [
    "How does ankle dorsiflexion affect sprint acceleration?",
    "Who is faster between Mbappé and Lamine Yamal?",
    "What exercises improve reactive strength for sprinters?",
    "How does sleep quality affect athletic recovery?",
    "What is Noah Lyles' sprint profile and what makes him fast?",
    "How does horizontal force production relate to 30m sprint performance?",
]

print(f"\n  Model : {llm_name}")
print(f"  Docs  : {len(raw_corpus)} scientific documents indexed")
print()
print("  Suggested questions:")
for i, s in enumerate(SUGGESTIONS, 1):
    print(f"    {i}. {s}")
print()
print("  Type a question, a number (1-6), or 'quit' to exit.")
print("  " + "─" * 46)

while True:
    try:
        user_input = input("\n  You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  Goodbye.")
        break

    if not user_input or user_input.lower() in ("quit", "exit", "q"):
        print("  Goodbye.")
        break

    if user_input.isdigit() and 1 <= int(user_input) <= len(SUGGESTIONS):
        question = SUGGESTIONS[int(user_input) - 1]
        print(f"  → {question}")
    else:
        question = user_input

    print("  Thinking…", end=" ", flush=True)
    t0 = time.time()

    try:
        answer = ask(question)
        elapsed = time.time() - t0
        _, docs = retrieve_context(question)

        print(f"\r  Answer ({elapsed:.1f}s):\n")
        words = answer.split()
        line  = "  "
        for word in words:
            if len(line) + len(word) > 80:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

        print(f"\n  Sources:")
        for doc in docs[:2]:
            print(f"    · {doc[:90]}…")

    except Exception as e:
        print(f"\r  Error: {e}")
