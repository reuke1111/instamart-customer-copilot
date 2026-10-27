"""Deterministic whitespace chunking, TF-IDF retrieval and grounded mock QA."""

from collections import Counter
import csv
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

from scripts.common import ROOT, save_json

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
POLICY_FILES = {f"{name}_policy.txt" for name in ("refund", "delivery", "escalation", "payment", "privacy")}
FALLBACK = "I could not find that information in the policy documents."
SYSTEM_PROMPT = (
    "POLICY ASSISTANT. Answer only from the supplied context. "
    "If the context does not contain the answer, say you could not find the information. "
    "Do not use general knowledge or invent policy rules."
)


def chunk_document(source, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if type(chunk_size) is not int or type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    words = text.split()
    chunks = []
    for start in range(0, len(words), chunk_size - overlap):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "chunk_id": f"{source}:{len(chunks):03d}", "source": source,
            "start_word": start, "end_word": end, "text": " ".join(words[start:end]),
        })
        if end == len(words):
            break
    return chunks


class PolicyRetriever:
    def __init__(self, root=ROOT, overlap=CHUNK_OVERLAP):
        self.root = Path(root)
        self.overlap = overlap
        self.paths = sorted((self.root / "data/policies").glob("*.txt"), key=lambda path: path.name)
        if {path.name for path in self.paths} != POLICY_FILES:
            raise ValueError("The complete five-document policy corpus is required")
        if any(not path.read_text(encoding="utf-8").strip() for path in self.paths):
            raise ValueError("Policy documents must not be empty")
        self.chunks = [
            chunk for path in self.paths
            for chunk in chunk_document(path.name, path.read_text(encoding="utf-8"), overlap=overlap)
        ]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in self.chunks])

    def retrieve(self, query, k=3):
        if not isinstance(query, str) or not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("Query must be text and k must be a positive integer")
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel().tolist()
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], self.chunks[index]["chunk_id"]))
        return [
            {"source": self.chunks[index]["source"], "score": scores[index], "text": self.chunks[index]["text"]}
            for index in ranked[:k]
        ]


@lru_cache(maxsize=4)
def default_retriever(root=ROOT):
    return PolicyRetriever(root)


def retrieve(query, k=3):
    return default_retriever().retrieve(query, k)


def grounded_text(text, results, chunks):
    """Trace extracted words to original positions; remove only proven overlap.

    Never join different sources or jump across a gap in the source document.
    This validates the model's extraction without selecting a replacement answer
    from reference facts or unretrieved material.
    """
    if not isinstance(text, str) or not text.strip() or text.startswith("I could not find"):
        return FALLBACK
    tokens, positions = [], []
    for result in results:
        chunk = next((item for item in chunks if item["source"] == result["source"] and item["text"] == result["text"]), None)
        if chunk is None:
            return FALLBACK
        words = result["text"].split()
        tokens.extend(words)
        positions.extend((chunk["source"], chunk["start_word"] + i) for i in range(len(words)))
    wanted = text.split()
    for start in range(len(tokens) - len(wanted) + 1):
        if tokens[start:start + len(wanted)] != wanted:
            continue
        span = positions[start:start + len(wanted)]
        source, first = span[0]
        last, accepted = first - 1, []
        for word, (current_source, offset) in zip(wanted, span):
            if current_source != source or offset < first or offset > last + 1:
                break
            if offset == last + 1:
                accepted.append(word)
                last = offset
            # Already covered source positions are the duplicate overlap.
        else:
            return " ".join(accepted)
    return FALLBACK


def answer(question, *, retriever=None, client=mock_llm):
    retriever = default_retriever() if retriever is None else retriever
    results = retriever.retrieve(question, k=3)
    if len(results) != 3:
        raise ValueError("Policy answering requires three context chunks")
    context = "\n".join(f"[{row['source']}]\n{row['text']}" for row in results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
    ]
    response = client.chat(messages)
    text = response.choices[0].message.content
    grounded = grounded_text(text, results, retriever.chunks)
    return {
        "answer": grounded, "raw_answer": text, "sources": [row["source"] for row in results],
        "retrieved": results, "tokens_used": int(response.usage.total_tokens),
    }


def evaluate(root=ROOT, overlap=CHUNK_OVERLAP):
    root = Path(root)
    retriever = PolicyRetriever(root, overlap=overlap)
    with (root / "data/raw/policy_eval.csv").open(encoding="utf-8", newline="") as stream:
        questions = list(csv.DictReader(stream))
    mock_llm.reset()
    rows = []
    for item in questions:
        result = answer(item["question"], retriever=retriever)
        answerable = item["expected_doc"] != "NONE"
        rows.append({
            **item, **result, "answerable": answerable,
            "source_correct": answerable and item["expected_doc"] in result["sources"],
            "fact_correct": answerable and item["expected_fact"] in result["answer"],
            "raw_fact_correct": answerable and item["expected_fact"] in result["raw_answer"],
            "unsupported_fallback": not answerable and result["answer"].startswith("I could not find"),
        })
    return {
        "overlap": overlap, "chunk_size": CHUNK_SIZE,
        "chunk_counts": dict(sorted(Counter(chunk["source"] for chunk in retriever.chunks).items())),
        "chunks": len(retriever.chunks), "questions": len(rows),
        "answerable": sum(row["answerable"] for row in rows),
        "source_correct": sum(row["source_correct"] for row in rows),
        "fact_correct": sum(row["fact_correct"] for row in rows),
        "raw_fact_correct": sum(row["raw_fact_correct"] for row in rows),
        "unanswerable": sum(not row["answerable"] for row in rows),
        "unsupported_fallbacks": sum(row["unsupported_fallback"] for row in rows),
        "total_tokens": sum(row["tokens_used"] for row in rows), "details": rows,
    }


def run(root=ROOT):
    root = Path(root)
    folder = root / "outputs/part_d"
    reports = {str(overlap): evaluate(root, overlap) for overlap in [10, 0]}
    for overlap, report in reports.items():
        save_json(folder / f"rag_overlap_{overlap}.json", report)
        save_json(folder / f"chunks_overlap_{overlap}.json", PolicyRetriever(root, int(overlap)).chunks)
    save_json(folder / "retrieval_example.json", PolicyRetriever(root).retrieve("refund policy late delivery"))
    return reports


if __name__ == "__main__":
    for overlap, report in run().items():
        print({key: value for key, value in report.items() if key != "details"})
        for row in report["details"]:
            if row["answerable"] and not row["fact_correct"]:
                print("Fact not found:", row["question"], "| expected:", row["expected_fact"], "| answer:", row["answer"])
