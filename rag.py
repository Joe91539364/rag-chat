"""
群聊 RAG 助手 —— 消息语义检索 + 问答

完整链路：加载数据 -> 分块 -> 向量化 -> 建索引 -> 语义检索 -> 生成答案

设计说明：
- 检索层做了抽象，支持两种实现，用 --retriever 切换：
    * numpy   手写余弦相似度（零依赖，便于理解原理，默认）
    * chroma  Chroma 向量库（专业引擎，数据量大时更高效，需 chromadb）
- 嵌入层做了抽象，支持两种实现，用 --embedder 切换：
    * bow   词袋 + TF-IDF 基线（零外部依赖，秒出）
    * bge   bge-m3 句子嵌入（本地 GPU 跑，语义更强，需 sentence-transformers）
- 生成层留了 LLM API 接口，默认用检索拼接占位

运行：
    python rag.py "你的问题"                          # 默认 bge 嵌入 + numpy 检索
    python rag.py "你的问题" --embedder bow           # 词袋基线
    python rag.py "你的问题" --retriever chroma       # Chroma 向量库
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "messages.json"


def load_messages(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    # 中文按单字 + 英文数字按词切分，做最简分词
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    return tokens


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class BagOfWordsEmbedder:
    """词袋嵌入基线：用 TF-IDF 权重构造向量。"""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.dim = len(vocab)
        self.idf = {w: 1.0 for w in vocab}

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        tf = Counter(tokens)
        vec = [0.0] * self.dim
        if not tokens:
            return vec
        for w, c in tf.items():
            if w in self.word2idx:
                idx = self.word2idx[w]
                tfidf = (1 + math.log(c)) * self.idf.get(w, 1.0)
                vec[idx] = tfidf
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class BGEEmbedder:
    """bge-m3 句子嵌入：中文语义更强，本地 GPU 跑。"""

    # 模型本地路径（由 download_model.py 下载到此处）
    MODEL_DIR = Path(__file__).parent / "models" / "bge-m3"

    def __init__(self):
        import os
        # 国内网络：禁用本地代理 + 走 hf-mirror 镜像（本地无模型时兜底下载）
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(key, None)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from sentence_transformers import SentenceTransformer
        # 优先从本地目录加载，避免网络/符号链接问题
        model_path = str(self.MODEL_DIR) if self.MODEL_DIR.exists() else "BAAI/bge-m3"
        self.model = SentenceTransformer(model_path)
        # 有 GPU 用 GPU，否则 CPU
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()


def build_vocab(messages: list[dict]) -> list[str]:
    counter = Counter()
    for m in messages:
        counter.update(tokenize(m["text"]))
    vocab = [w for w, c in counter.items() if c >= 1]
    return vocab


def make_embedder(name: str, messages: list[dict]):
    if name == "bge":
        return BGEEmbedder()
    vocab = build_vocab(messages)
    return BagOfWordsEmbedder(vocab)


class NumpyIndex:
    """手写余弦检索基线：逐条算相似度，便于理解原理，适合小数据量。"""

    def __init__(self, messages: list[dict], embedder):
        self.messages = messages
        self.embedder = embedder
        self.vectors = embedder.embed_batch([m["text"] for m in messages])
        print(f"[numpy 索引完成] 共 {len(messages)} 条消息，向量维度 {len(self.vectors[0])}")

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        qvec = self.embedder.embed(query)
        scored = [(self.messages[i], cosine(qvec, self.vectors[i]))
                  for i in range(len(self.messages))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class ChromaIndex:
    """Chroma 向量库检索：专业引擎，内部建索引，数据量大时高效。"""

    def __init__(self, messages: list[dict], embedder):
        import chromadb
        self.messages = messages
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=str(Path(__file__).parent / "chroma_db")
        )
        self.collection = self.client.get_or_create_collection(
            name="chat_messages",
            metadata={"hnsw:space": "cosine"},
        )
        # 生成唯一 id（用消息自带 id，若无则用索引）
        ids = [str(m.get("id", i)) for i, m in enumerate(messages)]
        texts = [m["text"] for m in messages]
        vectors = embedder.embed_batch(texts)
        # id 一并存入 metadata，检索时才能还原出完整消息（含 id）
        metadatas = [{"speaker": m["speaker"], "id": m.get("id", i)}
                     for i, m in enumerate(messages)]
        # 写入策略：空则新增；已有数据但维度与当前嵌入不符（换了嵌入模型）则重建
        new_dim = len(vectors[0])
        existing = self.collection.count()
        if existing == 0:
            self.collection.add(ids=ids, embeddings=vectors,
                                documents=texts, metadatas=metadatas)
        else:
            try:
                old_dim = self.collection.peek(1)
                old_dim = len(old_dim["embeddings"][0])
            except Exception:
                old_dim = None
            if old_dim != new_dim:
                self.client.delete_collection("chat_messages")
                self.collection = self.client.get_or_create_collection(
                    name="chat_messages",
                    metadata={"hnsw:space": "cosine"},
                )
                self.collection.add(ids=ids, embeddings=vectors,
                                    documents=texts, metadatas=metadatas)
        print(f"[chroma 索引完成] 共 {len(messages)} 条消息，"
              f"向量维度 {len(vectors[0])}")

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        qvec = self.embedder.embed(query)
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma 的 distance 与余弦相似度相反（越小越相似），转回相似度
        hits = []
        for doc, meta, dist in zip(res["documents"][0],
                                   res["metadatas"][0],
                                   res["distances"][0]):
            msg = {"id": meta["id"], "text": doc, "speaker": meta["speaker"]}
            hits.append((msg, 1.0 - dist))
        return hits


def make_index(messages: list[dict], retriever: str, embedder):
    if retriever == "chroma":
        return ChromaIndex(messages, embedder)
    return NumpyIndex(messages, embedder)


class RAGIndex:
    """统一索引入口：对外只暴露 embedder_name + retriever 两个开关。

    eval.py / generate.py 都用这个类，切换检索后端（numpy / chroma）
    和嵌入方式（bge / bow）都不影响调用方。
    """

    def __init__(self, messages: list[dict], embedder_name: str = "bge",
                 retriever: str = "numpy"):
        self.messages = messages
        self.embedder = make_embedder(embedder_name, messages)
        self.index = make_index(messages, retriever, self.embedder)

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        return self.index.search(query, top_k=top_k)


def generate_answer(query: str, hits: list[tuple[dict, float]]) -> str:
    """生成层占位：把检索到的上下文拼接。接入 LLM API 时在此处替换。"""
    if not hits or hits[0][1] <= 0.0:
        return "没有找到相关内容，试试换个问法。"
    context = "\n".join(f"[{m['speaker']}] {m['text']}" for m, _ in hits)
    return (
        f"问题：{query}\n\n"
        f"根据检索到的相关消息：\n{context}\n\n"
        f"(这里是占位回答，接入 LLM API 后由模型基于以上上下文生成答案)"
    )


def main():
    parser = argparse.ArgumentParser(description="群聊 RAG 助手")
    parser.add_argument("query", nargs="*", help="要检索的问题")
    parser.add_argument("--embedder", choices=["bge", "bow"], default="bge",
                        help="嵌入模型：bge=bge-m3 句子嵌入，bow=词袋基线")
    parser.add_argument("--retriever", choices=["numpy", "chroma"], default="numpy",
                        help="检索后端：numpy=手写余弦，chroma=向量库")
    args = parser.parse_args()

    query = " ".join(args.query) if args.query else "班会什么时候开"
    messages = load_messages(DATA_FILE)
    index = RAGIndex(messages, embedder_name=args.embedder,
                     retriever=args.retriever)
    hits = index.search(query, top_k=3)

    print(f"\n【问题】{query}\n")
    print("【检索结果】(按相似度排序)")
    for m, score in hits:
        bar = "#" * int(score * 20)
        print(f"  {score:.3f} {bar:<20} [{m['speaker']}] {m['text']}")
    print("\n" + "=" * 50 + "\n")
    print(generate_answer(query, hits))


if __name__ == "__main__":
    main()
