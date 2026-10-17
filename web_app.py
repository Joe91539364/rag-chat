"""
班长工作台 —— 本地网页工具

把 rag / generate / agent / memory 的能力包成一个本地 Web 界面。

启动：
    python web_app.py
    然后浏览器打开 http://127.0.0.1:5000

页面功能：
    - 对话区：问群聊相关问题（RAG 检索 + LLM 生成）
    - 记一笔：用自然语言记待办/提醒（agent 工具调用）
    - 待办面板：查看待办、勾选完成
    - 记忆面板：查看/检索沉淀的长期记忆
    - 导入消息：把微信群聊记录粘贴进来，扩展知识库
"""

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from agent import load_store as load_agent_store, save_store as save_agent_store, run_agent
from generate import load_config, generate
from importers import auto_parse
from memory import load_store as load_memory_store, save_store as save_memory_store, recall_memories
from rag import RAGIndex, load_messages, DATA_FILE

app = Flask(__name__)

# 全局索引，启动时构建一次；导入新消息后重建
_index = None


def get_index():
    global _index
    if _index is None:
        messages = load_messages(DATA_FILE)
        _index = RAGIndex(messages, embedder_name="bge", retriever="numpy")
    return _index


def rebuild_index():
    global _index
    messages = load_messages(DATA_FILE)
    _index = RAGIndex(messages, embedder_name="bge", retriever="numpy")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """问答：检索 + 生成。"""
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    backend = data.get("backend") or None
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"ok": False, "error": "问题不能为空"}), 400

    config = load_config()
    backend_name = backend or config["default_backend"]

    idx = get_index()
    hits = idx.search(query, top_k=top_k)
    context = [{"speaker": m["speaker"], "text": m["text"], "score": round(s, 4)}
               for m, s in hits]
    answer = generate(query, hits, backend_name)

    return jsonify({
        "ok": True,
        "query": query,
        "answer": answer,
        "context": context,
        "backend": backend_name,
    })


@app.route("/api/agent", methods=["POST"])
def api_agent():
    """记一笔：自然语言 → 待办/提醒。"""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "内容不能为空"}), 400

    result = run_agent(text)
    return jsonify({"ok": True, "result": result, "store": load_agent_store()})


@app.route("/api/todos", methods=["GET", "POST"])
def api_todos():
    """待办：GET 查看，POST {title, done} 更新完成状态。"""
    store = load_agent_store()
    if request.method == "GET":
        return jsonify({"ok": True, "store": store})

    data = request.get_json(force=True)
    title = (data.get("title") or "").strip()
    done = bool(data.get("done", False))
    for t in store["todos"]:
        if t["title"] == title:
            t["done"] = done
            break
    save_agent_store(store)
    return jsonify({"ok": True, "store": store})


@app.route("/api/memories", methods=["GET"])
def api_memories():
    """记忆：GET ?q=关键词 检索长期记忆。"""
    store = load_memory_store()
    q = (request.args.get("q") or "").strip()
    hits = recall_memories(store, q)
    return jsonify({"ok": True, "memories": hits, "total": len(store["memories"])})


@app.route("/api/import", methods=["POST"])
def api_import():
    """导入群聊消息：自动识别多种格式（逐条/电脑版复制/合并转发/JSON/CSV）。"""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "内容不能为空"}), 400

    # 自动嗅探格式并解析
    parsed, fmt = auto_parse(text)
    if not parsed:
        return jsonify({"ok": False, "error": "没能识别出消息，请检查粘贴的格式"}), 400

    messages = load_messages(DATA_FILE)
    next_id = max((m["id"] for m in messages), default=0) + 1

    added = 0
    for p in parsed:
        speaker = p["speaker"]
        content = p["text"]
        if not content:
            continue
        # 时间戳附进 text 前面，便于检索时保留时间线索
        full_text = content
        if p.get("time"):
            full_text = f"[{p['time']}] {content}"
        messages.append({"id": next_id, "speaker": speaker, "text": full_text})
        next_id += 1
        added += 1

    # 写回数据文件
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    # 重建索引，让新消息立即可检索
    rebuild_index()

    return jsonify({
        "ok": True,
        "added": added,
        "total": len(messages),
        "format": fmt,
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    """状态：后端配置、消息数量、记忆数量。"""
    config = load_config()
    messages = load_messages(DATA_FILE)
    mem_store = load_memory_store()
    agent_store = load_agent_store()
    return jsonify({
        "ok": True,
        "default_backend": config["default_backend"],
        "backends": {
            name: {
                "name": b["name"],
                "configured": bool(b["api_key"].strip()),
                "free": bool(b.get("free")),
            }
            for name, b in config["backends"].items()
        },
        "message_count": len(messages),
        "memory_count": len(mem_store["memories"]),
        "todo_count": len([t for t in agent_store["todos"] if not t["done"]]),
    })


def main():
    parser = argparse.ArgumentParser(description="班长工作台本地网页工具")
    parser.add_argument("--port", type=int, default=5000, help="端口（默认 5000）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    args = parser.parse_args()

    print("=" * 50)
    print("  班长工作台已启动")
    print(f"  请用浏览器打开 http://{args.host}:{args.port}")
    print("  按 Ctrl+C 停止")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
