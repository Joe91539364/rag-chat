"""
LLM 生成层：把检索到的上下文交给大模型生成最终答案

支持多个后端，可通过命令行或 config.json 随时切换：
    --backend zhipu       智谱 GLM-4-Flash（免费，推荐默认）
    --backend siliconflow 硅基流动（免费额度）
    --backend deepseek    DeepSeek 官方
    （未配置任何 key 时自动降级为「检索拼接」占位，不报错）

配置方式：编辑 config.json 填入对应平台的 api_key
    - 智谱：https://open.bigmodel.cn 注册，手机号即可，GLM-4-Flash 永久免费
    - 硅基流动：https://cloud.siliconflow.cn 注册，送免费额度

用法：
    python generate.py "体测是几号"                    # 用默认后端
    python generate.py "体测是几号" --backend zhipu    # 指定后端
    python generate.py --list                          # 列出所有后端及状态
"""

import argparse
import json
import os
import sys
from pathlib import Path

from rag import RAGIndex, load_messages, DATA_FILE

CONFIG_FILE = Path(__file__).parent / "config.json"
CONFIG_TEMPLATE = Path(__file__).parent / "config.example.json"

SYSTEM_PROMPT = (
    "你是一个班级群聊助手。请严格依据下面提供的群聊消息回答用户问题，"
    "不要编造消息里没有的信息。如果消息里没有答案，请明确说'消息里没有相关信息'。"
)


def load_config() -> dict:
    # 首次运行：如果 config.json 不存在，从模板复制
    if not CONFIG_FILE.exists() and CONFIG_TEMPLATE.exists():
        import shutil
        shutil.copy(CONFIG_TEMPLATE, CONFIG_FILE)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(hits):
    lines = [f"[{m['speaker']}] {m['text']}" for m, _ in hits]
    return "\n".join(lines)


def call_llm(system: str, user: str, backend_name: str = None) -> str:
    """通用 LLM 调用：自定义 system/user 提示词，供 memory.py / agent.py 复用。

    未配置 key 时返回空字符串，调用方自行降级。
    """
    import requests

    config = load_config()
    name = backend_name or config["default_backend"]
    backend = config["backends"].get(name)
    if not backend:
        raise ValueError(f"未知后端 '{name}'")
    if not backend["api_key"].strip():
        return ""

    payload = {
        "model": backend["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }
    resp = requests.post(
        f"{backend['base_url'].rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {backend['api_key'].strip()}"},
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_openai_compatible(backend: dict, query: str, context: str) -> str:
    """调用 OpenAI 兼容接口（智谱/硅基流动/DeepSeek 都兼容此格式）。"""
    import requests

    api_key = backend["api_key"].strip()
    base_url = backend["base_url"].rstrip("/")
    model = backend["model"]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"群聊消息：\n{context}\n\n问题：{query}"},
        ],
        "temperature": 0.3,
    }
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def fallback_concat(query: str, hits) -> str:
    """降级方案：未配置任何后端时，直接拼接检索结果。"""
    context = build_context(hits)
    return (
        f"（未配置大模型 API，以下为检索结果拼接）\n\n{context}\n\n"
        f"编辑 config.json 填入智谱/硅基流动的免费 api_key 后，"
        f"将由大模型生成自然语言回答。"
    )


def generate(query: str, hits, backend_name: str) -> str:
    config = load_config()
    backend = config["backends"].get(backend_name)

    if not backend:
        print(f"未知后端 '{backend_name}'，可用：{', '.join(config['backends'])}")
        sys.exit(1)

    api_key = backend["api_key"].strip()
    if not api_key:
        return fallback_concat(query, hits)

    context = build_context(hits)
    try:
        return call_openai_compatible(backend, query, context)
    except Exception as e:
        return f"（调用 {backend['name']} 失败：{e}）\n\n降级为检索拼接：\n{context}"


def list_backends():
    config = load_config()
    print("可用后端：")
    for name, b in config["backends"].items():
        has_key = "✓ 已配置" if b["api_key"].strip() else "✗ 未配置"
        free = "免费" if b.get("free") else "付费"
        default = "（默认）" if name == config["default_backend"] else ""
        print(f"  {name:<12} {b['name']} [{free}] {has_key} {default}")
    print("\n配置方法：编辑 config.json 填入 api_key")
    print("  智谱免费：https://open.bigmodel.cn（GLM-4-Flash 永久免费）")
    print("  硅基流动：https://cloud.siliconflow.cn（送免费额度）")


def main():
    parser = argparse.ArgumentParser(description="群聊 RAG 生成层")
    parser.add_argument("query", nargs="*", help="要问的问题")
    parser.add_argument("--backend", help="后端名称：zhipu / siliconflow / deepseek")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回条数（默认5）")
    parser.add_argument("--retriever", choices=["numpy", "chroma"], default="numpy",
                        help="检索后端：numpy=手写余弦，chroma=向量库")
    parser.add_argument("--list", action="store_true", help="列出所有后端")
    args = parser.parse_args()

    if args.list:
        list_backends()
        return

    query = " ".join(args.query) if args.query else "体测是几号"
    config = load_config()
    backend_name = args.backend or config["default_backend"]

    messages = load_messages(DATA_FILE)
    index = RAGIndex(messages, embedder_name="bge", retriever=args.retriever)
    hits = index.search(query, top_k=args.top_k)

    print(f"【问题】{query}")
    print(f"【后端】{backend_name} ({config['backends'][backend_name]['name']})\n")
    print("【检索到的相关消息】")
    for m, score in hits:
        print(f"  ({score:.3f}) [{m['speaker']}] {m['text']}")
    print("\n" + "=" * 50 + "\n")
    print("【回答】")
    print(generate(query, hits, backend_name))


if __name__ == "__main__":
    main()
