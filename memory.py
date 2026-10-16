"""
分层记忆：会话摘要 + 长期记忆沉淀

设计（记忆复利）：
- 短期  原始群聊消息（data/messages.json，已有）
- 中期  会话摘要：用 LLM 把一批消息压缩成结构化摘要（谁在何时说了什么关键事）
- 长期  长期记忆：从摘要中抽取「事实性信息」，沉淀为可检索的记忆条目，
        用一次就比上一次更懂这个班，实现复利

持久化：全部落本地 JSON（memory_store.json），零数据库依赖。

运行：
    python memory.py                 # 从全部消息生成摘要 + 沉淀长期记忆
    python memory.py --recall "体测"  # 检索长期记忆
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from generate import call_llm

STORE_FILE = Path(__file__).parent / "data" / "memory_store.json"

SUMMARY_PROMPT = (
    "你是班级群聊的整理助手。请把下面一批群聊消息整理成结构化摘要，"
    "用简洁的要点列出：时间线索、事项安排、已决定的结论、待办事项。"
    "只依据消息内容，不要编造。"
)

FACT_PROMPT = (
    "你是班级群聊的记忆抽取助手。请从下面这批群聊消息中抽取「长期有用的事实信息」，"
    "每条一行，格式：事实描述。只抽取确定的事实（日期、规则、数字、地点、流程等），"
    "忽略寒暄、闲聊、疑问句。每条事实要自包含、可独立理解，不要依赖上下文。"
    "如果没有值得记忆的事实，直接回复「无」。"
)


def load_store() -> dict:
    if STORE_FILE.exists():
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"summaries": [], "memories": []}


def save_store(store: dict):
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _call_llm(prompt: str, text: str) -> str:
    """调用默认后端 LLM；未配置 key 时返回空字符串（降级）。"""
    try:
        return call_llm(prompt, text).strip()
    except Exception as e:
        print(f"  [警告] LLM 调用失败：{e}")
        return ""


def _chunk_messages(messages: list[dict], size: int = 30) -> list[list[dict]]:
    """按条数把消息切成批次，避免一次塞给 LLM 过多内容。"""
    return [messages[i:i + size] for i in range(0, len(messages), size)]


def _fmt_messages(messages: list[dict]) -> str:
    return "\n".join(f"[{m['speaker']}] {m['text']}" for m in messages)


def build_summaries(messages: list[dict], batch_size: int = 30) -> list[str]:
    """把全部消息分批生成会话摘要。未配置 LLM 时降级为规则摘要。"""
    summaries = []
    for i, batch in enumerate(_chunk_messages(messages, batch_size), start=1):
        text = _fmt_messages(batch)
        summary = _call_llm(SUMMARY_PROMPT, text)
        if not summary:
            # 降级：按 speaker 统计发言次数，给一个粗糙的规则摘要
            cnt = Counter(m["speaker"] for m in batch)
            top = "、".join(f"{s}({c}条)" for s, c in cnt.most_common(5))
            summary = f"[规则摘要] 本批 {len(batch)} 条消息，主要发言：{top}"
        summaries.append(summary)
        print(f"  [摘要 {i}/{len(_chunk_messages(messages, batch_size))}] 完成")
    return summaries


def extract_memories(summaries: list[str]) -> list[str]:
    """从摘要中抽取长期记忆事实。未配置 LLM 时返回空。"""
    memories = []
    joined = "\n".join(summaries)
    result = _call_llm(FACT_PROMPT, joined)
    if not result or result == "无":
        return memories
    for line in result.splitlines():
        line = line.strip().lstrip("-• ").strip()
        if line and line != "无":
            memories.append(line)
    return memories


def recall_memories(store: dict, keyword: str) -> list[str]:
    """朴素关键词检索长期记忆（后续可换成向量检索）。"""
    kw = keyword.strip()
    if not kw:
        return store["memories"]
    return [m for m in store["memories"] if kw in m]


def main():
    parser = argparse.ArgumentParser(description="分层记忆")
    parser.add_argument("--recall", help="关键词检索长期记忆（不填则重建记忆）")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="摘要分批大小（默认30条/批）")
    args = parser.parse_args()

    store = load_store()

    if args.recall:
        hits = recall_memories(store, args.recall)
        print(f"\n【长期记忆】匹配 '{args.recall}' 共 {len(hits)} 条：")
        for m in hits:
            print(f"  - {m}")
        return

    from rag import load_messages, DATA_FILE
    messages = load_messages(DATA_FILE)
    print(f"加载 {len(messages)} 条消息，开始生成会话摘要...")
    summaries = build_summaries(messages, args.batch_size)
    print("\n抽取长期记忆...")
    memories = extract_memories(summaries)

    store["summaries"] = summaries
    store["memories"] = memories
    save_store(store)

    print(f"\n完成：{len(summaries)} 条摘要，{len(memories)} 条长期记忆")
    print(f"已保存到 {STORE_FILE}")
    if memories:
        print("\n【沉淀的长期记忆】")
        for m in memories:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
