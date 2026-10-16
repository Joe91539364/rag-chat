"""
Agent 化：工具调用（待办 / 提醒 / 日程）

思路：
- 用 LLM 判断用户意图，输出结构化「工具调用」JSON
- 本地执行对应工具，持久化到 data/agent_store.json
- 支持三种工具：
    * add_todo     添加待办
    * set_reminder 设置提醒
    * query_calendar 查询日程（待办 + 提醒汇总）

运行：
    python agent.py "提醒我周五下午3点开班会"
    python agent.py "帮我记一下，明天要交高数作业"
    python agent.py "我最近有什么待办？"
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from generate import call_llm

STORE_FILE = Path(__file__).parent / "data" / "agent_store.json"

TOOL_PROMPT = (
    "你是一个班级助手的意图识别器。根据用户输入，判断要调用哪个工具，"
    "并抽取参数，只输出 JSON，不要输出任何多余文字。\n"
    "可用工具及参数：\n"
    "1. add_todo：添加一条待办。参数 {title: 待办内容}\n"
    "2. set_reminder：设置提醒。参数 {title: 提醒内容, time: 时间描述}\n"
    "3. query_calendar：查询日程。参数 {}（空对象）\n"
    "如果用户是陈述待办/任务（如「帮我记一下XX」「明天要XX」），用 add_todo；\n"
    "如果用户强调某个时间点要做什么（如「提醒我X点XX」「周五XX」），用 set_reminder；\n"
    "如果用户是询问已有安排（如「我有什么待办」「最近有什么日程」），用 query_calendar。\n"
    "输出格式示例：{\"tool\": \"add_todo\", \"args\": {\"title\": \"交高数作业\"}}\n"
    "只输出这一行 JSON。"
)


def load_store() -> dict:
    if STORE_FILE.exists():
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"todos": [], "reminders": []}


def save_store(store: dict):
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---- 工具实现 ----

def add_todo(store: dict, title: str) -> str:
    item = {"title": title, "done": False, "created": _now()}
    store["todos"].append(item)
    return f"已添加待办：{title}"


def set_reminder(store: dict, title: str, time: str) -> str:
    item = {"title": title, "time": time, "created": _now()}
    store["reminders"].append(item)
    return f"已设置提醒：{title}（{time}）"


def query_calendar(store: dict) -> str:
    lines = []
    todos = [t for t in store["todos"] if not t["done"]]
    lines.append(f"待办（{len(todos)} 项）：")
    for t in todos:
        lines.append(f"  - {t['title']}")
    if not todos:
        lines.append("  （无）")
    lines.append(f"提醒（{len(store['reminders'])} 项）：")
    for r in store["reminders"]:
        lines.append(f"  - {r['title']}（{r['time']}）")
    if not store["reminders"]:
        lines.append("  （无）")
    return "\n".join(lines)


TOOLS = {
    "add_todo": lambda store, **a: add_todo(store, a.get("title", "")),
    "set_reminder": lambda store, **a: set_reminder(
        store, a.get("title", ""), a.get("time", "")),
    "query_calendar": lambda store, **a: query_calendar(store),
}


def parse_intent(text: str) -> dict:
    """用 LLM 判断意图；未配置 key 时用关键词规则降级。"""
    result = call_llm(TOOL_PROMPT, text).strip()
    if result:
        try:
            # 容错：截取第一个 { 到最后一个 }
            start = result.find("{")
            end = result.rfind("}")
            if start != -1 and end != -1:
                return json.loads(result[start:end + 1])
        except json.JSONDecodeError:
            pass
    return rule_fallback(text)


def rule_fallback(text: str) -> dict:
    """无 LLM 时的关键词规则降级。"""
    if any(k in text for k in ("待办", "日程", "有什么", "最近")):
        return {"tool": "query_calendar", "args": {}}
    if any(k in text for k in ("提醒", "几点", "记得")):
        return {"tool": "set_reminder",
                "args": {"title": text, "time": "未指定"}}
    return {"tool": "add_todo", "args": {"title": text}}


def run_agent(text: str) -> str:
    store = load_store()
    intent = parse_intent(text)
    tool = intent.get("tool")
    args = intent.get("args", {})

    if tool not in TOOLS:
        return f"无法识别意图：{text}"

    result = TOOLS[tool](store, **args)
    save_store(store)
    return result


def main():
    parser = argparse.ArgumentParser(description="班级 Agent（工具调用）")
    parser.add_argument("text", nargs="*", help="要执行的操作/问题")
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    if not text:
        print("请提供一句话，例如：python agent.py \"提醒我周五开班会\"")
        return

    print(f"【输入】{text}\n")
    print(run_agent(text))


if __name__ == "__main__":
    main()
