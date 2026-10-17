"""
微信群聊记录解析器 —— 自动识别多种粘贴格式

支持的格式（自动嗅探，无需手动选）：
1. 逐条快照：每行「昵称：内容」或「昵称: 内容」
2. 电脑版复制：三条一组
     2024年1月1日 10:30:00
     张三
     在吗
   （不同版本：时间戳可能无秒；昵称+内容可能并一行）
3. 合并转发复制：块状
     张三
     10:30
     在吗
4. 第三方工具（留痕 MemoTrace / WeChatMsg）导出：
   - TXT 时间戳行式
   - CSV（表头含 StrContent / SenderName / CreateTime 等）
   - JSON（sender/nick/msg 等字段）

用法：
    from importers import parse_text, parse_json, parse_csv
    messages = parse_text("粘贴的内容")
    messages = parse_json(json_str_or_file)
    messages = parse_csv(csv_str)
"""

import csv
import io
import json
import re

# ---- 时间戳正则 ----
# 2024-01-01 10:30:00 / 2024年1月1日 10:30 / 2024/1/1 10:30 / 10:30 / 01-01 10:30
RE_DATETIME = re.compile(
    r"^\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}[-/月]\d{1,2}日?)"
    r"\s+(\d{1,2}:\d{2}(:\d{2})?)\s*$"
)
RE_TIME_ONLY = re.compile(r"^\s*(\d{1,2}:\d{2}(:\d{2})?)\s*$")


def _clean(s: str) -> str:
    return s.replace("\u200b", "").replace("\u00a0", " ").strip()


def _split_speaker(line: str):
    """拆「昵称：内容」或「昵称: 内容」，返回 (speaker, content) 或 None。

    排除「时间戳」这类含冒号但并非「昵称：内容」的行。
    """
    if _is_datetime_line(line) or _is_time_line(line):
        return None
    for sep in ("：", ":"):
        if sep in line:
            speaker, _, content = line.partition(sep)
            speaker = _clean(speaker)
            content = _clean(content)
            if speaker and content:
                return speaker, content
    return None


def _is_datetime_line(line: str) -> bool:
    return bool(RE_DATETIME.match(line))


def _is_time_line(line: str) -> bool:
    return bool(RE_TIME_ONLY.match(line))


def parse_text(text: str) -> list[dict]:
    """解析粘贴文本，返回 [{speaker, text}]（时间戳若存在则附 time）。"""
    lines = [_clean(l) for l in text.splitlines()]
    lines = [l for l in lines if l]  # 去空行

    if not lines:
        return []

    # 模式 A：逐条「昵称：内容」，即所有行都能拆出 speaker
    split_all = [_split_speaker(l) for l in lines]
    if all(s is not None for s in split_all):
        return [{"speaker": s, "text": c} for s, c in split_all]

    # 模式 B：时间戳行式（电脑版复制 / 第三方 TXT）
    # 每条消息：可能的时间戳行 + 昵称行 + 内容（可多行）
    if _is_datetime_line(lines[0]) or _is_time_line(lines[0]):
        return _parse_timestamp_style(lines)

    # 模式 C：块状（合并转发）—— 昵称行 + 纯时间行 + 内容行
    return _parse_block_style(lines)


def _parse_timestamp_style(lines: list[str]) -> list[dict]:
    """时间戳行开头的格式：时间行 + 昵称行 + 内容（多行），直到下个时间行。"""
    messages = []
    i = 0
    n = len(lines)
    while i < n:
        # 跳过时间戳行
        if _is_datetime_line(lines[i]) or _is_time_line(lines[i]):
            ts = lines[i]
            i += 1
        else:
            ts = None
        if i >= n:
            break
        # 下一行：昵称 或「昵称：内容」
        pair = _split_speaker(lines[i])
        if pair:
            speaker, content = pair
            i += 1
        else:
            speaker = lines[i]
            i += 1
            # 收集内容行，直到下个时间戳行
            content_parts = []
            while i < n and not _is_datetime_line(lines[i]) and not _is_time_line(lines[i]):
                content_parts.append(lines[i])
                i += 1
            content = " ".join(content_parts)
        if speaker:
            msg = {"speaker": speaker, "text": content}
            if ts:
                msg["time"] = ts
            messages.append(msg)
    return messages


def _parse_block_style(lines: list[str]) -> list[dict]:
    """合并转发块状：昵称行 + 纯时间行 + 内容行。"""
    messages = []
    i = 0
    n = len(lines)
    while i < n:
        speaker = lines[i]
        i += 1
        ts = None
        # 可能的纯时间行
        if i < n and _is_time_line(lines[i]):
            ts = lines[i]
            i += 1
        # 内容行，直到下个「昵称+（时间）」块
        content_parts = []
        while i < n:
            # 判断下一行是否新块的昵称行（后面跟时间行或能拆出 speaker）
            if _is_time_line(lines[i]):
                # 纯时间行属于下一块的开头？实际合并转发里时间行前必有昵称行
                break
            if i + 1 < n and _is_time_line(lines[i + 1]) and not _is_time_line(lines[i]):
                # lines[i] 是下一块的昵称行
                break
            content_parts.append(lines[i])
            i += 1
        content = " ".join(content_parts)
        if speaker:
            msg = {"speaker": speaker, "text": content}
            if ts:
                msg["time"] = ts
            messages.append(msg)
    return messages


def parse_json(text: str) -> list[dict]:
    """解析第三方工具（留痕/WeChatMsg）导出的 JSON。"""
    data = json.loads(text)

    # 展平：可能直接是 list，或含 messages/items/data 键
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("messages") or data.get("items") or data.get("data") or []
        if isinstance(items, dict):
            items = list(items.values())
    else:
        return []

    messages = []
    for it in items:
        if not isinstance(it, dict):
            continue
        speaker = (
            it.get("sender") or it.get("nick") or it.get("talker")
            or it.get("from") or it.get("SenderName") or it.get("senderName")
            or it.get("nickName") or "未知"
        )
        content = (
            it.get("msg") or it.get("content") or it.get("text")
            or it.get("message") or it.get("StrContent") or ""
        )
        t = (
            it.get("time") or it.get("CreateTime") or it.get("createTime")
            or it.get("datetime") or it.get("CreateTimeStr")
        )
        if not content:
            continue
        msg = {"speaker": str(speaker), "text": str(content)}
        if t:
            msg["time"] = str(t)
        messages.append(msg)
    return messages


def parse_csv(text: str) -> list[dict]:
    """解析第三方工具导出的 CSV。"""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return []

    def find_key(candidates):
        for c in candidates:
            for fn in reader.fieldnames:
                if fn and c.lower() in fn.lower():
                    return fn
        return None

    key_speaker = find_key(["sender", "nick", "talker", "from", "name", "strtalker"])
    key_content = find_key(["content", "strcontent", "msg", "text", "message"])
    key_time = find_key(["createtime", "time", "datetime", "date"])

    messages = []
    for row in reader:
        content = row.get(key_content, "") if key_content else ""
        if not content:
            continue
        speaker = row.get(key_speaker, "未知") if key_speaker else "未知"
        msg = {"speaker": speaker, "text": content}
        if key_time and row.get(key_time):
            msg["time"] = row[key_time]
        messages.append(msg)
    return messages


def auto_parse(text: str) -> tuple[list[dict], str]:
    """自动嗅探格式，返回 (messages, 识别出的格式名)。"""
    t = text.strip()
    if not t:
        return [], "empty"
    if t.lstrip().startswith(("[", "{")):
        try:
            return parse_json(t), "json"
        except Exception:
            pass
    # 看是否像 CSV（首行含表头关键字）
    first = t.splitlines()[0] if t.splitlines() else ""
    if "," in first and any(k in first.lower() for k in ("content", "msg", "sender", "strcontent", "createtime")):
        try:
            msgs = parse_csv(t)
            if msgs:
                return msgs, "csv"
        except Exception:
            pass
    return parse_text(t), "text"
