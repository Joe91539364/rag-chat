"""
RAG 检索评测：计算 recall@k，量化检索效果

做法：
- 手工标注一批「问题 -> 正确答案的消息 id」
- 对每个问题做检索，看正确答案是否落在 top-k 结果里
- 统计 recall@1 / recall@3，用数字证明系统有效

运行：python eval.py [--embedder bge|bow] [--retriever numpy|chroma]
"""

import argparse
from rag import RAGIndex, load_messages, DATA_FILE

# 评测集：每个问题对应正确消息的 id（从扩充后的 messages.json 里挑）
# 设计原则：挑有「干扰项」的问题（同一主题有多条相关消息），
# 才能测出语义检索 vs 词袋的真实差距
QUERIES = [
    # 直接问题（关键词明确）
    {"q": "体测具体是几月几号", "answer_ids": [8]},
    {"q": "奖学金评定的成绩门槛是多少", "answer_ids": [6]},
    {"q": "青年大学习什么时候截止", "answer_ids": [2]},
    {"q": "班会在哪栋楼开", "answer_ids": [1]},
    {"q": "毛概读书笔记要写多少字", "answer_ids": [17]},
    {"q": "英语听力用哪个软件", "answer_ids": [15]},
    {"q": "离散数学考到第几章", "answer_ids": [22]},
    {"q": "计算机等级考试报名费多少", "answer_ids": [30]},
    {"q": "春游每人要交多少钱", "answer_ids": [33]},
    {"q": "宿舍晚上几点关门", "answer_ids": [54]},
    # 语义改写 + 有干扰项（考察真正语义理解）
    {"q": "跑步跳远这种体能考核安排在什么日子", "answer_ids": [8]},
    {"q": "评奖学金对绩点有什么要求", "answer_ids": [6]},
    {"q": "入党积极分子要完成的线上学习什么时候到期", "answer_ids": [2]},
    {"q": "集体开会讨论评优是几点钟", "answer_ids": [1]},
    {"q": "宿舍卫生检查重点抓哪些东西", "answer_ids": [4]},
    {"q": "英语听说作业用哪个APP做", "answer_ids": [15]},
    {"q": "四六级什么时候报名结束", "answer_ids": [26]},
    {"q": "困难生认定要准备什么材料", "answer_ids": [46]},
    {"q": "励志奖学金对成绩的要求是什么", "answer_ids": [50]},
    {"q": "图书馆借书能借多长时间", "answer_ids": [110]},
]


def evaluate(embedder_name: str, top_k: int = 3, retriever: str = "numpy"):
    messages = load_messages(DATA_FILE)
    index = RAGIndex(messages, embedder_name=embedder_name, retriever=retriever)

    id_to_idx = {m["id"]: i for i, m in enumerate(messages)}

    print(f"\n=== 评测：embedder={embedder_name}, retriever={retriever}, top_k={top_k} ===\n")
    hits = {k: 0 for k in range(1, top_k + 1)}
    total = len(QUERIES)

    for item in QUERIES:
        q = item["q"]
        results = index.search(q, top_k=top_k)
        # 取出命中消息的 id 列表
        result_ids = [m["id"] for m, _ in results]
        correct = item["answer_ids"]

        # 找到正确答案最早出现的位置
        rank = None
        for i, rid in enumerate(result_ids, start=1):
            if rid in correct:
                rank = i
                break

        status = "✓" if rank else "✗"
        print(f"  [{status}] {q}")
        if rank:
            for k in range(rank, top_k + 1):
                hits[k] += 1
        else:
            print(f"       未命中，返回: {result_ids}")

    print("\n" + "=" * 50)
    for k in range(1, top_k + 1):
        recall = hits[k] / total
        print(f"  recall@{k} = {recall:.3f}  ({hits[k]}/{total})")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["bge", "bow"], default="bge")
    parser.add_argument("--retriever", choices=["numpy", "chroma"], default="numpy")
    args = parser.parse_args()
    evaluate(args.embedder, retriever=args.retriever)
