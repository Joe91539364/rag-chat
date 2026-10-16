"""
下载 bge-m3 嵌入模型到本地 models/bge-m3/ 目录

背景：Windows + 国内网络下，huggingface_hub 下载 bge-m3 会踩三个坑：
1. 本地代理(127.0.0.1:51464)让 huggingface.co 返回 502 -> 走 hf-mirror.com
2. xet 加速传输在镜像下 401 -> 禁用 xet
3. hf-mirror 的 resolve 接口返回 302，huggingface_hub 跟随失败写入空文件
   -> 改用 curl -L 直接下载（curl 能正确跟随重定向）

运行：python download_model.py
"""

import os
import subprocess
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models" / "bge-m3"
BASE = "https://hf-mirror.com/BAAI/bge-m3/resolve/main"

# bge-m3 稠密检索必需的文件
FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "pytorch_model.bin",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
]


def download():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "1_Pooling").mkdir(exist_ok=True)

    for f in FILES:
        dest = MODEL_DIR / f
        url = f"{BASE}/{f}"
        print(f"下载 {f} ...")
        # 关键：-L 跟随重定向，否则会下载到 302 重定向文本而不是真实文件
        result = subprocess.run(
            ["curl", "-sL", "--retry", "3", "-o", str(dest), url],
            capture_output=True,
        )
        if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
            print(f"  ✗ 下载失败: {f}")
            return False
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  ✓ {f} ({size_mb:.1f} MB)")

    print(f"\n模型已下载到: {MODEL_DIR}")
    return True


if __name__ == "__main__":
    download()
