#!/usr/bin/env python3
"""xhs_interview_cleaner.py

洗小红书面试笔记 desc → 结构化 Q&A JSON（兼容 openai>=1.0.0 新 SDK）
--------------------------------------------------------------------------
$ export OPENAI_API_KEY="sk-..."
$ python xhs_interview_cleaner.py raw_notes.json cleaned_qas.json

依赖：
  pip install openai>=1.0 numpy scikit-learn tqdm
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# -------------------- OpenAI 新 SDK --------------------
try:
    from openai import OpenAI  # >=1.0.0
except ImportError as exc:  # 旧版本未安装
    raise RuntimeError(
        "❌ 未检测到 openai>=1.0.0，请先执行 `pip install --upgrade openai`"  # noqa: E501
    ) from exc

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ 请先 export OPENAI_API_KEY=sk-...")

client = OpenAI()  # 使用默认 key / base_url / organization

# -------------------- 配置 --------------------
MODEL_CHAT = "gpt-3.5-turbo"          # 精炼句子
MODEL_EMBED = "text-embedding-3-small"  # 1536 维稀释版

# -------------------- 1. 预处理 --------------------
_SANITIZE_RE = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9#@/\\.\+:\-()？? ]+")
_QUESTION_CANDIDATE_RE = re.compile(
    r"(?:^|[\n\r])"          # 行首或换行
    r"(?:[0-9]{0,2}[️⃣①②③④⑤⑥⑦⑧⑨]?)"  # 序号
    r"([^?？\n\r]{4,60})"    # 题干
    r"[?？]"                   # 问号
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _SANITIZE_RE.sub("", text)


def extract_candidates(desc: str) -> List[str]:
    desc_norm = normalize(desc)
    return [m.group(1).strip() + "?" for m in _QUESTION_CANDIDATE_RE.finditer(desc_norm)]

# -------------------- 2. 句子精炼 --------------------
_SYSTEM_PROMPT = (
    "你是一名资深 Java 后端面试官。\n"
    "- 输入是一组句子，可能是面试题，也可能只是描述。\n"
    "- 对于每个句子：如果它是完整面试题，输出 `Q: <精炼题干>`；否则输出空行。\n"
    "- 题干需书面化，结尾保留问号。"
)


def refine_questions(candidates: List[str]) -> List[str]:
    if not candidates:
        return []

    resp = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(candidates)},
        ],
        temperature=0,
    )
    refined: List[str] = []
    for line in resp.choices[0].message.content.splitlines():
        if line.startswith("Q:"):
            refined.append(line[2:].strip())
    return refined

# -------------------- 3. 语义聚类 --------------------

def get_embedding(text: str) -> List[float]:
    return client.embeddings.create(model=MODEL_EMBED, input=text).data[0].embedding


def cluster_questions(questions: List[str], eps: float = 0.4) -> Dict[str, str]:
    if not questions:
        return {}

    vecs = np.array([get_embedding(q) for q in tqdm(questions, desc="Embedding")])
    labels = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(vecs)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for q, lbl in zip(questions, labels):
        clusters[int(lbl)].append(q)

    # 每簇用最短一句作代表
    canon_map: Dict[str, str] = {}
    for qs in clusters.values():
        rep = min(qs, key=len)
        for q in qs:
            canon_map[q] = rep
    return canon_map

# -------------------- 4. 构建最终 Q&A --------------------

def build_qa(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    note_to_candidates: Dict[str, List[str]] = {}
    for note in notes:
        note_to_candidates[note["note_id"]] = extract_candidates(note.get("desc", ""))

    # 精炼
    note_to_refined: Dict[str, List[str]] = {}
    all_refined: List[str] = []
    for nid, cands in tqdm(note_to_candidates.items(), desc="Refine", total=len(note_to_candidates)):
        refined = refine_questions(cands)
        note_to_refined[nid] = refined
        all_refined.extend(refined)

    # 聚类
    canon_map = cluster_questions(all_refined)

    # 组装
    qa: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"question": "", "sources": []})
    for note in notes:
        for q in note_to_refined[note["note_id"]]:
            canon_q = canon_map[q]
            qa[canon_q]["question"] = canon_q
            qa[canon_q]["sources"].append(note)

    return list(qa.values())

# -------------------- 5. CLI --------------------

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python xhs_interview_cleaner.py <input_notes.json> <output_qas.json>")
        sys.exit(1)

    in_path, out_path = sys.argv[1:3]

    with open(in_path, "r", encoding="utf-8") as f:
        notes = json.load(f)
        if not isinstance(notes, list):
            raise ValueError("输入 JSON 须为数组！")

    qa_json = build_qa(notes)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_json, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(qa_json)} Q&A written → {out_path}")


if __name__ == "__main__":
    main()
