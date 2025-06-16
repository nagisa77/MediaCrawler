#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xhs_interview_cleaner.py

洗小红书面试笔记 desc → 结构化 Q&A JSON（兼容 openai>=1.0.0 新 SDK）
--------------------------------------------------------------------------

用法：
  $ export OPENAI_API_KEY="sk-..."
  $ python xhs_interview_cleaner.py raw_notes.json cleaned_qas.json

依赖：
  pip install --upgrade openai>=1.0 numpy scikit-learn tqdm
"""

from __future__ import annotations

import json
import html
import os
import re
import sys
import unicodedata
import time
import asyncio
from collections import defaultdict
from typing import List, Dict, Any

import aiomysql
import config

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# -------------------- OpenAI 新 SDK --------------------
try:
    from openai import OpenAI  # >=1.0.0
except ImportError as exc:
    raise RuntimeError(
        "❌ 未检测到 openai>=1.0.0，请先执行 `pip install --upgrade openai`"
    ) from exc

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ 请先 export OPENAI_API_KEY=sk-...")

client = OpenAI()  # 使用默认的 key / base_url / organization

# -------------------- 配置 --------------------
MODEL_CHAT = "gpt-3.5-turbo"           # 用于句子精炼
MODEL_EMBED = "text-embedding-3-small" # 用于向量嵌入（1536维稀释版）
CLUSTER_EPS = 0.4                      # DBSCAN 半径，可根据效果酌情调整

# 数据库中 question 字段允许的最大长度（schema 中为 varchar(512)）
QUESTION_MAX_LEN = 512

# 处理过的笔记ID记录文件，按平台区分
PROCESSED_ID_FILE = {
    "xhs": os.path.join("data", "xhs", "tmp", "processed_note_ids.json"),
    "zhihu": os.path.join("data", "zhihu", "tmp", "processed_content_ids.json"),
}


def get_record_id(item: Dict[str, Any]) -> str:
    """统一获取笔记或内容记录的唯一 ID"""
    return item.get("note_id") or item.get("content_id") or ""


def load_processed_ids(platform: str) -> set[str]:
    """读取已处理的 note_id 集合"""
    if config.SAVE_DATA_OPTION == "db" and platform in ("xhs", "zhihu"):
        async def _load_from_db() -> set[str]:
            pool = await aiomysql.create_pool(
                host=config.RELATION_DB_HOST,
                port=config.RELATION_DB_PORT,
                user=config.RELATION_DB_USER,
                password=config.RELATION_DB_PWD,
                db=config.RELATION_DB_NAME,
                autocommit=True,
            )
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    if platform == "xhs":
                        await cur.execute(
                            "SELECT note_id FROM xhs_note WHERE is_analyzed=1"
                        )
                        field = "note_id"
                    else:
                        await cur.execute(
                            "SELECT content_id FROM zhihu_content WHERE is_analyzed=1"
                        )
                        field = "content_id"
                    rows = await cur.fetchall()
            pool.close()
            await pool.wait_closed()
            return {row[field] for row in rows}

        return asyncio.run(_load_from_db())

    file_path = PROCESSED_ID_FILE.get(platform)
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_processed_ids(ids: set[str], platform: str) -> None:
    """保存已处理的 note_id 集合"""
    if config.SAVE_DATA_OPTION == "db" and platform in ("xhs", "zhihu"):
        async def _update_db():
            if not ids:
                return
            pool = await aiomysql.create_pool(
                host=config.RELATION_DB_HOST,
                port=config.RELATION_DB_PORT,
                user=config.RELATION_DB_USER,
                password=config.RELATION_DB_PWD,
                db=config.RELATION_DB_NAME,
                autocommit=True,
            )
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    for nid in ids:
                        if platform == "xhs":
                            await cur.execute(
                                "UPDATE xhs_note SET is_analyzed=1 WHERE note_id=%s",
                                (nid,),
                            )
                        else:
                            await cur.execute(
                                "UPDATE zhihu_content SET is_analyzed=1 WHERE content_id=%s",
                                (nid,),
                            )
            pool.close()
            await pool.wait_closed()

        asyncio.run(_update_db())
        return

    file_path = PROCESSED_ID_FILE.get(platform)
    if not file_path:
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, ensure_ascii=False, indent=2)

# -------------------- 1. 预处理：提取候选问题 --------------------
_SENTENCE_SPLIT_RE = re.compile(
    r"[。；;…\n\r]+|(?<=\d)[\)\.]|[\d]+\s*[、\.]"  # 中文句号/分号/换行/数字.) / 1. 2、 等
)

# 出现这些关键词即视为询问句（不要求问号）
_CN_WH_WORDS = ("什么", "如何", "为什么", "多少", "怎么", "哪", "能否", "是否", "原理", "流程", "区别", "优缺点")
_EN_WH_WORDS = ("what", "why", "how", "when", "which", "where", "can", "could", "should")
_PREFIX_HINTS = ("问", "请", "说说", "讲讲", "介绍", "描述")

def is_question_like(seg: str) -> bool:
    s = seg.strip()
    if s.endswith(("?", "？")):
        return True
    if s.startswith(_PREFIX_HINTS):
        return True
    if any(w in s.lower() for w in _EN_WH_WORDS) or any(w in s for w in _CN_WH_WORDS):
        return True
    return False

def normalize(text: str) -> str:
    text = html.unescape(text)               # &gt; → >
    text = unicodedata.normalize("NFKC", text)
    # 不再删除换行，让后续“行粒度”解析更容易
    text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9#@/\.\+:\-\(\)（）？? \n\r]", "", text)
    return text

def extract_candidates(desc: str) -> list[str]:
    desc_norm = normalize(desc)

    print("【DEBUG】标准化后的笔记描述：", desc_norm)

    # ① 句子/行级切分
    frags = _SENTENCE_SPLIT_RE.split(desc_norm)

    # ② 去掉空白、去重
    uniq = []
    for f in frags:
        f = f.strip()
        if 4 <= len(f) <= 120 and f not in uniq:
            uniq.append(f)

    # ③ 问句判定
    cands = []
    for seg in uniq:
        if is_question_like(seg):
            # 确保问号结尾，方便 GPT 识别
            seg_q = seg if seg.endswith(("?", "？")) else seg + "?"
            cands.append(seg_q)

    print("【DEBUG】提取的候选问题：", cands)

    return cands

# -------------------- 2. 句子精炼 (Chat GPT) --------------------
"""
思路：让 GPT 判断哪些候选是真正的"面试题"；并对题干做一些精炼处理。
如果 GPT 判定不是题，则输出空行，判定是题就输出 "Q: <精炼题干>"。
"""

_SYSTEM_PROMPT = (
    "你是一名资深后端面试官。\n"
    "- 输入是一组可能是面试题的句子（含问号）。\n"
    "- 你需要判断它是不是完整、可作为面试问题的句子。\n"
    "- 对于每个句子：如果它是一个清晰、可答的面试题，输出 `Q: <精炼题干>`；否则输出空行。\n"
    "- 题干需保持问句形式，并且语义完整、简洁。\n"
    "只输出处理结果，不要添加额外解释。\n"
)

def refine_questions(candidates: List[str]) -> List[str]:
    """调用 GPT 对候选句子进行判定和精炼，返回精炼后的面试题列表。"""
    if not candidates:
        return []
    # 将所有候选句拼成一段
    # 每行一个
    user_text = "\n".join(candidates)

    resp = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
    )
    # 记录用户输入和GPT返回内容，便于调试
    print("【DEBUG】发送给GPT的内容：")
    print(user_text)
    print("【DEBUG】GPT返回内容：")
    print(resp.choices[0].message.content)
    # 拆分 GPT 返回
    lines = resp.choices[0].message.content.splitlines()
    # 只取带 "Q:" 的行
    refined = []
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            refined_text = line[2:].strip()
            # 避免空题
            if refined_text:
                refined.append(refined_text)
    return refined

# -------------------- 3. 语义聚类 (DBSCAN) --------------------
"""
对于提炼后的面试题，我们用 OpenAI Embedding 得到向量，然后用 DBSCAN 根据余弦距离聚类。
同一个簇里的问题被视为“同义或高度相似”。
"""

def get_embedding(text: str) -> List[float]:
    """获取文本的向量表示。"""
    emb_resp = client.embeddings.create(model=MODEL_EMBED, input=text)
    return emb_resp.data[0].embedding

def cluster_questions(questions: List[str], eps: float = CLUSTER_EPS) -> Dict[str, str]:
    """
    返回一个映射：原始问题 -> 簇代表(即canonical question)。
    同簇问题都指向同一个代表。
    每簇选用最短的一句做代表。
    """
    if not questions:
        return {}

    # 计算 embeddings
    vecs = []
    for q in tqdm(questions, desc="Embedding"):
        vec = get_embedding(q)
        vecs.append(vec)
    vecs = np.array(vecs)

    # 余弦距离  => DBSCAN metric='cosine'
    # eps 可调；越大合并越多
    labels = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(vecs)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for q, lbl in zip(questions, labels):
        clusters[lbl].append(q)

    # 每个簇里选用最短文本作为canonical
    canon_map: Dict[str, str] = {}
    for q_list in clusters.values():
        rep = min(q_list, key=len)
        for q in q_list:
            canon_map[q] = rep
    return canon_map

# -------------------- 4. 构建最终 Q&A 结构 --------------------
"""
最终我们需要:
[
  {
    "question": "某个canonical question",
    "sources": [
      {
         ... note1 的信息 ...
      },
      {
         ... note2 的信息 ...
      }
    ]
  },
  ...
]
"""

def build_qa(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    核心流程：
      1) 从每个 note.desc 中提取候选问题
      2) 用 GPT refine
      3) 全部问题一起做聚类
      4) 按簇合并来源
    """
    # 1) 收集所有候选
    note_to_candidates: Dict[str, List[str]] = {}
    for note in notes:
        nid = get_record_id(note)
        if not nid:
            continue
        if note.get("platform") == "zhihu":
            desc = note.get("content_text", "")
        elif note.get("platform") == "xhs":
            desc = note.get("desc", "")
        else:
            continue
        cands = extract_candidates(desc)
        note_to_candidates[nid] = cands

    # 2) 精炼
    note_to_refined: Dict[str, List[str]] = {}
    all_refined: List[str] = []
    all_candidate_count = 0

    for nid, cands in tqdm(note_to_candidates.items(), desc="Refine"):
        refined = refine_questions(cands)  # GPT 判断+精炼
        note_to_refined[nid] = refined
        all_refined.extend(refined)
        all_candidate_count += len(cands)

    print(f"总共从 desc 中匹配到 {all_candidate_count} 个带问号短语，经 GPT 判定得到 {len(all_refined)} 个有效问题。")

    # 3) 聚类
    canon_map = cluster_questions(all_refined)

    # 4) 组装
    #   由于同个问题(簇代表)会出现在多个笔记中，所以需要把 note 汇总到同一个 key 里
    qa_dict: Dict[str, Dict[str, Any]] = {}

    def make_empty_qa(q: str) -> Dict[str, Any]:
        return {"question": q, "platform": [], "sources": [], "categories": []}

    for note in notes:
        nid = get_record_id(note)
        if not nid:
            continue
        refined_in_note = note_to_refined.get(nid, [])
        for q in refined_in_note:
            canon_q = canon_map[q]  # 该问题所属的canonical
            if canon_q not in qa_dict:
                qa_dict[canon_q] = make_empty_qa(canon_q)
            existing_ids = {get_record_id(src) for src in qa_dict[canon_q]["sources"]}
            if nid in existing_ids:
                continue
            if note.get("platform") == "zhihu":
                qa_dict[canon_q]["sources"].append({
                    "content_id": note.get("content_id", ""),
                    "content_type": note.get("content_type", ""),
                    "content_text": note.get("content_text", ""),
                    "content_url": note.get("content_url", ""),
                    "question_id": note.get("question_id", ""),
                    "title": note.get("title", ""),
                    "desc": note.get("desc", ""),
                    "created_time": note.get("created_time", ""),
                    "updated_time": note.get("updated_time", ""),
                    "voteup_count": note.get("voteup_count", 0),
                    "comment_count": note.get("comment_count", 0),
                    "source_keyword": note.get("source_keyword", ""),
                    "user_id": note.get("user_id", ""),
                    "user_link": note.get("user_link", ""),
                    "user_nickname": note.get("user_nickname", ""),
                    "user_avatar": note.get("user_avatar", ""),
                    "user_url_token": note.get("user_url_token", ""),
                    "last_modify_ts": note.get("last_modify_ts", 0),
                    "platform": note.get("platform", ""),
                })
            elif note.get("platform") == "xhs":
                qa_dict[canon_q]["sources"].append({
                    "note_id": note.get("note_id", ""),
                    "type": note.get("type", ""),
                    "video_url": note.get("video_url", ""),
                    "time": note.get("time", 0),
                    "last_update_time": note.get("last_update_time", 0),
                    "user_id": note.get("user_id", ""),
                    "nickname": note.get("nickname", ""),
                    "avatar": note.get("avatar", ""),
                    "liked_count": note.get("liked_count", ""),
                    "collected_count": note.get("collected_count", ""),
                    "comment_count": note.get("comment_count", ""),
                    "share_count": note.get("share_count", ""),
                    "ip_location": note.get("ip_location", ""),
                    "image_list": note.get("image_list", ""),
                    "tag_list": note.get("tag_list", ""),
                    "last_modify_ts": note.get("last_modify_ts", 0),
                    "note_url": note.get("note_url", ""),
                    "source_keyword": note.get("source_keyword", ""),
                    "xsec_token": note.get("xsec_token", ""),
                    "desc": note.get("desc", ""),
                    "platform": note.get("platform", ""),
                })
            # 合并分类
            note_categories = []
            raw_cat = note.get("categories", [])
            if isinstance(raw_cat, str):
                try:
                    note_categories = json.loads(raw_cat)
                except Exception:
                    note_categories = [c for c in raw_cat.split(",") if c]
            elif isinstance(raw_cat, list):
                note_categories = raw_cat
            qa_dict[canon_q]["categories"] = list(
                dict.fromkeys(qa_dict[canon_q]["categories"] + note_categories)
            )
            qa_dict[canon_q]["platform"] = list(
                dict.fromkeys(qa_dict[canon_q]["platform"] + [note.get("platform", "")])
            )

    # 转成 list
    return list(qa_dict.values())


async def store_to_db(qa_items: List[Dict[str, Any]]) -> None:
    """将 Q&A 结果保存到数据库，若问题已存在则合并来源"""
    pool = await aiomysql.create_pool(
        host=config.RELATION_DB_HOST,
        port=config.RELATION_DB_PORT,
        user=config.RELATION_DB_USER,
        password=config.RELATION_DB_PWD,
        db=config.RELATION_DB_NAME,
        autocommit=True,
    )
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            for item in qa_items:
                question = item["question"]
                if len(question) > QUESTION_MAX_LEN:
                    question = question[:QUESTION_MAX_LEN]

                # 查询是否已有该问题
                await cur.execute(
                    "SELECT sources, categories, platform FROM interview_question WHERE question=%s",
                    (question,),
                )
                row = await cur.fetchone()

                if row:
                    try:
                        existing_sources = json.loads(row.get("sources", "[]"))
                    except Exception:
                        existing_sources = []
                    try:
                        existing_categories = json.loads(row.get("categories", "[]"))
                    except Exception:
                        existing_categories = []
                    # 按来源ID去重合并新旧来源
                    merged = {get_record_id(src): src for src in existing_sources}
                    for src in item["sources"]:
                        merged[get_record_id(src)] = src
                    merged_sources = list(merged.values())
                    merged_categories = list(dict.fromkeys(existing_categories + item.get("categories", [])))
                    try:
                        existing_platform = json.loads(row.get("platform", "[]"))
                    except Exception:
                        existing_platform = []
                    merged_platform = list(dict.fromkeys(existing_platform + item.get("platform", [])))
                    await cur.execute(
                        "UPDATE interview_question SET sources=%s, categories=%s, platform=%s, add_ts=%s WHERE question=%s",
                        (
                            json.dumps(merged_sources, ensure_ascii=False),
                            json.dumps(merged_categories, ensure_ascii=False),
                            json.dumps(merged_platform, ensure_ascii=False),
                            int(time.time() * 1000),
                            question,
                        ),
                    )
                else:
                    await cur.execute(
                        "INSERT INTO interview_question(question, sources, categories, platform, add_ts) VALUES (%s, %s, %s, %s, %s)",
                        (
                            question,
                            json.dumps(item["sources"], ensure_ascii=False),
                            json.dumps(item.get("categories", []), ensure_ascii=False),
                            json.dumps(item.get("platform", []), ensure_ascii=False),
                            int(time.time() * 1000),
                        ),
                    )
    pool.close()
    await pool.wait_closed()


async def merge_existing_questions() -> None:
    """按语义合并数据库中已存在的面试题，合并其来源列表"""
    pool = await aiomysql.create_pool(
        host=config.RELATION_DB_HOST,
        port=config.RELATION_DB_PORT,
        user=config.RELATION_DB_USER,
        password=config.RELATION_DB_PWD,
        db=config.RELATION_DB_NAME,
        autocommit=True,
    )

    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("SELECT id, question, sources, categories, platform FROM interview_question")
            rows = await cur.fetchall()

            if not rows:
                pool.close()
                await pool.wait_closed()
                return

            questions = [row["question"] for row in rows]
            canon_map = cluster_questions(questions)

            clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                canon = canon_map[row["question"]]
                clusters[canon].append(row)

            for canon_q, items in clusters.items():
                merged: Dict[str, Any] = {}
                merged_categories_set: set[str] = set()
                merged_platform_set: set[str] = set()
                rep_id = None
                for row in items:
                    try:
                        src_list = json.loads(row.get("sources", "[]"))
                    except Exception:
                        src_list = []
                    try:
                        cat_list = json.loads(row.get("categories", "[]"))
                    except Exception:
                        cat_list = []
                    for src in src_list:
                        merged[get_record_id(src)] = src
                    merged_categories_set.update(cat_list)
                    try:
                        plat_list = json.loads(row.get("platform", "[]"))
                    except Exception:
                        plat_list = []
                    merged_platform_set.update(plat_list)
                    if row["question"] == canon_q:
                        rep_id = row["id"]

                if rep_id is None:
                    rep_id = items[0]["id"]
                    await cur.execute(
                        "UPDATE interview_question SET question=%s WHERE id=%s",
                        (canon_q, rep_id),
                    )

                merged_sources = list(merged.values())
                await cur.execute(
                    "UPDATE interview_question SET sources=%s, categories=%s, platform=%s, add_ts=%s WHERE id=%s",
                    (
                        json.dumps(merged_sources, ensure_ascii=False),
                        json.dumps(list(merged_categories_set), ensure_ascii=False),
                        json.dumps(list(merged_platform_set), ensure_ascii=False),
                        int(time.time() * 1000),
                        rep_id,
                    ),
                )

                for row in items:
                    if row["id"] != rep_id:
                        await cur.execute(
                            "DELETE FROM interview_question WHERE id=%s",
                            (row["id"],),
                        )

    pool.close()
    await pool.wait_closed()


async def analyze_questions() -> None:
    """简单统计数据库中的问题数量并按来源数量排序输出前5"""
    pool = await aiomysql.create_pool(
        host=config.RELATION_DB_HOST,
        port=config.RELATION_DB_PORT,
        user=config.RELATION_DB_USER,
        password=config.RELATION_DB_PWD,
        db=config.RELATION_DB_NAME,
        autocommit=True,
    )

    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("SELECT question, sources FROM interview_question")
            rows = await cur.fetchall()

    pool.close()
    await pool.wait_closed()

    stats = []
    for row in rows:
        try:
            srcs = json.loads(row.get("sources", "[]"))
        except Exception:
            srcs = []
        stats.append((row["question"], len(srcs)))

    stats.sort(key=lambda x: x[1], reverse=True)
    print("Top questions by sources:")
    for q, cnt in stats[:20]:
        print(f"{cnt}× {q}")


async def load_notes_from_db() -> List[Dict[str, Any]]:
    """从数据库加载待处理的笔记信息，包含小红书和知乎"""
    pool = await aiomysql.create_pool(
        host=config.RELATION_DB_HOST,
        port=config.RELATION_DB_PORT,
        user=config.RELATION_DB_USER,
        password=config.RELATION_DB_PWD,
        db=config.RELATION_DB_NAME,
        autocommit=True,
    )
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("SELECT * FROM xhs_note WHERE IFNULL(is_analyzed,0)=0")
            xhs_rows = await cur.fetchall()
            for r in xhs_rows:
                r["platform"] = "xhs"

            await cur.execute("SELECT * FROM zhihu_content WHERE IFNULL(is_analyzed,0)=0")
            zhihu_rows = await cur.fetchall()
            for r in zhihu_rows:
                r["platform"] = "zhihu"

            # 确保 xhs_rows 和 zhihu_rows 都是列表，即使为空也不会出错
            rows = list(xhs_rows) if xhs_rows else []
            if zhihu_rows:
                rows += list(zhihu_rows)
    pool.close()
    await pool.wait_closed()
    return list(rows)

# -------------------- 5. CLI 入口 --------------------

def main() -> None:
    if config.SAVE_DATA_OPTION == "db":
        notes = asyncio.run(load_notes_from_db())
        out_path = ""
        enable_db = True
    else:
        if len(sys.argv) < 3:
            print(
                "Usage: python xhs_interview_cleaner.py <input_notes.json> <output_qas.json> [--db]"
            )
            sys.exit(1)

        in_path, out_path = sys.argv[1], sys.argv[2]
        enable_db = len(sys.argv) > 3 and sys.argv[3] == "--db"

        with open(in_path, "r", encoding="utf-8") as f:
            notes = json.load(f)
            if not isinstance(notes, list):
                raise ValueError("输入 JSON 须为数组！")

    processed_ids_xhs = load_processed_ids("xhs")
    processed_ids_zhihu = load_processed_ids("zhihu")
    unique_notes = []
    new_ids_xhs = set()
    new_ids_zhihu = set()
    for note in notes:
        platform = note.get("platform", "xhs")
        nid = get_record_id(note)
        if platform == "xhs":
            if not nid or nid in processed_ids_xhs or nid in new_ids_xhs:
                continue
            new_ids_xhs.add(nid)
        else:
            if not nid or nid in processed_ids_zhihu or nid in new_ids_zhihu:
                continue
            new_ids_zhihu.add(nid)
        unique_notes.append(note)

    # if not unique_notes:
    #     print("No new notes to process.")
    #     return

    qa_json = build_qa(unique_notes)

    if config.SAVE_DATA_OPTION != "db":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(qa_json, f, ensure_ascii=False, indent=2)

    if config.SAVE_DATA_OPTION == "db":
        save_processed_ids(new_ids_xhs, "xhs")
        save_processed_ids(new_ids_zhihu, "zhihu")
    else:
        processed_ids_xhs.update(new_ids_xhs)
        processed_ids_zhihu.update(new_ids_zhihu)
        save_processed_ids(processed_ids_xhs, "xhs")
        save_processed_ids(processed_ids_zhihu, "zhihu")

    if enable_db:
        asyncio.run(store_to_db(qa_json))
        if config.ENABLE_MERGE_INTERVIEW_QUESTIONS:
            asyncio.run(merge_existing_questions())
        if config.ENABLE_ANALYZE_INTERVIEW_QUESTIONS:
            asyncio.run(analyze_questions())

    if config.SAVE_DATA_OPTION == "db":
        print(f"✅ 生成 {len(qa_json)} 条 Q&A，已写入数据库")
    else:
        print(f"✅ 生成 {len(qa_json)} 条 Q&A，写入 → {out_path}")


if __name__ == "__main__":
    main()
