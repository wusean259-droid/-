"""
用户自定义知识库 (RAG) — ChromaDB 本地向量检索
文档目录: my_investment_brain/  (用户自行添加 .md / .txt，可为空)
向量库:   knowledge_chroma/    (自动生成)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = os.path.join(MODULE_DIR, "my_investment_brain")
CHROMA_DIR = os.path.join(MODULE_DIR, "knowledge_chroma")
MANIFEST_FILE = os.path.join(CHROMA_DIR, "manifest.json")
COLLECTION_NAME = "investment_brain"

SUPPORTED_EXT = {".md", ".txt", ".markdown"}
SKIP_FILENAMES = {"00_使用说明.md", "README.md", "readme.md"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 3


def _ensure_dirs():
    os.makedirs(BRAIN_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)


def _read_secrets_embedding_config() -> dict:
    """从 secrets.toml 读取可选的 Embedding API 配置。"""
    cfg = {"api_key": "", "base_url": "https://api.openai.com/v1", "model": "text-embedding-3-small"}
    pattern = re.compile(r"^(\w+)\s*=\s*[\"']?([^\"'\n]+)[\"']?\s*$")
    for path in (
        os.path.join(MODULE_DIR, "secrets.toml"),
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
    ):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    m = pattern.match(line.strip())
                    if not m:
                        continue
                    k, v = m.group(1), m.group(2).strip()
                    if k == "EMBEDDING_API_KEY":
                        cfg["api_key"] = v
                    elif k == "EMBEDDING_BASE_URL":
                        cfg["base_url"] = v
                    elif k == "EMBEDDING_MODEL":
                        cfg["model"] = v
        except OSError:
            continue
    if not cfg["api_key"]:
        cfg["api_key"] = os.environ.get("EMBEDDING_API_KEY", "").strip()
    return cfg


def _build_embedding_function():
    """优先 OpenAI 兼容 Embedding API；否则用 Chroma 内置 ONNX 模型（无需额外 Key）。"""
    import chromadb.utils.embedding_functions as ef

    emb_cfg = _read_secrets_embedding_config()
    if emb_cfg["api_key"]:
        try:
            from openai import OpenAI

            class _OpenAICompatEmbedding(ef.EmbeddingFunction):
                def __init__(self, api_key, base_url, model):
                    self._client = OpenAI(api_key=api_key, base_url=base_url)
                    self._model = model

                def __call__(self, input: list[str]) -> list[list[float]]:
                    resp = self._client.embeddings.create(model=self._model, input=input)
                    return [item.embedding for item in resp.data]

            return _OpenAICompatEmbedding(emb_cfg["api_key"], emb_cfg["base_url"], emb_cfg["model"])
        except Exception:
            pass
    return ef.DefaultEmbeddingFunction()


def _get_collection():
    import chromadb

    _ensure_dirs()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = _build_embedding_function()
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_FILE):
        return {}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(manifest: dict):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def list_brain_documents() -> list[dict]:
    """列出知识库目录下所有文档及元信息。"""
    _ensure_dirs()
    docs = []
    for fname in sorted(os.listdir(BRAIN_DIR)):
        if fname.startswith(".") or fname in SKIP_FILENAMES:
            continue
        if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXT:
            continue
        fpath = os.path.join(BRAIN_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        docs.append({
            "filename": fname,
            "path": fpath,
            "size_kb": round(os.path.getsize(fpath) / 1024, 1),
            "hash": _file_hash(fpath),
        })
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按字符切块，尽量在段落边界断开。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at == -1:
                break_at = text.rfind("\n", start, end)
            if break_at > start + chunk_size // 3:
                end = break_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _index_single_file(collection, fname: str, fpath: str) -> int:
    """索引单个文件，返回 chunk 数量。"""
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    chunks = chunk_text(raw)
    if not chunks:
        return 0

    prefix = f"{fname}::"
    existing = collection.get(where={"source": fname})
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{prefix}{i}")
        documents.append(chunk)
        metadatas.append({"source": fname, "chunk_index": i})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def sync_knowledge_index(force: bool = False) -> dict:
    """
    增量同步：扫描 my_investment_brain/，新增/变更文件写入 ChromaDB，删除已移除文件。
    返回统计信息。
    """
    _ensure_dirs()
    collection = _get_collection()
    manifest = _load_manifest()
    docs = list_brain_documents()
    current_files = {d["filename"]: d["hash"] for d in docs}

    stats = {"indexed": 0, "removed": 0, "chunks_added": 0, "total_docs": len(docs), "total_chunks": 0}

    for fname, fhash in current_files.items():
        if not force and manifest.get(fname) == fhash:
            continue
        n = _index_single_file(collection, fname, os.path.join(BRAIN_DIR, fname))
        manifest[fname] = fhash
        stats["indexed"] += 1
        stats["chunks_added"] += n

    for stale in set(manifest.keys()) - set(current_files.keys()):
        existing = collection.get(where={"source": stale})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
        del manifest[stale]
        stats["removed"] += 1

    _save_manifest(manifest)
    stats["total_chunks"] = collection.count()
    return stats


def search_my_brain(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """语义检索知识库，返回最相关的文本片段。"""
    query = str(query).strip()
    if not query:
        return []
    _ensure_dirs()
    if not list_brain_documents():
        return []

    sync_knowledge_index()
    collection = _get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    hits = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        hits.append({
            "rank": i + 1,
            "source": meta.get("source", "未知"),
            "chunk_index": meta.get("chunk_index", 0),
            "content": doc,
            "distance": round(float(dist), 4) if dist is not None else None,
            "id": ids[i] if i < len(ids) else None,
        })
    return hits


def tool_search_my_brain(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Agent Function Calling 入口。"""
    hits = search_my_brain(query, top_k=top_k)
    if not list_brain_documents():
        return {
            "query": query,
            "count": 0,
            "message": "知识库当前为空，无需引用私有文档。请直接用量化工具与通用分析能力回答。",
            "snippets": [],
        }
    if not hits:
        return {
            "query": query,
            "count": 0,
            "message": "知识库中未找到相关片段。请结合实时数据工具与通用分析回答，勿编造笔记内容。",
            "snippets": [],
        }
    snippets = [
        {
            "rank": h["rank"],
            "source": h["source"],
            "excerpt": h["content"][:800],
            "relevance_distance": h["distance"],
        }
        for h in hits
    ]
    return {"query": query, "count": len(snippets), "snippets": snippets}


def save_text_document(title: str, text: str) -> str:
    """将粘贴的文字保存为 .md / .txt 文件。"""
    title = str(title).strip()
    text = str(text).strip()
    if not title:
        raise ValueError("请填写文件名")
    if not text:
        raise ValueError("内容不能为空")
    safe = os.path.basename(title)
    if not os.path.splitext(safe)[1]:
        safe += ".md"
    ext = os.path.splitext(safe)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError("文件名须以 .md 或 .txt 结尾")
    dest = os.path.join(BRAIN_DIR, safe)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return dest


def save_uploaded_document(filename: str, content: bytes) -> str:
    """保存用户上传的文档到知识库目录。"""
    _ensure_dirs()
    safe = os.path.basename(filename)
    ext = os.path.splitext(safe)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"不支持的格式 {ext}，请上传 .md 或 .txt")
    dest = os.path.join(BRAIN_DIR, safe)
    with open(dest, "wb") as f:
        f.write(content)
    return dest


def delete_brain_document(filename: str) -> bool:
    """删除知识库中的指定文档及其向量。"""
    safe = os.path.basename(filename)
    fpath = os.path.join(BRAIN_DIR, safe)
    if not os.path.isfile(fpath):
        return False
    os.remove(fpath)
    collection = _get_collection()
    existing = collection.get(where={"source": safe})
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])
    manifest = _load_manifest()
    manifest.pop(safe, None)
    _save_manifest(manifest)
    return True


def get_kb_stats() -> dict:
    docs = list_brain_documents()
    try:
        collection = _get_collection()
        chunk_count = collection.count()
    except Exception:
        chunk_count = 0
    emb_cfg = _read_secrets_embedding_config()
    return {
        "brain_dir": BRAIN_DIR,
        "document_count": len(docs),
        "chunk_count": chunk_count,
        "embedding_mode": "API" if emb_cfg["api_key"] else "本地 ONNX (Chroma 内置)",
        "documents": [d["filename"] for d in docs],
    }


def render_knowledge_base_panel():
    """Streamlit UI：知识库管理面板。"""
    import streamlit as st

    _ensure_dirs()
    st.markdown("**自定义知识库（可选）** — 你随时上传的私有文档，AI 在需要时会检索引用。")
    st.caption("不填也能正常用。有内容时相当于「开卷考试」；没有则走通用分析 + 实时数据工具。")
    st.caption(f"📁 本地文件夹：`{BRAIN_DIR}`")

    stats = get_kb_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("文档数", stats["document_count"])
    c2.metric("向量片段", stats["chunk_count"])
    c3.metric("Embedding", stats["embedding_mode"])

    st.markdown("""
**知识库 = 这个文件夹里的文字文件。** `.md` / `.txt` 就是普通文本，用记事本 / Cursor / Word 另存为 写进去都行。
写完以后要点 **「🔄 重建索引」**（网页上传则会自动索引）。
    """)

    tab_paste, tab_upload, tab_folder = st.tabs(["✍️ 网页里直接写", "📤 上传已有文件", "📁 用文件夹管理"])

    with tab_paste:
        st.caption("适合临时写一段笔记，不用自己建文件。")
        paste_name = st.text_input("文件名", placeholder="我的笔记.md", key="kb_paste_name")
        paste_body = st.text_area("内容（粘贴或打字均可）", height=180, key="kb_paste_body")
        if st.button("💾 保存并入库", key="kb_save_paste", use_container_width=True):
            try:
                save_text_document(paste_name, paste_body)
                with st.spinner("正在切块并向量化…"):
                    sync_stats = sync_knowledge_index(force=True)
                st.success(f"已保存，共 {sync_stats['total_chunks']} 个检索片段")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with tab_upload:
        st.caption("从电脑选一个已经写好的 .md 或 .txt 文件（可拖拽到框里）。")
        uploaded = st.file_uploader(
            "选择文件",
            type=["md", "txt", "markdown"],
            key="kb_file_uploader",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.caption(f"已选：**{uploaded.name}**")
            if st.button("💾 上传并入库", key="kb_save_upload", use_container_width=True):
                try:
                    save_uploaded_document(uploaded.name, uploaded.getvalue())
                    with st.spinner("正在切块并向量化…"):
                        sync_stats = sync_knowledge_index(force=True)
                    st.success(f"已入库，共 {sync_stats['total_chunks']} 个检索片段")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    with tab_folder:
        st.markdown(f"""
1. 打开文件夹：`{BRAIN_DIR}`
2. 新建或复制 `.md` / `.txt` 文件进去（**在里面写文字、粘贴文字都对**）
3. 回到这里点 **「🔄 重建索引」**

Word / PDF 不能直接用，请先 **另存为 .txt 或 .md** 再放进去。
        """)
    st.markdown("**已有文档**")
    docs = list_brain_documents()
    if not docs:
        st.info("知识库为空 — 这是正常状态。上传文档后会自动参与检索。")
    else:
        for doc in docs:
            dc1, dc2 = st.columns([4, 1])
            dc1.caption(f"📄 {doc['filename']}  ({doc['size_kb']} KB)")
            if dc2.button("🗑️", key=f"kb_del_{doc['filename']}", help="删除此文档"):
                delete_brain_document(doc["filename"])
                st.toast(f"已删除 {doc['filename']}", icon="🗑️")
                st.rerun()

    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("🔄 重建索引", key="kb_rebuild", use_container_width=True):
            with st.spinner("全量重建向量索引…"):
                s = sync_knowledge_index(force=True)
            st.success(f"完成：{s['total_docs']} 篇文档，{s['total_chunks']} 个片段")
            st.rerun()
    with bc2:
        test_q = st.text_input("试检索", placeholder="输入任意问题测试检索效果", key="kb_test_query")
        if st.button("🔍 试搜", key="kb_test_search", use_container_width=True) and test_q.strip():
            hits = search_my_brain(test_q.strip(), top_k=3)
            if not hits:
                st.warning("无匹配片段")
            else:
                for h in hits:
                    # 不可用 st.expander：外层 AI 中枢已有 expander，Streamlit 禁止嵌套
                    st.markdown(f"**#{h['rank']} · {h['source']}**（距离 {h['distance']}）")
                    st.markdown(h["content"])
                    st.divider()

    st.caption("⚙️ Embedding：默认用本地模型；可选在 secrets.toml 配置 EMBEDDING_API_KEY / BASE_URL / MODEL")
