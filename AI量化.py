import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import datetime 
import requests 
import json
import os
import re
import time
from openai import OpenAI
import plotly.graph_objects as go

try:
    from knowledge_rag import render_knowledge_base_panel, sync_knowledge_index, tool_search_my_brain
    KB_RAG_AVAILABLE = True
except ImportError:
    KB_RAG_AVAILABLE = False

    def tool_search_my_brain(query: str, top_k: int = 3) -> dict:
        return {"error": "知识库模块未安装，请执行 pip install chromadb", "query": query, "snippets": []}

    def render_knowledge_base_panel():
        import streamlit as st
        st.warning("请先安装依赖：`pip install chromadb`")

    def sync_knowledge_index(force: bool = False) -> dict:
        return {}

# ==========================================
# 💾 0. 硬盘持久化读写中枢 (绝对路径增强版)
# ==========================================
# 获取当前文件所在的绝对路径目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 强制将 json 文件锁定在这个目录下
HISTORY_FILE = os.path.join(CURRENT_DIR, "portfolio_history.json")
CHAT_HISTORY_FILE = os.path.join(CURRENT_DIR, "chat_history.json")
PRO_WORKSPACE_FILE = os.path.join(CURRENT_DIR, "pro_workspace.json")
BENCHMARK_CACHE_FILE = os.path.join(CURRENT_DIR, "benchmark_000300_cache.json")
BENCHMARK_CACHE_TTL = 86400

def get_deepseek_api_key() -> str:
    """
    读取 DeepSeek API Key。
    优先读本脚本同目录的 secrets.toml（与 streamlit run 启动路径无关）。
    """
    if st.session_state.get("_deepseek_api_key_cache"):
        return st.session_state["_deepseek_api_key_cache"]

    pattern = re.compile(r'DEEPSEEK_API_KEY\s*=\s*["\']?([^"\']+)["\']?')
    candidate_paths = [
        os.path.join(CURRENT_DIR, "secrets.toml"),
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(CURRENT_DIR, ".streamlit", "secrets.toml"),
    ]

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                match = pattern.search(f.read())
                if match:
                    key = match.group(1).strip()
                    st.session_state["_deepseek_api_key_cache"] = key
                    return key
        except Exception:
            continue

    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        st.session_state["_deepseek_api_key_cache"] = env_key
        return env_key

    try:
        key = st.secrets.get("DEEPSEEK_API_KEY", None)
        if key:
            key = str(key).strip()
            st.session_state["_deepseek_api_key_cache"] = key
            return key
    except Exception:
        pass

    raise RuntimeError(
        f"未找到 DeepSeek API Key。请将密钥写入：{os.path.join(CURRENT_DIR, 'secrets.toml')}"
    )

def get_deepseek_client() -> OpenAI:
    return OpenAI(api_key=get_deepseek_api_key(), base_url="https://api.deepseek.com")

def _load_benchmark_disk_cache():
    if not os.path.exists(BENCHMARK_CACHE_FILE):
        return None
    try:
        if time.time() - os.path.getmtime(BENCHMARK_CACHE_FILE) > BENCHMARK_CACHE_TTL:
            return None
        with open(BENCHMARK_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        df = pd.DataFrame(payload)
        df["净值日期"] = pd.to_datetime(df["净值日期"])
        df["基准点数"] = df["基准点数"].astype(float)
        return df.sort_values(by="净值日期")
    except Exception:
        return None

def _save_benchmark_disk_cache(df: pd.DataFrame):
    try:
        export = df.copy()
        export["净值日期"] = export["净值日期"].dt.strftime("%Y-%m-%d")
        with open(BENCHMARK_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(export.to_dict(orient="records"), f, ensure_ascii=False)
    except Exception:
        pass

def _fetch_benchmark_sina():
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "净值日期", "close": "基准点数"})
    return df[["净值日期", "基准点数"]].sort_values(by="净值日期")

def _fetch_benchmark_em():
    df = ak.index_zh_a_hist(symbol="000300", period="daily")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.rename(columns={"日期": "净值日期", "收盘": "基准点数"})
    return df[["净值日期", "基准点数"]].sort_values(by="净值日期")

# ==========================================
# 📡 全市场基金名录雷达 (缓存机制，每天只抓一次)
# ==========================================
@st.cache_data(ttl=86400)
def load_all_fund_names():
    try:
        df = ak.fund_name_em()
        return df[["基金代码", "基金简称", "基金类型"]]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_fund_name_index() -> dict:
    """全市场名称索引，每天只构建一次，避免重复拉取 1.5 万条名录。"""
    df = load_all_fund_names()
    if df.empty:
        return {}
    return dict(zip(df["基金代码"].astype(str), df["基金简称"].astype(str)))
    
def load_local_history():
    """启动时从硬盘读取历史记录"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.sidebar.error(f"⚠️ 本地记忆库读取失败: {e}")
            return []
    return []

def save_local_history(history_data, silent: bool = False):
    """有变动时将内存状态写入硬盘，并弹出视觉反馈"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
        if not silent:
            st.toast("历史组合已成功写入本地硬盘！", icon="✅")
    except Exception as e:
        st.error(f"写入本地硬盘失败: {e}")

def _compact_chat_for_disk(history: list) -> list:
    """落盘时去掉大图序列，只保留文字、方案快照与轻量指标。"""
    out = []
    for m in (history or [])[-50:]:
        item = {"role": m.get("role", "assistant"), "content": m.get("content", "")}
        if m.get("portfolio"):
            item["portfolio"] = m["portfolio"]
        light_vis = []
        for v in m.get("visuals") or []:
            if not isinstance(v, dict):
                continue
            if v.get("type") == "fund_metrics":
                light_vis.append({
                    "type": "fund_metrics",
                    "code": v.get("code"),
                    "name": v.get("name"),
                    "metrics": v.get("metrics"),
                })
            elif v.get("type") == "backtest":
                light_vis.append({
                    "type": "backtest",
                    "weights_pct": v.get("weights_pct"),
                    "metrics": v.get("metrics"),
                })
        if light_vis:
            item["visuals"] = light_vis
        out.append(item)
    return out

def load_chat_history() -> list:
    if not os.path.exists(CHAT_HISTORY_FILE):
        return []
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def save_chat_history(history: list, silent: bool = True):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_compact_chat_for_disk(history), f, ensure_ascii=False, indent=2)
        if not silent:
            st.toast("对话已保存", icon="✅")
    except Exception as e:
        st.warning(f"对话历史写入失败：{e}")

def load_pro_workspace() -> dict:
    if not os.path.exists(PRO_WORKSPACE_FILE):
        return {}
    try:
        with open(PRO_WORKSPACE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_pro_workspace(silent: bool = True):
    """把专家模式当前组合状态落到磁盘，切换页面后也能恢复。"""
    payload = {
        "fund_codes_input": st.session_state.get("fund_codes_input", ""),
        "portfolio_weights_preset": st.session_state.get("portfolio_weights_preset"),
        "engine_checkbox": bool(st.session_state.get("engine_checkbox", False)),
        "loaded_portfolio_name": st.session_state.get("loaded_portfolio_name"),
        "timing_radar_codes": st.session_state.get("timing_radar_codes"),
    }
    try:
        with open(PRO_WORKSPACE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if not silent:
            st.toast("专家模式工作区已保存", icon="💾")
    except Exception as e:
        st.warning(f"专家模式状态写入失败：{e}")

def ensure_portfolio_history_loaded():
    if "portfolio_history" not in st.session_state:
        st.session_state["portfolio_history"] = load_local_history()

def init_persistent_session_state():
    """跨模块切换时保持侧边栏与组合配置不丢失。"""
    ensure_portfolio_history_loaded()
    defaults = {
        "fund_codes_input": "005827, 005844",
        "fund_search_kw": "",
        "kyc_risk_text": "",
        "portfolio_weights_preset": None,
        "loaded_portfolio_name": None,
        "engine_checkbox": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 仅首次会话加载磁盘工作区；之后以内存/控件为准，由 save_pro_workspace 落盘
    if not st.session_state.get("_pro_workspace_loaded"):
        ws = load_pro_workspace()
        if ws:
            if ws.get("fund_codes_input"):
                st.session_state["fund_codes_input"] = ws["fund_codes_input"]
            if ws.get("portfolio_weights_preset") is not None:
                st.session_state["portfolio_weights_preset"] = ws["portfolio_weights_preset"]
            if "engine_checkbox" in ws:
                st.session_state["engine_checkbox"] = bool(ws["engine_checkbox"])
            if ws.get("loaded_portfolio_name") is not None:
                st.session_state["loaded_portfolio_name"] = ws["loaded_portfolio_name"]
            if ws.get("timing_radar_codes"):
                st.session_state["timing_radar_codes"] = ws["timing_radar_codes"]
        st.session_state["_pro_workspace_loaded"] = True

    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = load_chat_history()

def _clear_weight_widget_state():
    for k in list(st.session_state.keys()):
        if k.startswith(("w_", "bf_", "sf_")):
            del st.session_state[k]

def request_app_mode_switch(mode: str):
    """
    请求切换工作模式。
    - 只写 app_mode，并打上强制同步标记
    - main() 在创建 radio 之前才会写入控件 key（避免覆盖用户侧边栏点击）
    """
    st.session_state["app_mode"] = mode
    st.session_state["_force_mode_sync"] = True


def normalize_weights_pct(weights: dict) -> dict:
    raw = {str(c).strip(): float(w) for c, w in (weights or {}).items() if str(c).strip()}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {c: round(v / total * 100, 2) for c, v in raw.items()}

def inject_portfolio_snapshot(portfolio: dict, jump: bool = False, persist: bool = True) -> dict:
    """将方案注入专家模式工作区；可选立刻跳转。"""
    if not portfolio:
        return {"error": "方案为空"}
    weights = portfolio.get("weights") or portfolio.get("weights_pct") or {}
    codes = [str(c).strip() for c in (portfolio.get("codes") or list(weights.keys())) if str(c).strip()]
    if not codes:
        return {"error": "方案中没有基金代码"}
    preset = normalize_weights_pct(weights) if weights else {c: round(100.0 / len(codes), 2) for c in codes}
    if not preset:
        preset = {c: round(100.0 / len(codes), 2) for c in codes}

    st.session_state["fund_codes_input"] = ", ".join(codes)
    st.session_state["portfolio_weights_preset"] = preset
    st.session_state["timing_radar_codes"] = codes
    st.session_state["loaded_portfolio_name"] = portfolio.get("name") or "AI 方案"
    st.session_state["engine_checkbox"] = True
    _clear_weight_widget_state()
    st.session_state.pop("engine_sig", None)
    st.session_state.pop("fund_data_dict", None)
    st.session_state.pop("tab5_backtest", None)
    st.session_state.pop("tab5_weights_sig", None)
    st.session_state["auto_run_tab5_backtest"] = True
    if persist:
        save_pro_workspace()
    if jump:
        request_app_mode_switch("FOF 穿透与归因")
    return {
        "ok": True,
        "codes": codes,
        "weights_pct": preset,
        "name": st.session_state["loaded_portfolio_name"],
    }

def quick_save_named_portfolio(name: str, codes=None, weights_pct=None, metrics=None) -> dict:
    """把当前专家模式组合按名称归档到 portfolio_history。"""
    ensure_portfolio_history_loaded()
    name = str(name or "").strip()
    if not name:
        return {"error": "请填写组合名称"}
    codes = codes or [c.strip() for c in str(st.session_state.get("fund_codes_input", "")).split(",") if c.strip()]
    if not codes:
        return {"error": "没有可保存的基金代码"}
    weights_pct = weights_pct or st.session_state.get("portfolio_weights_preset")
    if not weights_pct:
        weights_pct = {c: round(100.0 / len(codes), 2) for c in codes}
    preset = normalize_weights_pct({c: weights_pct.get(c, 0) for c in codes})
    if not preset:
        return {"error": "权重无效"}
    # 历史库 signature 使用小数权重
    signature = {c: preset[c] / 100.0 for c in codes}
    metrics = metrics or {}
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_record = {
        "name": name,
        "signature": signature,
        "codes_str": " + ".join([f"{c}({preset[c]:.0f}%)" for c in codes]),
        "return": metrics.get("return", 0),
        "drawdown": metrics.get("drawdown", 0),
        "sharpe": metrics.get("sharpe", 0),
        "timestamp": current_time_str,
    }
    hist = st.session_state["portfolio_history"]
    existing_idx = next((i for i, x in enumerate(hist) if x.get("name") == name), None)
    if existing_idx is not None:
        hist.pop(existing_idx)
    hist.insert(0, new_record)
    save_local_history(hist)
    st.session_state["loaded_portfolio_name"] = name
    save_pro_workspace()
    return {"ok": True, "name": name}

def render_portfolio_jump_button(portfolio: dict, key: str):
    """对话气泡内的「用此方案打开专家模式」按钮。"""
    if not portfolio or not portfolio.get("codes") and not portfolio.get("weights"):
        return
    codes = portfolio.get("codes") or list((portfolio.get("weights") or {}).keys())
    label = portfolio.get("name") or "当前 AI 方案"
    st.info(f"附带方案快照：**{label}**（{len(codes)} 只基金）")
    if st.button("📊 用此方案打开专家模式", key=key, use_container_width=True):
        result = inject_portfolio_snapshot(portfolio, jump=True, persist=True)
        if result.get("error"):
            st.error(result["error"])
        else:
            st.rerun()



def apply_portfolio_from_history(item: dict, switch_to_fof: bool = True):
    """将历史组合载入侧边栏基金代码、权重预设，并可选跳转 FOF 看板。"""
    sig = item.get("signature") or {}
    if not sig:
        st.sidebar.error("该记录缺少配置明细，无法加载。")
        return
    codes = [str(c).strip() for c in sig.keys()]
    st.session_state["fund_codes_input"] = ", ".join(codes)
    st.session_state["portfolio_weights_preset"] = {
        str(c): round(float(w) * 100, 2) for c, w in sig.items()
    }
    st.session_state["timing_radar_codes"] = codes
    st.session_state["loaded_portfolio_name"] = item.get("name", "历史组合")
    st.session_state["engine_checkbox"] = True
    _clear_weight_widget_state()
    st.session_state.pop("engine_sig", None)
    st.session_state.pop("fund_data_dict", None)
    st.session_state.pop("tab5_backtest", None)
    st.session_state.pop("tab5_weights_sig", None)
    st.session_state["auto_run_tab5_backtest"] = True
    if switch_to_fof:
        request_app_mode_switch("FOF 穿透与归因")
    save_pro_workspace()
    st.toast(f"✅ 已加载组合：{item.get('name', '未命名')}", icon="📂")

def delete_portfolio_history_item(index: int) -> bool:
    """按索引删除一条历史组合，并落盘。"""
    ensure_portfolio_history_loaded()
    hist = st.session_state.get("portfolio_history") or []
    if index < 0 or index >= len(hist):
        return False
    removed = hist.pop(index)
    st.session_state["portfolio_history"] = hist
    save_local_history(hist)
    # 若删的是当前载入项，清标记
    if st.session_state.get("loaded_portfolio_name") == removed.get("name"):
        st.session_state["loaded_portfolio_name"] = None
    return True


def render_history_quick_load_sidebar(max_items: int = 8):
    """侧边栏历史组合一键快载 / 删除（全模块可见）。"""
    ensure_portfolio_history_loaded()
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 历史组合")
    history = st.session_state.get("portfolio_history", [])
    if not history:
        st.sidebar.caption("暂无记录。回测保存或快速命名后会出现在这里。")
        return

    loaded = st.session_state.get("loaded_portfolio_name")
    if loaded:
        st.sidebar.info(f"📂 当前载入：**{loaded}**")

    for idx, item in enumerate(history[:max_items]):
        name = item.get("name", f"组合{idx + 1}")
        short = name if len(name) <= 14 else name[:13] + "…"
        ret = item.get("return", 0)
        dd = item.get("drawdown", 0)
        help_txt = f"{item.get('codes_str', '')}\n收益 {ret:.1f}% · 回撤 {dd:.1f}%"
        b1, b2 = st.sidebar.columns([3, 1])
        with b1:
            if st.button(
                f"📂 {short}",
                key=f"sidebar_load_hist_{idx}",
                help=help_txt,
                use_container_width=True,
            ):
                apply_portfolio_from_history(item)
                st.rerun()
        with b2:
            if st.button(
                "🗑️",
                key=f"sidebar_del_hist_{idx}",
                help=f"删除「{name}」",
                use_container_width=True,
            ):
                if delete_portfolio_history_item(idx):
                    st.sidebar.toast(f"已删除：{name}", icon="🗑️")
                    st.rerun()

    if len(history) > max_items:
        st.sidebar.caption(f"共 {len(history)} 条，完整管理见专家模式「历史档案」或本页下方。")

# ==========================================
# ⚙️ 1. 全局唯一配置 (系统的“锚”)
# ==========================================
st.set_page_config(page_title="📊 FOF 机构级资产配置与归因看板", page_icon="📈", layout="wide")

if 'max_drawdown_limit' not in st.session_state:
    st.session_state['max_drawdown_limit'] = None

init_persistent_session_state()

# ==========================================
# 🧠 2. 数据获取与缓存层 (公共基础设施)
# ==========================================
@st.cache_data(ttl=3600)
def get_dynamic_risk_free_rate() -> float:
    try:
        bond_df = ak.bond_zh_us_rate()
        return float(bond_df["中国国债收益率10年"].dropna().iloc[-1]) / 100.0
    except Exception:
        return 0.02

@st.cache_data(ttl=3600)
def get_benchmark_data(refresh_token: int = 0):
    """沪深300基准：磁盘缓存优先，失败时不污染长期缓存（见 refresh_token）。"""
    cached = _load_benchmark_disk_cache()
    if cached is not None and not cached.empty:
        return cached

    for fetcher in (_fetch_benchmark_sina, _fetch_benchmark_em):
        try:
            df = fetcher()
            if df is not None and not df.empty:
                _save_benchmark_disk_cache(df)
                return df
        except Exception:
            continue
    return None

@st.cache_data(ttl=3600)
def get_fund_clean_data(fund_code: str):
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df['单位净值'] = df['单位净值'].astype(float)
        df['日增长率'] = df['日增长率'].astype(float) 
        return df[['净值日期', '单位净值', '日增长率']].sort_values(by='净值日期').reset_index(drop=True)
    except:
        return None

@st.cache_data(ttl=3600)
def get_fund_manager_info(fund_code: str):
    try:
        url = f"http://fundf10.eastmoney.com/jjjl_{fund_code}.html"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        res.encoding = 'utf-8'
        tables = pd.read_html(res.text)
        for df in tables:
            cols = [str(c) for c in df.columns]
            start_col = next((c for c in cols if '起始' in c), None)
            end_col = next((c for c in cols if '截止' in c or '结束' in c), None)
            name_col = next((c for c in cols if '姓名' in c or '经理' in c), None)
            if start_col and end_col and name_col:
                current = df[df[end_col].str.contains('至今', na=False)]
                if not current.empty:
                    return {"name": current.iloc[0][name_col], "start_date": current.iloc[0][start_col]}
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def get_fund_portfolio(fund_code: str):
    year = datetime.datetime.now().year
    for y in [str(year), str(year - 1), str(year - 2)]:
        try:
            df = ak.fund_portfolio_hold_em(symbol=fund_code, date=y)
            if df is not None and not df.empty:
                time_col = next((col for col in ['季度', '截止日期', '报告期', '日期'] if col in df.columns), None)
                if time_col:
                    latest = df[time_col].iloc[0]
                    top10 = df[df[time_col] == latest].head(10).copy()
                else:
                    top10 = df.head(10).copy()
                ratio_col = next(
                    (c for c in top10.columns if str(c).startswith("占净值比例")),
                    None,
                )
                name_col = next((c for c in top10.columns if "股票名称" in str(c) or str(c) == "证券名称"), None)
                code_col = next((c for c in top10.columns if "股票代码" in str(c) or "证券代码" in str(c)), None)
                if not name_col or not ratio_col:
                    continue
                out = pd.DataFrame({
                    "股票名称": top10[name_col].astype(str),
                    "占净值比例": pd.to_numeric(top10[ratio_col], errors="coerce"),
                })
                if code_col:
                    codes_raw = top10[code_col].apply(_normalize_stock_code)
                    out.insert(0, "股票代码", codes_raw)
                else:
                    out.insert(0, "股票代码", "")
                return out
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fund_industry(fund_code: str):
    year = datetime.datetime.now().year
    for y in [str(year), str(year - 1), str(year - 2)]:
        try:
            df = ak.fund_portfolio_industry_allocation_em(symbol=fund_code, date=y)
            if df is not None and not df.empty:
                time_col = next((col for col in ['季度', '截止日期', '报告期', '日期'] if col in df.columns), None)
                if time_col:
                    latest = df[time_col].iloc[0]
                    ind_df = df[df[time_col] == latest]
                else:
                    ind_df = df
                ratio_col = '占净值比例' if '占净值比例' in ind_df.columns else '占净值比例(%)'
                return ind_df.rename(columns={ratio_col: '占净值比例'})[['行业类别', '占净值比例']]
        except:
            continue
    return pd.DataFrame()

STYLE_INDEX_SYMBOLS = {
    "🔵 大盘白马 (沪深300)": "sh000300",
    "🟡 中小盘成长 (中证500)": "sh000905",
    "🔴 科技硬核 (创业板指)": "sz399006",
    "🛡️ 深度防守 (中证红利)": "sh000922",
    "🍷 消费信仰 (中证消费)": "sz399932",
    "💊 医药医疗 (中证医疗)": "sz399989",
    "💰 纯债固收 (上证国债)": "sh000012",
    "🥇 黄金代理 (黄金股指)": "sz399481",
}

# 季报行业名 → 风格代理指数（用于左右侧结构，不是个股实时 tick）
_INDUSTRY_STYLE_RULES = [
    (["医药", "医疗", "生物", "健康", "制药"], "💊 医药医疗 (中证医疗)"),
    (["消费", "食品", "饮料", "白酒", "家电", "零售", "社会服务"], "🍷 消费信仰 (中证消费)"),
    (["电子", "半导体", "计算机", "通信", "传媒", "科技", "软件", "互联网"], "🔴 科技硬核 (创业板指)"),
    (["银行", "非银", "保险", "证券", "金融"], "🔵 大盘白马 (沪深300)"),
    (["煤炭", "石油", "石化", "红利", "公用", "交运", "交通运输"], "🛡️ 深度防守 (中证红利)"),
    (["有色", "钢铁", "机械", "制造", "军工", "电力设备", "新能源"], "🟡 中小盘成长 (中证500)"),
    (["黄金", "贵金属"], "🥇 黄金代理 (黄金股指)"),
    (["债券", "固收", "利率", "信用", "可转债"], "💰 纯债固收 (上证国债)"),
]


@st.cache_data(ttl=3600)
def get_style_benchmarks():
    """RBS 风格雷达底座：全市场多维度风格基准库（日收益率）。"""
    merged_styles = pd.DataFrame()
    for name, symbol in STYLE_INDEX_SYMBOLS.items():
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"date": "净值日期", "close": name})
            df = df[["净值日期", name]].set_index("净值日期")
            df[name] = df[name].pct_change().fillna(0)
            if merged_styles.empty:
                merged_styles = df
            else:
                merged_styles = pd.merge(
                    merged_styles, df, left_index=True, right_index=True, how="outer"
                )
        except Exception:
            continue
    return merged_styles.dropna()


@st.cache_data(ttl=3600)
def get_style_index_levels() -> pd.DataFrame:
    """风格指数收盘价序列（用于计算真实涨跌与 MA 结构）。"""
    merged = pd.DataFrame()
    for name, symbol in STYLE_INDEX_SYMBOLS.items():
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"date": "净值日期", "close": name})
            df = df[["净值日期", name]].set_index("净值日期")
            if merged.empty:
                merged = df
            else:
                merged = pd.merge(merged, df, left_index=True, right_index=True, how="outer")
        except Exception:
            continue
    return merged.dropna(how="all")


def _index_trend_from_close(close: pd.Series) -> dict:
    """用真实收盘价计算涨跌与相对 MA30 的左右侧结构。"""
    s = close.dropna().astype(float)
    if len(s) < 35:
        return {}
    last = float(s.iloc[-1])
    ma30 = s.rolling(30, min_periods=30).mean()
    ma_now = float(ma30.iloc[-1])
    if pd.isna(ma_now) or ma_now <= 0:
        return {}

    def _ret(n: int):
        if len(s) > n:
            return round((last / float(s.iloc[-(n + 1)]) - 1) * 100, 2)
        return None

    ma_up = bool(ma30.iloc[-1] > ma30.iloc[-6]) if len(ma30.dropna()) >= 6 else False
    above = last > ma_now
    if above and ma_up:
        side = "偏右侧（价>MA30 且均线上行）"
    elif above:
        side = "震荡偏强（价>MA30，均线走平/下）"
    elif ma_up:
        side = "左侧/修复早期（价<MA30，均线仍上）"
    else:
        side = "偏左侧/弱势（价<MA30 且均线走弱）"

    return {
        "近5日%": _ret(5),
        "近20日%": _ret(20),
        "近60日%": _ret(60),
        "相对MA30%": round((last / ma_now - 1) * 100, 2),
        "结构判定": side,
    }


def match_industry_to_style(industry_name: str) -> str:
    text = str(industry_name or "")
    for keys, style in _INDUSTRY_STYLE_RULES:
        if any(k in text for k in keys):
            return style
    return "🔵 大盘白马 (沪深300)"


def build_sector_proxy_snapshot(industry_df: pd.DataFrame | None, top_n: int = 5) -> list[dict]:
    """
    把重仓行业映射到可交易风格指数，给出可核查的涨跌/MA结构。
    说明：这是「行业代理指数」实况，不是单只股票 tick。
    """
    levels = get_style_index_levels()
    rows: list[dict] = []
    if industry_df is not None and not industry_df.empty and "行业类别" in industry_df.columns:
        view = industry_df.head(top_n)
        for _, r in view.iterrows():
            ind = str(r.get("行业类别", ""))
            style = match_industry_to_style(ind)
            item = {
                "重仓行业": ind,
                "占净值比例": r.get("占净值比例", "-"),
                "代理指数": style,
                "数据说明": "行业→风格指数代理，非个股实时行情",
            }
            if style in levels.columns:
                item.update(_index_trend_from_close(levels[style]))
            else:
                item["结构判定"] = "代理指数暂无数据"
            rows.append(item)
    if not rows:
        # 无行业穿透时，至少给宽基实况，避免模型瞎编
        for style in ["🔵 大盘白马 (沪深300)", "🔴 科技硬核 (创业板指)", "💰 纯债固收 (上证国债)"]:
            if style not in levels.columns:
                continue
            item = {"重仓行业": "(无季报行业)", "代理指数": style, "数据说明": "宽基代理兜底"}
            item.update(_index_trend_from_close(levels[style]))
            rows.append(item)
    return rows


def _normalize_stock_code(raw) -> str:
    """清洗季报股票代码：600519.0 / 600519.SH / 00700 → 可拉行情格式。"""
    s = str(raw or "").strip().upper()
    if not s or s in ("NAN", "NONE", "-"):
        return ""
    # 去掉浮点尾巴、交易所后缀
    s = s.replace(".HK", "").replace(".SH", "").replace(".SZ", "")
    if "." in s:
        left, right = s.split(".", 1)
        if right.isdigit() and len(right) <= 2 and left.isdigit():
            s = left  # 600519.0 → 600519
        elif left.isdigit():
            s = left
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return ""
    if len(digits) <= 5:
        return digits.zfill(5)
    if len(digits) >= 6:
        return digits[-6:]  # 防 6005190 这类脏数据
    return digits


def classify_stock_symbol(raw_code: str) -> tuple[str, str]:
    """识别市场：返回 (market, code)，market in {'a','hk','unknown'}。"""
    code = _normalize_stock_code(raw_code)
    if not code:
        return "unknown", ""
    if len(code) <= 5:
        return "hk", code.zfill(5)
    if len(code) == 6:
        return "a", code
    return "unknown", code


def _pick_col(df: pd.DataFrame, names: tuple[str, ...], fallback_idx: int | None = None):
    for c in df.columns:
        cs = str(c).lower()
        if c in names or cs in {n.lower() for n in names}:
            return c
    if fallback_idx is not None and len(df.columns) > fallback_idx:
        return df.columns[fallback_idx]
    return None


def _pick_ohlc_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close_col = _pick_col(df, ("收盘", "close", "Close"), fallback_idx=3)
    date_col = _pick_col(df, ("日期", "date", "Date"), fallback_idx=0)
    if close_col is None:
        return pd.Series(dtype=float)
    closes = pd.to_numeric(df[close_col], errors="coerce")
    if date_col is not None:
        idx = pd.to_datetime(df[date_col], errors="coerce")
        ser = pd.Series(closes.values, index=idx).dropna()
        ser = ser[~ser.index.isna()].sort_index()
        return ser
    return closes.dropna()


def _normalize_ohlc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """统一为 date/open/high/low/close 列，按日期升序。"""
    if df is None or df.empty:
        return pd.DataFrame()
    date_col = _pick_col(df, ("日期", "date", "Date"), fallback_idx=0)
    open_col = _pick_col(df, ("开盘", "open", "Open"), fallback_idx=1)
    # 东财 hist 常见：日期,代码,开盘,收盘,最高,最低
    close_col = _pick_col(df, ("收盘", "close", "Close"), fallback_idx=3)
    high_col = _pick_col(df, ("最高", "high", "High"), fallback_idx=4)
    low_col = _pick_col(df, ("最低", "low", "Low"), fallback_idx=5)
    # sina daily: date,open,high,low,close
    if high_col is None:
        high_col = _pick_col(df, ("最高", "high", "High"), fallback_idx=2)
    if low_col is None:
        low_col = _pick_col(df, ("最低", "low", "Low"), fallback_idx=3)
    if close_col is None:
        close_col = _pick_col(df, ("收盘", "close", "Close"), fallback_idx=4)
    need = [date_col, open_col, high_col, low_col, close_col]
    if any(c is None for c in need):
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "open": pd.to_numeric(df[open_col], errors="coerce"),
        "high": pd.to_numeric(df[high_col], errors="coerce"),
        "low": pd.to_numeric(df[low_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
    }).dropna()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def detect_stock_candle_signals(ohlc: pd.DataFrame, lookback: int = 8) -> dict:
    """
    基于重仓股真实 OHLC 的逃顶/企稳形态（非基金净值）。
    - 长上影线后接连阴：高优先级逃顶
    - 长下影线后突破 MA20/MA30 两连阳：右侧企稳买入
    """
    if ohlc is None or ohlc.empty or len(ohlc) < 35:
        return {"形态信号": "样本不足", "逃顶信号": False, "企稳买入信号": False}
    df = ohlc.tail(max(lookback + 5, 40)).copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_ratio = (upper / rng).fillna(0)
    lower_ratio = (lower / rng).fillna(0)
    is_yin = c < o
    is_yang = c > o

    long_upper = (upper_ratio >= 0.55) & (upper > body * 1.2)
    long_lower = (lower_ratio >= 0.55) & (lower > body * 1.2)

    # 近 lookback 内出现长上影，且之后出现至少 2 根阴线
    escape = False
    escape_detail = ""
    recent = df.tail(lookback).copy()
    lu_idx = recent.index[long_upper.reindex(recent.index).fillna(False)]
    if len(lu_idx) > 0:
        last_lu = int(lu_idx[-1])
        after = df.loc[last_lu + 1 :]
        yin_after = int(is_yin.reindex(after.index).fillna(False).sum()) if len(after) else 0
        # 长上影后连续下跌（至少两连阴，或阴线>=2）
        consec = 0
        for i in after.index:
            if bool(is_yin.loc[i]):
                consec += 1
            else:
                break
        if yin_after >= 2 or consec >= 2:
            escape = True
            escape_detail = f"长上影后阴线{yin_after}根（连续阴{consec}）"

    # 企稳：近 10 日有长下影，且最新收盘 > MA20 与 MA30，且近两日连阳
    ma20 = c.rolling(20, min_periods=20).mean()
    ma30 = c.rolling(30, min_periods=30).mean()
    stabilize = False
    stab_detail = ""
    tail10 = df.tail(10)
    has_ll = bool(long_lower.reindex(tail10.index).fillna(False).any())
    if has_ll and len(df) >= 2:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        ma20_v, ma30_v = float(ma20.iloc[-1]), float(ma30.iloc[-1])
        if (
            not pd.isna(ma20_v) and not pd.isna(ma30_v)
            and float(last["close"]) > ma20_v and float(last["close"]) > ma30_v
            and bool(is_yang.iloc[-1]) and bool(is_yang.iloc[-2])
        ):
            stabilize = True
            stab_detail = "长下影后收盘站上MA20/MA30且两连阳"

    # 最近一根影线提示
    last_ur = float(upper_ratio.iloc[-1])
    last_lr = float(lower_ratio.iloc[-1])
    tip = []
    if last_ur >= 0.55:
        tip.append("最新长上影")
    if last_lr >= 0.55:
        tip.append("最新长下影")
    if escape:
        tip.append("逃顶形态")
    if stabilize:
        tip.append("右侧企稳")

    return {
        "形态信号": "；".join(tip) if tip else "无明显影线形态",
        "逃顶信号": escape,
        "企稳买入信号": stabilize,
        "逃顶细节": escape_detail or "-",
        "企稳细节": stab_detail or "-",
        "最新上影占比": round(last_ur, 2),
        "最新下影占比": round(last_lr, 2),
        "近2日阴阳": ("阳" if bool(is_yang.iloc[-2]) else "阴") + ("阳" if bool(is_yang.iloc[-1]) else "阴"),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_close_series(symbol: str) -> pd.Series:
    """拉取单股前复权收盘价（稳定版：直连行情 API，不经过 OHLC 解析链）。"""
    market, code = classify_stock_symbol(symbol)
    if not code or market == "unknown":
        return pd.Series(dtype=float)
    try:
        if market == "hk":
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            return _pick_ohlc_close_series(df)

        prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
        try:
            df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
            ser = _pick_ohlc_close_series(df)
            if not ser.empty:
                return ser
        except Exception:
            pass
        start = (datetime.date.today() - datetime.timedelta(days=420)).strftime("%Y%m%d")
        end = datetime.date.today().strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"
        )
        return _pick_ohlc_close_series(df)
    except Exception:
        return pd.Series(dtype=float)


def build_holdings_trend_snapshot(portfolio_df: pd.DataFrame | None, top_n: int = 8) -> list[dict]:
    """
    重仓股单股实盘趋势（近5/20/60日涨跌 + 相对MA30结构）。
    稳定版：仅收盘价趋势，不依赖 OHLC/影线链路。
    """
    if portfolio_df is None or portfolio_df.empty:
        return []
    view = portfolio_df.head(top_n)
    rows: list[dict] = []
    for _, r in view.iterrows():
        name = str(r.get("股票名称", "") or "")
        raw_code = _normalize_stock_code(r.get("股票代码", ""))
        ratio = r.get("占净值比例", "-")
        item = {
            "股票名称": name,
            "股票代码": raw_code or "-",
            "占净值比例": ratio,
        }
        if not raw_code:
            item["结构判定"] = "无股票代码"
            item["数据说明"] = "季报缺代码"
            rows.append(item)
            continue
        market, _norm = classify_stock_symbol(raw_code)
        item["市场"] = "港股" if market == "hk" else ("A股" if market == "a" else "未知")
        closes = fetch_stock_close_series(raw_code)
        if closes is None or closes.empty:
            item["结构判定"] = "行情拉取失败"
            item["数据说明"] = "单股行情暂不可用"
            rows.append(item)
            continue
        trend = _index_trend_from_close(closes)
        if not trend:
            item["结构判定"] = "样本不足"
            item["数据说明"] = "单股历史过短"
        else:
            item.update(trend)
            item["数据说明"] = "单股前复权收盘推导"
            item["最新收盘"] = round(float(closes.iloc[-1]), 4)
        rows.append(item)
    return rows


def _holding_weight(row: dict) -> float:
    try:
        w = float(str(row.get("占净值比例", 0)).replace("%", "").strip() or 0)
    except (TypeError, ValueError):
        w = 0.0
    return max(0.0, w)


def _side_score(structure: str) -> float:
    """结构判定 → 方向分。"""
    s = str(structure or "")
    if "偏右侧" in s:
        return 1.0
    if "震荡偏强" in s:
        return 0.55
    if "左侧/修复" in s or "修复早期" in s:
        return -0.35
    if "偏左侧" in s or "弱势" in s:
        return -1.0
    return 0.0


def _holding_row_failed(side: str) -> bool:
    s = str(side or "").strip()
    if not s:
        return True
    return any(m in s for m in ("拉取失败", "无法拉", "无股票代码", "样本不足", "暂无数据"))


def summarize_holdings_breadth(holdings_proxy: list[dict]) -> dict:
    """按占净值比例加权汇总重仓股多空（大权重个股主导结论）。"""
    rows = holdings_proxy or []
    right_n = left_n = fail_n = 0
    w_right = w_left = w_total = 0.0
    ret20_wsum = ret20_w = 0.0
    leaders = []
    for h in rows:
        side = str(h.get("结构判定", ""))
        w = _holding_weight(h)
        if _holding_row_failed(side):
            fail_n += 1
            continue
        score = _side_score(side)
        if score == 0.0:
            w_total += w
            continue
        w_total += w
        if score > 0:
            right_n += 1
            w_right += w * score
        else:
            left_n += 1
            w_left += w * abs(score)

        r20 = h.get("近20日%")
        try:
            if r20 is not None and r20 != "-":
                ret20_wsum += float(r20) * w
                ret20_w += w
        except (TypeError, ValueError):
            pass
        leaders.append({
            "股票名称": h.get("股票名称"),
            "占净值比例": w,
            "近20日%": r20,
            "结构判定": side,
        })

    leaders = sorted(leaders, key=lambda x: float(x.get("占净值比例") or 0), reverse=True)[:5]
    usable = right_n + left_n
    net = w_right - w_left
    denom = max(w_right + w_left, 1e-6)
    strength = round(net / denom, 3)

    if not rows:
        conclusion = "无股票持仓（债/货基等）"
    elif usable <= 0:
        conclusion = "重仓股行情暂不可用"
    elif strength >= 0.25:
        conclusion = "重仓股（权重加权）偏强"
    elif strength <= -0.25:
        conclusion = "重仓股（权重加权）偏弱"
    else:
        conclusion = "重仓股（权重加权）分化"

    avg20 = round(ret20_wsum / ret20_w, 2) if ret20_w > 0 else None
    return {
        "样本数": len(rows),
        "有效样本": usable,
        "偏强/右侧只数": right_n,
        "偏弱/左侧只数": left_n,
        "强势权重分": round(w_right, 2),
        "弱势权重分": round(w_left, 2),
        "权重净强度": strength,
        "加权近20日%": avg20,
        "前五大有效重仓": leaders,
        "广度结论": conclusion,
        "主判据说明": "按本基金占净值比例加权，大权重个股占主导",
    }


def build_directional_stance_hint(
    holdings_breadth: dict | None,
    sector_proxy: list | None,
    engine_advice: dict | None,
    is_passive: bool,
    holdings_proxy: list | None = None,
) -> dict:
    """
    默认立场：本基金重仓股（按占净值比例）为主判据，行业代理仅辅助，净值引擎再次之。
    """
    hb = holdings_breadth or {}
    # 主判据：权重净强度（已按占净值比例）
    hold_strength = hb.get("权重净强度")
    try:
        hold_strength = float(hold_strength) if hold_strength is not None else 0.0
    except (TypeError, ValueError):
        hold_strength = 0.0
    usable = int(hb.get("有效样本") or 0)
    avg20 = hb.get("加权近20日%")

    # 辅助：行业代理（等权，系数很小）
    sec_score = 0.0
    sec_right = sec_left = 0
    for s in sector_proxy or []:
        sc = _side_score(str(s.get("结构判定", "")))
        if sc > 0:
            sec_right += 1
        elif sc < 0:
            sec_left += 1
        sec_score += sc
    if (sec_right + sec_left) > 0:
        sec_score = sec_score / (sec_right + sec_left)

    # 被动基金：引擎可抬到中等权重；主动：引擎几乎不主导
    engine_txt = str((engine_advice or {}).get("建议", ""))
    eng = 0.0
    if any(k in engine_txt for k in ("买入", "加仓")):
        eng = 1.0
    elif any(k in engine_txt for k in ("减仓", "清仓", "防守", "空仓")):
        eng = -1.0
    elif "关注" in engine_txt or "边界" in engine_txt:
        eng = -0.3

    if usable > 0:
        # 主:辅:引擎 ≈ 0.75 : 0.15 : 0.10（主动）；被动引擎提到 0.35
        if is_passive:
            score = 0.50 * hold_strength + 0.15 * sec_score + 0.35 * eng
            weights_txt = "被动权重 重仓股50% / 行业15% / 引擎35%"
        else:
            score = 0.75 * hold_strength + 0.15 * sec_score + 0.10 * eng
            weights_txt = "主动权重 重仓股75% / 行业15% / 净值引擎10%"
    else:
        # 无持仓行情（如纯债）：退回行业+引擎
        if is_passive:
            score = 0.35 * sec_score + 0.65 * eng
        else:
            score = 0.70 * sec_score + 0.30 * eng
        weights_txt = "无有效重仓行情，暂用行业代理+引擎兜底"

    # 加权近20日进一步倾斜（仍属重仓股主判据）
    if isinstance(avg20, (int, float)):
        if avg20 >= 3:
            score += 0.15
        elif avg20 <= -3:
            score -= 0.15

    top = (hb.get("前五大有效重仓") or [])[:3]
    top_txt = "、".join(
        f"{t.get('股票名称')}({t.get('占净值比例')}%,{t.get('结构判定','-')},20日{t.get('近20日%','-')}%)"
        for t in top
    ) or "无"

    if usable > 0 and score >= 0.22:
        stance = "买入" if score >= 0.45 else "分批买入"
        reason = f"主判据=本基金重仓股权重偏强（净强度{hold_strength:.2f}；代表：{top_txt}）。行业仅辅助确认。"
        delta_hint = "+3~+8" if stance == "买入" else "+2~+5"
    elif usable > 0 and score <= -0.22:
        stance = "减仓止盈" if score <= -0.45 else "暂不操作"
        reason = f"主判据=本基金重仓股权重偏弱（净强度{hold_strength:.2f}；代表：{top_txt}）。应防守，行业仅辅助。"
        delta_hint = "-4~-12（可到0）" if "减仓" in stance else "-1~-5"
    elif usable > 0:
        wr = float(hb.get("强势权重分") or 0)
        wl = float(hb.get("弱势权重分") or 0)
        if wr >= wl * 1.15:
            stance = "分批买入"
            reason = f"结构分化但大权重偏强（代表：{top_txt}），主判据支持谨慎加仓"
            delta_hint = "+1~+4"
        elif wl >= wr * 1.15:
            stance = "暂不操作"
            reason = f"结构分化但大权重偏弱（代表：{top_txt}），主判据支持先不加/小减"
            delta_hint = "-1~-5"
        else:
            stance = "暂不操作"
            reason = f"大权重多空接近（代表：{top_txt}），明确暂不操作，禁止持有观望"
            delta_hint = "0"
    elif not (holdings_proxy or []):
        weights_txt = "无股票穿透（债/货基/指数），用行业代理+引擎"
        reason = "该基金季报无股票持仓，请依据行业代理与净值引擎研判，勿写「重仓股数据缺失」"
        stance = "分批买入" if score >= 0.25 else ("减仓止盈" if score <= -0.35 else "暂不操作")
        delta_hint = "+1~+3" if score >= 0.25 else ("-3~-8" if score <= -0.35 else "0")
    elif score >= 0.25:
        stance = "分批买入"
        reason = "有持仓名单但行情暂不可用，仅按行业/引擎谨慎分批"
        delta_hint = "+1~+3"
    elif score <= -0.25:
        stance = "暂不操作"
        reason = "有持仓名单但行情暂不可用，辅助信号偏弱，暂不操作"
        delta_hint = "0~-3"
    else:
        stance = "暂不操作"
        reason = "重仓股行情暂不可用，先看行业代理与引擎"
        delta_hint = "0"

    return {
        "默认决断": stance,
        "打分": round(score, 3),
        "权重方案": weights_txt,
        "主判据": "本基金重仓股（按占净值比例加权）整体趋势",
        "辅助判据": "行业代理指数",
        "理由": reason,
        "建议变动区间pt": delta_hint,
        "重仓股净值强度": hold_strength,
        "重仓股": f"强{hb.get('偏强/右侧只数', 0)}/弱{hb.get('偏弱/左侧只数', 0)}/有效{usable}",
        "行业代理": f"强{sec_right}/弱{sec_left}",
        "加权近20日%": avg20,
        "代表重仓": top_txt,
    }


def enforce_decisive_verdict(ai_result: dict, stance_hint: dict | None) -> dict:
    """若模型仍吐出「持有观望」，按系统默认立场纠偏（保留其文字解释）。"""
    out = dict(ai_result or {})
    verdict = str(out.get("综合决断", "")).strip()
    # 已有明确买卖/暂停方向则不改
    decisive_ok = any(w in verdict for w in ("买入", "加仓", "减仓", "止盈", "暂不", "不宜", "卖出"))
    if decisive_ok and "观望" not in verdict:
        return out

    hint = (stance_hint or {}).get("默认决断")
    if not hint:
        return out

    soft = ("观望" in verdict) or verdict in ("", "-", "需人工阅读", "持有")
    if soft:
        out["综合决断"] = hint
        note = str((stance_hint or {}).get("理由", "") or "")
        old_assess = str(out.get("买卖点评估", "") or "")
        if note and note not in old_assess:
            out["买卖点评估"] = (old_assess + "；" if old_assess else "") + f"【系统立场纠偏】{note}"
        one = str(out.get("总监一句话", "") or "")
        if "观望" in one or not one or one == "-":
            out["总监一句话"] = f"系统纠偏为{hint}"
    return out


def resolve_baseline_weight_pct(code: str, selected_codes: list[str] | None = None) -> float | None:
    """读取 AI/手动注入的组合初始仓位（%）。无预设时按已选标的等权估算。"""
    preset = st.session_state.get("portfolio_weights_preset") or {}
    code = str(code).strip()
    if code in preset:
        try:
            return round(float(preset[code]), 2)
        except (TypeError, ValueError):
            pass
    codes = [str(c).strip() for c in (selected_codes or []) if str(c).strip()]
    if codes:
        return round(100.0 / len(codes), 2)
    return None


def apply_timing_position_math(ai_result: dict, baseline_pct: float | None) -> dict:
    """把 AI 数字仓位与初始仓位对齐：初始 → 变动 → 变动后；允许单腿降到 0（空仓该腿）。"""
    out = dict(ai_result or {})
    base = None if baseline_pct is None else float(baseline_pct)

    def _to_float(v):
        if v is None or v == "" or v == "-":
            return None
        try:
            return float(str(v).replace("%", "").strip())
        except (TypeError, ValueError):
            return None

    target = _to_float(out.get("建议目标仓位%"))
    delta = _to_float(out.get("建议仓位变动百分点"))

    if base is not None:
        if target is None and delta is not None:
            target = base + delta
        verdict = str(out.get("综合决断", ""))
        # 若决断已方向化但仓位几乎不动，按决断补幅度（避免全员观望后变“名义买入、仓位不变”）
        if target is not None and abs(float(target) - base) < 0.5:
            if any(w in verdict for w in ("买入", "加仓")) and "分批" not in verdict:
                target = min(100.0, base + 5.0)
            elif "分批" in verdict:
                target = min(100.0, base + 3.0)
            elif any(w in verdict for w in ("减仓", "止盈")):
                target = max(0.0, base - 5.0)
            elif "暂不" in verdict or "不宜" in verdict:
                target = base
        if target is None:
            if any(w in verdict for w in ("买入", "加仓")) and "分批" not in verdict:
                target = min(100.0, base + 5.0)
            elif "分批" in verdict:
                target = min(100.0, base + 3.0)
            elif any(w in verdict for w in ("减仓", "止盈")):
                target = max(0.0, base - 5.0)
            elif "暂不" in verdict or "不宜" in verdict:
                target = base
            else:
                target = base
        target = max(0.0, min(100.0, float(target)))
        delta = round(target - base, 2)
        out["初始仓位%"] = round(base, 2)
        out["建议仓位变动百分点"] = delta
        out["建议目标仓位%"] = round(target, 2)
        out["变动后仓位%"] = round(target, 2)
    else:
        if target is not None:
            target = max(0.0, min(100.0, float(target)))
            out["建议目标仓位%"] = round(target, 2)
            out["变动后仓位%"] = round(target, 2)
        out["初始仓位%"] = None
        out["建议仓位变动百分点"] = delta
    return out

def calculate_risk_metrics(df: pd.DataFrame, rf_rate: float) -> dict:
    if df.empty: return {}
    daily_ret = df['日增长率'] / 100.0
    days = len(df)
    total_ret = df['单位净值'].iloc[-1] / df['单位净值'].iloc[0]
    
    ann_ret_plat = (total_ret ** (250 / days)) - 1
    vol_plat = daily_ret.std() * np.sqrt(250)
    sharpe_plat = (ann_ret_plat - 0.015) / vol_plat if vol_plat != 0 else 0
    
    ann_ret_real = (total_ret ** (252 / days)) - 1
    vol_real = daily_ret.std() * np.sqrt(252)
    sharpe_real = (ann_ret_real - rf_rate) / vol_real if vol_real != 0 else 0
    
    df_copy = df.copy()
    df_copy['max_here'] = df_copy['单位净值'].cummax()
    df_copy['drawdown'] = (df_copy['单位净值'] - df_copy['max_here']) / df_copy['max_here']
    max_dd = df_copy['drawdown'].min()
    
    return {
        "总收益率": (total_ret - 1) * 100,      
        "最大回撤": max_dd * 100,
        "平台夏普": sharpe_plat,
        "真实夏普": sharpe_real
    }

def resolve_fund_fees(fund_name: str, default_buy: float, default_sell: float) -> tuple[float, float]:
    """根据基金类型智能匹配买卖费率，未识别则沿用用户输入。"""
    if not fund_name:
        return default_buy, default_sell
    name_upper = fund_name.upper()
    if any(k in name_upper for k in ["ETF", "指数"]):
        return 0.0001, 0.0001
    if name_upper.endswith("C"):
        return 0.0, 0.0
    if name_upper.endswith("A"):
        return 0.0012, 0.0050
    return default_buy, default_sell

def is_passive_fund(fund_name: str) -> bool:
    """被动/指数类基金才适用均线择时引擎。"""
    if not fund_name:
        return False
    name_upper = fund_name.upper()
    return any(k in name_upper for k in ["ETF", "指数", "联接"])

def compute_rsi(nav: pd.Series, period: int = 14) -> pd.Series:
    """Wilder 平滑 RSI。"""
    delta = nav.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_timing_signals(df_nav: pd.DataFrame, is_passive: bool) -> pd.DataFrame:
    """
    高敏捷双核择时：
    - MA30 判定战略趋势
    - 20 日 SMA + 2σ 布林带（均值回归）
    - 14 日 RSI 动量过滤
    主动基金强制满仓，不做择时。
    """
    if '净值日期' in df_nav.columns:
        work = df_nav.set_index('净值日期')
    else:
        work = df_nav.copy()

    nav = work['单位净值'].astype(float)
    result = work[['单位净值']].copy()

    if not is_passive:
        result['Signal'] = 1.0
        result['目标仓位'] = 1.0
        result['仓位变化'] = 0.0
        return result

    ma20 = nav.rolling(window=20, min_periods=20).mean()
    ma30 = nav.rolling(window=30, min_periods=30).mean()
    bb_std = nav.rolling(window=20, min_periods=20).std()
    bb_mid = ma20
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    rsi = compute_rsi(nav, period=14)
    ma30_up = ma30 > ma30.shift(1)

    result['MA20'] = ma20
    result['MA30'] = ma30
    result['BB_upper'] = bb_upper
    result['BB_lower'] = bb_lower
    result['RSI'] = rsi

    signals = []
    current = 1.0
    warmup = max(30, 20, 14)

    for i in range(len(result)):
        if i < warmup or pd.isna(bb_lower.iloc[i]) or pd.isna(rsi.iloc[i]):
            signals.append(current)
            continue

        price = nav.iloc[i]
        proposed = current
        oversold = rsi.iloc[i] < 30
        overbought = rsi.iloc[i] > 70
        trend_up = bool(ma30_up.iloc[i])
        touch_lower = price <= bb_lower.iloc[i]
        touch_upper = price >= bb_upper.iloc[i]
        break_ma = (price < ma20.iloc[i]) or (price < ma30.iloc[i])

        if trend_up and touch_lower:
            proposed = 1.0
        elif touch_upper and overbought:
            proposed = 0.5
        elif break_ma and not oversold and not trend_up:
            proposed = 0.0

        if oversold and break_ma:
            proposed = max(proposed, current)

        current = proposed
        signals.append(current)

    result['Signal'] = signals
    result['目标仓位'] = pd.Series(signals, index=result.index).shift(1).fillna(1.0)
    result['仓位变化'] = result['目标仓位'].diff().fillna(0.0)
    result['ATR20'] = nav.diff().abs().rolling(window=20, min_periods=20).mean()
    return result

def _build_signal_reason(
    trend_up: bool, touch_lower: bool, touch_upper: bool,
    overbought: bool, oversold: bool, break_ma: bool, proposed: float, current: float,
) -> str:
    if oversold and break_ma and proposed >= current:
        return "超卖拦截：跌破均线但 RSI<30，禁止杀跌，锁仓等反弹"
    if trend_up and touch_lower:
        return "买入逻辑：MA30 趋势向上 + 净值触及布林下轨（超跌均值回归）"
    if touch_upper and overbought:
        return "减仓逻辑：触及布林上轨 + RSI>70（超买止盈）"
    if break_ma and not oversold and not trend_up:
        return "防守逻辑：跌破 MA20/MA30 且非超卖区，趋势转弱减仓"
    return "持有观望：未触发明确买卖条件"

def compute_timing_signals_v2(df_nav: pd.DataFrame, is_passive: bool) -> pd.DataFrame:
    """V2.0 完整版：指标 + 触发原因 + 理论事件标记（供实况雷达使用）。"""
    if '净值日期' in df_nav.columns:
        work = df_nav.set_index('净值日期')
    else:
        work = df_nav.copy()
    nav = work['单位净值'].astype(float)

    ma20 = nav.rolling(window=20, min_periods=20).mean()
    ma30 = nav.rolling(window=30, min_periods=30).mean()
    bb_std = nav.rolling(window=20, min_periods=20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    rsi = compute_rsi(nav, period=14)

    if not is_passive:
        base = work[['单位净值']].copy()
        base['MA20'] = ma20
        base['MA30'] = ma30
        base['BB_upper'] = bb_upper
        base['BB_lower'] = bb_lower
        base['RSI'] = rsi
        base['ATR20'] = nav.diff().abs().rolling(window=20, min_periods=20).mean()
        base['Signal'] = 1.0
        base['目标仓位'] = 1.0
        base['仓位变化'] = 0.0
        base['触发原因'] = "主动基金：不适用机械择时，指标仅供视觉参考"
        base['理论事件'] = "不适用"
        return base

    base = compute_timing_signals(df_nav, is_passive)
    reasons, events = [], []
    prev_sig = 1.0
    for i in range(len(base)):
        row = base.iloc[i]
        if pd.isna(row.get('BB_lower')) or pd.isna(row.get('RSI')):
            reasons.append("指标预热期")
            events.append("")
            continue
        price = nav.iloc[i]
        oversold = row['RSI'] < 30
        overbought = row['RSI'] > 70
        trend_up = row['MA30'] > base['MA30'].iloc[i - 1] if i > 0 else False
        touch_lower = price <= row['BB_lower']
        touch_upper = price >= row['BB_upper']
        break_ma = (price < row['MA20']) or (price < row['MA30'])
        proposed = row['Signal']
        reason = _build_signal_reason(trend_up, touch_lower, touch_upper, overbought, oversold, break_ma, proposed, prev_sig)
        reasons.append(reason)

        evt = ""
        if proposed > prev_sig:
            evt = "理论买入"
        elif proposed < prev_sig:
            evt = "理论减仓" if proposed > 0 else "理论清仓"
        elif i == len(base) - 1 and (touch_lower or touch_upper or (break_ma and not oversold)):
            evt = "关注"
        events.append(evt)
        prev_sig = proposed

    base['触发原因'] = reasons
    base['理论事件'] = events
    return base

def get_live_timing_advice(signals: pd.DataFrame, fund_name: str, is_passive: bool) -> dict:
    """基于最新交易日输出实况操作建议（前瞻，非历史复盘）。"""
    if signals.empty:
        return {"状态": "无数据", "建议": "无法测算", "目标仓位": "-", "触发原因": "-"}

    if not is_passive:
        return {
            "状态": "主动基金",
            "建议": "不建议机械择时",
            "目标仓位": "100%",
            "触发原因": "主动管理基金均线失真，请用 RBS 风格雷达 + 季报穿透跟踪",
            "RSI14": "-",
            "MA30趋势": "-",
        }

    last = signals.iloc[-1]
    pos = float(last['目标仓位'])
    rsi_v = last.get('RSI', np.nan)
    ma30_up = False
    if len(signals) > 1 and not pd.isna(last.get('MA30')) and not pd.isna(signals['MA30'].iloc[-2]):
        ma30_up = last['MA30'] > signals['MA30'].iloc[-2]

    if pos >= 0.99:
        action = "满仓持有 / 逢低关注"
        status = "多头"
    elif pos >= 0.45:
        action = "半仓防守 / 观望"
        status = "中性"
    else:
        action = "空仓或极低仓位防守"
        status = "防守"

    # 最新 bar 若出现理论事件，优先提示
    evt = str(last.get('理论事件', ''))
    if evt == "理论买入":
        action = "【信号】考虑分批买入/加仓"
        status = "买入窗口"
    elif evt in ("理论减仓", "理论清仓"):
        action = "【信号】考虑止盈/减仓"
        status = "卖出窗口"
    elif evt == "关注":
        action = "【关注】接近触发边界，暂不操作"
        status = "观察"

    return {
        "状态": status,
        "建议": action,
        "目标仓位": f"{pos*100:.0f}%",
        "触发原因": str(last.get('触发原因', '')),
        "RSI14": f"{rsi_v:.1f}" if not pd.isna(rsi_v) else "-",
        "MA30趋势": "向上" if ma30_up else "向下/走平",
        "最新净值": f"{last['单位净值']:.4f}",
        "布林上轨": f"{last.get('BB_upper', 0):.4f}" if not pd.isna(last.get('BB_upper')) else "-",
        "布林下轨": f"{last.get('BB_lower', 0):.4f}" if not pd.isna(last.get('BB_lower')) else "-",
    }

def plot_timing_radar_chart(signals: pd.DataFrame, fund_code: str, fund_name: str, show_history: bool = False, history_days: int = 60):
    """净值 + MA + 布林带 + RSI 双子图；默认仅高亮最新实况点。"""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.10,
        subplot_titles=("净值与波动轨道", "RSI14 动量过滤"),
    )
    idx = signals.index
    nav = signals['单位净值']

    fig.add_trace(go.Scatter(x=idx, y=nav, name="单位净值", line=dict(color="#2c3e50", width=2)), row=1, col=1)
    if 'MA30' in signals.columns:
        fig.add_trace(go.Scatter(x=idx, y=signals['MA30'], name="MA30 战略趋势", line=dict(color="#e67e22", width=1.5)), row=1, col=1)
    if 'MA20' in signals.columns:
        fig.add_trace(go.Scatter(x=idx, y=signals['MA20'], name="SMA20 中轨", line=dict(color="#3498db", width=1, dash="dot")), row=1, col=1)
    if 'BB_upper' in signals.columns and 'BB_lower' in signals.columns:
        fig.add_trace(go.Scatter(x=idx, y=signals['BB_upper'], name="BB上轨 +2σ", line=dict(color="#95a5a6", width=1, dash="dash"), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=signals['BB_lower'], name="BB下轨 -2σ",
            line=dict(color="#95a5a6", width=1, dash="dash"),
            fill='tonexty', fillcolor='rgba(149,165,166,0.12)',
        ), row=1, col=1)

    last_idx = idx[-1]
    fig.add_trace(go.Scatter(
        x=[last_idx], y=[nav.iloc[-1]], mode='markers',
        name="最新实况", showlegend=True,
        marker=dict(size=14, color="#e74c3c", symbol='star', line=dict(width=2, color='white')),
    ), row=1, col=1)

    if show_history and '理论事件' in signals.columns:
        recent = signals.tail(history_days)
        buys = recent[recent['理论事件'] == '理论买入']
        sells = recent[recent['理论事件'].isin(['理论减仓', '理论清仓'])]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys['单位净值'], mode='markers', name=f"近{history_days}日理论买点",
                marker=dict(symbol='triangle-up', size=8, color='rgba(231,76,60,0.5)'),
            ), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells['单位净值'], mode='markers', name=f"近{history_days}日理论卖点",
                marker=dict(symbol='triangle-down', size=8, color='rgba(46,204,113,0.5)'),
            ), row=1, col=1)

    if 'RSI' in signals.columns:
        fig.add_trace(
            go.Scatter(x=idx, y=signals['RSI'], name="RSI14", line=dict(color="#9b59b6", width=1.5), showlegend=False),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#e74c3c", opacity=0.6, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#27ae60", opacity=0.6, row=2, col=1)

    fig.update_layout(
        height=640,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dddddd",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=36, b=100),
    )
    fig.update_annotations(font=dict(size=13))
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return fig

def extract_timing_snapshot(signals: pd.DataFrame) -> dict:
    """汇总技术面快照，供 AI 大势研判使用。"""
    if signals.empty:
        return {}
    last = signals.iloc[-1]
    nav = signals['单位净值'].astype(float)
    snap = {
        "最新交易日": str(signals.index[-1])[:10],
        "单位净值": round(float(last['单位净值']), 4),
    }
    if 'RSI' in signals.columns and not pd.isna(last.get('RSI')):
        rsi_v = float(last['RSI'])
        snap["RSI14"] = round(rsi_v, 1)
        snap["RSI区间"] = "超买(>70)" if rsi_v > 70 else ("超卖(<30)" if rsi_v < 30 else "中性")
    if len(nav) >= 6:
        snap["近5日涨跌幅%"] = round((nav.iloc[-1] / nav.iloc[-6] - 1) * 100, 2)
    if len(nav) >= 21:
        snap["近20日涨跌幅%"] = round((nav.iloc[-1] / nav.iloc[-21] - 1) * 100, 2)
    if len(nav) >= 61:
        snap["近60日涨跌幅%"] = round((nav.iloc[-1] / nav.iloc[-61] - 1) * 100, 2)
    if 'MA30' in signals.columns:
        ma30 = signals['MA30'].dropna()
        if len(ma30) >= 2:
            snap["MA30趋势"] = "向上" if ma30.iloc[-1] > ma30.iloc[-2] else "向下/走平"
        if len(ma30) >= 6:
            snap["MA30近5日方向"] = "上升" if ma30.iloc[-1] > ma30.iloc[-6] else "下降"
        if not pd.isna(last.get('MA30')):
            snap["净值相对MA30%"] = round((nav.iloc[-1] / float(last['MA30']) - 1) * 100, 2)
    if 'BB_upper' in signals.columns and 'BB_lower' in signals.columns:
        bu, bl = last.get('BB_upper'), last.get('BB_lower')
        if not pd.isna(bu) and not pd.isna(bl) and float(bu) > float(bl):
            pos_pct = (nav.iloc[-1] - float(bl)) / (float(bu) - float(bl)) * 100
            snap["布林带位置%"] = round(pos_pct, 1)
            if pos_pct <= 10:
                snap["均值回归位置"] = "贴近下轨（偏超跌）"
            elif pos_pct >= 90:
                snap["均值回归位置"] = "贴近上轨（偏超买）"
            else:
                snap["均值回归位置"] = "轨道中部"
    if 'ATR20' in signals.columns and not pd.isna(last.get('ATR20')):
        snap["ATR20波动率"] = round(float(last['ATR20']), 4)
    if '理论事件' in signals.columns:
        evt = str(last.get('理论事件', '')).strip()
        if evt:
            snap["引擎理论事件"] = evt
    if '触发原因' in signals.columns:
        snap["引擎触发原因"] = str(last.get('触发原因', ''))
    return snap

def _parse_ai_json_response(text: str) -> dict:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "综合决断": "需人工阅读",
            "适宜度": "-",
            "大趋势": text[:200],
            "买卖点评估": text,
            "仓位建议": "-",
            "建议目标仓位%": None,
            "建议仓位变动百分点": None,
            "风险警示": "-",
            "总监一句话": "AI 返回格式异常，请重试",
        }

def _df_to_prompt_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    """把持仓/行业表压成 prompt 文本，避免依赖 to_markdown。"""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return "暂无数据"
    view = df.head(max_rows).copy()
    try:
        return view.to_string(index=False)
    except Exception:
        return str(view.to_dict(orient="records"))

def run_ai_timing_judgment(
    fund_code: str,
    fund_name: str,
    snapshot: dict,
    engine_advice: dict,
    is_passive: bool,
    portfolio_df: pd.DataFrame | None = None,
    industry_df: pd.DataFrame | None = None,
    kyc_limit: float | None = None,
    baseline_weight_pct: float | None = None,
    sector_proxy: list | None = None,
    holdings_proxy: list | None = None,
) -> dict:
    """调用 DeepSeek：基金净值 + 行业代理 + 重仓股实盘 + 初始仓位，综合给出仓位变动。"""
    client = get_deepseek_client()
    kyc_text = f"最大回撤心理底线 {abs(kyc_limit):.0f}%" if kyc_limit is not None else "未设定"
    if portfolio_df is None:
        portfolio_df = pd.DataFrame()
    if industry_df is None:
        industry_df = pd.DataFrame()
    if sector_proxy is None:
        sector_proxy = build_sector_proxy_snapshot(industry_df)
    if holdings_proxy is None:
        holdings_proxy = build_holdings_trend_snapshot(portfolio_df)
    holdings_breadth = summarize_holdings_breadth(holdings_proxy)
    stance_hint = build_directional_stance_hint(
        holdings_breadth, sector_proxy, engine_advice, is_passive, holdings_proxy
    )

    port_text = _df_to_prompt_table(portfolio_df, max_rows=10)
    ind_text = _df_to_prompt_table(industry_df, max_rows=5)
    base_txt = f"{baseline_weight_pct:.2f}%" if baseline_weight_pct is not None else "未设定（请给合理绝对目标仓位%）"

    prompt = f"""你是极具攻击性与纪律性的 FOF 量化择时总监。
证据层用新变量，立场层沿用旧铁律：必须明确方向，禁止永久「持有观望」。

## 判据优先级（绝对服从，这是本次核心）
1. **主判据（占大头）**：本基金真实重仓股行情，且按【占净值比例】加权——大权重个股的涨跌/结构，才真正驱动基金净值变动。
2. **辅助判据**：行业代理指数，只用来印证或提示风格环境，不能压过重仓股权重结论。
3. **次辅助**：基金净值/引擎指标（主动基金不得因净值横盘否决重仓股右侧结论；被动基金可提高引擎权重）。

禁止编造未给出的新闻/点位；左右侧须引用重仓股或代理指数的结构判定/涨跌数字。

## 标的画像
- 基金：{fund_name}（{fund_code}）
- 属性：{"被动/指数类：引擎可作重要参考，但仍要用成分/重仓校验" if is_passive else "主动管理类：重仓股（按占比）偏强即可买入/加仓；偏弱则减仓/暂不；禁止因净值横盘而观望"}
- 用户组合中该腿【初始仓位】：{base_txt}

## 系统默认立场（按重仓股权重算出来的；无强反证必须采纳）
{json.dumps(stance_hint, ensure_ascii=False, indent=2)}

## 主数据：重仓股单股行情（按占净值比例看）
**广度/加权摘要**：{json.dumps(holdings_breadth, ensure_ascii=False)}
**个股明细**：
{json.dumps(holdings_proxy, ensure_ascii=False, indent=2)}

## 辅助数据：行业代理指数
{json.dumps(sector_proxy, ensure_ascii=False, indent=2)}

## 底层名单（季报）
行业：
{ind_text}
持仓：
{port_text}

## 次辅助：引擎 / 净值
{json.dumps(engine_advice, ensure_ascii=False, indent=2)}
{json.dumps(snapshot, ensure_ascii=False, indent=2)}

## 用户风控
{kyc_text}

## 研判铁律
1. 杜绝中庸：不要「持有观望」。从【买入/分批买入/减仓止盈/暂不操作/不宜追涨】中选。
2. 主判据：前几大重仓（高占净值比例）整体趋势 —— 多数偏强/近20日明显正收益 → 「买入」或「分批买入」；多数偏弱 → 「减仓止盈」或「暂不操作」。
3. 若广度摘要写「无股票持仓（债/货基等）」，禁止写「重仓股数据缺失」，改依据行业代理+引擎表态。
4. 行业与个股冲突时：听重仓股整体；行业分歧写进风险警示即可。
5. 仓位相对【初始仓位】给数字；允许留现金。
6. 综合决断须与「系统默认立场」同向。

输出必须是纯 JSON（不要 markdown 代码块）：
{{
  "综合决断": "买入|分批买入|减仓止盈|暂不操作|不宜追涨 之一",
  "适宜度": "高|中|低",
  "大趋势": "必须先点名前两大权重重仓股的结构/涨跌，再可提行业",
  "买卖点评估": "先写重仓股总体趋势如何决定加/减，影线最多一句旁证",
  "仓位建议": "一句话说明怎么调",
  "建议目标仓位%": 数字（0~100，允许0）,
  "建议仓位变动百分点": 数字（相对初始，加仓为正、减仓为负）,
  "风险警示": "需警惕的情形",
  "总监一句话": "不超过30字，必须含方向词与代表性重仓股"
}}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content or ""
    parsed = _parse_ai_json_response(raw)
    parsed["_raw"] = raw
    parsed["_sector_proxy"] = sector_proxy
    parsed["_holdings_proxy"] = holdings_proxy
    parsed["_holdings_breadth"] = holdings_breadth
    parsed["_stance_hint"] = stance_hint
    parsed = enforce_decisive_verdict(parsed, stance_hint)
    return apply_timing_position_math(parsed, baseline_weight_pct)

def render_ai_timing_verdict(ai_result: dict):
    """结论前置：决断 + 初始/变动/变动后仓位，细节折叠。"""
    verdict = ai_result.get("综合决断", "观望")
    suitability = ai_result.get("适宜度", "-")
    one_liner = ai_result.get("总监一句话", "")

    buy_words = ("买入", "加仓", "逢低")
    sell_words = ("减仓", "止盈", "卖出", "不宜追涨", "暂不操作")
    if any(w in verdict for w in buy_words):
        banner, icon = "success", "🟢"
    elif any(w in verdict for w in sell_words):
        banner, icon = "warning", "🔴"
    else:
        banner, icon = "info", "🟡"

    st.markdown(f"### {icon} 核心定调：**{verdict}**（适宜度：{suitability}）")
    if banner == "success":
        st.success(f"**决策精要**：{one_liner}")
    elif banner == "warning":
        st.warning(f"**决策精要**：{one_liner}")
    else:
        st.info(f"**决策精要**：{one_liner}")

    c1, c2, c3 = st.columns(3)
    init_w = ai_result.get("初始仓位%")
    delta_w = ai_result.get("建议仓位变动百分点")
    after_w = ai_result.get("变动后仓位%", ai_result.get("建议目标仓位%"))
    c1.metric("初始仓位", f"{init_w:.1f}%" if isinstance(init_w, (int, float)) else "未设定")
    if isinstance(delta_w, (int, float)):
        c2.metric("建议变动", f"{delta_w:+.1f}pt")
    else:
        c2.metric("建议变动", "-")
    c3.metric("变动后仓位", f"{after_w:.1f}%" if isinstance(after_w, (int, float)) else "-")

    breadth = ai_result.get("_holdings_breadth") or {}
    if breadth:
        st.caption(
            f"重仓股主判据：{breadth.get('广度结论', '-')} "
            f"（权重净强度 {breadth.get('权重净强度', '-')}，"
            f"偏强{breadth.get('偏强/右侧只数', 0)}/偏弱{breadth.get('偏弱/左侧只数', 0)}，"
            f"加权20日 {breadth.get('加权近20日%', '-')}%）"
        )

    with st.expander("🔍 展开查看底层穿透分析与深度研判", expanded=False):
        st.markdown(f"**买卖点评估**：{ai_result.get('买卖点评估', '-')}")
        st.markdown(f"**行业大趋势**：{ai_result.get('大趋势', '-')}")
        st.markdown(f"**仓位建议**：{ai_result.get('仓位建议', '-')}")
        holdings = ai_result.get("_holdings_proxy") or []
        if holdings:
            st.markdown("**重仓股单股行情（综合研判核心）**")
            st.dataframe(pd.DataFrame(holdings), hide_index=True, use_container_width=True)
        proxy = ai_result.get("_sector_proxy") or []
        if proxy:
            st.markdown("**行业代理指数实况**")
            st.dataframe(pd.DataFrame(proxy), hide_index=True, use_container_width=True)
        if ai_result.get("风险警示"):
            st.error(f"⚠️ **风险警示**：{ai_result['风险警示']}")
def _timing_ai_cache_key(code: str, signals: pd.DataFrame, baseline_weight_pct: float | None = None) -> str:
    day = "na" if signals is None or signals.empty else str(signals.index[-1])[:10]
    w = "nw" if baseline_weight_pct is None else f"{float(baseline_weight_pct):.2f}"
    # h7: 回退稳定版收盘价重仓趋势
    return f"{code}_{day}_{w}_h7"

def _fee_for_change(change: float, buy_fee: float, sell_fee: float, punitive: bool) -> float:
    if abs(change) < 1e-9:
        return 0.0
    rate = 0.015 if punitive else (buy_fee if change > 0 else sell_fee)
    return abs(change) * rate

def backtest_fund_with_friction_guard(
    temp_daily: pd.DataFrame,
    signals: pd.DataFrame,
    buy_fee: float,
    sell_fee: float,
) -> tuple:
    """单基金回测 + 摩擦成本熔断器（含 C 类不满 7 天赎回惩罚）。"""
    df = temp_daily.copy()
    df['原始日增长率'] = df['日增长率'] / 100.0
    df = df.set_index('净值日期')

    raw_targets = signals['Signal'].shift(1).fillna(1.0)
    targets, fees, streaks = [], [], []
    strategy_nav = 1.0
    fee_history = []
    locked = False
    change_streak = 0
    holding_days = 0  # 连续持有（目标仓位>0）交易日
    prev_target = 1.0
    warning_msg = None

    for dt in df.index:
        if locked:
            target = prev_target
        else:
            target = float(raw_targets.loc[dt])

        change = target - prev_target
        if abs(change) > 1e-9:
            change_streak += 1
        else:
            change_streak = 0

        # C 类核心：卖出且持有未满 7 个交易日 → 1.5% 惩罚费率（与“连续调仓天数”脱钩）
        # 先按“卖出前已持有天数”判定，再更新计数
        punitive_short = (change < -1e-9) and (holding_days > 0) and (holding_days < 7)
        # 保留极端高频调仓的额外惩罚
        punitive_freq = change_streak >= 7
        fee = _fee_for_change(change, buy_fee, sell_fee, punitive_short or punitive_freq)
        daily_ret = df.at[dt, '原始日增长率'] * target - fee

        strategy_nav *= (1 + daily_ret)
        fee_history.append(fee)

        if not locked and len(fee_history) >= 30:
            rolling_fee = sum(fee_history[-30:])
            if rolling_fee > 0.03 * strategy_nav:
                locked = True
                warning_msg = "触发高频调仓摩擦成本熔断，系统已自动转入被动锁仓防守状态"

        # 更新持仓天数：有仓累加；清仓归零；从空仓重新开仓则从 1 起计
        if target > 1e-9:
            if prev_target <= 1e-9:
                holding_days = 1
            else:
                holding_days += 1
        else:
            holding_days = 0

        targets.append(target)
        fees.append(fee)
        streaks.append(change_streak)
        prev_target = target

    out = df.copy()
    out['目标仓位'] = targets
    out['仓位变化'] = pd.Series(targets, index=df.index).diff().fillna(0.0)
    out['手续费损耗'] = fees
    out['调仓连续天数'] = streaks
    out['策略净值'] = np.nan  # 占位，外层组合层重算
    out['日收益_含费'] = out['原始日增长率'] * out['目标仓位'] - out['手续费损耗']
    return out, locked, warning_msg

def backtest_portfolio_with_friction_guard(
    valid_codes: list,
    fund_data_dict: dict,
    weights: dict,
    buy_fees: dict,
    sell_fees: dict,
) -> tuple:
    """组合级回测试算：双核择时 + 摩擦熔断，再按权重合成。"""
    fund_frames = {}
    signal_cache = {}
    warnings = []

    for code in valid_codes:
        df = fund_data_dict.get(code)
        temp_daily = df[['净值日期', '单位净值', '日增长率']].copy()
        fund_name = get_fund_display_name(code)
        is_passive = is_passive_fund(fund_name)
        signals = compute_timing_signals(temp_daily, is_passive)
        signal_cache[code] = signals

        buy_fee, sell_fee = resolve_fund_fees(fund_name, buy_fees[code], sell_fees[code])
        bt_df, locked, warn = backtest_fund_with_friction_guard(temp_daily, signals, buy_fee, sell_fee)
        if warn:
            label = fund_name or code
            warnings.append(f"⚡ **{label} ({code})**：{warn}")

        bt_df[code] = bt_df['日收益_含费']
        bt_df[f'{code}_原始'] = bt_df['原始日增长率']
        fund_frames[code] = bt_df[[code, f'{code}_原始', '目标仓位', '仓位变化', '手续费损耗']].rename(columns={
            '目标仓位': f'{code}_目标仓位',
            '仓位变化': f'{code}_仓位变化',
            '手续费损耗': f'{code}_手续费',
        })

    if not fund_frames:
        return None, signal_cache, warnings

    port_daily_df = None
    for code in valid_codes:
        frame = fund_frames[code]
        if port_daily_df is None:
            port_daily_df = frame
        else:
            port_daily_df = pd.merge(port_daily_df, frame, left_index=True, right_index=True, how='inner')

    if port_daily_df is None or port_daily_df.empty:
        return None, signal_cache, warnings

    port_daily_df['[我的组合]_日收益'] = 0.0
    port_daily_df['[静态持有]_日收益'] = 0.0
    port_daily_df['[组合]_手续费'] = 0.0

    for code in valid_codes:
        port_daily_df['[我的组合]_日收益'] += port_daily_df[code] * weights[code]
        port_daily_df['[静态持有]_日收益'] += port_daily_df[f'{code}_原始'] * weights[code]
        port_daily_df['[组合]_手续费'] += port_daily_df[f'{code}_手续费'] * weights[code]

    if len(port_daily_df) >= 30:
        rolling_fee = port_daily_df['[组合]_手续费'].rolling(window=30, min_periods=30).sum()
        strategy_nav = (1 + port_daily_df['[我的组合]_日收益']).cumprod()
        breach = rolling_fee > 0.03 * strategy_nav
        if breach.any():
            lock_from = breach.idxmax()
            warnings.append(
                f"⚡ **组合整体**：触发高频调仓摩擦成本熔断（自 {lock_from.strftime('%Y-%m-%d')} 起），"
                "系统已自动转入被动锁仓防守状态"
            )
            locked_slice = port_daily_df.index >= lock_from
            port_daily_df.loc[locked_slice, '[我的组合]_日收益'] = 0.0
            for code in valid_codes:
                lock_pos = float(port_daily_df.loc[lock_from, f'{code}_目标仓位'])
                port_daily_df.loc[locked_slice, code] = (
                    port_daily_df.loc[locked_slice, f'{code}_原始'] * lock_pos
                )
                port_daily_df.loc[locked_slice, '[我的组合]_日收益'] += (
                    port_daily_df.loc[locked_slice, code] * weights[code]
                )

    return port_daily_df, signal_cache, warnings

def get_fund_display_name(code: str) -> str:
    return get_fund_name_index().get(str(code).strip(), "")

def ensure_engine_data(codes: list, refresh_token: int = 0):
    """
    引擎数据只加载一次，存入 session_state。
    避免 Streamlit 每次交互都重新打穿所有 akshare 接口。
    """
    sig = (tuple(sorted(codes)), refresh_token)
    if (
        st.session_state.get("engine_sig") == sig
        and st.session_state.get("fund_data_dict") is not None
    ):
        return (
            st.session_state["rf_rate"],
            st.session_state["bench_df"],
            st.session_state["fund_data_dict"],
            st.session_state["industry_dict"],
            st.session_state.get("portfolio_dict", {}),
            st.session_state.get("manager_dict", {}),
        )

    rf_rate = get_dynamic_risk_free_rate()
    bench_df = get_benchmark_data(refresh_token=refresh_token)
    fund_data_dict, industry_dict, portfolio_dict, manager_dict = {}, {}, {}, {}
    for code in codes:
        fund_data_dict[code] = get_fund_clean_data(code)
        industry_dict[code] = get_fund_industry(code)
        portfolio_dict[code] = get_fund_portfolio(code)
        manager_dict[code] = get_fund_manager_info(code)

    st.session_state.update({
        "engine_sig": sig,
        "rf_rate": rf_rate,
        "bench_df": bench_df,
        "fund_data_dict": fund_data_dict,
        "industry_dict": industry_dict,
        "portfolio_dict": portfolio_dict,
        "manager_dict": manager_dict,
        "tab5_backtest": None,
    })
    return rf_rate, bench_df, fund_data_dict, industry_dict, portfolio_dict, manager_dict

# ==========================================
# 🧠 Tab4 归因扩展：持有周期 & 资金承载研判
# ==========================================
HIGH_VOL_THEME_KEYWORDS = [
    "半导体", "芯片", "集成电路", "医药", "医疗", "生物", "疫苗", "创新药",
    "科创", "计算机", "人工智能", "AI", "算力", "新能源", "光伏", "锂电", "储能", "券商", "军工",
]
STABLE_THEME_KEYWORDS = [
    "红利", "宽基", "沪深300", "中证500", "蓝筹", "价值", "低波", "稳", "债券", "固收", "货币", "理财",
]
BLUE_CHIP_KEYWORDS = [
    "茅台", "五粮液", "工商银行", "建设银行", "农业银行", "中国银行", "招商银行", "兴业银行",
    "中国移动", "中国电信", "中国石油", "中国石化", "中国神华", "长江电力", "宁德时代", "比亚迪",
    "中国人寿", "中国平安", "美的集团", "格力电器", "海尔", "伊利", "海天", "紫金矿业", "中芯国际",
]

def _theme_text(fund_name: str, industry_df: pd.DataFrame) -> str:
    text = fund_name or ""
    if industry_df is not None and not industry_df.empty:
        text += " " + " ".join(industry_df["行业类别"].astype(str).head(5).tolist())
    return text

def is_high_vol_theme(fund_name: str, industry_df: pd.DataFrame) -> bool:
    text = _theme_text(fund_name, industry_df)
    return any(k in text for k in HIGH_VOL_THEME_KEYWORDS)

def is_stable_theme(fund_name: str, industry_df: pd.DataFrame) -> bool:
    text = _theme_text(fund_name, industry_df)
    return any(k in text for k in STABLE_THEME_KEYWORDS)

def is_blue_chip_stock(stock_name: str) -> bool:
    name = str(stock_name)
    return any(kw in name for kw in BLUE_CHIP_KEYWORDS)

def portfolio_all_blue_chip(portfolio_df: pd.DataFrame) -> bool:
    if portfolio_df is None or portfolio_df.empty or len(portfolio_df) < 5:
        return False
    return all(is_blue_chip_stock(name) for name in portfolio_df["股票名称"])

@st.cache_data(ttl=3600)
def get_fund_turnover_rate(fund_code: str):
    """尝试抓取基金年化换手率，失败则返回 None 由代理逻辑兜底。"""
    try:
        detail = ak.fund_individual_detail_info_xq(symbol=fund_code)
        if detail is not None and not detail.empty:
            for col in detail.columns:
                if "换手" in str(col):
                    val = pd.to_numeric(detail[col], errors="coerce").dropna()
                    if not val.empty:
                        return float(val.iloc[0])
            for _, row in detail.iterrows():
                row_text = " ".join(str(v) for v in row.values)
                if "换手" in row_text:
                    nums = pd.Series(row.values).astype(str).str.extract(r"([\d.]+)")[0].dropna()
                    if not nums.empty:
                        return float(nums.iloc[0])
    except Exception:
        pass
    try:
        overview = ak.fund_overview_em(symbol=fund_code)
        if overview is not None and not overview.empty:
            text_blob = overview.to_string()
            match = re.search(r"换手率[^\d]*([\d.]+)", text_blob)
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return None

def assess_holding_duration(style_tag: str, fund_name: str, industry_df: pd.DataFrame) -> str:
    if "锐度进攻派" in style_tag and is_high_vol_theme(fund_name, industry_df):
        return "⚡ 短期战术波段型 | 建议 1-3 个月 | 紧跟布林带上下轨"
    if "稳健画线派" in style_tag or is_stable_theme(fund_name, industry_df):
        return "🏰 中长期战略底仓型 | 建议 6 个月以上 | 跨越波动周期"
    if "⚔️" in style_tag:
        return "🎯 中短期进攻型 | 建议 1-3 个月 | 紧盯行业催化与止盈"
    if "🐢" in style_tag or "⚠️" in style_tag:
        return "⏳ 中期观察型 | 建议 3-6 个月 | 需配合大盘节奏再评估"
    return "📋 灵活配置型 | 建议 3-6 个月 | 按季报与风格漂移动态调仓"

def assess_capital_capacity(
    fund_code: str,
    df_fund: pd.DataFrame,
    tenure_max_dd: float,
    portfolio_df: pd.DataFrame,
    fund_name: str,
    industry_df: pd.DataFrame,
    style_tag: str,
    risk_level: str,
    rf_rate: float,
) -> str:
    one_year_ago = df_fund["净值日期"].max() - pd.DateOffset(years=1)
    metrics_1y = calculate_risk_metrics(df_fund[df_fund["净值日期"] >= one_year_ago], rf_rate)
    dd_1y = metrics_1y.get("最大回撤", 0)

    turnover = get_fund_turnover_rate(fund_code)
    high_turnover = turnover is not None and turnover >= 150
    if not high_turnover and tenure_max_dd < -35:
        if "锐度进攻" in style_tag or "极高风险" in risk_level or is_high_vol_theme(fund_name, industry_df):
            high_turnover = True

    if dd_1y > -15 and portfolio_all_blue_chip(portfolio_df):
        return "🐋 鲸鱼级大资金友好 | 承载上限高 | 适合 50 万+ 大额配置"
    if tenure_max_dd < -35 and high_turnover:
        suffix = f"（换手率 {turnover:.0f}%）" if turnover is not None else "（高波+高换代理判定）"
        return f"🐜 游资蚂蚁型 | 容量极小 | 建议单笔 ≤5 万 {suffix}"
    return "💼 标准零售适配型 | 建议单笔 5-50 万 | 兼顾流动性与冲击成本"

def style_tab4_dataframe(result_df: pd.DataFrame):
    def _style_holding(val):
        text = str(val)
        if "短期" in text or "⚡" in text:
            return "background-color: #fff3cd; color: #856404; font-weight: 600"
        if "中长期" in text or "🏰" in text:
            return "background-color: #d4edda; color: #155724; font-weight: 600"
        return "background-color: #e8f4fd; color: #0c5460"

    def _style_capacity(val):
        text = str(val)
        if "鲸鱼" in text or "🐋" in text:
            return "background-color: #d1ecf1; color: #0c5460; font-weight: 700"
        if "蚂蚁" in text or "🐜" in text:
            return "background-color: #f8d7da; color: #721c24; font-weight: 700"
        return "background-color: #f8f9fa; color: #495057"

    def _style_style(val):
        text = str(val)
        if "稳健画线" in text:
            return "background-color: #d4edda; color: #155724"
        if "锐度进攻" in text:
            return "background-color: #ffe5d0; color: #c0392b"
        return ""

    styler = result_df.style
    if "🚀 最佳持有周期" in result_df.columns:
        styler = styler.map(_style_holding, subset=["🚀 最佳持有周期"])
    if "💰 资金承载能力" in result_df.columns:
        styler = styler.map(_style_capacity, subset=["💰 资金承载能力"])
    if "🏷️ 演算投资风格" in result_df.columns:
        styler = styler.map(_style_style, subset=["🏷️ 演算投资风格"])
    return styler

# ==========================================
# 🧠 3. 核心计算模型 (全局类定义隔离)
# ==========================================
class MacroAttributionModel:
    def __init__(self):
        self.baseline_factors = {
            "真实利率": 0.1,    
            "科技预期": 0.8,    
            "政策流动性": 0.5,  
            "经济动能": 0.2,    
            "地缘风险": 0.4     
        }
        
        self.base_allocation = {"权益资产": 45.0, "固收资产": 35.0, "现金资产": 20.0}
        self.base_equity_structure = {"红利低波": 50.0, "高端制造": 30.0, "科技AI": 20.0}
        self.base_bond_structure = {"信用债": 60.0, "利率债": 40.0}

        self.sensitivity_matrix = {
            "真实利率": {
                "alloc": {"权益资产": -15.0, "固收资产": 5.0, "现金资产": 10.0},
                "eq": {"红利低波": -20.0, "高端制造": 10.0, "科技AI": 10.0}, 
                "bd": {"信用债": -30.0, "利率债": 30.0}
            },
            "科技预期": {
                "alloc": {"权益资产": 10.0, "固收资产": -5.0, "现金资产": -5.0},
                "eq": {"红利低波": -25.0, "高端制造": 5.0, "科技AI": 20.0},
                "bd": {"信用债": 0.0, "利率债": 0.0}
            },
            "政策流动性": {
                "alloc": {"权益资产": 15.0, "固收资产": 10.0, "现金资产": -25.0},
                "eq": {"红利低波": 5.0, "高端制造": 10.0, "科技AI": 15.0},
                "bd": {"信用债": 15.0, "利率债": -15.0} 
            },
            "经济动能": {
                "alloc": {"权益资产": 15.0, "固收资产": -10.0, "现金资产": -5.0},
                "eq": {"红利低波": -15.0, "高端制造": 25.0, "科技AI": -10.0},
                "bd": {"信用债": 20.0, "利率债": -20.0}
            },
            "地缘风险": {
                "alloc": {"权益资产": -20.0, "固收资产": 10.0, "现金资产": 10.0},
                "eq": {"红利低波": 20.0, "高端制造": 10.0, "科技AI": -30.0}, 
                "bd": {"信用债": -20.0, "利率债": 20.0}
            }
        }

    def simulate(self, current_factors: dict):
        alloc = self.base_allocation.copy()
        eq_struct = self.base_equity_structure.copy()
        bd_struct = self.base_bond_structure.copy()
        
        logs = []

        for factor, current_val in current_factors.items():
            baseline_val = self.baseline_factors[factor]
            delta = current_val - baseline_val
            
            if abs(delta) > 0.01: 
                direction = "上行" if delta > 0 else "下行"
                logs.append(f"因子 [{factor}] {direction} (偏离度: {delta:+.2f})，触发敏感度矩阵调仓。")
                
                for asset, weight in self.sensitivity_matrix[factor]["alloc"].items():
                    alloc[asset] += weight * delta
                for asset, weight in self.sensitivity_matrix[factor]["eq"].items():
                    eq_struct[asset] += weight * delta
                for asset, weight in self.sensitivity_matrix[factor]["bd"].items():
                    bd_struct[asset] += weight * delta

        def normalize(d):
            d = {k: max(0.0, v) for k, v in d.items()}
            total = sum(d.values())
            return {k: round((v / total) * 100, 2) for k, v in d.items()} if total > 0 else d

        return {
            "logs": logs,
            "allocation": normalize(alloc),
            "equity_structure": normalize(eq_struct),
            "bond_structure": normalize(bd_struct)
        }

# ==========================================
# 🤖 Agentic AI：ReAct 风控总监工具箱
# ==========================================
AGENT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_funds",
            "description": (
                "按主题/行业/风格关键词在全市场模糊检索基金。"
                "仅在「全新配置」或需要替补标的时使用；用户追问/微调当前方案时不要仅为了换新而调用。"
                "关键词如：红利、医疗、消费、债券、黄金、纳斯达克、可转债。"
                "不要把历史代码直接当 keyword；已知代码请用 get_fund_metrics。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "主题/行业/风格词（红利、医疗、纯债等）。不要填历史组合里的6位代码。",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_metrics",
            "description": (
                "获取单只已知基金代码的收益、回撤、夏普，以及实时技术快照"
                "（live_timing_snapshot：RSI/MA30/布林位置/近N日涨跌；live_engine_advice：引擎买卖建议）。"
                "用户问「现在能买吗/该不该减仓」时必须调用本工具，禁止只凭历史累计收益空谈。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "6位基金代码"}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_macro_allocation",
            "description": "运行宏观敏感度沙盘，输入因子刻度(-1~1)推演大类/权益/固收配额。",
            "parameters": {
                "type": "object",
                "properties": {
                    "factors": {
                        "type": "object",
                        "description": "宏观因子字典，键：真实利率/科技预期/政策流动性/经济动能/地缘风险",
                        "additionalProperties": {"type": "number"},
                    }
                },
                "required": ["factors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_my_brain",
            "description": (
                "检索用户自定义知识库（投资原则、买卖纪律、风格偏好）。"
                "知识库有文档时，配置/策略类问题应调用，用用户自己的原则约束推荐。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索用的关键词或问题表述"},
                    "top_k": {"type": "integer", "description": "返回片段数量，默认4", "default": 4},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backtest_portfolio",
            "description": "对指定权重组合运行双核择时+摩擦熔断历史回测，返回收益、回撤、夏普与熔断警告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "weights": {
                        "type": "object",
                        "description": "基金代码到权重(%)的映射，如 {\"005827\": 30, \"270048\": 70}",
                        "additionalProperties": {"type": "number"},
                    }
                },
                "required": ["weights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_pro_dashboard",
            "description": (
                "将最终配置方案注入专家模式并准备跳转。用户确认方案后、或明确要求去专业看板深度回测时调用。"
                "会写入基金代码、权重，开启量化引擎，并切换到 FOF 穿透与归因面板。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "基金代码列表；若提供 weights 可省略",
                    },
                    "weights": {
                        "type": "object",
                        "description": "代码→权重(%)，如 {\"005827\": 40, \"270048\": 60}",
                        "additionalProperties": {"type": "number"},
                    },
                    "note": {
                        "type": "string",
                        "description": "给用户看的跳转说明，默认：已注入专家模式参数",
                    },
                },
                "required": [],
            },
        },
    },
]

def build_portfolio_history_context() -> str:
    history = load_local_history()
    if not history:
        return "用户暂无历史组合测算记录。请通过全市场主题搜索独立发现候选基金。"
    lines = [
        "以下仅供风格对比。不要为了「看起来全新」而无故整套换仓。",
        "用户若在迭代当前方案，应优先沿用「进行中的工作方案」。",
        "",
    ]
    for item in history[:8]:
        lines.append(
            f"- [{item.get('timestamp', '未知时间')}] {item.get('name', '未命名')} | "
            f"配置: {item.get('codes_str', '')} | 累计收益 {item.get('return', 0):.2f}% | "
            f"最大回撤 {item.get('drawdown', 0):.2f}% | 夏普 {item.get('sharpe', 0):.3f}"
        )
    return "\n".join(lines)

def _kb_status_and_prefetch(user_query: str = "") -> tuple:
    """返回知识库状态说明 + 对当前问题的预检索片段。"""
    if not KB_RAG_AVAILABLE:
        return "知识库模块未安装。", []
    try:
        from knowledge_rag import get_kb_stats, search_my_brain
        stats = get_kb_stats()
        if stats.get("document_count", 0) <= 0:
            return "知识库当前为空，无需调用 search_my_brain。", []
        docs = "、".join(stats.get("documents", [])[:12]) or "若干文档"
        status = (
            f"知识库已就绪：{stats['document_count']} 篇文档（{docs}），"
            f"{stats.get('chunk_count', 0)} 个向量片段。"
            f"配置/策略类问题必须优先遵守这些原则。"
        )
        hits = search_my_brain(user_query, top_k=4) if str(user_query).strip() else []
        return status, hits
    except Exception as e:
        return f"知识库状态读取失败：{e}", []

def compute_fund_metrics_package(code: str) -> dict:
    code = str(code).strip()
    df_fund = get_fund_clean_data(code)
    if df_fund is None or df_fund.empty:
        return {"error": f"无法获取基金 {code} 的净值数据"}

    rf_rate = get_dynamic_risk_free_rate()
    fund_name = get_fund_display_name(code)
    industry_df = get_fund_industry(code)
    portfolio_df = get_fund_portfolio(code)
    bench_df = get_benchmark_data()

    metrics_all = calculate_risk_metrics(df_fund, rf_rate)
    one_year_ago = df_fund["净值日期"].max() - pd.DateOffset(years=1)
    metrics_1y = calculate_risk_metrics(df_fund[df_fund["净值日期"] >= one_year_ago], rf_rate)

    config = get_fund_manager_info(code)
    style_tag = "无基准暂不评定"
    alpha = None
    bench_ret = None
    manager_max_dd = metrics_all.get("最大回撤", 0)
    risk_level = "中"

    if manager_max_dd < -35:
        risk_level = "极高风险 (大开大合)"
    elif manager_max_dd < -20:
        risk_level = "中高风险 (进取波动)"
    elif manager_max_dd < -10:
        risk_level = "中低风险 (均衡控制)"
    else:
        risk_level = "低风险 (严控回撤)"

    if config:
        start_date = pd.to_datetime(config["start_date"])
        df_m = df_fund[df_fund["净值日期"] >= start_date]
        if not df_m.empty and bench_df is not None and not bench_df.empty:
            bench_m = bench_df[bench_df["净值日期"] >= start_date]
            merged = pd.merge(df_m[["净值日期", "单位净值"]], bench_m, on="净值日期", how="inner")
            if not merged.empty:
                fund_ret = (merged["单位净值"].iloc[-1] / merged["单位净值"].iloc[0]) - 1
                bench_ret = (merged["基准点数"].iloc[-1] / merged["基准点数"].iloc[0]) - 1
                alpha = fund_ret - bench_ret
                tenure_metrics = calculate_risk_metrics(df_m, rf_rate)
                manager_max_dd = tenure_metrics.get("最大回撤", manager_max_dd)
                merged["bench_max"] = merged["基准点数"].cummax()
                bench_max_dd = ((merged["基准点数"] - merged["bench_max"]) / merged["bench_max"]).min() * 100
                if manager_max_dd > bench_max_dd and alpha > 0.1:
                    style_tag = "🛡️ 稳健画线派 (回撤小于大盘且有超额)"
                elif manager_max_dd <= bench_max_dd and alpha > 0.15:
                    style_tag = "⚔️ 锐度进攻派 (高波动换取高超额)"
                elif manager_max_dd <= bench_max_dd and alpha < 0:
                    style_tag = "⚠️ 裸多单边派 (承担更高风险却跑输大盘)"
                elif manager_max_dd > bench_max_dd and alpha < 0:
                    style_tag = "🐢 钝化防守派 (回撤小但无进攻能力)"
                else:
                    style_tag = "⚖️ 均衡跟随派 (贴合基准波动)"

    holding = assess_holding_duration(style_tag, fund_name, industry_df)
    capacity = assess_capital_capacity(
        code, df_fund, manager_max_dd, portfolio_df, fund_name, industry_df, style_tag, risk_level, rf_rate
    )

    # 实时盯盘：MA/RSI/布林等技术快照，供 Copilot「现在能买吗」直接调用
    passive = is_passive_fund(fund_name or "")
    signals = compute_timing_signals_v2(df_fund, passive)
    timing_snap = extract_timing_snapshot(signals)
    live_advice = get_live_timing_advice(signals, fund_name or "", passive)

    return {
        "code": code,
        "name": fund_name or "未知",
        "cum_return_pct": round(metrics_all.get("总收益率", 0), 2),
        "max_drawdown_pct": round(metrics_all.get("最大回撤", 0), 2),
        "sharpe": round(metrics_all.get("真实夏普", 0), 3),
        "return_1y_pct": round(metrics_1y.get("总收益率", 0), 2),
        "drawdown_1y_pct": round(metrics_1y.get("最大回撤", 0), 2),
        "sharpe_1y": round(metrics_1y.get("真实夏普", 0), 3),
        "alpha_pct": round(alpha * 100, 2) if alpha is not None else None,
        "beta_benchmark_return_pct": round(bench_ret * 100, 2) if bench_ret is not None else None,
        "style_tag": style_tag,
        "risk_level": risk_level,
        "holding_duration": holding,
        "capital_capacity": capacity,
        "manager": config.get("name") if config else None,
        "manager_since": config.get("start_date") if config else None,
        "is_passive": passive,
        "live_timing_snapshot": timing_snap,
        "live_engine_advice": {
            "状态": live_advice.get("状态"),
            "建议": live_advice.get("建议"),
            "目标仓位": live_advice.get("目标仓位"),
            "触发原因": live_advice.get("触发原因"),
            "RSI14": live_advice.get("RSI14"),
            "MA30趋势": live_advice.get("MA30趋势"),
        },
    }

def tool_search_funds(keyword: str) -> dict:
    keyword = str(keyword).strip()
    if not keyword:
        return {"error": "关键词不能为空", "results": []}
    all_funds = load_all_fund_names()
    if all_funds.empty:
        return {"error": "全市场基金名录加载失败", "results": []}

    # 纯6位数字：视为查码，并提醒应用主题词做全市场发现
    code_only = bool(re.fullmatch(r"\d{6}", keyword))
    matched = all_funds[
        all_funds["基金简称"].str.contains(keyword, na=False, case=False)
        | all_funds["基金代码"].astype(str).str.contains(keyword, na=False)
    ].head(10)
    results = matched.to_dict(orient="records")
    out = {"keyword": keyword, "count": len(results), "results": results}
    if code_only:
        out["hint"] = (
            "你正在用基金代码检索。若任务是全市场配置，请改用主题关键词（红利/医疗/消费/债券等）再搜索。"
        )
    return out

def tool_simulate_macro_allocation(factors: dict) -> dict:
    model = MacroAttributionModel()
    merged = model.baseline_factors.copy()
    if factors:
        for k, v in factors.items():
            if k in merged:
                merged[k] = float(v)
    result = model.simulate(merged)
    return {
        "factors_used": merged,
        "allocation": result["allocation"],
        "equity_structure": result["equity_structure"],
        "bond_structure": result["bond_structure"],
        "trigger_logs": result["logs"],
    }

def tool_backtest_portfolio(weights: dict) -> dict:
    if not weights:
        return {"error": "weights 不能为空"}
    codes = [str(k).strip() for k in weights.keys()]
    fund_data_dict = {c: get_fund_clean_data(c) for c in codes}
    valid_codes = [c for c in codes if fund_data_dict.get(c) is not None and not fund_data_dict[c].empty]
    if not valid_codes:
        return {"error": "所有基金代码均无法获取净值数据", "weights": weights}

    raw_w = {c: float(weights[c]) for c in valid_codes}
    total = sum(raw_w.values())
    if total <= 0:
        return {"error": "权重之和必须大于 0"}
    norm_w = {c: raw_w[c] / total for c in valid_codes}
    buy_fees = {c: 0.0015 for c in valid_codes}
    sell_fees = {c: 0.0050 for c in valid_codes}

    port_df, _, warnings = backtest_portfolio_with_friction_guard(
        valid_codes, fund_data_dict, norm_w, buy_fees, sell_fees
    )
    if port_df is None or port_df.empty:
        return {"error": "组合回测对齐失败，共同交易日不足", "weights": norm_w}

    nav = (1 + port_df["[我的组合]_日收益"]).cumprod()
    eval_df = pd.DataFrame({
        "净值日期": port_df.index,
        "日增长率": port_df["[我的组合]_日收益"] * 100.0,
        "单位净值": nav,
    })
    rf_rate = get_dynamic_risk_free_rate()
    metrics = calculate_risk_metrics(eval_df, rf_rate)
    static_nav = (1 + port_df["[静态持有]_日收益"]).cumprod()

    chart_df = pd.DataFrame({
        "动态择时": nav,
        "静态持有": static_nav,
    }).tail(250)
    chart_reset = chart_df.reset_index()
    chart_reset.columns = ["date", "动态择时", "静态持有"]
    chart_reset["date"] = chart_reset["date"].astype(str)

    return {
        "weights_pct": {c: round(norm_w[c] * 100, 2) for c in valid_codes},
        "dynamic_strategy": {
            "cum_return_pct": round(metrics.get("总收益率", 0), 2),
            "max_drawdown_pct": round(metrics.get("最大回撤", 0), 2),
            "sharpe": round(metrics.get("真实夏普", 0), 3),
        },
        "static_hold": {
            "cum_return_pct": round((static_nav.iloc[-1] - 1) * 100, 2),
        },
        "friction_warnings": warnings,
        "common_trading_days": len(port_df),
        "_visual": {
            "type": "backtest",
            "weights_pct": {c: round(norm_w[c] * 100, 2) for c in valid_codes},
            "metrics": {
                "cum_return_pct": round(metrics.get("总收益率", 0), 2),
                "max_drawdown_pct": round(metrics.get("最大回撤", 0), 2),
                "sharpe": round(metrics.get("真实夏普", 0), 3),
                "static_return_pct": round((static_nav.iloc[-1] - 1) * 100, 2),
            },
            "nav_chart": chart_reset.to_dict(orient="list"),
            "warnings": warnings[:5] if warnings else [],
        },
    }

def tool_route_to_pro_dashboard(codes=None, weights=None, note: str = "") -> dict:
    """注入专家模式参数。不自动跳转，由聊天气泡按钮一键打开，避免丢对话上下文。"""
    portfolio = {
        "codes": codes or [],
        "weights": weights or {},
        "name": (note or "").strip() or f"AI方案 {datetime.datetime.now().strftime('%m-%d %H:%M')}",
    }
    result = inject_portfolio_snapshot(portfolio, jump=False, persist=True)
    if result.get("error"):
        return result
    # 供本轮回复挂载跳转按钮
    st.session_state["_latest_ai_portfolio"] = {
        "codes": result["codes"],
        "weights": result["weights_pct"],
        "name": result["name"],
    }
    return {
        "ok": True,
        "routed": False,
        "codes": result["codes"],
        "weights_pct": result["weights_pct"],
        "message": "参数已写入专家模式工作区。请提示用户点击回复下方的「用此方案打开专家模式」按钮。",
    }


def _strip_visual_for_llm(observation: dict) -> dict:
    if not isinstance(observation, dict):
        return observation
    return {k: v for k, v in observation.items() if k != "_visual"}

def build_fund_metrics_visual(result: dict) -> dict | None:
    """从 get_fund_metrics 结果构造聊天内联卡片。"""
    if not result or result.get("error"):
        return None
    code = str(result.get("code", "")).strip()
    if not code:
        return None
    df = get_fund_clean_data(code)
    nav_chart = {}
    if df is not None and not df.empty:
        tail = df.tail(250)[["净值日期", "单位净值"]].copy()
        nav_chart = {
            "date": tail["净值日期"].dt.strftime("%Y-%m-%d").tolist(),
            "单位净值": tail["单位净值"].astype(float).tolist(),
        }
    return {
        "type": "fund_metrics",
        "code": code,
        "name": result.get("name") or code,
        "metrics": {
            "cum_return_pct": result.get("cum_return_pct"),
            "max_drawdown_pct": result.get("max_drawdown_pct"),
            "sharpe": result.get("sharpe"),
            "return_1y_pct": result.get("return_1y_pct"),
            "style_tag": result.get("style_tag"),
            "live_advice": (result.get("live_engine_advice") or {}).get("建议"),
            "RSI14": (result.get("live_timing_snapshot") or {}).get("RSI14"),
            "MA30趋势": (result.get("live_timing_snapshot") or {}).get("MA30趋势"),
        },
        "nav_chart": nav_chart,
    }

def render_chat_visual(visual: dict):
    """在 assistant 气泡中渲染轻量图表。"""
    if not visual or not isinstance(visual, dict):
        return
    vtype = visual.get("type")
    if vtype == "fund_metrics":
        title = f"{visual.get('name', '')}（{visual.get('code', '')}）"
        st.markdown(f"**📈 {title}**")
        m = visual.get("metrics") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("累计收益", f"{m.get('cum_return_pct', 0):.1f}%")
        c2.metric("最大回撤", f"{m.get('max_drawdown_pct', 0):.1f}%")
        c3.metric("夏普", f"{m.get('sharpe', 0):.2f}")
        c4.metric("近1年收益", f"{m.get('return_1y_pct', 0):.1f}%")
        if m.get("style_tag"):
            st.caption(f"风格：{m['style_tag']}")
        chart = visual.get("nav_chart") or {}
        if chart.get("date") and chart.get("单位净值"):
            cdf = pd.DataFrame({"单位净值": chart["单位净值"]}, index=pd.to_datetime(chart["date"]))
            st.line_chart(cdf, height=220)
    elif vtype == "backtest":
        st.markdown("**🏆 组合回测摘要**")
        m = visual.get("metrics") or {}
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("动态收益", f"{m.get('cum_return_pct', 0):.1f}%")
        b2.metric("最大回撤", f"{m.get('max_drawdown_pct', 0):.1f}%")
        b3.metric("夏普", f"{m.get('sharpe', 0):.2f}")
        b4.metric("静态收益", f"{m.get('static_return_pct', 0):.1f}%")
        chart = visual.get("nav_chart") or {}
        if chart.get("date"):
            cdf = pd.DataFrame({
                "动态择时": chart.get("动态择时", []),
                "静态持有": chart.get("静态持有", []),
            }, index=pd.to_datetime(chart["date"]))
            st.line_chart(cdf, height=240)
        weights = visual.get("weights_pct") or {}
        if weights:
            with st.expander("查看组合权重明细", expanded=False):
                st.dataframe(
                    pd.DataFrame([{"代码": k, "权重%": v} for k, v in weights.items()]),
                    hide_index=True,
                    use_container_width=True,
                )
        warns = visual.get("warnings") or []
        if warns:
            with st.expander("摩擦熔断提示", expanded=False):
                for w in warns:
                    st.warning(w)

def tool_get_fund_metrics(code: str) -> dict:
    result = compute_fund_metrics_package(code)
    visual = build_fund_metrics_visual(result)
    if visual:
        result = dict(result)
        result["_visual"] = visual
    return result

def dispatch_agent_tool(name: str, arguments: dict) -> dict:
    try:
        if name == "search_funds":
            return tool_search_funds(arguments.get("keyword", ""))
        if name == "get_fund_metrics":
            return tool_get_fund_metrics(arguments.get("code", ""))
        if name == "simulate_macro_allocation":
            return tool_simulate_macro_allocation(arguments.get("factors", {}))
        if name == "backtest_portfolio":
            return tool_backtest_portfolio(arguments.get("weights", {}))
        if name == "search_my_brain":
            return tool_search_my_brain(
                arguments.get("query", ""),
                top_k=int(arguments.get("top_k", 3)),
            )
        if name == "route_to_pro_dashboard":
            return tool_route_to_pro_dashboard(
                codes=arguments.get("codes"),
                weights=arguments.get("weights"),
                note=arguments.get("note", ""),
            )
        return {"error": f"未知工具: {name}"}
    except Exception as exc:
        return {"error": str(exc)}

def build_current_portfolio_context() -> str:
    """把当前工作方案注入上下文，便于追问与迭代，而非每次推倒重来。"""
    snap = st.session_state.get("_latest_ai_portfolio") or {}
    if not snap:
        # 退回专家模式工作区
        codes = [c.strip() for c in str(st.session_state.get("fund_codes_input", "")).split(",") if c.strip()]
        weights = st.session_state.get("portfolio_weights_preset") or {}
        if codes:
            snap = {
                "name": st.session_state.get("loaded_portfolio_name") or "专家模式当前组合",
                "codes": codes,
                "weights": {c: weights.get(c) for c in codes} if weights else {},
            }
    if not snap or not (snap.get("codes") or snap.get("weights")):
        return "当前尚无「进行中的方案」。若用户首次要配置，再启动全市场主题检索。"
    codes = snap.get("codes") or list((snap.get("weights") or {}).keys())
    weights = snap.get("weights") or snap.get("weights_pct") or {}
    lines = [
        f"名称：{snap.get('name', '未命名')}",
        f"代码：{', '.join(map(str, codes))}",
    ]
    if weights:
        wtxt = ", ".join([f"{c}:{weights.get(c)}%" for c in codes if c in weights])
        lines.append(f"权重：{wtxt}")
    lines.append("用户说「再给一遍 / 解释一下 / 调权重 / 换掉某只」时，默认以上述方案为底座迭代，禁止无故整套重做。")
    return "\n".join(lines)

def build_agent_system_message(extra_context=None, user_query: str = "") -> str:
    history_ctx = build_portfolio_history_context()
    kyc_limit = st.session_state.get("max_drawdown_limit")
    kyc_text = f"{kyc_limit:.2f}%" if kyc_limit is not None else "未设定"
    kb_status, kb_hits = _kb_status_and_prefetch(user_query)
    current_scheme = build_current_portfolio_context()

    ctx_block = ""
    if extra_context:
        ctx_block = f"\n## 当前看板实时上下文\n```json\n{json.dumps(extra_context, ensure_ascii=False, indent=2)}\n```\n"

    kb_block = f"\n## 用户自定义知识库状态\n{kb_status}\n"
    if kb_hits:
        kb_block += "\n### 已预检索的相关笔记（请严格参照，并在回答中点名来源文件）\n"
        for h in kb_hits:
            kb_block += f"\n**【{h.get('source', '未知')}】**\n{h.get('content', '')[:600]}\n"

    return f"""你是一位拥有 15 年实战经验的 FOF 智能风控总监 Agent。

## 历史组合记忆（风格对比参考，不是强制候选池）
{history_ctx}

## 进行中的工作方案（会话底座，优先沿用）
{current_scheme}

## 用户 KYC 回撤底线
{kyc_text}
{kb_block}
{ctx_block}
## 意图分流（比工具更优先）
先判断本轮用户意图，再行动：

**A. 澄清 / 追问**（为什么选它、风险在哪、持有多久、适不适合我）
- 直接基于「进行中的工作方案」与已有工具结果解释。
- **不要**重新全市场搜基金，**不要**另起一套配置。

**B. 迭代改进**（再给一遍、微调、把某只换成…、债券再高点、回撤再压一点）
- **默认沿用当前方案**为底座，只改用户点名的部分（权重、替换 1～2 只、加减仓）。
- 可用 get_fund_metrics / backtest_portfolio 验证改动；search_funds 仅在需要替补标的时按主题词搜。
- 输出时明确写出「相对上一版改了什么」。

**C. 全新配置**（另做一套、推倒重来、换风格、我要重新配）
- 才启动全市场主题检索：search_funds（红利/医疗/债券等关键词）→ 体检 → 回测。
- 禁止用历史代码当 search_funds 的 keyword；可用 get_fund_metrics 查已知代码。

**D. 落地专家模式**
- 调用 route_to_pro_dashboard 注入；提醒用户点回复下的「用此方案打开专家模式」。

歧义时：若已有进行中方案，且用户未明确说「重做/另配」，一律按 **B 迭代** 处理。

## 工作模式（ReAct）
1. Thought：先判定 A/B/C/D。
2. Act：按意图选工具；迭代时少搜多调。
3. Observe：读 JSON 再决定。
4. 信息够了就停止工具，给最终答复。

## 最终输出要求
- 隐藏 Thought/Act/工具名。
- 首次全新建：给出标的、权重、金额（若有资金）、持有周期、KYC 匹配。
- 迭代时：先复述当前底座，再列变更点，避免整表伪装成新方案。
- 结尾附一句「总监决断」。"""

def _chat_history_as_messages(max_turns: int = 8) -> list:
    """把已有对话注入 Agent，使其能围绕同一方案追问/改进。"""
    hist = st.session_state.get("agent_chat_history") or []
    msgs = []
    # 取末尾若十对用户/助手
    tail = hist[-(max_turns * 2):]
    for m in tail:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        # 截断过长，避免占满上下文
        if len(content) > 2500:
            content = content[:2500] + "…"
        # 若该轮挂了方案，补一行摘要，便于模型对齐
        port = m.get("portfolio")
        if role == "assistant" and port:
            codes = port.get("codes") or list((port.get("weights") or {}).keys())
            content += f"\n\n[本轮方案快照] {port.get('name', '')} | 代码: {', '.join(map(str, codes[:12]))}"
        msgs.append({"role": role, "content": content})
    return msgs

def run_react_agent(user_query: str, extra_context=None, max_steps: int = 8, collect_trace: bool = False, collect_visuals: bool = False):
    client = get_deepseek_client()
    prior = _chat_history_as_messages(max_turns=8)
    # 去掉与本轮重复的最后一条相同 user（_process 可能已写入 history）
    if prior and prior[-1].get("role") == "user" and prior[-1].get("content") == user_query:
        prior = prior[:-1]

    messages = [
        {"role": "system", "content": build_agent_system_message(extra_context, user_query=user_query)},
        *prior,
        {"role": "user", "content": user_query},
    ]
    trace = []
    visuals = []
    fund_visual_count = 0
    latest_portfolio = None
    if collect_trace and KB_RAG_AVAILABLE:
        try:
            _, kb_hits = _kb_status_and_prefetch(user_query)
            if kb_hits:
                trace.append({
                    "step": 0,
                    "tool": "search_my_brain(预检索)",
                    "args": {"query": user_query[:80], "hits": len(kb_hits)},
                    "ok": True,
                })
        except Exception:
            pass

    def _pack(answer):
        # 回测/路由产生的方案挂到 session，供气泡按钮使用；迭代时尽量保留/合并底座
        if latest_portfolio:
            st.session_state["_latest_ai_portfolio"] = latest_portfolio
        if collect_trace and collect_visuals:
            return answer, trace, visuals, latest_portfolio
        if collect_trace:
            return answer, trace
        if collect_visuals:
            return answer, visuals, latest_portfolio
        return answer

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=AGENT_TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                observation = dispatch_agent_tool(tc.function.name, args)
                if collect_visuals and isinstance(observation, dict) and observation.get("_visual"):
                    vis = observation["_visual"]
                    # 基金卡片最多 3 张，回测保留最新一张
                    if vis.get("type") == "fund_metrics":
                        if fund_visual_count < 3:
                            visuals.append(vis)
                            fund_visual_count += 1
                    elif vis.get("type") == "backtest":
                        visuals = [v for v in visuals if v.get("type") != "backtest"]
                        visuals.append(vis)
                        w = vis.get("weights_pct") or {}
                        if w:
                            base_name = None
                            if st.session_state.get("_latest_ai_portfolio"):
                                base_name = st.session_state["_latest_ai_portfolio"].get("name")
                            latest_portfolio = {
                                "codes": list(w.keys()),
                                "weights": w,
                                "name": base_name or f"AI方案 {datetime.datetime.now().strftime('%m-%d %H:%M')}",
                            }
                if tc.function.name == "route_to_pro_dashboard" and isinstance(observation, dict) and observation.get("ok"):
                    latest_portfolio = {
                        "codes": observation.get("codes") or [],
                        "weights": observation.get("weights_pct") or {},
                        "name": st.session_state.get("loaded_portfolio_name") or "AI 方案",
                    }
                if collect_trace:
                    trace.append({
                        "step": step + 1,
                        "tool": tc.function.name,
                        "args": args,
                        "ok": "error" not in observation,
                    })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(_strip_visual_for_llm(observation), ensure_ascii=False),
                })
            continue

        if msg.content:
            return _pack(msg.content.strip())

    messages.append({
        "role": "user",
        "content": "请基于已收集的工具结果，直接输出给用户的最终配置建议（不要调用工具）。若本轮是迭代，请先说明相对上一版的变更。",
    })
    final = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.2,
    )
    answer = (final.choices[0].message.content or "暂时无法生成诊断，请稍后重试。").strip()
    return _pack(answer)

AGENT_QUICK_PROMPTS = [
    "我想买半导体，10万块，最多拿两个月，帮我看看怎么配",
    "多行业稳定收益，50万资金，回撤不超过15%，给我一套配置",
    "帮我找几只红利低波基金，并回测 60%红利+40%纯债组合",
    "科技预期过热、流动性宽松时，宏观沙盘建议怎么配？",
]

def _process_agent_query(user_query: str, extra_context=None, show_trace: bool = False):
    """执行一轮 Agent 对话并写入 session / 磁盘历史。"""
    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = load_chat_history()

    if any(
        m.get("role") == "user" and m.get("content") == user_query
        for m in st.session_state["agent_chat_history"][-2:]
    ):
        return

    st.session_state["agent_chat_history"].append({"role": "user", "content": user_query})
    save_chat_history(st.session_state["agent_chat_history"])

    with st.chat_message("assistant"):
        with st.spinner("副驾思考中：推理 → 调工具 → 观察 → 再推理 …"):
            try:
                answer, trace, visuals, portfolio = run_react_agent(
                    user_query,
                    extra_context=extra_context,
                    collect_trace=True,
                    collect_visuals=True,
                )
                if portfolio is None:
                    portfolio = st.session_state.get("_latest_ai_portfolio")

                if visuals:
                    for vis in visuals:
                        render_chat_visual(vis)
                    st.divider()

                if show_trace and trace:
                    with st.expander(f"本次自动调用了 {len(trace)} 次工具（点击展开）", expanded=False):
                        for item in trace:
                            icon = "✅" if item["ok"] else "⚠️"
                            st.markdown(f"{icon} **步骤 {item['step']}** · `{item['tool']}` · 参数 `{item['args']}`")

                st.markdown(answer)
                if portfolio:
                    render_portfolio_jump_button(portfolio, key=f"jump_live_{len(st.session_state['agent_chat_history'])}")

                st.session_state["agent_chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "visuals": visuals or [],
                    "portfolio": portfolio,
                })
                save_chat_history(st.session_state["agent_chat_history"])
            except Exception as exc:
                err = f"⚠️ Agent 调用失败，请检查网络或 API Key：{exc}"
                st.error(err)
                st.session_state["agent_chat_history"].append({"role": "assistant", "content": err})
                save_chat_history(st.session_state["agent_chat_history"])

def render_ai_advisor_chat(extra_context=None, input_key: str = "ai_hub_chat_input", show_trace: bool = True):
    """全宽对话区（主流 AI 布局：中间主对话 + 底部输入）。"""
    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = load_chat_history()

    history = st.session_state["agent_chat_history"]
    if not history:
        st.markdown(
            """
            <div style="max-width:720px;margin:3.5rem auto 1.5rem auto;text-align:center;">
              <div style="font-size:1.75rem;font-weight:600;letter-spacing:0.02em;margin-bottom:0.6rem;">你好，我是投研副驾</div>
              <div style="opacity:0.72;font-size:0.95rem;line-height:1.6;">
                用大白话说你的资金、期限和风险底线。<br/>
                我会搜全市场、量化体检、回测，并在对话里直接出图。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(2)
        for i, prompt in enumerate(AGENT_QUICK_PROMPTS[:4]):
            with chip_cols[i % 2]:
                if st.button(prompt, key=f"chat_chip_{i}", use_container_width=True):
                    st.session_state["agent_pending_query"] = prompt
                    st.rerun()
    else:
        st.markdown(
            """
            <style>
            div[data-testid="stChatMessage"] {
              max-width: 860px;
              margin-left: auto;
              margin-right: auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        for i, msg in enumerate(history):
            with st.chat_message(msg["role"]):
                for vis in msg.get("visuals") or []:
                    render_chat_visual(vis)
                if msg.get("visuals"):
                    st.divider()
                st.markdown(msg["content"])
                if msg.get("role") == "assistant" and msg.get("portfolio"):
                    render_portfolio_jump_button(msg["portfolio"], key=f"jump_hist_{i}")

    pending = st.session_state.pop("agent_pending_query", None)
    if pending:
        with st.chat_message("user"):
            st.markdown(pending)
        _process_agent_query(pending, extra_context=extra_context, show_trace=show_trace)
        return

    user_query = st.chat_input("问副驾任何配置问题…", key=input_key)
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        _process_agent_query(user_query, extra_context=extra_context, show_trace=show_trace)

def render_agent_risk_director(extra_context=None):
    """兼容旧入口：轻量 Agent 聊天。"""
    render_ai_advisor_chat(extra_context=extra_context, input_key="tab5_agent_chat_input", show_trace=False)

def run_ai_advisor_hub():
    """🧠 AI 副驾 — 主流全宽对话界面（侧边栏只做导航/KYC/记忆）。"""
    if "portfolio_history" not in st.session_state:
        st.session_state["portfolio_history"] = load_local_history()

    if KB_RAG_AVAILABLE and not st.session_state.get("kb_index_synced"):
        try:
            sync_knowledge_index()
        except Exception:
            pass
        st.session_state["kb_index_synced"] = True

    # 顶栏极简：状态 + 动作，不占对话宽度
    kyc_limit = st.session_state.get("max_drawdown_limit")
    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.markdown("### 🧠 AI 副驾")
        if kyc_limit is not None:
            st.caption(f"回撤底线已锁定 **{abs(kyc_limit):.0f}%** · 可在左侧侧边栏修改")
        else:
            st.caption("尚未设定回撤底线 · 请在左侧「KYC」填写（对话里说出来也可以）")
    with top_r:
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("🗑️ 清空", key="hub_clear_chat", use_container_width=True, help="清空对话"):
                st.session_state["agent_chat_history"] = []
                save_chat_history([])
                st.rerun()
        with a2:
            if st.button("📖 知识库", key="hub_kb_toggle", use_container_width=True):
                st.session_state["show_kb_drawer"] = not st.session_state.get("show_kb_drawer", False)
                st.rerun()
        with a3:
            if st.button("📊 专家", key="hub_goto_pro", use_container_width=True):
                request_app_mode_switch("FOF 穿透与归因")
                st.rerun()

    if st.session_state.get("show_kb_drawer"):
        with st.expander("自定义知识库", expanded=True):
            render_knowledge_base_panel()

    hub_context = {
        "kyc_drawdown_limit_pct": st.session_state.get("max_drawdown_limit"),
        "recent_portfolio_history_count": len(st.session_state.get("portfolio_history", [])),
        "current_codes": st.session_state.get("fund_codes_input"),
    }
    render_ai_advisor_chat(extra_context=hub_context, input_key="ai_hub_chat_input", show_trace=True)

# ==========================================
# 📦 4. 业务逻辑沙盒封装 (严格物理隔离)
# ==========================================
def render_live_timing_radar(
    codes: list,
    fund_data_dict: dict,
    portfolio_dict: dict,
    industry_dict: dict,
    audience: str = "pro",
    key_prefix: str = "timing",
):
    """
    实况择时雷达（专家 Tab / 独立「实时操作建议」共用）。
    audience='consumer'：结论与仓位变动前置，技术图折叠，面向普通投资者。
    """
    is_consumer = audience == "consumer"
    if is_consumer:
        st.subheader("买 / 卖 / 减仓建议（一图看懂）")
        st.markdown(
            "根据你的**持仓权重** + **本基金重仓股实盘**（大权重为主）给出可执行建议。"
            "行业与净值只作辅助。无需阅读原始穿透报表。"
        )
    else:
        st.subheader("📡 实况择时信号雷达 (V2.0)")
        st.markdown(
            "在你的**初始组合仓位**上叠加择时变动；"
            "**主判据=本基金重仓股行情（按占净值比例加权）**，行业代理仅辅助，净值/引擎再次之。"
            "允许组合留现金（合计可 < 100%）。"
        )

    weight_preset = st.session_state.get("portfolio_weights_preset") or {}
    if weight_preset:
        st.caption(
            "初始仓位："
            + "、".join(f"{c}:{w:.1f}%" for c, w in list(weight_preset.items())[:8])
            + ("…" if len(weight_preset) > 8 else "")
        )
    else:
        st.caption("尚未检测到组合权重，将按所选基金等权估算初始仓位。")

    name_index = get_fund_name_index()
    options = [c for c in codes if c]
    if not options:
        st.warning("请先配置基金代码。")
        return

    timing_codes = st.multiselect(
        "选择要给出买卖建议的基金",
        options=options,
        default=options,
        key=f"{key_prefix}_codes",
    )
    tr_col1, tr_col2, tr_col3 = st.columns([2, 2, 1])
    with tr_col1:
        show_timing_history = st.checkbox(
            "叠加近 N 日理论触发点（复盘，默认关）",
            value=False,
            key=f"{key_prefix}_show_history",
        )
    with tr_col2:
        history_window = st.slider(
            "复盘窗口（交易日）", 20, 120, 60, step=10, key=f"{key_prefix}_history_days"
        )
    with tr_col3:
        st.write("")
        st.write("")
        force_refresh = st.button(
            "🔄 重跑AI",
            key=f"{key_prefix}_force_refresh",
            help="清除缓存并强制重新研判",
        )

    if "timing_ai_results" not in st.session_state:
        st.session_state["timing_ai_results"] = {}
    if force_refresh:
        for k in list(st.session_state["timing_ai_results"].keys()):
            if any(k.startswith(f"{c}_") or k == c for c in timing_codes):
                del st.session_state["timing_ai_results"][k]
        try:
            fetch_stock_close_series.clear()
            get_fund_portfolio.clear()
        except Exception:
            pass
        st.rerun()

    if not timing_codes:
        st.info("请至少选择一只基金。")
        return

    prepared = []
    for code in timing_codes:
        df_nav = fund_data_dict.get(code)
        if df_nav is None or df_nav.empty:
            prepared.append({"code": code, "error": f"无法获取基金 {code} 的数据。"})
            continue

        fund_name = name_index.get(code, code)
        passive = is_passive_fund(fund_name)
        signals = compute_timing_signals_v2(df_nav, passive)
        advice = get_live_timing_advice(signals, fund_name, passive)
        snapshot = extract_timing_snapshot(signals)
        baseline_w = resolve_baseline_weight_pct(code, timing_codes)
        cache_key = _timing_ai_cache_key(code, signals, baseline_w)
        port_df = get_fund_portfolio(code)
        portfolio_dict[code] = port_df
        if "portfolio_dict" in st.session_state:
            st.session_state["portfolio_dict"][code] = port_df
        ind_df = industry_dict.get(code, pd.DataFrame())
        if ind_df is None:
            ind_df = pd.DataFrame()
        sector_proxy = build_sector_proxy_snapshot(ind_df)
        holdings_proxy = build_holdings_trend_snapshot(port_df)

        if cache_key not in st.session_state["timing_ai_results"]:
            with st.spinner(f"正在生成 {fund_name} 的买卖建议…"):
                try:
                    ai_result = run_ai_timing_judgment(
                        fund_code=code,
                        fund_name=fund_name,
                        snapshot=snapshot,
                        engine_advice=advice,
                        is_passive=passive,
                        portfolio_df=port_df,
                        industry_df=ind_df,
                        kyc_limit=st.session_state.get("max_drawdown_limit"),
                        baseline_weight_pct=baseline_w,
                        sector_proxy=sector_proxy,
                        holdings_proxy=holdings_proxy,
                    )
                    st.session_state["timing_ai_results"][cache_key] = ai_result
                except Exception as e:
                    fail = {
                        "综合决断": "研判失败",
                        "适宜度": "-",
                        "买卖点评估": str(e),
                        "大趋势": "-",
                        "仓位建议": "-",
                        "风险警示": "请检查 secrets.toml 中的 API Key 或网络连接",
                        "总监一句话": "请稍后重试",
                        "_sector_proxy": sector_proxy,
                        "_holdings_proxy": holdings_proxy,
                        "_holdings_breadth": summarize_holdings_breadth(holdings_proxy),
                    }
                    st.session_state["timing_ai_results"][cache_key] = apply_timing_position_math(
                        fail, baseline_w
                    )

        ai_cached = st.session_state["timing_ai_results"].get(cache_key)
        if ai_cached and ai_cached.get("变动后仓位%") is None:
            ai_cached = apply_timing_position_math(ai_cached, baseline_w)
            st.session_state["timing_ai_results"][cache_key] = ai_cached
        if ai_cached and (not ai_cached.get("_holdings_proxy")) and holdings_proxy:
            ai_cached = dict(ai_cached)
            ai_cached["_holdings_proxy"] = holdings_proxy
            ai_cached["_holdings_breadth"] = summarize_holdings_breadth(holdings_proxy)
            st.session_state["timing_ai_results"][cache_key] = ai_cached

        prepared.append({
            "code": code,
            "fund_name": fund_name,
            "passive": passive,
            "signals": signals,
            "advice": advice,
            "ai_cached": ai_cached,
            "baseline_w": baseline_w,
            "port_df": port_df,
            "ind_df": ind_df,
            "sector_proxy": sector_proxy,
            "holdings_proxy": holdings_proxy,
        })

    summary_rows = []
    for item in prepared:
        if item.get("error"):
            continue
        ai_cached = item["ai_cached"]
        advice = item["advice"]
        summary_rows.append({
            "基金代码": item["code"],
            "基金名称": item["fund_name"],
            "AI决断": (ai_cached or {}).get("综合决断", "研判中"),
            "适宜度": (ai_cached or {}).get("适宜度", "-"),
            "初始仓位%": (ai_cached or {}).get("初始仓位%", item["baseline_w"]),
            "建议变动pt": (ai_cached or {}).get("建议仓位变动百分点", "-"),
            "变动后仓位%": (ai_cached or {}).get("变动后仓位%", "-"),
            "总监一句话": (ai_cached or {}).get("总监一句话", "-"),
            "引擎建议": advice.get("建议"),
        })

    if summary_rows and is_consumer:
        st.write("#### 今日操作一览")

        def _color_action(val):
            s = str(val)
            if any(w in s for w in ("买入", "加仓")):
                return "background-color: #d4edda; color: #155724; font-weight: bold"
            if any(w in s for w in ("减仓", "卖出", "止盈", "不宜", "暂不", "清仓")):
                return "background-color: #f8d7da; color: #721c24; font-weight: bold"
            return ""

        df_render = pd.DataFrame(summary_rows)
        style_cols = [c for c in ("AI决断", "引擎建议") if c in df_render.columns]
        try:
            styled_df = df_render.style.map(_color_action, subset=style_cols) if style_cols else df_render
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_render, use_container_width=True, hide_index=True)
        after_vals = [
            float(r["变动后仓位%"])
            for r in summary_rows
            if isinstance(r.get("变动后仓位%"), (int, float))
        ]
        if after_vals:
            invested = round(sum(after_vals), 2)
            cash = round(max(0.0, 100.0 - invested), 2)
            c1, c2 = st.columns(2)
            c1.metric("变动后权益合计", f"{invested:.1f}%")
            c2.metric("现金/空仓余地", f"{cash:.1f}%")
        st.divider()

    for item in prepared:
        if item.get("error"):
            st.error(item["error"])
            continue
        code = item["code"]
        fund_name = item["fund_name"]
        passive = item["passive"]
        signals = item["signals"]
        advice = item["advice"]
        ai_cached = item["ai_cached"]
        port_df = item["port_df"]
        ind_df = item["ind_df"]
        sector_proxy = item["sector_proxy"]
        holdings_proxy = item["holdings_proxy"]

        st.divider()
        st.markdown(f"### {fund_name} (`{code}`)")
        if passive:
            st.caption("被动/指数类 · 引擎参考权重更高")
        else:
            st.caption("主动管理类 · 重仓股（按占比）为主判据")

        if ai_cached:
            render_ai_timing_verdict(ai_cached)

        if not is_consumer:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("引擎状态", advice.get("状态", "-"))
            m2.metric("引擎建议", advice.get("建议", "-"))
            m3.metric("引擎参考仓位", advice.get("目标仓位", "-"))
            m4.metric("RSI14 / MA30", f"{advice.get('RSI14', '-')} / {advice.get('MA30趋势', '-')}")

        with st.expander("查看依据（重仓股 / 行业 / 引擎）", expanded=False):
            st.caption(f"引擎触发：{advice.get('触发原因', '-')}")
            if holdings_proxy:
                st.markdown("**重仓股单股行情**")
                st.dataframe(pd.DataFrame(holdings_proxy), hide_index=True, use_container_width=True)
            if not ind_df.empty:
                st.markdown("**重仓行业**")
                st.dataframe(ind_df.head(5), hide_index=True, use_container_width=True)
            if sector_proxy:
                st.markdown("**行业代理指数**")
                st.dataframe(pd.DataFrame(sector_proxy), hide_index=True, use_container_width=True)
            if not port_df.empty:
                st.markdown("**季报持仓名单**")
                st.dataframe(port_df.head(10), hide_index=True, use_container_width=True)

        chart_box = st.expander("技术面看盘（净值 / 布林 / RSI）", expanded=not is_consumer)
        with chart_box:
            fig_timing = plot_timing_radar_chart(
                signals, code, fund_name,
                show_history=show_timing_history,
                history_days=history_window,
            )
            st.plotly_chart(fig_timing, use_container_width=True)

    if summary_rows and not is_consumer:
        st.divider()
        st.write("#### 多基金仓位变动一览（初始 → 择时后）")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        after_vals = [
            float(r["变动后仓位%"])
            for r in summary_rows
            if isinstance(r.get("变动后仓位%"), (int, float))
        ]
        if after_vals:
            invested = round(sum(after_vals), 2)
            cash = round(max(0.0, 100.0 - invested), 2)
            over = round(max(0.0, invested - 100.0), 2)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("变动后权益合计", f"{invested:.1f}%")
            mc2.metric("现金/空仓余地", f"{cash:.1f}%")
            if over > 0:
                mc3.metric("超额提示", f"+{over:.1f}%")
            else:
                mc3.metric("是否需满仓", "否，可留现金" if cash > 0 else "已近满仓")


def run_timing_action_desk():
    """独立「实时操作择时建议」：无需加载完整专家模式。"""
    st.title("📡 实时操作择时建议")
    st.markdown(
        "给普通人的买卖清单：**今天这只基该加、该减还是先空着**。"
        "直接加载历史组合或输入代码即可，不必进入专家模式看报表。"
    )

    ensure_portfolio_history_loaded()
    history = st.session_state.get("portfolio_history") or []

    with st.container(border=True):
        st.markdown("##### ① 选组合 / 填代码")
        src = st.radio(
            "标的来源",
            ["历史组合", "手动输入代码"],
            horizontal=True,
            key="timing_desk_src",
        )
        codes = []
        if src == "历史组合":
            if not history:
                st.info("暂无历史组合。可先手动输入代码，或在 AI 副驾 / 专家模式生成并保存方案。")
            else:
                names = [h.get("name", f"组合{i+1}") for i, h in enumerate(history)]
                pick = st.selectbox("选择历史组合", names, key="timing_desk_hist_pick")
                item = history[names.index(pick)]
                codes_str = item.get("codes_str") or ", ".join(
                    str(c) for c in (item.get("codes") or [])
                )
                st.caption(f"成分：{codes_str}")
                if st.button("载入该组合权重并测算", type="primary", key="timing_desk_load_hist"):
                    apply_portfolio_from_history(item, switch_to_fof=False)
                    st.rerun()
                codes = [
                    c.strip()
                    for c in str(st.session_state.get("fund_codes_input", "")).split(",")
                    if c.strip()
                ]
                if not codes:
                    raw_codes = item.get("codes") or [x.strip() for x in codes_str.split(",") if x.strip()]
                    codes = [str(c).strip() for c in raw_codes if str(c).strip()]
                    w = item.get("weights") or item.get("weights_pct") or {}
                    if w:
                        st.session_state["portfolio_weights_preset"] = {
                            str(k): float(v) for k, v in w.items()
                        }
        else:
            raw = st.text_input(
                "基金代码（逗号分隔）",
                value=st.session_state.get("fund_codes_input", ""),
                key="timing_desk_codes_input",
                placeholder="例如 005827, 110011, 000001",
            )
            codes = [c.strip() for c in raw.split(",") if c.strip()]
            if codes:
                st.session_state["fund_codes_input"] = ", ".join(codes)

    if not codes:
        st.warning("请先选择历史组合或填写基金代码。")
        return

    preset = st.session_state.get("portfolio_weights_preset") or {}
    with st.expander("② 初始仓位（可改，默认等权）", expanded=not bool(preset)):
        new_w = {}
        cols = st.columns(min(4, max(1, len(codes))))
        for i, code in enumerate(codes):
            init = float(preset.get(code, round(100.0 / len(codes), 2)))
            new_w[code] = cols[i % len(cols)].number_input(
                f"{code} %",
                min_value=0.0,
                max_value=100.0,
                value=init,
                step=1.0,
                key=f"timing_desk_w_{code}",
            )
        if st.button("应用上述仓位", key="timing_desk_apply_w"):
            st.session_state["portfolio_weights_preset"] = new_w
            st.success("已更新初始仓位")
            st.rerun()
        if not st.session_state.get("portfolio_weights_preset"):
            st.session_state["portfolio_weights_preset"] = {
                c: round(100.0 / len(codes), 2) for c in codes
            }

    refresh_token = st.session_state.get("data_refresh_token", 0)
    with st.spinner("加载基金净值与持仓数据…"):
        _rf, _bench, fund_data_dict, industry_dict, portfolio_dict, _mgr = ensure_engine_data(
            codes, refresh_token
        )

    st.markdown("##### ③ 操作建议")
    render_live_timing_radar(
        codes=codes,
        fund_data_dict=fund_data_dict,
        portfolio_dict=portfolio_dict,
        industry_dict=industry_dict,
        audience="consumer",
        key_prefix="desk_timing",
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("去专家模式看完整穿透/回测", use_container_width=True, key="desk_to_pro"):
            st.session_state["fund_codes_input"] = ", ".join(codes)
            st.session_state["engine_checkbox"] = True
            request_app_mode_switch("FOF 穿透与归因")
            st.rerun()
    with c2:
        if st.button("回 AI 副驾继续对话", use_container_width=True, key="desk_to_ai"):
            request_app_mode_switch("🧠 AI 智能投顾")
            st.rerun()

def run_fof_dashboard():
    """专家模式 Pro：FOF 穿透归因看板（读取 AI 注入的 session_state）。"""
    import time # 引入时间模块生成时间戳
    
    # --- 升级：从本地 JSON 加载历史组合记忆库 ---
    if "portfolio_history" not in st.session_state:
        st.session_state["portfolio_history"] = load_local_history()

    st.title("📊 专家模式 (Pro) · FOF 穿透与归因")
    loaded = st.session_state.get("loaded_portfolio_name")
    if loaded:
        st.success(f"当前参数来源：**{loaded}**（可由 AI 副驾或历史快载注入）")
    st.markdown("多维度穿透底层资产。基金代码与权重优先由 **AI 副驾** 注入；也可在下方手动调整。")
    st.divider()

    # —— 业务输入后置到主区；状态会写入 pro_workspace.json，切回 AI 也不会丢 ——
    has_codes = bool(str(st.session_state.get("fund_codes_input", "")).strip())
    with st.expander("⚙️ 组合配置 / 搜基 / 引擎 / 保存", expanded=True):
        cfg1, cfg2 = st.columns([2, 1])
        with cfg1:
            fund_inputs = st.text_input(
                "拟配置基金代码（逗号分隔）",
                key="fund_codes_input",
            )
            save_c1, save_c2 = st.columns([3, 1])
            with save_c1:
                save_name = st.text_input(
                    "快速保存并命名当前组合",
                    value=st.session_state.get("loaded_portfolio_name") or "",
                    placeholder="例如：稳收益50万方案",
                    key="pro_quick_save_name",
                )
            with save_c2:
                st.write("")
                st.write("")
                if st.button("💾 保存", key="pro_quick_save_btn", use_container_width=True):
                    codes_now = [c.strip() for c in str(st.session_state.get("fund_codes_input", "")).split(",") if c.strip()]
                    result = quick_save_named_portfolio(save_name, codes=codes_now)
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.success(f"已保存为「{result['name']}」，可在左侧历史快载一键回顾")
                        st.rerun()
            search_kw = st.text_input(
                "全市场模糊搜索（拼音/名称/主题）",
                placeholder="例如：蓝筹 / 煤炭 / 易方达…",
                key="fund_search_kw",
            )
            if search_kw:
                all_funds_df = load_all_fund_names()
                if not all_funds_df.empty:
                    matched = all_funds_df[
                        all_funds_df['基金简称'].str.contains(search_kw, na=False, case=False)
                        | all_funds_df['基金代码'].str.contains(search_kw, na=False)
                    ]
                    if matched.empty:
                        st.warning(f"未找到包含「{search_kw}」的基金。")
                    else:
                        st.dataframe(matched.head(10), hide_index=True, use_container_width=True)
                        st.caption("复制上表【基金代码】填入上方输入框即可。")
        with cfg2:
            engine_on = st.checkbox("🚀 开启量化引擎", key="engine_checkbox")
            if st.button("刷新数据缓存", use_container_width=True, key="clear_data_cache"):
                st.cache_data.clear()
                st.session_state.pop("engine_sig", None)
                st.session_state.pop("fund_data_dict", None)
                st.session_state.pop("tab5_backtest", None)
                if os.path.exists(BENCHMARK_CACHE_FILE):
                    try:
                        os.remove(BENCHMARK_CACHE_FILE)
                    except OSError:
                        pass
                st.session_state["data_refresh_token"] = st.session_state.get("data_refresh_token", 0) + 1
                save_pro_workspace()
                st.rerun()
            if st.button("返回 AI 副驾", use_container_width=True, key="pro_back_copilot"):
                save_pro_workspace()
                request_app_mode_switch("🧠 AI 智能投顾")
                st.rerun()
            if st.button("仅保存工作区", use_container_width=True, key="pro_save_workspace"):
                save_pro_workspace(silent=False)
    # 任一配置变化都落盘，保证切回 AI 再进 Pro 不丢
    save_pro_workspace()

    codes = [c.strip() for c in str(st.session_state.get("fund_codes_input", "")).split(",") if c.strip()]
    refresh_token = st.session_state.get("data_refresh_token", 0)
    engine_on = st.session_state.get("engine_checkbox", False)

    if not engine_on:
        st.info("请在上方展开 **「组合配置」** 勾选 **开启量化引擎**，或让 AI 副驾一键注入后跳转。")
        return

    if not codes:
        st.warning("尚未配置基金代码。可回 AI 副驾让它生成方案并跳转，或在上方手动输入。")
        return

    with st.spinner("⏳ 首次加载市场基准与基金数据（仅一次）…"):
        rf_rate, bench_df, fund_data_dict, industry_dict, portfolio_dict, manager_dict = ensure_engine_data(
            codes, refresh_token
        )

    m1, m2 = st.columns(2)
    m1.info(f"🌐 动态无风险利率：**{rf_rate*100:.3f}%**（10年期国债）")
    if bench_df is None:
        m2.warning("沪深300基准获取失败，Tab4 Alpha 将受限。可点「刷新数据缓存」。")
    else:
        m2.success("沪深300基准已就绪，Alpha 引擎可用。")

    tab1, tab2, tab3, tab4, tab5, tab_timing, tab6 = st.tabs([
        "🧩 单体净值与持仓透视", "🤝 组合相关性矩阵", "⚖️ 宏观板块对冲雷达",
        "🧠 经理真实任期归因", "🏆 投资组合整体回测", "📡 实况择时雷达", "📚 历史档案与对比",
    ])

    # TAB 1: 单体透视
    with tab1:
        for code in codes:
            df = fund_data_dict.get(code)
            if df is None or df.empty:
                st.error(f"❌ 无法获取基金 {code} 的数据。")
                continue
                
            st.subheader(f"🏷️ 基金代码: {code}")
            
            chart_data = df.set_index('净值日期')['单位净值']
            st.line_chart(chart_data, use_container_width=True)
            
            metrics_tot = calculate_risk_metrics(df, rf_rate)
            one_year_ago = df['净值日期'].max() - pd.DateOffset(years=1)
            metrics_1y = calculate_risk_metrics(df[df['净值日期'] >= one_year_ago], rf_rate)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🏃 成立以来 (穿越牛熊指标)")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("累计收益率", f"{metrics_tot['总收益率']:.2f}%")
                mc2.metric("最大回撤", f"{metrics_tot['最大回撤']:.2f}%")
                mc3.metric("真实夏普", f"{metrics_tot['真实夏普']:.3f}")
            with col2:
                st.markdown("#### ⚡ 近一年 (近期战斗力)")
                mc4, mc5, mc6 = st.columns(3)
                mc4.metric("近一年收益", f"{metrics_1y['总收益率']:.2f}%")
                mc5.metric("近一年回撤", f"{metrics_1y['最大回撤']:.2f}%")
                mc6.metric("真实夏普", f"{metrics_1y['真实夏普']:.3f}")
            
            st.markdown("##### 📦 底层资产穿透")
            pc1, pc2 = st.columns(2)
            with pc1:
                ind_df = industry_dict.get(code)
                if not ind_df.empty:
                    st.write("**前五大板块配置**")
                    st.dataframe(ind_df.head(5), use_container_width=True, hide_index=True)
                else:
                    st.write("暂无板块数据")
                with pc2:
                    port_df = portfolio_dict.get(code, pd.DataFrame())
                if not port_df.empty:
                    st.write("**前十大重仓个股**")
                    st.dataframe(port_df, use_container_width=True, hide_index=True)
                else:
                    st.write("暂无个股数据")
            st.divider()

    # TAB 2: 组合化学反应
    with tab2:
        st.subheader("🔗 历年绝对收益对比 与 底层日度相关系数矩阵")
        
        # 分离两个 DataFrame：一个看年度表象，一个算日度真实摩擦
        annual_merged_df = pd.DataFrame()
        daily_merged_df = pd.DataFrame()

        for code in codes:
            df = fund_data_dict.get(code)
            if df is not None and not df.empty:
                # --- 1. 结算历年绝对收益 (供直观展示) ---
                df['年份'] = df['净值日期'].dt.year
                df['日收益率'] = df['日增长率'] / 100.0
                annual_ret = df.groupby('年份')['日收益率'].apply(lambda x: (1 + x).prod() - 1)
                annual_ret.name = f'[{code}]_年度收益率'
                
                if annual_merged_df.empty:
                    annual_merged_df = pd.DataFrame(annual_ret)
                else:
                    annual_merged_df = pd.merge(annual_merged_df, annual_ret, on='年份', how='outer')
                
                # --- 2. 拼接底层日度涨跌幅 (供精准计算相关性) ---
                # 仅提取日期和当天的涨跌幅，将日期设为索引以便对齐
                temp_daily = df[['净值日期', '日增长率']].copy()
                temp_daily = temp_daily.rename(columns={'日增长率': f'[{code}]'})
                temp_daily = temp_daily.set_index('净值日期')
                
                if daily_merged_df.empty:
                    daily_merged_df = temp_daily
                else:
                    # 使用 inner join，严格对齐在同一天都有交易记录的数据点
                    daily_merged_df = pd.merge(daily_merged_df, temp_daily, left_index=True, right_index=True, how='inner')
        
        # 展示：历年绝对收益对比图
        if not annual_merged_df.empty:
            st.write("**历年绝对收益对比图**")
            # 按年份倒序排列，填补缺失值
            annual_merged_df = annual_merged_df.sort_index(ascending=False).fillna(0)
            display_df = (annual_merged_df * 100).map(lambda x: f"{x:.2f}%" if x != 0 else "缺失")
            st.dataframe(display_df, use_container_width=True)
            
        # 展示：真实的皮尔逊相关系数矩阵
        if not daily_merged_df.empty:
            # 加入统计学防呆设计：只有共同运行天数超过30天，计算相关性才有意义
            if len(daily_merged_df) >= 30:
                st.write(f"**底层相关系数矩阵 (热力图) —— 基于 {len(daily_merged_df)} 个共同交易日的真实波动计算**")
                corr_matrix = daily_merged_df.corr()
                # 限制热力图的色阶范围在 -1 到 1 之间，保留4位小数体现细腻度
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn_r', vmin=-1, vmax=1).format("{:.4f}"), use_container_width=True)
                st.info("💡 **配置原则提示**：颜色越绿，说明两只基金互补对冲能力越强；颜色越红(接近1)，说明它们是同质化的高风险绑定。")
            else:
                st.warning(f"⚠️ 两只基金共同运行的交易日过少（仅 {len(daily_merged_df)} 天），无法计算具备统计学意义的相关系数。")
                
    # TAB 3: 宏观板块雷达
    with tab3:
        st.subheader("🏢 宏观板块配置重叠度扫描")
        ind_merged = pd.DataFrame()
        for code in codes:
            df = industry_dict.get(code)
            if df is not None and not df.empty:
                temp = df[['行业类别', '占净值比例']].copy()
                temp['占净值比例'] = pd.to_numeric(temp['占净值比例'], errors='coerce')
                temp = temp.rename(columns={'占净值比例': f'[{code}]_权重(%)'})
                ind_merged = temp if ind_merged.empty else pd.merge(ind_merged, temp, on='行业类别', how='outer')
                
        if not ind_merged.empty:
            ind_merged = ind_merged.fillna(0)
            ind_merged['最高权重'] = ind_merged.drop(columns=['行业类别']).max(axis=1)
            ind_merged = ind_merged[ind_merged['最高权重'] >= 2.0].drop(columns=['最高权重'])
            first_col = [col for col in ind_merged.columns if '权重' in col][0]
            ind_merged = ind_merged.sort_values(by=first_col, ascending=False).reset_index(drop=True)
            
            st.dataframe(ind_merged.style.highlight_max(axis=1, color='lightgreen', subset=[c for c in ind_merged.columns if '权重' in c]), use_container_width=True)
            st.info("💡 **架构师点评**：横向对比每一行，如果多只基金在某一板块权重都很高，这就是你组合最大的系统性风险敞口！")

    # TAB 4: 基金经理任期归因 与 风格画像
    with tab4:
        st.subheader("🧠 剥离时代红利，拷问基金经理真实 Alpha 与 风险画像")
        st.markdown(
            "在 **Alpha / Beta / 最大回撤** 基础上，系统将进一步反推 "
            "**🚀 最佳持有周期** 与 **💰 资金承载能力**，辅助您匹配正确的资金体量与持有节奏。"
        )
        results = []
        for code in codes:
            config = manager_dict.get(code)
            df_fund = fund_data_dict.get(code)
            if not config or df_fund is None or df_fund.empty:
                continue

            start_date = pd.to_datetime(config['start_date'])
            df_m = df_fund[df_fund['净值日期'] >= start_date]
            if df_m.empty:
                continue

            fund_name = get_fund_display_name(code)
            industry_df = industry_dict.get(code, pd.DataFrame())
            portfolio_df = portfolio_dict.get(code, pd.DataFrame())

            fund_ret = (df_m['单位净值'].iloc[-1] / df_m['单位净值'].iloc[0]) - 1
            manager_metrics = calculate_risk_metrics(df_m, rf_rate)
            manager_max_dd = manager_metrics.get("最大回撤", 0)

            bench_ret = None
            alpha = None
            bench_max_dd = None

            if bench_df is not None and not bench_df.empty:
                bench_m = bench_df[bench_df['净值日期'] >= start_date]
                if not bench_m.empty:
                    merged = pd.merge(df_m[['净值日期', '单位净值']], bench_m, on='净值日期', how='inner')
                    if not merged.empty:
                        fund_ret = (merged['单位净值'].iloc[-1] / merged['单位净值'].iloc[0]) - 1
                        bench_ret = (merged['基准点数'].iloc[-1] / merged['基准点数'].iloc[0]) - 1
                        alpha = fund_ret - bench_ret
                        merged['bench_max'] = merged['基准点数'].cummax()
                        bench_drawdown = (merged['基准点数'] - merged['bench_max']) / merged['bench_max']
                        bench_max_dd = bench_drawdown.min() * 100

            style_tag = "未知"
            risk_level = "中"

            if manager_max_dd < -35:
                risk_level = "极高风险 (大开大合)"
            elif manager_max_dd < -20:
                risk_level = "中高风险 (进取波动)"
            elif manager_max_dd < -10:
                risk_level = "中低风险 (均衡控制)"
            else:
                risk_level = "低风险 (严控回撤)"

            if bench_max_dd is not None and alpha is not None:
                if manager_max_dd > bench_max_dd and alpha > 0.1:
                    style_tag = "🛡️ 稳健画线派 (回撤小于大盘且有超额)"
                elif manager_max_dd <= bench_max_dd and alpha > 0.15:
                    style_tag = "⚔️ 锐度进攻派 (高波动换取高超额)"
                elif manager_max_dd <= bench_max_dd and alpha < 0:
                    style_tag = "⚠️ 裸多单边派 (承担更高风险却跑输大盘)"
                elif manager_max_dd > bench_max_dd and alpha < 0:
                    style_tag = "🐢 钝化防守派 (回撤小但无进攻能力)"
                else:
                    style_tag = "⚖️ 均衡跟随派 (贴合基准波动)"
            else:
                style_tag = "无基准暂不评定"

            holding_duration = assess_holding_duration(style_tag, fund_name, industry_df)
            capital_capacity = assess_capital_capacity(
                code, df_fund, manager_max_dd, portfolio_df,
                fund_name, industry_df, style_tag, risk_level, rf_rate
            )

            results.append({
                "🎯 基金代码": code,
                "👤 现任掌舵人": config['name'],
                "📅 上任日期": config['start_date'],
                "💰 任职绝对收益": f"{fund_ret*100:.2f}%",
                "📈 同期大盘(Beta)": f"{bench_ret*100:.2f}%" if bench_ret is not None else "缺失",
                "🔥 超额能力(Alpha)": f"{alpha*100:.2f}%" if alpha is not None else "缺失",
                "🛡️ 任职最大回撤": f"{manager_max_dd:.2f}%",
                "📊 风险评级": risk_level,
                "🏷️ 演算投资风格": style_tag,
                "🚀 最佳持有周期": holding_duration,
                "💰 资金承载能力": capital_capacity,
            })

        if results:
            result_df = pd.DataFrame(results)
            st.dataframe(style_tab4_dataframe(result_df), use_container_width=True, hide_index=True)
            st.markdown("""
| 标签 | 含义速查 |
|------|----------|
| ⚡ **短期战术波段型** | 高波行业主题 + 进攻风格，适合 1-3 个月战术操作 |
| 🏰 **中长期战略底仓型** | 画线派 / 红利宽基，适合 6 个月以上底仓持有 |
| 🐋 **鲸鱼级大资金友好** | 低回撤 + 千亿蓝筹重仓，可承载 50 万+ 大额 |
| 🐜 **游资蚂蚁型** | 高回撤 + 高换手，单笔建议不超过 5 万元 |
            """)
            st.success(
                "✅ **架构师点评**：Alpha 决定「值不值得买」，"
                "🚀 持有周期决定「拿多久」，💰 资金承载决定「放多少钱」。"
                "三者同时匹配，才是合格的 FOF 配置决策。"
            )
        else:
            st.warning("暂未获取到有效的任期归因数据。")
            # TAB 5: 投资组合整体回测 (无损融合版：含动态择时、费率控制与 AI 诊断)
    
    with tab5:
        st.subheader("🏆 投资组合整体实盘试算 (双核择时 + 摩擦熔断)")
        st.markdown("采用 **MA30 趋势 + 布林带均值回归 + RSI 动量** 双核择时，并叠加 **7 日惩罚费率 / 30 日摩擦熔断** 防线。")
        
        # 1. 提取并严格对齐所有基金的每日涨跌幅数据
        valid_codes = [code for code in codes if fund_data_dict.get(code) is not None and not fund_data_dict.get(code).empty]
        
        if not valid_codes:
            st.warning("⚠️ 没有足够的数据来构建投资组合，请检查基金代码。")
        else:
            # ========================================
            # 💡 步骤一：调节资金配额与账户实际费率 (%)
            # ========================================
            st.write("#### ⚖️ 步骤一：调节资金配额与摩擦成本 (%)")
            
            with st.form("weight_allocation_form"):
                # 三列宽度的表头，文字全展示
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.markdown("**🧩 基金代码**")
                col2.markdown("**📥 买入费率(%)**")
                col3.markdown("**📤 卖出费率(%)**")
                
                weights_input, buy_fees, sell_fees = {}, {}, {}
                weight_preset = st.session_state.get("portfolio_weights_preset") or {}
                if weight_preset and set(weight_preset.keys()) != set(valid_codes):
                    weight_preset = {}
                    st.session_state["portfolio_weights_preset"] = None
                default_w = 100.0 / len(valid_codes)
                
                for code in valid_codes:
                    c1, c2, c3 = st.columns([2, 1, 1])
                    init_w = float(weight_preset.get(code, default_w))
                    with c1:
                        weights_input[code] = st.number_input(
                            f"{code} 权重(%)", value=init_w,
                            min_value=0.0, max_value=100.0, step=1.0, key=f"w_{code}",
                        )
                    with c2:
                        buy_fees[code] = st.number_input("买费(%)", value=0.15, min_value=0.0, max_value=5.0, step=0.01, key=f"bf_{code}") / 100.0
                    with c3:
                        sell_fees[code] = st.number_input("卖费(%)", value=0.50, min_value=0.0, max_value=5.0, step=0.01, key=f"sf_{code}") / 100.0
                        
                submitted = st.form_submit_button("✅ 确认并拉起量化回测引擎")
            
            total_w = sum(weights_input.values())
            weights = {k: v / total_w for k, v in weights_input.items()} if total_w > 0 else {k: 1.0/len(valid_codes) for k in valid_codes}
            if total_w != 100.0:
                st.info(f"💡 提示：权重总和为 {total_w:.2f}%，系统已在后台自动归一化为 100%。")

            weights_sig = (
                tuple(sorted((c, round(weights[c], 4)) for c in valid_codes)),
                tuple(sorted((c, round(buy_fees[c], 5)) for c in valid_codes)),
                tuple(sorted((c, round(sell_fees[c], 5)) for c in valid_codes)),
            )

            if submitted or st.session_state.pop("auto_run_tab5_backtest", False):
                with st.spinner("⏳ 正在运行双核择时回测引擎…"):
                    st.session_state["tab5_backtest"] = backtest_portfolio_with_friction_guard(
                        valid_codes, fund_data_dict, weights, buy_fees, sell_fees
                    )
                    st.session_state["tab5_weights_sig"] = weights_sig
                    if st.session_state.get("loaded_portfolio_name"):
                        st.success(f"📂 已自动回测载入组合：**{st.session_state['loaded_portfolio_name']}**")

            backtest_bundle = st.session_state.get("tab5_backtest")
            if backtest_bundle is None or st.session_state.get("tab5_weights_sig") != weights_sig:
                st.info("👆 请调节权重与费率后，点击 **「确认并拉起量化回测引擎」** 开始回测（避免每次操作都重算）。")
            else:
                port_daily_df, signal_cache, friction_warnings = backtest_bundle
                if port_daily_df is None or port_daily_df.empty:
                    st.warning("⚠️ 组合数据对齐失败，请检查基金代码是否有足够的共同交易日。")
                else:
                    for msg in friction_warnings:
                        st.warning(msg)

                nav_df = pd.DataFrame(index=port_daily_df.index)
                nav_df['🌟 动态择时策略 (智能防守)'] = (1 + port_daily_df['[我的组合]_日收益']).cumprod()
                nav_df['🐢 静态持有组合 (原始基准)'] = (1 + port_daily_df['[静态持有]_日收益']).cumprod()

                for code in valid_codes:
                    nav_df[f'单体: {code}'] = (1 + port_daily_df[f'{code}_原始']).cumprod()

                st.divider()
                st.write("#### 📈 步骤二：高级组合净值走势对比")
                st.caption("组合层仅对比「动态择时 vs 静态持有」净值曲线。单只基金的 MA/布林带实况信号请前往 **📡 实况择时雷达** Tab。")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df['🌟 动态择时策略 (智能防守)'],
                                         mode='lines', name='🌟 动态择时策略', line=dict(width=3, color='#e74c3c')))
                fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df['🐢 静态持有组合 (原始基准)'],
                                         mode='lines', name='🐢 静态持有(装死)', line=dict(width=2, color='#95a5a6', dash='dash')))

                fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                st.write("#### 📊 步骤三：组合风险与收益最终评估")
                eval_df = pd.DataFrame({
                    '净值日期': port_daily_df.index,
                    '日增长率': port_daily_df['[我的组合]_日收益'] * 100.0,
                    '单位净值': nav_df['🌟 动态择时策略 (智能防守)']
                })
                port_metrics = calculate_risk_metrics(eval_df, rf_rate)

                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("🌟 组合累计收益", f"{port_metrics.get('总收益率', 0):.2f}%")
                pm2.metric("🛡️ 组合最大回撤", f"{port_metrics.get('最大回撤', 0):.2f}%")
                pm3.metric("⚔️ 组合真实夏普", f"{port_metrics.get('真实夏普', 0):.3f}")

                st.success("✅ **架构师点评**：对比上面卡片的数据和单只基金的数据。如果你发现【组合最大回撤】小于你买的那些基金的平均回撤，说明你的动态择时防守生效了！")
                st.divider()

                user_limit = st.session_state.get('max_drawdown_limit')
                actual_drawdown = port_metrics.get('最大回撤', 0)
                if user_limit is not None:
                    if actual_drawdown < user_limit:
                        st.error(f"⚠️ **红色警报**：该组合的真实最大回撤为 **{actual_drawdown:.2f}%**，已经击穿了您向 AI 投顾承诺的心理底线（**{user_limit:.2f}%**）！建议立刻调低高波基金权重，或增加防守型固收资产！")
                    else:
                        st.success(f"🛡️ **风控通过**：组合最大回撤（{actual_drawdown:.2f}%）安全控制在您的心理预期（{user_limit:.2f}%）之内。")

                st.write("#### 🤖 步骤四：AI 配置建议")
                st.info(
                    "💡 **完整 AI 投顾已升级至独立中枢**，可自动调用检索、量化、宏观沙盘、组合回测四大工具。"
                    "请在左侧控制台切换至 **「🧠 AI 智能投顾」** 进行对话。"
                )
                if st.button("🧠 前往 AI 智能投顾中枢", key="goto_ai_hub"):
                    request_app_mode_switch("🧠 AI 智能投顾")
                    if port_metrics:
                        st.session_state["agent_pending_query"] = (
                            f"请基于当前组合 {valid_codes}、权重 {[round(weights[c]*100) for c in valid_codes]}%、"
                            f"累计收益 {port_metrics.get('总收益率', 0):.1f}%、"
                            f"最大回撤 {port_metrics.get('最大回撤', 0):.1f}% 给出优化建议。"
                        )
                    st.rerun()

                if submitted:
                    combo_signature = {c: weights[c] for c in valid_codes}
                    existing_idx = None
                    for idx, item in enumerate(st.session_state["portfolio_history"]):
                        if item["signature"] == combo_signature:
                            existing_idx = idx
                            break

                    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_record = {
                        "name": st.session_state["portfolio_history"][existing_idx]["name"] if existing_idx is not None else f"自选组合 {current_time_str[5:16]}",
                        "signature": combo_signature,
                        "codes_str": " + ".join([f"{c}({w*100:.0f}%)" for c, w in combo_signature.items()]),
                        "return": port_metrics.get('总收益率', 0),
                        "drawdown": port_metrics.get('最大回撤', 0),
                        "sharpe": port_metrics.get('真实夏普', 0),
                        "timestamp": current_time_str
                    }

                    if existing_idx is not None:
                        st.session_state["portfolio_history"].pop(existing_idx)
                    st.session_state["portfolio_history"].insert(0, new_record)
                    save_local_history(st.session_state["portfolio_history"])

                st.divider()
                st.write("#### 🔍 步骤六：RBS 风格漂移透视雷达 (主动基金照妖镜)")
                st.markdown("放弃滞后的季报，利用 60 日滚动相关性 (Rolling Correlation) 算法，实时反推基金经理的底层持仓是否发生严重偏移。")

                if st.button("🚀 启动 RBS 风格透视雷达"):
                    with st.spinner("正在利用算法反推基金经理底层持仓，请稍候..."):
                        style_df = get_style_benchmarks()
                        if style_df is not None and not style_df.empty:
                            for code in valid_codes:
                                fund_daily = port_daily_df[[f'{code}_原始']].dropna()
                                eval_data = pd.merge(fund_daily, style_df, left_index=True, right_index=True, how='inner')

                                if len(eval_data) > 60:
                                    rolling_corr = pd.DataFrame(index=eval_data.index)
                                    style_cols = [c for c in eval_data.columns if c != f'{code}_原始']

                                    for style in style_cols:
                                        rolling_corr[style] = eval_data[f'{code}_原始'].rolling(window=60).corr(eval_data[style])

                                    fig_rbs = go.Figure()
                                    colors = ['#3498db', '#f1c40f', '#e74c3c']

                                    for idx, col in enumerate(style_cols):
                                        fig_rbs.add_trace(go.Scatter(
                                            x=rolling_corr.index,
                                            y=rolling_corr[col],
                                            mode='lines',
                                            name=col,
                                            line=dict(width=2, color=colors[idx % len(colors)])
                                        ))

                                    fig_rbs.update_layout(
                                        height=400,
                                        title=f"🎯 基金 [{code}] 历史风格拟合轨迹",
                                        yaxis_title="相关系数 (越接近1拟合度越高)",
                                        hovermode="x unified",
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    st.plotly_chart(fig_rbs, use_container_width=True)
                                else:
                                    st.warning(f"基金 {code} 运行时间不足 60 天，无法启动 RBS 雷达。")
                        else:
                            st.error("⚠️ 风格基准数据拉取失败，请检查网络或更换 API。")

    # ==========================================
    # TAB: 实况择时信号雷达 (V2.0) —— 复用独立面板
    # ==========================================
    with tab_timing:
        render_live_timing_radar(
            codes=codes,
            fund_data_dict=fund_data_dict,
            portfolio_dict=portfolio_dict,
            industry_dict=industry_dict,
            audience="pro",
            key_prefix="pro_timing",
        )

    # ==========================================
    # TAB 6: 历史组合档案与对比雷达 (新增)
    # ==========================================
    with tab6:
        st.subheader("📚 历史组合记忆库与风险收益雷达")
        st.markdown("在这里管理你测算过的所有配置。通过跨组合的“风险收益散点图”，直观挑选最契合你风险承受能力的终极方案。")
        
        if not st.session_state["portfolio_history"]:
            st.info("💡 暂无组合历史记录。请先在【🏆 投资组合整体回测】页面调节配额并点击 [确认并应用当前配额] 按钮。")
        else:
            st.write("#### 🚀 步骤零：一键加载回顾（等同重新输入组合）")
            st.caption("点击「加载回顾」将自动填入侧边栏基金代码、权重，并开启量化引擎 + 运行回测。")
            for idx, item in enumerate(st.session_state["portfolio_history"]):
                lc1, lc2, lc3, lc4 = st.columns([3, 2, 1, 1])
                with lc1:
                    st.markdown(f"**{item.get('name', '未命名')}**")
                with lc2:
                    st.caption(
                        f"{item.get('codes_str', '')}  \n"
                        f"收益 {item.get('return', 0):.1f}% · 回撤 {item.get('drawdown', 0):.1f}% · "
                        f"{item.get('timestamp', '')}"
                    )
                with lc3:
                    if st.button("📂 加载", key=f"tab6_load_{idx}", use_container_width=True):
                        apply_portfolio_from_history(item)
                        st.rerun()
                with lc4:
                    if st.button("🗑️ 删除", key=f"tab6_del_{idx}", use_container_width=True):
                        name = item.get("name", "未命名")
                        if delete_portfolio_history_item(idx):
                            st.toast(f"已删除：{name}", icon="🗑️")
                            st.rerun()
            st.divider()
            st.write("#### 📝 步骤一：管理与自定义命名你的组合")
            hist_df = pd.DataFrame(st.session_state["portfolio_history"])
            
            # 组装展示给用户的交互式表格
            hist_df = pd.DataFrame(st.session_state["portfolio_history"])
            display_df = hist_df[['name', 'codes_str', 'return', 'drawdown', 'sharpe', 'timestamp']].copy()
            display_df.columns = ['🏷️ 组合自定义名称 (点此修改)', '🧩 配置明细', '📈 累计收益(%)', '🛡️ 最大回撤(%)', '⚔️ 真实夏普', '⏳ 最后测算时间']
            
            # ==========================================
            # 修复核心：引入专业的回调函数 (Callback)
            # ==========================================
            def update_portfolio_names():
                changes = st.session_state["history_editor"].get("edited_rows", {})
                has_changed = False  # 标记是否真的发生了修改
                
                for row_idx, edit_dict in changes.items():
                    if '🏷️ 组合自定义名称 (点此修改)' in edit_dict:
                        st.session_state["portfolio_history"][row_idx]['name'] = edit_dict['🏷️ 组合自定义名称 (点此修改)']
                        has_changed = True
                        
                # === 新增：如果名字真的改了，就把最新状态固化到硬盘 ===
                if has_changed:
                    save_local_history(st.session_state["portfolio_history"])

            # 渲染可编辑表格，并将它和上面的回调函数死死绑定
            st.data_editor(
                display_df,
                disabled=['🧩 配置明细', '📈 累计收益(%)', '🛡️ 最大回撤(%)', '⚔️ 真实夏普', '⏳ 最后测算时间'],
                use_container_width=True,
                key="history_editor",
                on_change=update_portfolio_names  # <--- 机构级前端的魔法就在这一行
            )

            st.divider()
            st.write("#### ⚖️ 步骤二：跨组合散点对比 (评估风险接受度)")
            
            # 提取所有已被命名的组合，提供多选框
            all_names = [item['name'] for item in st.session_state["portfolio_history"]]
            selected_names = st.multiselect(
                "🔍 勾选你需要放入竞技场的组合 (默认全选)：", 
                options=all_names,
                default=all_names
            )
            
            if selected_names:
                # 过滤出被选中的数据集合
                compare_data = [item for item in st.session_state["portfolio_history"] if item['name'] in selected_names]
                comp_df = pd.DataFrame(compare_data)
                
                # 为了图表直观，将回撤取绝对值作为X轴风险
                comp_df['风险敞口(最大回撤绝对值%)'] = comp_df['drawdown'].abs()
                comp_df['获取报酬(累计收益率%)'] = comp_df['return']
                comp_df['组合标识'] = comp_df['name']
                
                # 绘制风险/收益散点图 (利用原生图表引擎)
                st.scatter_chart(
                    comp_df, 
                    x='风险敞口(最大回撤绝对值%)', 
                    y='获取报酬(累计收益率%)', 
                    color='组合标识',
                    size=250
                )
                
                st.success("✅ **架构师点评**：在这张【风险 vs 收益】散点图中，**左上角**是资管的终极圣杯（承担极小的回撤，换取极高的收益）。\n\n**如何做决策？**\n* 如果你风险承受能力低：请果断选择横坐标最靠左的组合，守住本金；\n* 如果你追求极致进攻：请选择纵坐标最靠上的组合。")


def run_macro_sandbox():
    """宏观状态机：连续变量推演引擎"""
    st.title("🧭 宏观状态机：连续变量推演系统")
    st.markdown("底层采用敏感度矩阵模型。拉动滑块进行微调，右侧仓位会根据设定的[敏感系数]实时计算资金流向。")
    st.divider()
    
    # 实例化模型
    model = MacroAttributionModel()

    st.sidebar.header("🎛️ 调节宏观变量刻度")
    st.sidebar.markdown("*(滑块初始值对应现实基准数据)*")

    current_factors = {}

    current_factors["真实利率"] = st.sidebar.slider(
        "真实利率 (名义利率 - 通胀)", 
        min_value=-1.0, max_value=1.0, value=model.baseline_factors["真实利率"], step=0.1,
        help="负数=倒挂亏购买力，0=中性，正数=高真实利率压制估值",
        key="macro_slider_real_rate",
    )

    current_factors["科技预期"] = st.sidebar.slider(
        "科技/AI预期", 
        min_value=-1.0, max_value=1.0, value=model.baseline_factors["科技预期"], step=0.1,
        help="-1=泡沫破裂，0=回归基本面，1=极度狂热",
        key="macro_slider_tech",
    )

    current_factors["政策流动性"] = st.sidebar.slider(
        "政策流动性", 
        min_value=-1.0, max_value=1.0, value=model.baseline_factors["政策流动性"], step=0.1,
        help="-1=大幅缩表加息，0=中性，1=极端放水",
        key="macro_slider_liquidity",
    )

    current_factors["经济动能"] = st.sidebar.slider(
        "经济增长动能", 
        min_value=-1.0, max_value=1.0, value=model.baseline_factors["经济动能"], step=0.1,
        help="-1=深度衰退，0=企稳，1=全面过热",
        key="macro_slider_growth",
    )

    current_factors["地缘风险"] = st.sidebar.slider(
        "地缘博弈与风险", 
        min_value=0.0, max_value=1.0, value=model.baseline_factors["地缘风险"], step=0.1,
        help="0=平稳/预期修复，1=黑天鹅避险",
        key="macro_slider_geopolitics",
    )

    # 核心计算
    result = model.simulate(current_factors)

    # 主展示区
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 矩阵触发日志")
        if not result["logs"]:
            st.info("当前因子与现实基准一致，保持基础配置权重。")
        else:
            st.warning("⚠️ 侦测到偏离基准的变量微调：")
            for log in result["logs"]:
                st.write(f"- {log}")
                
        st.divider()
        with st.expander("👀 查看底层敏感系数 (Sensitivity Matrix)"):
            st.json(model.sensitivity_matrix)

    with col2:
        st.subheader("📊 实时推演配额 (基于矩阵乘数)")
        
        df_alloc = pd.DataFrame({
            "大类资产": ["权益资产", "固收资产", "现金资产"],
            "仓位比重 (%)": [result["allocation"]["权益资产"], result["allocation"]["固收资产"], result["allocation"]["现金资产"]]
        }).set_index("大类资产")
        
        df_eq = pd.DataFrame({
            "权益内部": ["红利低波", "高端制造", "科技AI"],
            "仓位比重 (%)": [result["equity_structure"]["红利低波"], result["equity_structure"]["高端制造"], result["equity_structure"]["科技AI"]]
        }).set_index("权益内部")
        
        df_bd = pd.DataFrame({
            "固收内部": ["信用债", "利率债"],
            "仓位比重 (%)": [result["bond_structure"]["信用债"], result["bond_structure"]["利率债"]]
        }).set_index("固收内部")

        tab_a, tab_b, tab_c = st.tabs(["🌐 大类资产总体分布", "⚔️ 权益内部结构", "🛡️ 固收内部结构"])
        
        with tab_a:
            st.bar_chart(df_alloc, color="#1f77b4")
        with tab_b:
            st.bar_chart(df_eq, color="#ff7f0e")
        with tab_c:
            st.bar_chart(df_bd, color="#2ca02c")

# ==========================================
# 🚦 5. 主控路由 (状态机)
# ==========================================
def render_global_sidebar_kyc():
    """全局侧边栏 KYC（各模式共用，避免再塞进专家看板业务区）。"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ KYC 回撤底线")
    limit = st.session_state.get("max_drawdown_limit")
    if limit is not None:
        st.sidebar.success(f"已锁定：**{abs(limit):.0f}%**")
    user_risk_text = st.sidebar.text_area(
        "大白话描述亏损底线",
        placeholder="例如：最多亏15%…",
        key="kyc_risk_text",
        height=68,
    )
    if st.sidebar.button("✨ 测算底线", key="sidebar_kyc_btn"):
        if user_risk_text.strip():
            try:
                client = get_deepseek_client()
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": f"提取最大回撤百分比数字，只返回数字：{user_risk_text}"}],
                    temperature=0.1,
                )
                val = float(response.choices[0].message.content.strip().replace("%", ""))
                st.session_state["max_drawdown_limit"] = -abs(val)
                st.sidebar.success(f"底线：**{abs(val):.0f}%**")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"解析失败：{e}")
        else:
            st.sidebar.warning("请先输入描述")

def main():
    st.sidebar.title("🎛️ 投研中枢")
    st.sidebar.caption("副驾对话主导 · 专家面板深挖")

    mode_options = ["🧠 AI 智能投顾", "📡 实时操作建议", "FOF 穿透与归因", "宏观敏感度沙盘"]
    mode_labels = {
        "🧠 AI 智能投顾": "🧠 AI 副驾 (Copilot)",
        "📡 实时操作建议": "📡 买卖择时（推荐）",
        "FOF 穿透与归因": "📊 专家模式 (Pro)",
        "宏观敏感度沙盘": "🧭 宏观沙盘",
    }
    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = mode_options[0]

    # 仅当按钮/AI 主动跳转时，才在创建 radio 前写入控件值。
    # 若每轮都覆盖 main_app_mode_radio，会把用户在侧边栏的点击改回去。
    if st.session_state.pop("_force_mode_sync", False):
        desired = st.session_state.get("app_mode", mode_options[0])
        if desired not in mode_options:
            desired = mode_options[0]
            st.session_state["app_mode"] = desired
        st.session_state["main_app_mode_radio"] = desired
    elif "main_app_mode_radio" not in st.session_state:
        st.session_state["main_app_mode_radio"] = st.session_state["app_mode"]

    app_mode = st.sidebar.radio(
        "工作模式",
        mode_options,
        format_func=lambda x: mode_labels.get(x, x),
        key="main_app_mode_radio",
    )
    st.session_state["app_mode"] = app_mode

    render_global_sidebar_kyc()
    render_history_quick_load_sidebar()
    st.sidebar.divider()

    if app_mode == "🧠 AI 智能投顾":
        run_ai_advisor_hub()
    elif app_mode == "📡 实时操作建议":
        run_timing_action_desk()
    elif app_mode == "FOF 穿透与归因":
        run_fof_dashboard()
    elif app_mode == "宏观敏感度沙盘":
        run_macro_sandbox()

if __name__ == "__main__":
    main()