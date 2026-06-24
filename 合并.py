import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import datetime 
import requests 
import json  # 新增：用于处理本地记忆库
import os    # 新增：用于检测本地文件是否存在

# ==========================================
# 💾 0. 硬盘持久化读写中枢 (绝对路径增强版)
# ==========================================
import os
import json

# 魔法修正：获取当前“合并.py”文件所在的绝对路径目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 强制将 json 文件锁定在这个目录下
HISTORY_FILE = os.path.join(CURRENT_DIR, "portfolio_history.json")

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

def save_local_history(history_data):
    """有变动时将内存状态写入硬盘，并弹出视觉反馈"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
        # 加上极其关键的视觉反馈，右下角会弹出提示！
        st.toast("💾 历史记录已成功写入本地硬盘！", icon="✅")
    except Exception as e:
        st.error(f"⚠️ 写入本地硬盘失败: {e}")
# ==========================================
# ⚙️ 1. 全局唯一配置 (系统的“锚”)
# ==========================================
st.set_page_config(page_title="📊 FOF 机构级资产配置与归因看板", page_icon="📈", layout="wide")

# ==========================================
# 🧠 2. 数据获取与缓存层 (公共基础设施)
# ==========================================
@st.cache_data(ttl=3600)
def get_dynamic_risk_free_rate() -> float:
    try:
        bond_df = ak.bond_zh_us_rate()
        return float(bond_df['中国国债收益率10年'].dropna().iloc[-1]) / 100.0
    except:
        return 0.02

@st.cache_data(ttl=3600)
def get_benchmark_data():
    try:
        df = ak.index_zh_a_hist(symbol="000300", period="daily")
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.rename(columns={'日期': '净值日期', '收盘': '基准点数'})
        return df[['净值日期', '基准点数']].sort_values(by='净值日期')
    except Exception:
        pass 
        
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'date': '净值日期', 'close': '基准点数'})
        return df[['净值日期', '基准点数']].sort_values(by='净值日期')
    except Exception:
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
                    top10 = df[df[time_col] == latest].head(10)
                else:
                    top10 = df.head(10)
                ratio_col = '占净值比例' if '占净值比例' in top10.columns else '占净值比例(%)'
                return top10.rename(columns={ratio_col: '占净值比例'})[['股票名称', '占净值比例']]
        except:
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
# 📦 4. 业务逻辑沙盒封装 (严格物理隔离)
# ==========================================
def run_fof_dashboard():
    """FOF 机构级资产配置与归因引擎"""
    import time # 引入时间模块生成时间戳
    
    # --- 升级：从本地 JSON 加载历史组合记忆库 ---
    if "portfolio_history" not in st.session_state:
        st.session_state["portfolio_history"] = load_local_history()

    st.title("📊 FOF 机构级资产配置与归因看板")
    st.markdown("通过多维度量化分析，穿透底层资产，揭示基金经理真实能力与组合对冲效应。")
    st.divider()

    st.sidebar.header("⚙️ 投资组合配置区")
    fund_inputs = st.sidebar.text_input("输入拟配置基金代码 (逗号分隔)", "005827, 005844")
    codes = [c.strip() for c in fund_inputs.split(",") if c.strip()]

    rf_rate = get_dynamic_risk_free_rate()
    st.sidebar.info(f"🌐 当前动态无风险利率: **{rf_rate*100:.3f}%**\n\n(锚定: 中国10年期国债)")
    bench_df = get_benchmark_data()
    
    if bench_df is None:
        st.sidebar.error("⚠️ 沪深300大盘基准获取失败。")
    else:
        st.sidebar.success("✅ 沪深300大盘基准接入成功！Alpha 引擎已就绪。")

    if st.sidebar.checkbox("🚀 开启量化引擎", value=False):
        if not codes:
            st.warning("请在左侧输入至少一只基金代码！")
            st.stop()

        # --- 新增：加入了第六个 Tab 标签 ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🧩 单体净值与持仓透视", "🤝 组合相关性矩阵", "⚖️ 宏观板块对冲雷达", "🧠 经理真实任期归因", "🏆 投资组合整体回测", "📚 历史档案与对比"])
        fund_data_dict = {}
        industry_dict = {}

        with st.spinner('⏳ 正在向金融数据库发起请求并清洗数据，请稍候...'):
            for code in codes:
                fund_data_dict[code] = get_fund_clean_data(code)
                industry_dict[code] = get_fund_industry(code)

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
                    port_df = get_fund_portfolio(code)
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
            results = []
            for code in codes:
                config = get_fund_manager_info(code)
                df_fund = fund_data_dict.get(code)
                if not config or df_fund is None or df_fund.empty: 
                    continue
                    
                start_date = pd.to_datetime(config['start_date'])
                df_m = df_fund[df_fund['净值日期'] >= start_date]
                if df_m.empty: 
                    continue
                
                # 1. 计算经理任期内的绝对收益
                fund_ret = (df_m['单位净值'].iloc[-1] / df_m['单位净值'].iloc[0]) - 1
                
                # 2. 计算经理任期内的核心风险指标 (复用引擎计算最大回撤)
                manager_metrics = calculate_risk_metrics(df_m, rf_rate)
                manager_max_dd = manager_metrics.get("最大回撤", 0)
                
                bench_ret = None
                alpha = None
                bench_max_dd = None
                
                # 3. 引入大盘对标，进行 Beta、Alpha 及抗风险能力计算
                if bench_df is not None and not bench_df.empty:
                    bench_m = bench_df[bench_df['净值日期'] >= start_date]
                    if not bench_m.empty:
                        merged = pd.merge(df_m[['净值日期', '单位净值']], bench_m, on='净值日期', how='inner')
                        if not merged.empty:
                            fund_ret = (merged['单位净值'].iloc[-1] / merged['单位净值'].iloc[0]) - 1
                            bench_ret = (merged['基准点数'].iloc[-1] / merged['基准点数'].iloc[0]) - 1
                            alpha = fund_ret - bench_ret
                            
                            # 计算同期大盘的最大回撤，用于对比
                            merged['bench_max'] = merged['基准点数'].cummax()
                            bench_drawdown = (merged['基准点数'] - merged['bench_max']) / merged['bench_max']
                            bench_max_dd = bench_drawdown.min() * 100

                # 4. 智能评估：投资风格与风险极性判定引擎
                style_tag = "未知"
                risk_level = "中"
                
                # 风险极性判定 (绝对回撤幅度)
                if manager_max_dd < -35:
                    risk_level = "极高风险 (大开大合)"
                elif manager_max_dd < -20:
                    risk_level = "中高风险 (进取波动)"
                elif manager_max_dd < -10:
                    risk_level = "中低风险 (均衡控制)"
                else:
                    risk_level = "低风险 (严控回撤)"
                    
                # 投资风格判定 (相对大盘的表现特征)
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

                results.append({
                    "🎯 基金代码": code,
                    "👤 现任掌舵人": config['name'],
                    "📅 上任日期": config['start_date'],
                    "💰 任职绝对收益": f"{fund_ret*100:.2f}%",
                    "📈 同期大盘(Beta)": f"{bench_ret*100:.2f}%" if bench_ret is not None else "缺失",
                    "🔥 超额能力(Alpha)": f"{alpha*100:.2f}%" if alpha is not None else "缺失",
                    "🛡️ 任职最大回撤": f"{manager_max_dd:.2f}%",
                    "📊 风险评级": risk_level,
                    "🏷️ 演算投资风格": style_tag
                })
                
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                st.success("✅ **架构师点评**：同期大盘 (Beta)可以横向对比：如果他的绝对收益是 30%，大盘也是 30%，那 Alpha 就是 0，风险评级再高也不值得买；只有那些 Beta 一般但 Alpha 极高，且打着『🛡️ 稳健画线派』标签的人，才是真正的配置核心。")
            else:
                st.warning("暂未获取到有效的任期归因数据。")
                # TAB 5: 投资组合整体回测 (从分到总)
        with tab5:
            st.subheader("🏆 投资组合整体实盘试算 (截至最新刷新日)")
            st.markdown("通过动态分配权重，将多只基金融合成一个整体策略，检验配置后的真实收益与抗风险能力。")
            
            # 1. 提取并严格对齐所有基金的每日涨跌幅数据
            port_daily_df = pd.DataFrame()
            valid_codes = []
            
            for code in codes:
                df = fund_data_dict.get(code)
                if df is not None and not df.empty:
                    temp_daily = df[['净值日期', '日增长率']].copy()
                    temp_daily['日增长率'] = temp_daily['日增长率'] / 100.0  # 转化为小数参与计算
                    temp_daily = temp_daily.rename(columns={'日增长率': code})
                    temp_daily = temp_daily.set_index('净值日期')
                    
                    if port_daily_df.empty:
                        port_daily_df = temp_daily
                    else:
                        port_daily_df = pd.merge(port_daily_df, temp_daily, left_index=True, right_index=True, how='inner')
                    valid_codes.append(code)
            
            if port_daily_df.empty or len(valid_codes) < 1:
                st.warning("⚠️ 没有足够的数据来构建投资组合，请检查基金代码。")
            else:
                # 2. 动态权重配置 UI (引入表单缓冲与精细刻度)
                st.write("#### ⚖️ 步骤一：调节资金配额 (%)")
                
                # 使用 st.form 建立缓冲池，拦截每一次加减导致的强制刷新
                with st.form("weight_allocation_form"):
                    cols = st.columns(len(valid_codes))
                    weights_input = {}
                    default_w = 100.0 / len(valid_codes)
                    
                    for i, code in enumerate(valid_codes):
                        with cols[i]:
                            # step=1.0 满足更细致的调节粒度，用户也可手动输入小数
                            weights_input[code] = st.number_input(
                                f"{code} 仓位占比", 
                                value=default_w, 
                                min_value=0.0, 
                                max_value=100.0, 
                                step=1.0
                            )
                    
                    # 只有点击这个确认按钮，系统才会拿着最新的权重去计算净值并重绘网页
                    submitted = st.form_submit_button("✅ 确认并应用当前配额")
                
                # 权重归一化 (防呆设计：防止用户输入的总和不等于100%)
                total_w = sum(weights_input.values())
                weights = {k: v / total_w for k, v in weights_input.items()} if total_w > 0 else {k: 1.0/len(valid_codes) for k in valid_codes}
                
                if total_w != 100.0:
                    st.info(f"💡 提示：您当前输入的权重总和为 {total_w:.2f}%，系统已在后台自动归一化为 100% 进行严谨试算。")
                # 3. 计算投资组合的整体每日涨跌幅与净值曲线
                port_daily_df['[我的组合]_日收益'] = 0.0
                for code in valid_codes:
                    port_daily_df['[我的组合]_日收益'] += port_daily_df[code] * weights[code]
                
                # 计算比较用的累计净值曲线 (起点归一化为 1.0)
                nav_df = pd.DataFrame(index=port_daily_df.index)
                nav_df['🌟 组合整体策略'] = (1 + port_daily_df['[我的组合]_日收益']).cumprod()
                for code in valid_codes:
                    nav_df[f'单体: {code}'] = (1 + port_daily_df[code]).cumprod()
                
                st.divider()
                st.write("#### 📈 步骤二：组合 vs 单体净值走势对比 (起点统一为 1.0)")
                st.line_chart(nav_df, use_container_width=True)
                
                # 4. 计算并输出最终的组合风险评估
                st.write("#### 📊 步骤三：组合风险与收益最终评估")
                
                # 构造符合之前计算引擎格式的临时 DataFrame
                eval_df = pd.DataFrame({
                    '净值日期': port_daily_df.index,
                    '日增长率': port_daily_df['[我的组合]_日收益'] * 100.0,  # 还原为百分比喂给引擎
                    '单位净值': nav_df['🌟 组合整体策略']
                })
                
                port_metrics = calculate_risk_metrics(eval_df, rf_rate)
                
                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("🌟 组合累计收益", f"{port_metrics.get('总收益率', 0):.2f}%")
                pm2.metric("🛡️ 组合最大回撤", f"{port_metrics.get('最大回撤', 0):.2f}%")
                pm3.metric("⚔️ 组合真实夏普", f"{port_metrics.get('真实夏普', 0):.3f}")
                
                st.success("✅ **架构师点评**：对比上面卡片的数据和单只基金的数据。如果你发现【组合最大回撤】小于你买的那些基金的平均回撤，并且【组合真实夏普】变高了，这说明你配置的几只基金起到了完美的“内部化学反应”，你成功用更低的风险换取了更高的收益效率！")
                # ==========================================
                # 修复：将自动归档与更新机制 严格绑定在按钮提交事件上
                # ==========================================
                if submitted:  # <--- 加上这个至关重要的状态锁
                    # 为当前配置生成唯一的签名
                    combo_signature = {c: weights[c] for c in valid_codes}
                    existing_idx = None
                    
                    # 遍历历史记录，如果发现配额和代码完全一致的旧记录，就打上标记
                    for idx, item in enumerate(st.session_state["portfolio_history"]):
                        if item["signature"] == combo_signature:
                            existing_idx = idx
                            break
                    
                    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 组装当前的配置档案
                    new_record = {
                        "name": st.session_state["portfolio_history"][existing_idx]["name"] if existing_idx is not None else f"自选组合 {current_time_str[5:16]}",
                        "signature": combo_signature,
                        "codes_str": " + ".join([f"{c}({w*100:.0f}%)" for c, w in combo_signature.items()]),
                        "return": port_metrics.get('总收益率', 0),
                        "drawdown": port_metrics.get('最大回撤', 0),
                        "sharpe": port_metrics.get('真实夏普', 0),
                        "timestamp": current_time_str
                    }
                    
                    # 若存在同配额的旧记录，先剔除；无论新旧，都把这条最新档案插入到历史记录的【最顶部】
                    if existing_idx is not None:
                        st.session_state["portfolio_history"].pop(existing_idx)
                    st.session_state["portfolio_history"].insert(0, new_record)
                    
                    # === 新增：将最新状态同步到本地硬盘 ===
                    save_local_history(st.session_state["portfolio_history"])
        # ==========================================
        # TAB 6: 历史组合档案与对比雷达 (新增)
        # ==========================================
        with tab6:
            st.subheader("📚 历史组合记忆库与风险收益雷达")
            st.markdown("在这里管理你测算过的所有配置。目前公用存档但具体信息不会暴露 通过跨组合的“风险收益散点图”，直观挑选最契合你风险承受能力的终极方案。")
            
            if not st.session_state["portfolio_history"]:
                st.info("💡 暂无组合历史记录。请先在【🏆 投资组合整体回测】页面调节配额并点击 [确认并应用当前配额] 按钮。")
            else:
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
        help="负数=倒挂亏购买力，0=中性，正数=高真实利率压制估值"
    )

    current_factors["科技预期"] = st.sidebar.slider(
        "科技/AI预期", 
        min_value=-1.0, max_value=1.0, value=model.baseline_factors["科技预期"], step=0.1,
        help="-1=泡沫破裂，0=回归基本面，1=极度狂热"
    )

    current_factors["政策流动性"] = st.sidebar.slider(
        "政策流动性", 
        min_value=-1.0, max_value=1.0, value=model.baseline_factors["政策流动性"], step=0.1,
        help="-1=大幅缩表加息，0=中性，1=极端放水"
    )

    current_factors["经济动能"] = st.sidebar.slider(
        "经济增长动能", 
        min_value=-1.0, max_value=1.0, value=model.baseline_factors["经济动能"], step=0.1,
        help="-1=深度衰退，0=企稳，1=全面过热"
    )

    current_factors["地缘风险"] = st.sidebar.slider(
        "地缘博弈与风险", 
        min_value=0.0, max_value=1.0, value=model.baseline_factors["地缘风险"], step=0.1,
        help="0=平稳/预期修复，1=黑天鹅避险"
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
def main():
    st.sidebar.title("🎛️ 投研中枢控制台")
    
    app_mode = st.sidebar.radio(
        "请选择量化分析引擎",
        ["FOF 穿透与归因", "宏观敏感度沙盘"],
        index=0
    )
    
    st.sidebar.divider() 
    
    # 路由分发 
    if app_mode == "FOF 穿透与归因":
        run_fof_dashboard()
    elif app_mode == "宏观敏感度沙盘":
        run_macro_sandbox()

if __name__ == "__main__":
    main()
