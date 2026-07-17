# FOF 智能投顾 / 量化看板

本文件夹包含该应用的**全部运行文件**，与 `profolio` 下其他项目（如期货雷达）相互独立。

## 目录说明

| 文件/文件夹 | 说明 |
|-------------|------|
| `AI量化.py` | 主程序入口 |
| `knowledge_rag.py` | 自定义知识库 (RAG) |
| `requirements.txt` | Python 依赖 |
| `secrets.toml` | API 密钥（勿提交 Git） |
| `portfolio_history.json` | 历史组合记录 |
| `benchmark_000300_cache.json` | 沪深300基准缓存 |
| `my_investment_brain/` | 自定义知识库文档（可选） |
| `knowledge_chroma/` | 向量索引（自动生成） |
| `legacy/` | 旧版/实验脚本，不参与运行 |

## 启动方式

```bash
cd FOF智能投顾
pip install -r requirements.txt
streamlit run AI量化.py
```

Windows 也可双击 **`启动.bat`**。

## 在 Cursor 中打开

建议将工作区根目录设为 **`FOF智能投顾`** 或 **`profolio`**，便于与其他项目区分。
