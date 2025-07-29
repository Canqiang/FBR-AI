# FBR AI Growth Engine Demo

```
fbr-ai-growth-engine/
├── config/
│   ├── __init__.py
│   ├── config.json.template      # 配置模板
│   ├── config.json               # 实际配置（gitignore）
│   └── settings.py               # 配置管理
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── connectors.py        # 数据库连接器
│   │   ├── repositories.py      # 数据访问层
│   │   └── models.py           # 数据模型
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── predictive.py        # 预测分析
│   │   ├── causal.py           # 因果推断
│   │   ├── anomaly.py          # 异常检测
│   │   └── optimization.py     # 优化算法
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── azure_client.py     # Azure OpenAI客户端
│   │   ├── prompts.py          # Prompt模板
│   │   ├── agents.py           # LLM代理
│   │   └── chains.py           # LangChain集成
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── core.py             # 核心引擎
│   │   ├── scheduler.py        # 任务调度
│   │   └── pipelines.py        # 分析流水线
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI应用
│   │   ├── routes.py           # API路由
│   │   └── schemas.py          # 请求/响应模型
│   │
│   └── ui/
│       ├── __init__.py
│       ├── streamlit_app.py    # Streamlit界面
│       └── components.py        # UI组件
│
├── tests/
│   ├── __init__.py
│   ├── test_analytics.py
│   ├── test_llm.py
│   └── test_engine.py
│
├── scripts/
│   ├── setup_db.py             # 数据库初始化
│   └── run_demo.py             # Demo运行脚本
│
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── README.md
└── .gitignore
```