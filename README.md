# ESP_System
面向电商服务场景，设计并实现一个基于多智能体协同的智能服务系统，覆盖售前商品咨询、售中线下服务触达、售后技术支持三类核心业务能力。系统支持商品搜索与对比、服务站查询与导航、私域知识库问答、多轮会话记忆和流式交互展示，具备较强的业务扩展性与工程落地能力。

## 项目简介

这个项目不是单一问答机器人，而是一个基于 `LangGraph` 搭建的多智能体协同系统。系统会先识别用户意图，再将问题分发给最合适的专家智能体处理：

- `technical_specialist`：处理技术支持、故障排查、知识库问答
- `service_specialist`：处理服务站查询、位置解析、线下导航
- `product_specialist`：处理商品搜索、详情查询、商品对比
- `main_supervisor`：负责意图判断、任务编排和最终回答整合

项目同时提供独立的知识库服务与两个前端应用：

- `agent_web_ui`：多智能体对话界面
- `knowlege_platform_ui`：知识库上传与管理界面

## 核心亮点

- 基于 `LangGraph + langgraph-supervisor` 实现调度型多智能体架构
- 支持短期会话记忆、长期语义记忆和结构化用户画像
- 技术问答支持独立 RAG 服务，包含向量检索、BM25 检索和重排能力
- 对话接口使用 `SSE` 流式输出，可展示处理过程与最终答案
- 支持 `HITL` 人工审核中断与恢复，便于高风险场景兜底
- 将地图、服务站、商品查询等能力封装为工具，支持智能体按权限调用
- 提供 RAG 检索评估与 `ragas` 答案质量评估脚本

## 系统架构

```text
User
  -> agent_web_ui
  -> FastAPI (backend/app)
  -> main_supervisor
      -> technical_specialist
          -> knowledge service
          -> web search
      -> service_specialist
          -> service station tools
          -> map / navigation tools
      -> product_specialist
          -> product search tools
  -> SSE stream back to frontend

knowledge_platform_ui
  -> FastAPI (backend/knowledge)
  -> document upload / split / embedding / retrieval / evaluation
```

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 后端主服务 | FastAPI, LangGraph, LangChain, Pydantic |
| 记忆与状态 | LangMem, PostgreSQL checkpoint, Redis user profile |
| 知识库 | LlamaIndex, Milvus/向量存储, Elasticsearch(BM25), RAGAS |
| 前端 | Vue 3, Vite, Element Plus, Marked |
| 工具接入 | MCP, 地图能力, 商品检索, 本地服务工具 |

## 主要能力

### 1. 多智能体路由

系统会根据用户输入自动判断当前问题属于技术支持、服务导航还是商品咨询，并将任务交给对应专家智能体处理。

### 2. 长短期记忆

- 短期记忆：基于 LangGraph 的会话状态管理
- 长期记忆：基于语义召回存储用户长期偏好、历史问题和上下文
- 用户画像：支持 Redis 持久化结构化用户信息

### 3. 技术知识库问答

知识库服务支持文档上传、切分、向量化、混合检索和答案生成，适合 IT 支持、内部知识问答等场景。

### 4. 服务站与导航

服务智能体可以解析用户位置、查询附近服务站，并生成导航信息，适合售后、维修和线下到店场景。

### 5. 商品查询与对比

商品智能体支持商品搜索、详情查询和对比推荐，适合电商或导购类问题。

### 6. 流式交互与人工审核

前端可以实时展示处理过程；当命中高风险场景时，系统支持进入人工审核并继续执行。

## 目录结构

```text
its_multi_agent/
├─ backend/
│  ├─ app/                  # 多智能体主服务
│  │  ├─ api/               # 对话 / 会话 / HITL 接口
│  │  ├─ graph/             # LangGraph 编排、记忆、工具、状态
│  │  ├─ multi_agent/       # 兼容层入口
│  │  ├─ prompts/           # 各智能体提示词
│  │  └─ services/          # 对话执行、会话存储等
│  └─ knowledge/            # 独立知识库服务
│     ├─ api/               # 文档上传、知识查询接口
│     ├─ services/          # ingestion / retrieval / query
│     ├─ evaluation/        # 检索评估与 ragas 评估
│     └─ docker-compose.yml # Milvus / ES / Redis 等依赖
├─ front/
│  ├─ agent_web_ui/         # 对话前端
│  └─ knowlege_platform_ui/ # 知识库平台前端
```

## 关键接口

### 主服务 `backend/app`

- `POST /api/query`：流式多智能体对话
- `POST /api/hitl/resume`：人工审核后恢复执行
- `POST /api/hitl/export`：导出审核回流样本
- `POST /api/user_sessions`：获取用户历史会话

### 知识库服务 `backend/knowledge`

- `POST /upload`：上传知识文档并入库
- `POST /query`：知识库问答

## 快速启动

### 1. 安装后端依赖

```powershell
cd backend/app
pip install -r requirements.txt

cd ../knowledge
pip install -e .
```

### 2. 安装前端依赖

```powershell
cd front/agent_web_ui
npm install

cd ../knowlege_platform_ui
npm install
```

### 3. 启动基础组件

在 `backend/knowledge` 目录下启动知识库相关依赖：

```powershell
docker compose up -d
```

默认会启动：

- Milvus
- Elasticsearch
- Redis
- MinIO
- etcd

### 4. 配置环境变量

主服务 `backend/app/.env` 至少需要配置一组可用模型服务：

```env
SF_API_KEY=
SF_BASE_URL=
MAIN_MODEL_NAME=Qwen/Qwen3-32B
KNOWLEDGE_BASE_URL=http://127.0.0.1:8001
BAIDUMAP_AK=
```

或：

```env
AL_BAILIAN_API_KEY=
AL_BAILIAN_BASE_URL=
```

知识库服务 `backend/knowledge/.env` 常用配置示例：

```env
API_KEY=
BASE_URL=
MODEL=
EMBEDDING_MODEL=
BM25_ELASTICSEARCH_URL=http://localhost:9200
```

### 5. 启动服务

启动主服务：

```powershell
cd backend/app
python -m api.main
```

启动知识库服务：

```powershell
cd backend/knowledge
python -m api.main
```

启动对话前端：

```powershell
cd front/agent_web_ui
npm run dev
```

启动知识库平台前端：

```powershell
cd front/knowlege_platform_ui
npm run dev
```

默认端口：

- 主服务：`127.0.0.1:8000`
- 知识库服务：`127.0.0.1:8001`
- 知识库平台前端：`127.0.0.1:3000`

## 评估能力

项目内置两类评估流程：

- 检索评估：用于衡量召回质量，如 `Recall@k`、`MRR@k`
- RAGAS 评估：用于衡量最终回答质量，如 `faithfulness`、`answer_relevancy`

示例：

```powershell
cd backend/knowledge
python -m evaluation.eval_rag --dataset evaluation/sample_eval_dataset.jsonl
python -m evaluation.eval_ragas --dataset evaluation/sample_ragas_dataset.jsonl --output evaluation/last_ragas_result.json
```

## 适用场景

- 企业 IT 支持与知识问答
- 售后服务与线下网点导航
- 商品导购、搜索与对比
- 需要多智能体协同与工具调用的智能客服场景

## 后续可继续优化

- 增加统一的根目录 `.gitignore`，避免提交 `node_modules`、日志、向量库数据和本地依赖目录
- 为 README 补充系统截图、接口时序图和演示 GIF
- 增加 Docker 化一键启动脚本，降低部署门槛


