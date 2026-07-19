# CampusMind Web - 校园智能问答助手

基于 RAG + ReAct Agent 的校园智能问答系统，支持学生手册知识库检索、多轮对话、会话管理等功能。

## 功能特性

- 🔐 **用户认证**：注册、登录、密码修改
- 🤖 **智能问答**：基于 ReAct Agent 的多轮对话，支持流式输出
- 📚 **RAG 检索**：知识库文档检索，支持 PDF、Word、TXT 格式
- 💬 **会话管理**：历史会话持久化、会话切换、重命名、删除
- 🔍 **消息搜索**：在当前会话中搜索历史消息，关键词高亮
- 👁️ **密码显示**：登录/注册时支持密码显示/隐藏切换
- 📱 **响应式布局**：适配桌面和移动设备
- ⚡ **流式响应**：实时展示思考过程、工具调用和答案输出

## 技术栈

### 后端
- FastAPI - Web 框架
- SQLAlchemy + SQLite - 数据库
- LangChain + ChromaDB - RAG 向量检索
- DashScope (通义千问) - LLM 模型
- ReAct Agent - 智能推理框架

### 前端
- 原生 HTML / CSS / JavaScript
- Tailwind CSS - 样式框架

## 项目结构

```
campusmind-web/
├── backend/
│   ├── main.py              # FastAPI 主入口
│   ├── database.py          # 数据库模型
│   ├── config/
│   │   └── config.yaml      # 配置文件
│   ├── rag/
│   │   ├── react_agent.py   # ReAct Agent
│   │   ├── vector_store.py  # 向量库管理
│   │   └── tools/           # 工具集
│   └── services/            # 业务服务
├── web/
│   └── index.html           # 前端页面
├── requirements.txt         # Python 依赖
├── render.yaml              # Render 部署配置
└── README.md
```

## 快速开始

### 环境要求
- Python 3.11+
- DashScope API Key

### 本地运行

1. **克隆项目**
```bash
git clone https://github.com/purple0165/campusmind.git
cd campusmind
```

2. **创建虚拟环境并安装依赖**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. **配置 API Key**

编辑 `backend/config/config.yaml`，设置你的 DashScope API Key：
```yaml
model:
  api_key: "your-dashscope-api-key"
  chat_model: "qwen-turbo"
```

4. **启动服务**
```bash
python backend/main.py
```

5. **访问应用**

打开浏览器访问 http://localhost:8080

## 部署到 Render

### 一键部署

1. Fork 本仓库到你的 GitHub
2. 登录 [Render](https://render.com/)
3. 点击 **New +** → **Web Service**
4. 选择你的 `campusmind` 仓库并连接
5. 配置部署参数：
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python backend/main.py`
   - **Instance Type**: Free（免费版）
6. 添加环境变量：
   - `DASHSCOPE_API_KEY` = 你的 DashScope API Key
   - `PYTHON_VERSION` = `3.11.6`
7. 点击 **Create Web Service** 等待部署完成

### 免费版限制
- 15 分钟无活动后服务休眠，首次访问需等待唤醒
- 磁盘存储为临时，重启后数据（数据库、向量库）会丢失
- 每月有限额的运行时长

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/change-password` | POST | 修改密码 |
| `/api/query/react/stream` | POST | 流式问答（ReAct） |
| `/api/sessions` | GET | 获取会话列表 |
| `/api/sessions/{session_id}` | GET | 获取会话详情 |
| `/api/sessions` | POST | 创建会话 |
| `/api/sessions/{session_id}/rename` | POST | 重命名会话 |
| `/api/sessions/{session_id}/delete` | POST | 删除会话 |
| `/api/session/clear` | POST | 清空当前会话 |
| `/api/documents/upload` | POST | 上传文档 |
| `/api/documents` | GET | 获取文档列表 |
| `/api/health` | GET | 健康检查 |

## 使用说明

1. **注册账号**：首次使用请先注册一个账号
2. **上传文档**：在文档管理页面上传学生手册等知识库文档（PDF/Word/TXT）
3. **开始对话**：在聊天框输入问题，AI 会基于知识库进行智能回答
4. **查看思考过程**：点击 AI 回复下方的"思考过程"可查看 ReAct Agent 的推理步骤
5. **管理会话**：左侧边栏可创建、切换、重命名、删除会话
6. **搜索消息**：点击顶部搜索图标，可在当前会话中搜索历史消息

## 许可证

MIT License
