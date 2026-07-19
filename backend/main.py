import os
import sys
import uuid
import json
from pathlib import Path
from typing import Dict, Generator, List, Optional

_backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_backend_dir))
sys.path.insert(0, str(_backend_dir.parent))

import yaml
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from auth.auth_service import AuthService
from db.database import init_db

init_db()

_config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if _config_path.exists():
    with open(_config_path, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {
        "model": {"api_key": "", "chat_model": "qwen3-max", "embedding_model": "text-embedding-v2", "temperature": 0.3},
        "paths": {"data_dir": "./data", "prompts_dir": "./prompts", "vector_db": "./vectorstore"},
        "rag": {"top_k": 4, "chunk_size": 500, "chunk_overlap": 100},
    }

_env_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
if _env_api_key:
    CONFIG.setdefault("model", {})["api_key"] = _env_api_key

_env_chat_model = os.getenv("CHAT_MODEL", "").strip()
if _env_chat_model:
    CONFIG.setdefault("model", {})["chat_model"] = _env_chat_model

try:
    from rag.rag_service import RAGService
    from rag.react_agent import ReActAgent

    rag_service = RAGService(CONFIG)
    react_agent = ReActAgent(rag_service)
    HAS_RAG = True
except Exception as e:
    rag_service = None
    react_agent = None
    HAS_RAG = False
    print(f"RAG service initialization failed: {e}")

app = FastAPI(
    title="CampusMind Web",
    description="校园问答助手 - 清新网页版",
    version="1.0.0",
)


def get_current_user(authorization: Optional[str] = Header(None)):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

    if not token:
        raise HTTPException(status_code=401, detail="未提供认证令牌")

    user_data = AuthService.verify_token(token)
    if not user_data:
        raise HTTPException(status_code=401, detail="认证令牌无效或已过期")

    return user_data


def get_current_user_optional(authorization: Optional[str] = Header(None)):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

    if not token:
        return None

    return AuthService.verify_token(token)


def _save_conversation(session_id: str, user_id: int, title: str = "新对话"):
    """保存或创建会话"""
    from db.database import Conversation, get_db

    db = next(get_db())
    conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()

    if conv:
        conv.title = title
        db.commit()
        return conv

    conv = Conversation(
        session_id=session_id,
        user_id=user_id,
        title=title,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def _save_message(session_id: str, user_id: int, role: str, content: str, steps=None):
    """保存消息到数据库"""
    import json

    from db.database import Conversation, Message, get_db

    db = next(get_db())
    conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()

    if not conv:
        conv = Conversation(
            session_id=session_id,
            user_id=user_id,
            title=content[:50] if role == "user" else "新对话",
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)

    if role == "user" and len(content) > 0 and conv.title == "新对话":
        conv.title = content[:50] + ("..." if len(content) > 50 else "")
        db.commit()

    message = Message(
        conversation_id=conv.id,
        session_id=session_id,
        role=role,
        content=content,
        steps=json.dumps(steps, ensure_ascii=False) if steps else None,
    )
    db.add(message)
    db.commit()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again later."},
    )


@app.get("/api/health", response_model=Dict[str, str])
async def health_check():
    return {"status": "ok", "service": "CampusMind Web", "has_rag": str(HAS_RAG)}


@app.post("/api/auth/register")
async def register(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
):
    success, message = AuthService.register(username, password)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@app.post("/api/auth/login")
async def login(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
):
    success, message, user_info = AuthService.login(username, password)
    if not success:
        raise HTTPException(status_code=401, detail=message)
    return {"message": message, "user": user_info}


@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    return {"message": "登出成功"}


@app.post("/api/auth/change-password")
async def change_password(
    old_password: str = Body(..., embed=True),
    new_password: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user),
):
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少6位")

    success, message = AuthService.change_password(
        current_user["user_id"], old_password, new_password
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"message": message}


@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["user_id"],
        "username": current_user["username"],
        "role": current_user["role"],
    }


@app.post("/api/query")
async def query(
    question: str = Body(..., embed=True),
    session_id: Optional[str] = Body(None),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    sid = session_id or str(uuid.uuid4())

    if HAS_RAG and rag_service:
        answer = rag_service.rag_summarize(question)
    else:
        mock_answers = [
            "你好！我是 CampusMind 校园智能助手。关于你的问题，我可以为你提供以下帮助：\n\n1. 校园规章制度查询\n2. 课程安排咨询\n3. 图书馆借阅信息\n4. 校园活动通知\n\n请问你具体想了解什么？",
            "感谢你的提问！校园图书馆开放时间为周一至周五 8:00-22:00，周末 9:00-21:00。你可以通过校园卡或学生证进入图书馆。",
            "校园网络连接方式：\n- 有线网络：使用网线连接宿舍或教室的网络端口\n- 无线网络：搜索 CMCC-Campus 或 Eduroam 信号\n- VPN：在校外访问校园资源需要连接 VPN\n\n如有其他问题，请随时提问！",
            "食堂就餐时间：\n- 早餐：7:00-9:00\n- 午餐：11:00-13:30\n- 晚餐：17:00-19:30\n\n校园内共有三个食堂，分别位于东区、西区和北区，提供多种口味的餐食选择。",
            "期末考试安排通常在每学期结束前两周公布，你可以通过教务系统查询具体的考试时间和地点。建议提前做好复习计划，合理安排时间！",
        ]
        answer = mock_answers[hash(question) % len(mock_answers)]

    if current_user:
        _save_message(sid, current_user["user_id"], "user", question)
        _save_message(sid, current_user["user_id"], "assistant", answer)

    return {"question": question, "answer": answer, "session_id": sid, "has_rag": HAS_RAG}


@app.post("/api/query/react")
async def query_react(
    question: str = Body(..., embed=True),
    session_id: Optional[str] = Body(None),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """ReAct Agent 模式：思考-行动-观察循环，集成 MCP 工具"""
    sid = session_id or str(uuid.uuid4())

    if HAS_RAG and react_agent:
        result = react_agent.run(question)
        if current_user:
            _save_message(sid, current_user["user_id"], "user", question)
            _save_message(sid, current_user["user_id"], "assistant", result["answer"], result["steps"])
        return {
            "question": question,
            "answer": result["answer"],
            "steps": result["steps"],
            "session_id": sid,
            "has_react": result["has_react"],
        }

    if current_user:
        _save_message(sid, current_user["user_id"], "user", question)
        _save_message(sid, current_user["user_id"], "assistant", "ReAct Agent 未初始化。")

    return {
        "question": question,
        "answer": "ReAct Agent 未初始化。",
        "steps": [],
        "session_id": sid,
        "has_react": False,
    }


@app.get("/api/tools")
async def list_tools():
    """列出所有可用的 MCP 工具"""
    try:
        from mcp.tool_registry import tool_registry
        return {"tools": tool_registry.list_tools()}
    except ImportError:
        return {"tools": []}


@app.post("/api/tools/execute")
async def execute_tool(
    name: str = Body(..., embed=True),
    params: dict = Body(default={}),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """直接执行指定工具"""
    try:
        from mcp.tool_registry import tool_registry
        result = tool_registry.execute(name, **params)
        return {"name": name, "result": result}
    except ImportError:
        return {"error": "MCP 工具未初始化"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/query/stream")
async def query_stream(
    question: str = Body(..., embed=True),
    session_id: Optional[str] = Body(None),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    sid = session_id or str(uuid.uuid4())

    def generate() -> Generator[str, None, None]:
        try:
            if HAS_RAG and rag_service:
                for chunk in rag_service.rag_summarize_stream(question):
                    yield chunk
            else:
                yield "RAG服务未初始化，请检查依赖和配置。"
        except Exception as e:
            yield f"\n[生成回答时出错: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/api/query/react/stream")
async def query_react_stream(
    question: str = Body(..., embed=True),
    session_id: Optional[str] = Body(None),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    sid = session_id or str(uuid.uuid4())
    final_answer = ""
    final_steps = []

    def generate() -> Generator[str, None, None]:
        nonlocal final_answer, final_steps
        try:
            if HAS_RAG and react_agent:
                for event in react_agent.run_stream(question):
                    event["session_id"] = sid
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                    if event["type"] == "steps":
                        final_steps = event["steps"]
                    elif event["type"] == "answer":
                        final_answer = event["answer"]
            else:
                yield json.dumps({
                    "type": "answer",
                    "answer": "ReAct Agent 未初始化。",
                    "steps": [],
                    "session_id": sid,
                    "has_react": False,
                }, ensure_ascii=False) + "\n"
                final_answer = "ReAct Agent 未初始化。"
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": str(e),
                "session_id": sid,
            }, ensure_ascii=False) + "\n"

    if current_user:
        _save_message(sid, current_user["user_id"], "user", question)

    response = StreamingResponse(generate(), media_type="application/x-ndjson")

    if current_user:
        from fastapi import BackgroundTasks
        from starlette.background import BackgroundTask

        def save_assistant_message():
            if final_answer:
                _save_message(sid, current_user["user_id"], "assistant", final_answer, final_steps)

        response.background = BackgroundTask(save_assistant_message)

    return response


@app.post("/api/session/clear")
async def clear_session(
    session_id: str = Body(..., embed=True),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")

    from db.database import Conversation, Message, get_db

    db = next(get_db())
    conv = db.query(Conversation).filter(
        Conversation.session_id == session_id,
        Conversation.user_id == current_user["user_id"],
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    db.query(Message).filter(Message.conversation_id == conv.id).delete()
    conv.title = "新对话"
    db.commit()

    return {"message": "会话已清空", "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    from db.database import Conversation, get_db

    db = next(get_db())
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user["user_id"]
    ).order_by(Conversation.updated_at.desc()).all()

    return {
        "sessions": [
            {
                "session_id": conv.session_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            }
            for conv in conversations
        ]
    }


@app.get("/api/sessions/{session_id}")
async def get_session_messages(session_id: str, current_user: dict = Depends(get_current_user)):
    from db.database import Conversation, Message, get_db

    db = next(get_db())
    conv = db.query(Conversation).filter(
        Conversation.session_id == session_id,
        Conversation.user_id == current_user["user_id"]
    ).first()

    if not conv:
        return {"messages": [], "title": "新对话"}

    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.created_at.asc()).all()

    import json

    return {
        "session_id": session_id,
        "title": conv.title,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "steps": json.loads(msg.steps) if msg.steps else None,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in messages
        ]
    }


@app.post("/api/sessions/{session_id}/delete")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    from db.database import Conversation, Message, get_db

    db = next(get_db())
    conv = db.query(Conversation).filter(
        Conversation.session_id == session_id,
        Conversation.user_id == current_user["user_id"]
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    db.query(Message).filter(Message.session_id == session_id).delete()
    db.delete(conv)
    db.commit()

    return {"message": "会话删除成功"}


@app.post("/api/sessions/{session_id}/rename")
async def rename_session(
    session_id: str,
    title: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user),
):
    from db.database import Conversation, get_db

    if not title or not title.strip():
        raise HTTPException(status_code=400, detail="标题不能为空")

    title = title.strip()[:200]

    db = next(get_db())
    conv = db.query(Conversation).filter(
        Conversation.session_id == session_id,
        Conversation.user_id == current_user["user_id"]
    ).first()

    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    conv.title = title
    db.commit()

    return {"message": "重命名成功", "session_id": session_id, "title": title}


@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    indexed_chunks = 0

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".pdf", ".docx"]:
            continue

        file_path = data_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_count += 1

    if HAS_RAG and rag_service:
        file_paths = [data_dir / f.filename for f in files if Path(f.filename).suffix.lower() in [".pdf", ".docx"]]
        indexed_chunks = rag_service.ingest_uploaded_files(file_paths)

    return {"saved_count": saved_count, "indexed_chunks": indexed_chunks}


@app.get("/api/documents")
async def list_documents(current_user: Optional[dict] = Depends(get_current_user_optional)):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    documents = []
    stats = {"total_chunks": 0}

    if data_dir.exists():
        for file_path in data_dir.glob("*.*"):
            ext = file_path.suffix.lower()
            if ext in [".pdf", ".docx"]:
                documents.append({
                    "source": str(file_path),
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                })

    if HAS_RAG and rag_service:
        stats = rag_service.get_vectorstore_stats()

    return {"documents": documents, "stats": stats}


@app.delete("/api/documents/{source}")
async def delete_document(
    source: str,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    try:
        file_path = Path(source)
        if file_path.exists():
            file_path.unlink()

        if HAS_RAG and rag_service:
            rag_service.delete_document(source)

        return {"message": f"Document {source} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


_web_dir = Path(__file__).resolve().parent.parent / "web"
if _web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_web_dir)), name="static")


@app.get("/")
async def index():
    index_file = _web_dir / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return {"message": "CampusMind Web. Frontend not found"}


@app.get("/{path:path}")
async def spa_catch_all(path: str):
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    index_file = _web_dir / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)