"""ReAct Agent - 思考-行动-观察循环模式（集成 MCP 工具）"""
import json
import re
from typing import Dict, Generator, List, Optional

try:
    from langchain_core.documents import Document
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

try:
    from mcp.tool_registry import tool_registry
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _build_react_prompt(tools_desc: str) -> str:
    return f"""你是一个智能问答助手，使用 ReAct（思考-行动-观察）模式回答问题。

你可以使用以下工具：

{tools_desc}

请严格按照以下格式回答：

Thought: 分析用户问题，思考需要调用什么工具
Action: 工具名称
Action Input: JSON格式的参数

（然后你会收到工具执行结果 Observation）

当你收集到足够信息后，用以下格式给出最终答案：

Thought: 我已经获得了足够的信息来回答问题
Final Answer: 给用户的最终回答

注意事项：
1. 每次只执行一个工具调用
2. Action Input 必须是合法的 JSON 格式
3. 如果第一次调用没有找到完整答案，可以再次调用工具
4. 最多执行3次工具调用，然后必须给出最终答案
5. 最终回答要自然、友好、清晰
6. 对于实时信息（时间、天气等），必须调用对应工具获取"""


class ReActAgent:
    """ReAct Agent: 思考→行动→观察 循环，集成 MCP 工具"""

    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.max_iterations = 4

    def _get_tools_description(self) -> str:
        """获取所有可用工具的描述"""
        if HAS_MCP:
            tools = tool_registry.list_tools()
            lines = []
            for t in tools:
                params = t.get("parameters", {}).get("properties", {})
                required = t.get("parameters", {}).get("required", [])
                param_str = ", ".join(
                    f"{k}: {v.get('description', '')}" for k, v in params.items()
                )
                req_str = f" (必填: {', '.join(required)})" if required else ""
                lines.append(f"- {t['name']}: {t['description']}\n  参数: {param_str}{req_str}")
            return "\n".join(lines)
        return "- search: 在知识库中搜索相关信息。参数: query (搜索关键词)"

    def _execute_tool(self, action: str, action_input: str) -> str:
        """执行工具调用"""
        # 解析 action_input（JSON 格式）
        try:
            params = json.loads(action_input) if action_input else {}
        except json.JSONDecodeError:
            # 如果不是 JSON，尝试作为简单字符串参数
            params = {"query": action_input, "city": action_input, "expression": action_input, "date1": action_input}

        # 知识库搜索特殊处理：注入 rag_service
        if action == "search_knowledge" or action == "search":
            return self._search_knowledge(params.get("query", ""))

        if HAS_MCP:
            return tool_registry.execute(action, **params)

        return f"工具 {action} 不可用"

    def _search_knowledge(self, query: str) -> str:
        """知识库搜索"""
        if not HAS_LANGCHAIN or self.rag_service is None or self.rag_service.vector_store is None:
            return "知识库未配置，无法检索。"

        top_k = self.rag_service.config.get("rag", {}).get("top_k", 4)
        docs = self.rag_service.vector_store.similarity_search(query=query, k=top_k)

        if not docs:
            return "未检索到相关文档。"

        blocks = []
        for idx, doc in enumerate(docs, start=1):
            blocks.append(f"[文档{idx}] {doc.page_content[:300]}")
        return "\n\n".join(blocks)

    def _parse_response(self, text: str) -> Dict:
        """解析大模型返回的 Thought/Action/Action Input / Final Answer"""
        result = {"thought": "", "action": None, "action_input": None, "final_answer": None}

        # 提取 Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n[A-Z]|\Z)", text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 提取 Final Answer
        final_match = re.search(r"Final Answer:\s*(.*?)(?=\n[A-Z]|\Z)", text, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # 提取 Action
        action_match = re.search(r"Action:\s*(.*?)(?=\n|$)", text)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # 提取 Action Input（可能跨行）
        input_match = re.search(r"Action Input:\s*(.*?)(?=\n[A-Z]|\Z)", text, re.DOTALL)
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        return result

    def run(self, question: str) -> Dict:
        """运行 ReAct 循环"""
        tools_desc = self._get_tools_description()
        system_prompt = _build_react_prompt(tools_desc)

        # 尝试获取大模型
        chat_model = None
        if self.rag_service and self.rag_service.model_factory:
            try:
                chat_model = self.rag_service.model_factory.get_chat_model()
            except Exception:
                pass

        # 模型不可用时降级：智能匹配工具直接执行
        if chat_model is None:
            return self._fallback_response(question)

        steps: List[Dict] = []
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        for i in range(self.max_iterations):
            prompt = self._build_prompt(messages)
            try:
                resp = chat_model.invoke(prompt)
                ai_text = str(getattr(resp, "content", resp))
            except Exception as e:
                return {
                    "steps": steps,
                    "answer": f"模型调用出错: {str(e)}",
                    "has_react": True,
                }

            parsed = self._parse_response(ai_text)
            step = {
                "iteration": i + 1,
                "thought": parsed["thought"],
                "action": parsed["action"],
                "action_input": parsed["action_input"],
                "observation": None,
            }
            steps.append(step)

            # 有最终答案，结束
            if parsed["final_answer"]:
                return {
                    "steps": steps,
                    "answer": parsed["final_answer"],
                    "has_react": True,
                }

            # 执行工具
            if parsed["action"] and parsed["action_input"]:
                observation = self._execute_tool(parsed["action"], parsed["action_input"])
                step["observation"] = observation[:800] if observation else "工具返回空结果"
                messages.append({"role": "assistant", "content": ai_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            elif parsed["action"]:
                # 有 Action 但没有 Action Input
                observation = self._execute_tool(parsed["action"], "{}")
                step["observation"] = observation[:800]
                messages.append({"role": "assistant", "content": ai_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # 模型没有按格式回答，直接当作最终答案
                return {
                    "steps": steps,
                    "answer": ai_text,
                    "has_react": True,
                }

        # 达到最大迭代，强制生成最终答案
        messages.append({
            "role": "user",
            "content": "你已经调用了多次工具，请现在给出最终答案。格式：Final Answer: 你的回答",
        })
        prompt = self._build_prompt(messages)
        try:
            resp = chat_model.invoke(prompt)
            ai_text = str(getattr(resp, "content", resp))
            parsed = self._parse_response(ai_text)
            final = parsed["final_answer"] or ai_text
        except Exception as e:
            final = f"生成最终答案时出错: {str(e)}"

        return {
            "steps": steps,
            "answer": final,
            "has_react": True,
        }

    def run_stream(self, question: str) -> Generator[Dict, None, None]:
        """运行 ReAct 循环（流式版本）"""
        tools_desc = self._get_tools_description()
        system_prompt = _build_react_prompt(tools_desc)

        chat_model = None
        if self.rag_service and self.rag_service.model_factory:
            try:
                chat_model = self.rag_service.model_factory.get_chat_model()
            except Exception:
                pass

        if chat_model is None:
            fallback = self._fallback_response(question)
            yield {"type": "steps", "steps": fallback["steps"]}
            yield {"type": "answer", "answer": fallback["answer"], "has_react": True}
            return

        steps: List[Dict] = []
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        for i in range(self.max_iterations):
            prompt = self._build_prompt(messages)
            try:
                ai_text = ""
                for chunk in chat_model.stream(prompt):
                    content = getattr(chunk, "content", chunk)
                    if content:
                        ai_text += str(content)
                        yield {"type": "thinking", "iteration": i + 1, "partial": str(content)}
            except Exception as e:
                yield {"type": "error", "error": f"模型调用出错: {str(e)}"}
                return

            parsed = self._parse_response(ai_text)
            step = {
                "iteration": i + 1,
                "thought": parsed["thought"],
                "action": parsed["action"],
                "action_input": parsed["action_input"],
                "observation": None,
            }
            steps.append(step)

            if parsed["final_answer"]:
                yield {"type": "steps", "steps": steps}
                yield {"type": "answer", "answer": parsed["final_answer"], "has_react": True}
                return

            if parsed["action"] and parsed["action_input"]:
                observation = self._execute_tool(parsed["action"], parsed["action_input"])
                step["observation"] = observation[:800] if observation else "工具返回空结果"
                yield {"type": "observation", "iteration": i + 1, "observation": observation[:800]}
                messages.append({"role": "assistant", "content": ai_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            elif parsed["action"]:
                observation = self._execute_tool(parsed["action"], "{}")
                step["observation"] = observation[:800]
                yield {"type": "observation", "iteration": i + 1, "observation": observation[:800]}
                messages.append({"role": "assistant", "content": ai_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                yield {"type": "steps", "steps": steps}
                yield {"type": "answer", "answer": ai_text, "has_react": True}
                return

        messages.append({
            "role": "user",
            "content": "你已经调用了多次工具，请现在给出最终答案。格式：Final Answer: 你的回答",
        })
        prompt = self._build_prompt(messages)
        try:
            ai_text = ""
            for chunk in chat_model.stream(prompt):
                content = getattr(chunk, "content", chunk)
                if content:
                    ai_text += str(content)
                    yield {"type": "thinking", "iteration": self.max_iterations + 1, "partial": str(content)}
            parsed = self._parse_response(ai_text)
            final = parsed["final_answer"] or ai_text
        except Exception as e:
            final = f"生成最终答案时出错: {str(e)}"

        yield {"type": "steps", "steps": steps}
        yield {"type": "answer", "answer": final, "has_react": True}

    def _build_prompt(self, messages: List[Dict]) -> str:
        """构造文本提示"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"用户：{content}")
            elif role == "assistant":
                parts.append(f"助手：{content}")
        return "\n\n".join(parts)

    def _fallback_response(self, question: str) -> Dict:
        """模型不可用时的智能降级：多步工具调用"""
        q_lower = question.lower()

        steps = []
        observations = []

        # 第一步：检查是否需要实时时间/天气等工具
        if any(kw in q_lower for kw in ["时间", "几点", "日期", "今天", "现在", "time", "date"]):
            observation = tool_registry.execute("get_current_time", timezone="Asia/Shanghai") if HAS_MCP else "时间工具不可用"
            steps.append({
                "iteration": 1,
                "thought": "用户问题涉及时间，先获取当前时间",
                "action": "get_current_time",
                "action_input": "",
                "observation": observation[:500],
            })
            observations.append(observation)

        if any(kw in q_lower for kw in ["天气", "气温", "weather", "温度"]):
            cities = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安", "重庆", "天津"]
            city = "杭州"
            for c in cities:
                if c in question:
                    city = c
                    break
            observation = tool_registry.execute("get_weather", city=city) if HAS_MCP else "天气工具不可用"
            steps.append({
                "iteration": len(steps) + 1,
                "thought": f"用户问题涉及天气，查询{city}天气",
                "action": "get_weather",
                "action_input": json.dumps({"city": city}, ensure_ascii=False),
                "observation": observation[:500],
            })
            observations.append(observation)

        if any(kw in q_lower for kw in ["计算", "等于", "加", "减", "乘", "除", "calculator"]):
            expr_match = re.search(r"[\d+\-*/.() ]+", question)
            expr = expr_match.group().strip() if expr_match else question
            observation = tool_registry.execute("calculator", expression=expr) if HAS_MCP else "计算器工具不可用"
            steps.append({
                "iteration": len(steps) + 1,
                "thought": "用户问题涉及计算，使用计算器",
                "action": "calculator",
                "action_input": json.dumps({"expression": expr}, ensure_ascii=False),
                "observation": observation[:500],
            })
            observations.append(observation)

        if any(kw in q_lower for kw in ["多少天", "天后", "距离", "相差"]):
            days_match = re.search(r"(\d+)\s*天", question)
            add_days = int(days_match.group(1)) if days_match else 0
            observation = tool_registry.execute("date_diff", add_days=add_days) if HAS_MCP else "日期计算工具不可用"
            steps.append({
                "iteration": len(steps) + 1,
                "thought": "用户问题涉及日期计算",
                "action": "date_diff",
                "action_input": json.dumps({"add_days": add_days}, ensure_ascii=False),
                "observation": observation[:500],
            })
            observations.append(observation)

        # 第二步：如果有知识库，检索相关信息
        knowledge_observation = ""
        if self.rag_service and self.rag_service.vector_store:
            knowledge_observation = self._search_knowledge(question)
            steps.append({
                "iteration": len(steps) + 1,
                "thought": "检索知识库获取相关信息",
                "action": "search_knowledge",
                "action_input": json.dumps({"query": question}, ensure_ascii=False),
                "observation": knowledge_observation[:500],
            })
            observations.append(knowledge_observation)

        # 构建最终回答
        if observations:
            final_answer = "\n\n".join(observations)
            # 如果有知识库结果，做一个简单的综合回答
            if knowledge_observation and "图书馆" in q_lower:
                time_info = ""
                for o in observations:
                    if "当前时间" in o:
                        time_info = o.split("\n")[0]
                        break
                if time_info:
                    final_answer = f"{time_info}\n\n{knowledge_observation}"
        else:
            final_answer = "ReAct Agent 未配置，请检查模型 API Key 和依赖。"

        return {
            "steps": steps,
            "answer": final_answer,
            "has_react": True,
        }
