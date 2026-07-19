"""MCP 工具框架 - 工具注册与调用系统"""
import json
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Callable, Dict, List, Optional


class MCPTool:
    """单个工具定义"""

    def __init__(self, name: str, description: str, parameters: Dict, handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema 格式
        self.handler = handler

    def execute(self, **kwargs) -> str:
        """执行工具"""
        try:
            result = self.handler(**kwargs)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"工具执行出错: {str(e)}"

    def to_schema(self) -> Dict:
        """返回工具的 schema 描述（给大模型看）"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class MCPToolRegistry:
    """MCP 工具注册表 - 管理所有可用工具"""

    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._register_builtin_tools()

    def register(self, tool: MCPTool):
        """注册工具"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict]:
        """列出所有工具的 schema"""
        return [tool.to_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        """执行指定工具"""
        tool = self._tools.get(name)
        if not tool:
            return f"未找到工具: {name}。可用工具: {', '.join(self._tools.keys())}"
        return tool.execute(**kwargs)

    def _register_builtin_tools(self):
        """注册内置工具"""
        # 1. 实时时间工具
        self.register(MCPTool(
            name="get_current_time",
            description="获取当前实时时间和日期。可指定时区。",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区名称，如 'Asia/Shanghai', 'UTC', 'America/New_York'。默认 'Asia/Shanghai'",
                    }
                }
            },
            handler=self._get_current_time,
        ))

        # 2. 天气查询工具
        self.register(MCPTool(
            name="get_weather",
            description="查询指定城市的实时天气信息，包括温度、天气状况、湿度、风力等。",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京', '上海', '杭州'",
                    }
                },
                "required": ["city"],
            },
            handler=self._get_weather,
        ))

        # 3. 计算器工具
        self.register(MCPTool(
            name="calculator",
            description="数学计算器，支持加减乘除、幂运算等。输入数学表达式。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2+3*4', '100/5', '2**10'",
                    }
                },
                "required": ["expression"],
            },
            handler=self._calculator,
        ))

        # 4. 知识库搜索工具
        self.register(MCPTool(
            name="search_knowledge",
            description="在校园知识库中搜索相关信息。用于回答关于校园的问题。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题",
                    }
                },
                "required": ["query"],
            },
            handler=self._search_knowledge,
        ))

        # 5. 日期计算工具
        self.register(MCPTool(
            name="date_diff",
            description="计算两个日期之间的天数差，或从今天起若干天后的日期。",
            parameters={
                "type": "object",
                "properties": {
                    "date1": {"type": "string", "description": "第一个日期 YYYY-MM-DD 格式"},
                    "date2": {"type": "string", "description": "第二个日期 YYYY-MM-DD 格式（可选）"},
                    "add_days": {"type": "integer", "description": "从今天起加几天（可选）"},
                },
            },
            handler=self._date_diff,
        ))

    # ===== 工具实现 =====

    def _get_current_time(self, timezone: str = "Asia/Shanghai") -> str:
        """获取当前时间"""
        tz_map = {
            "Asia/Shanghai": dt_timezone(timedelta(hours=8)),
            "UTC": dt_timezone.utc,
            "America/New_York": dt_timezone(timedelta(hours=-5)),
            "Europe/London": dt_timezone(timedelta(hours=0)),
            "Japan": dt_timezone(timedelta(hours=9)),
        }
        tz = tz_map.get(timezone, dt_timezone(timedelta(hours=8)))
        now = datetime.now(tz)
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return (
            f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekdays[now.weekday()]}\n"
            f"时区: {timezone}"
        )

    def _get_weather(self, city: str) -> str:
        """查询天气（使用免费 wttr.in API）"""
        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            current = data.get("current_condition", [{}])[0]
            area = data.get("nearest_area", [{}])[0]

            temp = current.get("temp_C", "N/A")
            feels = current.get("FeelsLikeC", "N/A")
            humidity = current.get("humidity", "N/A")
            wind_speed = current.get("windspeedKmph", "N/A")
            wind_dir = current.get("winddir16Point", "N/A")
            weather_desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
            visibility = current.get("visibility", "N/A")
            pressure = current.get("pressure", "N/A")

            city_name = area.get("areaName", [{}])[0].get("value", city)
            country = area.get("country", [{}])[0].get("value", "")

            return (
                f"📍 {city_name}, {country}\n"
                f"🌡️ 温度: {temp}°C（体感 {feels}°C）\n"
                f"🌤️ 天气: {weather_desc}\n"
                f"💧 湿度: {humidity}%\n"
                f"💨 风力: {wind_speed}km/h {wind_dir}\n"
                f"👁️ 能见度: {visibility}km\n"
                f"📊 气压: {pressure}hPa"
            )
        except urllib.error.URLError:
            return f"天气查询失败：网络无法访问 wttr.in 服务。请稍后重试。"
        except Exception as e:
            return f"天气查询失败: {str(e)}"

    def _calculator(self, expression: str) -> str:
        """数学计算器"""
        # 安全的字符过滤
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误: 表达式包含非法字符，仅支持数字和 + - * / ( )"
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except ZeroDivisionError:
            return "错误: 除数不能为零"
        except Exception as e:
            return f"计算错误: {str(e)}"

    def _search_knowledge(self, query: str) -> str:
        """知识库搜索（占位，实际由 ReActAgent 注入）"""
        return f"知识库搜索: {query}（需要注入 rag_service）"

    def _date_diff(self, date1: str = "", date2: str = "", add_days: int = 0) -> str:
        """日期计算"""
        try:
            if add_days:
                today = datetime.now()
                target = today + timedelta(days=add_days)
                weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
                return f"今天起 {add_days} 天后是: {target.strftime('%Y年%m月%d日')} {weekdays[target.weekday()]}"

            if date1 and date2:
                d1 = datetime.strptime(date1, "%Y-%m-%d")
                d2 = datetime.strptime(date2, "%Y-%m-%d")
                diff = abs((d2 - d1).days)
                return f"{date1} 和 {date2} 相差 {diff} 天"

            if date1:
                d1 = datetime.strptime(date1, "%Y-%m-%d")
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                diff = (d1 - today).days
                if diff > 0:
                    return f"距离 {date1} 还有 {diff} 天"
                elif diff < 0:
                    return f"{date1} 已经过去 {abs(diff)} 天"
                else:
                    return f"{date1} 就是今天"

            return "请提供 date1, date2 或 add_days 参数"
        except ValueError:
            return "日期格式错误，请使用 YYYY-MM-DD 格式"
        except Exception as e:
            return f"日期计算错误: {str(e)}"


# 全局工具注册表实例
tool_registry = MCPToolRegistry()
