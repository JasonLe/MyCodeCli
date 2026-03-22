# http://learn.shareai.run/zh/s03/
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Callable, Any

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
load_dotenv()

WORK_DIR = Path(os.getenv("WORK_DIR"))

llm = ChatNVIDIA(
    model="qwen/qwen3.5-122b-a10b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.6,
    top_p=0.95,
    max_completion_tokens=16384,
)

class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

TODO = TodoManager()

@tool
def update_todo(items: list) -> str:
    """更新待办事项列表。每项应包含id、text和status（pending/in_progress/completed）。"""
    # 显式地通过实例调用逻辑
    return TODO.update(items)

@tool
def run_bash(command: str):
    """执行shell命令"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(d in command for d in dangerous):
        return "错误: 危险命令禁止"

    try:
        r = subprocess.run(command, shell=True, cwd = os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(没有输出)"
    except subprocess.TimeoutExpired:
        return "错误: 超时120s"


def safe_path(p: str):
    path = (WORK_DIR / p).resolve()
    if not path.is_relative_to(WORK_DIR):
        raise ValueError(f"路径错误:{p}")
    return path

@tool
def read_file(p: str, limit: int = None):
    """读取文件"""
    text = safe_path(p).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)

@tool()
def write_file(path: str, content: str) -> str:
    """写入文件，路径相对于工作目录"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS: Dict[str, Callable[..., Any]] = {
    "run_bash":       lambda **kwargs: run_bash.invoke(kwargs),
    "read_file":  lambda **kwargs: read_file.invoke(kwargs),
    "write_file": lambda **kwargs: write_file.invoke(kwargs),
    "update_todo": lambda **kwargs: update_todo.invoke(kwargs),
}

tools = [run_bash, read_file, write_file, update_todo]

llm = llm.bind_tools(tools)

def agent_loop(messages: list):
    while True:
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            print("没有工具调用")
            return

        for tool_call in response.tool_calls:
            print(f"调用工具:{tool_call['name']}")
            tool = TOOL_HANDLERS.get(tool_call["name"])
            if not tool:
                raise ValueError(f'工具名称不匹配: {tool_call["name"]}')

            output = tool(**tool_call["args"])
            print(output[:200])
            messages.append(ToolMessage(content=output[:200], tool_call_id=tool_call['id']))


if __name__ == '__main__':
    history_messages: List[BaseMessage] = [
        SystemMessage(content=f"您是 {WORK_DIR} 的一名编码员。使用待办事项工具规划多步骤任务。开始前标记为“进行中”，完成后标记为“已完成”。建议优先使用工具而非文字描述。")
    ]
    while True:
        try:
            query = input("输入:")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history_messages.append(HumanMessage(content=query))
        agent_loop(history_messages)
        print()
