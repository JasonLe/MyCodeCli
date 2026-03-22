# http://learn.shareai.run/zh/s02/
import os
import subprocess
from pathlib import Path
from typing import List

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


TOOL_HANDLERS = {
    "bash": lambda **kwargs: run_bash.invoke(kwargs),
    "read_file": lambda **kwargs: read_file.invoke(kwargs),
}


llm = llm.bind_tools([run_bash, read_file])

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
                raise ValueError(f'工具名称不匹配{tool_call["name"]}')

            output = tool(**tool_call["args"])
            print(output[:200])
            messages.append(ToolMessage(content=output[:200], tool_call_id=tool_call['id']))


if __name__ == '__main__':
    history_messages: List[BaseMessage] = [
        SystemMessage(content=f"你是{os.getcwd()}的一名编码助手。使用工具完成任务。行动起来，无需解释。")
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
