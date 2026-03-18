# MyCodeCli

这是一个基于 LangChain 和 NVIDIA AI 端点的简单 AI 编码助手。它可以根据用户指令自动执行 Shell 命令并进行多轮对话。

## 功能特点

- **智能对话**: 使用 NVIDIA 托管的 `qwen3.5-122b` 模型。
- **工具调用**: 集成了 `run_bash` 工具，能够自动在本地执行 Shell 命令（如：列出文件、运行脚本等）。
- **循环执行 (Agent Loop)**: 支持多步决策，AI 可以根据命令执行结果决定下一步行动。
- **安全检查**: 内置基础的命令过滤功能，防止执行高危命令。

## 快速开始

### 环境依赖

确保已安装以下 Python 库：

```bash
pip install langchain-core langchain-nvidia-ai-endpoints
```

### 配置 API KEY

在 `L1/AgentLoop.py` 中，将 `api_key="xxx"` 替换为您真实的 NVIDIA API Key。

### 运行

```bash
python L1/AgentLoop.py
```

## 项目结构

- `L1/AgentLoop.py`: 核心代理逻辑实现。
- `.gitignore`: 忽略无需提交的系统与 IDE 配置文件。

## 注意事项

- **安全第一**: 虽然代码中有简单的命令检查，但请谨慎授予 AI 执行 shell 命令的权限。
- **隐私**: 提交代码前，请确保已移除代码中的敏感 API 密钥。建议使用环境变量来管理密钥。
