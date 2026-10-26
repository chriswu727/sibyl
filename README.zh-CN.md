# Sibyl

[English](README.md) | 简体中文

**为你现有的 AI agent 提供研究能力的 MCP：跨来源调查、检查证据、追踪反证和研究缺口，并在需要时导出带引用的报告。**

你的 agent 负责思考，Sibyl 提供持久化的研究工作区。它可以连接支持本地 MCP 的 agent，
包括通过订阅使用的 agent。Sibyl 不调用另一个生成式模型，不读取订阅凭证，也不需要第二个模型 API key。
宿主的订阅规则、额度限制和隐私条款仍然适用。

本文件是仓库唯一维护的中文文档。英文 README、功能目标和实施计划是规范版本。

> **当前是源码预览，不是新发布的 PyPI 版本。** 本文描述源码中的 host-only 架构。
> 已发布的 beta 0.4.0b1 和稳定版 0.3.0 仍包含旧架构。请按下方步骤安装源码。
> 当前尚未证明 Sibyl 的研究质量超过你的 agent 自带的搜索。

## 给 agent 的研究能力，给人的可选报告

一次搜索返回线索，Sibyl 将调查过程连接起来：

- 将问题拆成研究子问题，并跨多次调用保存计划。
- 把搜索结果和原始页面读入同一份证据库。
- 按需检查原文段落、稳定标识、内容哈希和来源信息。
- 保存有支持、有争议或未解决的判断及其证据链接。
- 调查反证，围绕具体缺口继续检索。
- 直接复用结构化研究结果，或导出 Markdown、JSON 和 PDF。

宿主决定查什么、阅读证据并判断它支持什么。Sibyl 负责检索、保存状态、检查引用关系和限制资源使用。
它不替宿主判断绝对真伪，也不独立运行一个研究 agent。

默认 MCP 入口提供 **8 个工具**，返回紧凑的结构化简报，原文按需读取。
研究结果不需要先变成一篇长报告，才能被后续任务使用。

## 快速开始

需要 Python 3.11+，以及支持本地 stdio MCP 和 `structuredContent` 的宿主。
请使用全新的虚拟环境安装源码：

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/sibyl-mcp --list-tools
```

最后一条命令应显示 `profile: agent` 和 8 个工具。对于接受 `mcpServers` JSON 的客户端，
可以使用以下配置，并将两个路径替换为本机的绝对路径：

```json
{
  "mcpServers": {
    "sibyl": {
      "command": "/absolute/path/to/sibyl/.venv/bin/sibyl-mcp",
      "env": {
        "SIBYL_OUTPUT_ROOT": "/absolute/path/to/private-sibyl-research"
      }
    }
  }
}
```

Windows 的可执行文件位于 `.venv\Scripts\sibyl-mcp.exe`。
具体配置语法取决于宿主。只读取文本结果的客户端可以添加
`"args": ["--profile", "keyless"]`，使用 12 工具的兼容入口及完整文本响应。
这不是远程 HTTP MCP 服务。

默认搜索不需要 key。可选 Tavily 搜索使用独立凭证，并可能产生额外费用；
agent 的订阅不包含 Tavily 的费用。参见[安装说明](llms-install.md)。

可以这样对你的 agent 提问：

> 用 Sibyl 调查 SQLite WAL 是否适合一个多进程应用。阅读原始文档，调查限制和反证，
> 返回与证据关联的结论及未解决的问题。只有我要求时才生成报告。

## 研究流程

| 工具 | 用途 |
|---|---|
| `start_research` | 创建有预算限制的持久化研究任务和子问题 |
| `continue_research` | 搜索或将指定 URL 读入任务 |
| `get_research` | 读取简报、原文段落、来源目录或完整状态 |
| `assess_research` | 保存宿主的判断、支持证据、挑战证据和缺口 |
| `complete_research` | 按需定稿带引用的宿主报告 |
| `save_report` | 导出报告，默认 Markdown |
| `cancel_research` | 取消工作，或显式删除任务记录 |
| `gather_bundle` | 不创建任务，直接检索一份证据包 |

创建任务时提供 `question` 和 `sub_questions`。每轮检索关联 `question_id`；
调查相反证据时使用 `purpose="counterevidence"`。简报省略原文，不代表没有原文：
评估证据前应调用 `get_research(job_id, passage_id=...)` 阅读对应段落。

评估会保留稳定的支持/挑战段落 ID、结论以及可选的 `follow_up_queries`。
检查 `research_plan` 中的缺口，说明下一轮检索的理由，读取新证据后重新评估。
使用 `get_research(view="sources", revision=..., offset=0, limit=10)`
可以分页查看当前引用窗口之外的已保存来源。

需要报告时，向 `complete_research` 提交当前 revision 和 `[Source N]` 引用，
再调用 `save_report(job_id=..., format="md")`。
引用检查验证目标是否存在，**不证明原文支持该主张，更不保证主张符合现实**。
研究不完整时必须保留缺口；`allow_incomplete=true` 会产生 partial 报告。
任务 completed 表示报告已定稿，不代表所有结论都正确。定稿报告不可修改。

[详细用法和限制](docs/USAGE.md#persistent-research-jobs) ·
[Agent 协议](docs/agent-protocol.md)

### 查看完整示例

[![离线双轮研究回放](assets/research-replay.svg)](docs/examples/sqlite-wal/README.md)

[阅读报告](docs/examples/sqlite-wal/report.md) ·
[检查证据](docs/examples/sqlite-wal/sources.json) ·
[复现回放](docs/examples/sqlite-wal/README.md)

这是使用已记录来源和宿主撰写推理的 MCP 回放，不是实时准确率评测。
它检查研究缺口、反证调查、导出和重启恢复。
[十案例实时宿主研究记录](docs/examples/host-suite/README.md)
另外披露了来源发现失败、宿主提供 URL 和明确拒绝下结论的情况。

## 可选能力

```bash
.venv/bin/python -m pip install -e '.[pdf]'
.venv/bin/python -m pip install -e '.[report]'
.venv/bin/python -m pip install -e '.[monitor]'
```

`pdf` 用于读取文本 PDF，仅支持 POSIX，不做 OCR；`report` 只提供 PDF 导出，不包含模型 SDK；
`monitor` 提供需显式启用和认证的本地运行监控。Markdown 和 JSON 导出不需要额外依赖。

`extract` 为兼容读取工具提供 Trafilatura；`rerank` 提供可选的本地 FlashRank 排序。
默认词法排序无需下载模型。参见[用法](docs/USAGE.md)和[运行指南](docs/OPERATIONS.md)。

不通过 agent 时，可以使用 `sibyl gather "a focused question" --format json`
或异步 Python API `sibyl.gather_bundle` 检查证据；它们都不会生成答案。

## 从旧架构升级

源码已经移除 LiteLLM、模型提供商配置、自主执行、通用模型分析、SWOT/时间线生成和金融/图表工具。
`report` 扩展现在仅表示 PDF 渲染。使用默认 `agent` 或兼容 `keyless` 入口；
旧的 `report`、`finance` 和 `full` profile 会被拒绝。

建议新建虚拟环境：升级软件包不会自动卸载旧的间接依赖。
删除 `SIBYL_CONFIG` 中的旧模型设置，不要把宿主登录 token 交给 Sibyl。
已有任务仍可读取，已完成报告仍可导出；旧 autonomous 任务不能继续执行。
如需继续调查，请新建 host 任务。参见[迁移说明](docs/USAGE.md#migration)。

## 质量目标和当前限制

Sibyl 的差异是持久化、由证据推动的研究流程，不是拥有更强的模型。
是否比内置搜索产生更好的答案，还需要同一宿主、相同来源访问条件和预算下的对照评测。

[2026-10-21 实时检索记录](evals/results/live-retrieval-keyless-budget-fix-2026-10-21.json)
对 66 个问题进行了三次重复测试：答案覆盖率为 83.3%，ready 状态的答案精度为 97.3%；
但可回答问题的 ready 比例为 66.0%，未达到 75% 门槛，重复稳定性为 68.2%，未达到 90% 门槛。
这些指标依据标准答案/别名在证据中的匹配，不是报告语义准确率。
十案例实时宿主研究中，有七个案例需要宿主提供 URL。
精简接口和删除内置模型并未自动解决这些来源发现问题。

公共搜索可能限流，页面可能阻止读取。来源分类、时间和重复内容判断都是启发式信号。
PDF 读取有资源与长度限制，不保证表格和公式还原。
来源文本可能包含提示注入，宿主必须将其视为不可信证据，而不是指令。

研究状态保存在 `SIBYL_OUTPUT_ROOT/.research` 下的本地 SQLite 中，有容量限制和显式删除机制，
但不自动过期，也不进行静态加密。一个可信本地宿主拥有一个运行实例；网络检查、revision 检查和
导出限制降低风险，但不等于安全沙箱。

[重构功能目标](docs/REFACTOR_GOALS.md)明确区分事实、推断、推荐和未知，并标明已实现与计划中的能力。
我们追求每个重要结论都有可核查的依据，不承诺“百分百真实”或“零幻觉”。

[发布证据](docs/LAUNCH_READINESS.md) · [实施计划](docs/QUALITY_BUILD_PLAN.md) ·
[安全](SECURITY.md) · [隐私](PRIVACY.md)

## 参与开发

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 中的离线测试、真实 MCP 回放、干净安装检查和依赖审计。
付费评测需要明确授权。[架构说明](docs/MCP_ARCHITECTURE.md)记录各模块的职责。

[更新日志](CHANGELOG.md) · [MIT 许可证](LICENSE)
