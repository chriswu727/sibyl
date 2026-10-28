# Sibyl

[English](README.md) | 简体中文

**为你现有的 AI agent 提供研究能力的 MCP：跨来源调查、检查证据、追踪反证和研究缺口，并在需要时导出带引用的报告。**

你的 agent 负责搜索和思考，Sibyl 在它自带的搜索之外提供持久化的研究工作区，并不替代它的搜索。它可以连接支持本地 MCP 的 agent，
包括通过订阅使用的 agent。Sibyl 不调用另一个生成式模型，不读取订阅凭证，也不需要第二个模型 API key。
宿主的订阅规则、额度限制和隐私条款仍然适用。

本文件是仓库唯一维护的中文文档。英文 README、功能目标和实施计划是规范版本。

> **当前是源码预览，不是新发布的 PyPI 版本。** 本文描述源码中的 host-only 架构。
> 已发布的 beta 0.4.0b1 和稳定版 0.3.0 仍包含旧架构。请按下方步骤安装源码。
> 目前还没有配对评测证明：在 agent 自带搜索的基础上加入 Sibyl，能得到证据支持更充分的回答。

## 给 agent 的研究能力，给人的可选报告

一次搜索返回线索，Sibyl 将调查过程连接起来：

- 将问题拆成研究子问题，并跨多次调用保存计划。
- 把 agent 自己找到的页面读入同一份证据库，也可以使用免 key 搜索。
- 按需检查原文段落、稳定标识、内容哈希和来源信息。
- 事实主张引用的原文如果不在所引段落中逐字出现，直接拒绝。
- 保存有支持、有争议或未解决的判断及其证据链接。
- 调查反证，围绕具体缺口继续检索。
- 直接复用结构化研究结果，或导出 Markdown、JSON 和 PDF。

宿主决定查什么、阅读证据并判断它支持什么。Sibyl 负责检索、保存状态、检查引用关系和限制资源使用。
它不替宿主判断绝对真伪，也不独立运行一个研究 agent。

### 什么时候用

当结论需要被核对或复用时使用 Sibyl：需要引用证据的决策、应当考虑反证的比较、
跨多个问题或多次会话的调查、要交给别人的结论，或需要一份报告。
只是快速查一个事实时，agent 自带的搜索就够了。Sibyl 会增加步骤，证据链重要时才值得。

默认 MCP 入口提供 **8 个工具**，返回紧凑的结构化简报；每次读取或评估后返回单个问题的更新，原文按需读取。
研究结果不需要先变成一篇长报告，才能被后续任务使用。

## 快速开始

需要 Python 3.11+，以及支持本地 stdio MCP 的宿主。agent 模式的结果同时放在 `structuredContent`
和同样内容的 JSON 文本中，只把文本交给模型的宿主也能拿到完整结果。
请使用全新的虚拟环境安装源码：

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/sibyl-mcp --list-tools
```

最后一条命令应列出 8 个工具。对于接受 `mcpServers` JSON 的客户端，
可以使用以下配置，并将两个路径替换为本机的绝对路径。`SIBYL_OUTPUT_ROOT` 可省略；
省略时研究数据保存在当前用户的数据目录，由本机的多个会话共享：

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
具体配置语法取决于宿主。结果以 MCP `structuredContent` 返回，并同时附上相同的 JSON 文本。
原有的 `--profile keyless` 文本兼容入口已移除。这不是远程 HTTP MCP 服务。

默认搜索不需要 key。可选 Tavily 搜索使用独立凭证，并可能产生额外费用；
agent 的订阅不包含 Tavily 的费用。参见[安装说明](llms-install.md)。

可以这样对你的 agent 提问：

> 用 Sibyl 调查 SQLite WAL 是否适合一个多进程应用。阅读原始文档，调查限制和反证，
> 返回与证据关联的结论及未解决的问题。只有我要求时才生成报告。

## 研究流程

| 工具 | 用途 |
|---|---|
| `start_research` | 创建有预算限制的持久化研究任务和子问题 |
| `continue_research` | 将 agent 找到的 URL 读入任务，或使用免 key 搜索 |
| `get_research` | 读取简报、批量读取原文段落、来源目录或完整状态 |
| `assess_research` | 保存宿主的判断、支持证据、挑战证据和缺口 |
| `complete_research` | 按需定稿带引用的宿主报告 |
| `save_report` | 导出报告，默认 Markdown |
| `cancel_research` | 取消工作，或显式删除任务记录 |
| `gather_bundle` | 不创建任务，直接检索一份证据包 |

创建任务时提供 `question` 和 `sub_questions`。宿主有自带搜索时，用它查找来源，
再用 `continue_research(url=..., query=...)` 读入任务；只提供 query 时使用 Sibyl 的免 key 搜索。
每轮检索关联 `question_id`；调查相反证据时使用 `purpose="counterevidence"`。
每次读取返回一份更新：新来源的主要段落直接内联（过长的段落会被截断），并附上该问题的缺口和下一步搜索建议。
完整简报省略原文，不代表没有原文：评估证据前应调用
`get_research(job_id, passage_ids=[...])` 一次读取最多 12 个完整段落。

评估会保留稳定的支持/挑战段落 ID、结论以及可选的 `follow_up_queries`。
检查 `research_plan` 中的缺口，说明下一轮检索的理由，读取新证据后重新评估。
使用 `get_research(view="sources", revision=..., offset=0, limit=10)`
可以分页查看当前引用窗口之外的已保存来源。

新的 MCP 任务还通过 `assess_research(claims=...)` 记录每条关键事实、推断和推荐。
事实关联原文段落并逐字引用原文；引文在所引段落中找不到时（仅忽略大小写、空白、重音编码和排版符号差异，数字不做归一）会被拒绝。
推理关联所依据的主张，推荐保留标准与假设，未知信息保持明确。
基础事实改变或出现新检索时，依赖它的判断也需要复核。
参见[主张与证据协议](docs/research-claims.md)。

需要报告时，向 `complete_research` 提交当前 revision 和可选的 `claim_ids`，
再调用 `save_report(job_id=..., format="md")`。报告从宿主已经记录的主张生成，
自动带入引用、前提、标准和假设；未被选择的未知或争议信息仍然保留。
还需要 JSON 时，再调用一次 `save_report(format="json", job_id=...)` 并使用返回的路径，
不能通过替换后缀猜测文件路径。导出结果会列出已确认的真实路径、格式、大小、哈希及各格式的失败信息；
请求失败时也会返回结构化修复指引和可读取到的当前 revision，不会悄悄接受无效主张。
旧版（contract-1）任务不能再定稿，已完成的仍可读取和导出。
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
`report` 扩展现在仅表示 PDF 渲染。Sibyl 只提供一套工具；
旧的 `keyless`、`report`、`finance` 和 `full` profile 会被拒绝。

建议新建虚拟环境：升级软件包不会自动卸载旧的间接依赖。
删除 `SIBYL_CONFIG` 中的旧模型设置，不要把宿主登录 token 交给 Sibyl。
已有任务仍可读取，已完成报告仍可导出；旧 autonomous 任务不能继续执行。
如需继续调查，请新建 host 任务。参见[迁移说明](docs/USAGE.md#migration)。
[兼容边界清单](docs/MIGRATION.md)区分已经删除的执行能力、保留的旧数据读取能力和历史评测证据；
[现行架构](docs/MCP_ARCHITECTURE.md)记录新设计及模块职责。

## 质量目标和当前限制

Sibyl 的差异是持久化、由证据推动的研究流程，不是拥有更强的模型。
是否比内置搜索产生更好的答案，还需要同一宿主、相同来源访问条件和预算下的对照评测。

发布用检索门槛（每个问题读取三个固定的宿主搜索 URL）的[最新运行](evals/results/host-source-retrieval-honest-ua-2026-10-27.json)只有 ready 比例未通过：
答案覆盖率 87.0%、稳定性 98.5%、ready 状态的答案精度 100% 均达标，
但可回答问题的 ready 比例为 30.2%，主要因为来自单一域名的含答案读取被评为 `limited`。
维护者随后把该比例改为只报告、不卡发布，因此这次运行满足修订后的第一道门槛；剩下的研究质量门槛是同宿主对照评测。其[两次](evals/results/paired-host-2026-10-27/manifest.json)[运行](evals/results/paired-host-rerun-2026-10-28/manifest.json)均未通过。
合并两次结果，Sibyl 的引文支持率最高（79.4%，内置搜索为 74.9%），依据充分度评分也最高，事实题和陷阱题结果相同，成本约为内置搜索的两倍；
单次运行之间的差异在随机波动范围内。
[首次运行](evals/results/host-source-retrieval-2026-10-27.json)时，页面读取尚未以 Sibyl 身份标识，所有 Wikipedia 读取都被拒绝。
[最新免密钥发现记录](evals/results/live-retrieval-hedged-2026-10-27.json)
在 GitHub 托管的运行器上对 66 个问题进行了三次重复测试：陷阱问题全部安全处理，
但答案覆盖率为 74.1%（门槛 80%），可回答问题的 ready 比例为 35.2%（门槛 75%），
重复稳定性为 48.5%（门槛 90%），ready 状态的答案精度为 90.5%（门槛 95%），均未达标。
在该数据中心网络上，DuckDuckGo 和 Mojeek 几乎总是拒绝免密钥请求。
这些指标依据标准答案/别名在证据中的匹配，不是报告语义准确率。
十案例实时宿主研究中，有七个案例需要宿主提供 URL。
精简接口和删除内置模型并未自动解决这些来源发现问题。
目前没有该评测的完整通过记录。
[两次低 effort 宿主复测](evals/results/host-agent-structured-retest-2026-10-26.json)
都成功交付了 partial 状态的 Markdown/JSON，但 SQLite 推荐仍有一个重要限定条件未被所引片段支持。
此后加入了逐字引文检查：编造或放错段落的引文会被拒绝，但真实引文仍可能被过度解读。
一次[两个任务的 Claude Code 配对小测](evals/results/host-claude-code-paired-2026-10-27.json)（每个条件各跑一次，非盲评）
结果一胜一负：使用 Sibyl 时，宿主为每条有支持的事实都核对了原文引句，并发现了只用内置搜索的基线漏掉的一个严重 SQLite 损坏 bug；
但在 NVIDIA 问题上，它选到的来源比基线弱，成本约为基线的五倍。这不能证明 Sibyl 更好。
导出成功不代表研究语义质量通过。

检索会从有长度限制的提取正文中为每个来源选择最多三个不重叠片段，结合 Unicode
关键词和正文、表格语境，而不是只匹配目录标题。缺少章节时可用
`continue_research(url=..., query=...)` 定向读取：最多提取 60,000 字符，返回证据不超过
16,000 字符，保留准确位置和哈希；未返回的部分仍视为未读。这些是相关性启发式，
不等于语义支持校验，也不会自动翻译查询。
精确片段读取会显示有界来源长度、是否为摘录，以及预算允许时的进一步读取入口。
每项限定条件仍需逐一核对引用原文；这些提示不等于独立语义校验或完整阅读证明。
已标明的 Crossref 元数据和搜索摘要保留为线索，不计入就绪判断的实质正文或独立正文域名。
下载到 HTML 也可能只是摘要或目录记录，宿主仍需检查实际证据范围。

公共搜索可能限流，页面可能阻止读取。来源分类、时间和重复内容判断都是启发式信号。
PDF 读取有资源与长度限制，不保证表格和公式还原。
来源文本可能包含提示注入，宿主必须将其视为不可信证据，而不是指令。

研究状态保存在输出根目录的 `.research` 下的本地 SQLite 中，有容量限制和显式删除机制，
但不自动过期，也不进行静态加密。输出根目录为 `SIBYL_OUTPUT_ROOT`；未设置时使用当前用户的数据目录，
旧的 `./reports` 不会被移动。同一用户的本地会话共享这份存储：某个会话正在运行的任务，
在其他会话中只读。这是同一用户的共享研究空间，不是项目之间的隔离。
详见[共享研究存储](docs/USAGE.md#shared-research-store)。网络检查、revision 检查和
导出限制降低风险，但不等于安全沙箱。

基于结论记录的人读报告只呈现一次结论和推理，JSON 保留完整兼容记录。
PDF 导出需要字体覆盖实际文本；缺少字形或渲染失败时会明确报错，不再保存缺少正文却显示成功的文件。
此时仍可使用 Markdown/JSON。

[重构功能目标](docs/REFACTOR_GOALS.md)明确区分事实、推断、推荐和未知，并标明已实现与计划中的能力。
我们追求每个重要结论都有可核查的依据，不承诺“百分百真实”或“零幻觉”。

[发布证据](docs/LAUNCH_READINESS.md) · [实施计划](docs/QUALITY_BUILD_PLAN.md) ·
[安全](SECURITY.md) · [隐私](PRIVACY.md)

## 参与开发

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 中的离线测试、真实 MCP 回放、干净安装检查和依赖审计。
付费评测需要明确授权。[架构说明](docs/MCP_ARCHITECTURE.md)记录各模块的职责。

[更新日志](CHANGELOG.md) · [MIT 许可证](LICENSE)
