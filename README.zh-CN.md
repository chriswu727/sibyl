# Sibyl

[English](README.md) | 简体中文

**把 AI agent 的调研结论和它所引用的网页原文逐条核对。**

本文件是仓库唯一维护的中文文档，以英文 README 为准。

你的 agent 用它自己的搜索和抓取工具做调研，把每条结论连同依据的 URL 和逐字引文记进一个小的 JSON 账本（ledger）。
Sibyl 随后自己去读每个被引用的网页，并报告：

- 在网页正文里找不到的引文（比对前统一大小写、引号、破折号和空白；`...` 可跳过同一段落中的词）；
- 读不到的网页，其引文因此无法核实；
- 结论里出现、但支持它的引文中都没有的数字；
- 结论里出现、但在引文附近找不到的名称；
- 只有一个网站支持的结论，以及谈同一主题却给出不同数字的结论；
- 依赖于存在上述阻断问题的结论的推论。

核对通过后可以生成 Markdown 报告，包含结论、有争议和未解决的结论、反证搜索、研究缺口，以及每个来源所读文本的哈希。

Sibyl 以 **agent skill**（一个教 agent 按流程操作的 `SKILL.md`）加上它调用的 `sibyl` 命令的形式提供。它不搜索、不排序，也不调用任何模型。

> **Alpha 版，仅源码。** 这是对 Sibyl 早期调研 MCP 服务器的重写，旧服务器已退役，原因见[为什么重写](#为什么重写)。
> PyPI 上已发布的 0.4.0b1 和 0.3.0 是旧服务器。

## 安装

需要 Python 3.11 或更高版本：

```bash
git clone https://github.com/chriswu727/sibyl.git
python -m pip install -e './sibyl[pdf]'   # [pdf] 用于读取文本 PDF（仅 POSIX）
sibyl skill install                        # 写入 ~/.claude/skills/sibyl/SKILL.md
```

宿主从其他目录读取 skill 时，用 `sibyl skill install --dir <路径>`；`sibyl skill show` 打印 skill 内容。agent 需要有 shell 工具才能运行 `sibyl`。

## 使用

让 agent 给出有调研、带引用的回答即可，skill 会告诉它何时使用账本。也可以手动试用：

```bash
sibyl template > research/topic.json      # 从示例开始
# 编辑结论、URL 和引文
sibyl check research/topic.json           # 退出码 0 通过，1 有阻断问题，2 账本格式无效
sibyl report research/topic.json          # 写出 research/topic.md
```

`sibyl check` 会在账本旁写出 `topic.check.json`，记录每个来源的读取状态和文本 SHA-256、每条引文的位置以及所有问题。
`--save-pages DIR` 会保存从每个网页读到的文本。账本在核对之后被修改过时，`sibyl report` 会拒绝生成；核对未通过时，也会拒绝，
除非加上 `--allow-issues`，这时报告会标出每条未通过的结论。

所有字段和检查项见[账本格式](docs/ledger-format.md)（英文）。

## 核对通过意味着什么

意味着核对时每条引文都出现在 Sibyl 从对应网页提取的文本中，且结论里的数字和名称与引文相符。
它**不**意味着引文支持该结论、引文没有断章取义、来源正确或回答完整。结论仍然是 agent 自己的。
网页会变化：记录的哈希标识的是当时读到的内容，而不是网页现在的内容。

Sibyl 只读取公开的 HTTP(S) 网页，并如实表明自己的身份。拒绝自动化客户端、需要登录或只能靠 JavaScript 渲染的网页读不到；
PDF 最多读取 40 页，不做 OCR。

## 为什么重写

Sibyl 起初是一个调研 MCP 服务器，在 agent 自身调研能力还较弱的时候，帮它规划搜索、给来源排序、引导整个持久化的调查过程。
在一个预先登记的 20 题配对评测中，两次运行都显示：使用该服务器的 agent 并不比只用内置搜索的同一 agent 稳定地更好，
差异在两次运行之间的随机波动范围内，成本却约为两倍（[记录](evals/README.md)）。
agent 自己做不到的，是确定性地核对自己的引文，所以 Sibyl 只保留了这一部分。

## 安全与隐私

网页文本是不可信数据，可能包含针对 agent 的指令；skill 要求 agent 绝不执行这些指令。
详见 [SECURITY.md](SECURITY.md) 和 [PRIVACY.md](PRIVACY.md)。贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)。

MIT 许可证。
