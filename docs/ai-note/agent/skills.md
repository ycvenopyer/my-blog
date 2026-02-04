## 1.什么是Skills

### 1.1 定义

**Skills是模块化的能力包，包含指令、脚本和资源，让Claude在需要时自动加载和使用。**

`模块化`：Skills是一个个独立的文件夹，每个Skill做一件事。

`能力包`：每个Skill文件夹里包含：

- SKILL.md（核心指令文件，必需）
- scripts/（可执行脚本，可选）
- references/（参考文档，可选）
- assets/（模块和资源，可选）

`自动加载`：你不需要手动告诉Claude现在用XX Skill。Claude会根据你的任务描述，自动判断需要哪个Skill，然后加载。

### 1.2 核心价值：把重复的指令打包，按需加载

**渐进式披露**：分阶段、按需加载。

一个Skill包含很多内容：核心指令、参考文档、执行脚本、模块资源。但Claude不会一次性把所有内容都加载进上下文。采用三层加载机制：

第一层：元数据——总是加载

- SKILL.md开头的YAML部分，就两个字段：name和description

第二层：指令——触发时加载

- SKILL.md的主体部分，详细的操作指南

第三层：*资源——引用时加载

- scripts/目录里的脚本、references/目录里的参考文档、assets/目录里的模板

优势：

- 节省tokens开销

- Skills可以包含可执行脚本，脚本在本地执行，返回结果给Claude，可以封装确定性的执行能力


## 2.Skills vs MCP

MCP(Model Context Protocol)：一个连接协议，让Claude能够访问外部系统：数据库、API、文件系统、各种SaaS服务。当需要连接外部系统时使用。它的核心目标是为AI模型与外部数据源及工具之间建立一个标准化、安全、高效的通信桥梁。类比于USB。

Skills是使用手册。当有重复性的工作流程时使用。

两者互补，在一个复杂的场景下需要同时使用。

**MCP让Claude能碰到外部系统，Skills告诉Claude碰到之后怎么用**。**MCP 负责“连通外部世界”， Skills 负责“干具体的事”。**

Skills比MCP更简洁（只需markdown+YAML元数据和一些脚本），tokens消耗更少，跨平台兼容，且门槛低生态好。

Skills的设计符合LLM的本质：用文本描述能力，让模型理解并执行。可以用Skills封装专业知识和工作流程。

MCP则像是传统软件工程的思路：定义接口、实现服务、处理通信。


## 3.Skill开发

### 3.1 适合Skills的情况

- 有固定工作流
- 团队协作
- token消耗大的

### 3.2 创建Skill

Skill封装了你的工作流程、你的经验沉淀、你的SOP，将这些东西写成SKILL.md，让AI来做即可。

- 想清楚你要解决什么问题
- 把你的工作流说清楚
- 提供足够的context和参考资料

SKILL.md的关键字段：

`YAML Frontmatter`:文件必须以YAML Frontmatter开头，包含两个必需字段：

- name:
  - 最多64个字符
  - 只能用小写字母、数字、连字符
  - 不能以连字符开头或结尾
  - 不能有连续的连字符
- description：
  - 最多1024个字符
  - 要包含做什么和什么时候用
  - 触发关键词很重要

`Markdown主体`:(可选但建议有)包含：

- 核心目标
- 执行步骤
- 示例输入/输出
- 注意事项

一个更完整的Skill结构：

~~~
my-skill/
├── SKILL.md                 # 核心指令
├── scripts/
│   └── process.py           # 可执行脚本
├── references/
│   └── DETAILED_GUIDE.md    # 详细参考文档
└── assets/
    └── template.md          # 模板资源
~~~

Skills拆解：Skills要精准简洁，按需加载省token，触发更精准，且Skills可组合。

Skills也可分优先级。

### 3.3 Skills设计的五个最佳实践

- description决定一切：做什么（核心功能）+什么时候用（触发场景）+触发关键词

- 单一职责：每个Skill只做一件事

- 渐进式披露：核心内容放SKILL.md，详细内容放references/

  ~~~
  # SKILL.md
  
  ## 快速流程
  1. 第一步
  2. 第二步
  3. 第三步
  
  ## 常见场景
  - 场景A：做法
  - 场景B：做法
  
  ## 详细参考
  - 更多细节见：[DETAILED_GUIDE.md](references/DETAILED_GUIDE.md)
  - 边界情况见：[EDGE_CASES.md](references/EDGE_CASES.md)
  ~~~

- 脚本优于生成代码

- 从简单开始，逐步迭代：从最小可行版本开始：写一个简单的SKILL.md，用几次，发现问题，添加遗漏的规则，添加常见错误的处理，逐步完善

### 3.4 Skills分类体系：Skills库的构建

- 按来源分

  ~~~
  Skills来源
  ├── 官方Skills（Anthropic提供）
  │   ├── 文档处理：docx, pdf, pptx, xlsx
  │   ├── 医疗健康：FHIR开发, 临床试验协议
  │   └── 生命科学：scVI-tools, Nextflow
  ├── 合作伙伴Skills
  │   └── Notion, Atlassian, Figma, Canva, Stripe, Zapier...
  └── 自定义Skills
      ├── 社区开源
      └── 个人/团队创建
  ~~~

- 按功能分：

  ~~~
  Skills功能分类
  ├── 文档与创意
  │   ├── 文档生成（PDF/Word/PPT/Excel）
  │   ├── 视觉设计（插画、动图）
  │   └── 内容创作（品牌指南、风格指南）
  ├── 开发与工程
  │   ├── 前端开发
  │   ├── 后端架构
  │   ├── 测试质量
  │   ├── DevOps
  │   └── 代码审查
  ├── 工作流与自动化
  │   ├── 协作流程
  │   ├── 知识管理
  │   └── 项目管理
  └── 垂直领域
      ├── 财务分析
      ├── 法律合规
      ├── 医疗健康
      └── 安全审计
  ~~~

- 按作用域分：

  ~~~
  Skills作用域
  ├── 个人级（~/.claude/skills/）
  │   └── 个人偏好、通用能力
  ├── 项目级（.claude/skills/）
  │   └── 项目规范、团队约定
  └── 组织级（API统一管理）
      └── 企业标准、合规要求
  ~~~


## 4.Reference

[Anthropic官方Skills仓库](https://github.com/anthropics/skills)

[Agent Skills开放标准](https://agentskills.io)

[Simon Willison的分析](https://simonwillison.net/2025/Oct/16/claude-skills/)

[Skills官方文档](https://code.claude.com/docs/en/skills)

[obra/superpowers](https://github.com/obra/superpowers)

[Sionic AI案例](https://huggingface.co/blog/sionic-ai/claude-code-skills-training)

