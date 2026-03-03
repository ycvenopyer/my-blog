## 什么是Superpowers

Superpowers是一个为AI编程代理（如Claude Code、Codex、OpenCode）打造的完整软件开发工作流系统。它的核心理念是：通过一套可组合的技能（Skills）和初始指令，让AI代理在编写代码时自动遵循最佳实践，而不是像“没有经验的初级工程师”那样随意行事。

## 设计哲学

- 测试驱动开发（TDD）：永远先写测试。没有看到测试失败，就不能确定测试是否真正测试了正确的行为。

- 系统化而非临时化：用流程替代猜测。每个技能都有明确的决策流程图，作为可执行规范。

- 复杂度削减：以简洁为首要目标。技能反复强调YAGNI（You Aren't Gonna Need It）原则，积极删除不必要的功能。

- 证据而非声明：在宣布任务完成之前，必须验证。看到测试通过，看到代码运行，而不是“我觉得应该可以了”。

## 工作流程：7 步强制执行的开发流程

| 步骤 | 说明 | 关键特性 |
|------|------|----------|
| 1. 头脑风暴 | 编码前先细化需求，通过提问完善设计 | 分块展示设计文档供确认 |
| 2. Git 工作树管理 | 设计确认后创建隔离的分支工作区 | 验证干净的测试基线 |
| 3. 编写执行计划 | 将工作拆解为 2-5 分钟可完成的小任务 | 每个任务包含文件路径、完整代码、验证步骤 |
| 4. 子代理驱动开发 | 为每个任务分配独立子代理 | 执行「规格合规性 + 代码质量」两阶段评审 |
| 5. 测试驱动开发 | 严格遵循 RED-GREEN-REFACTOR 流程 | 编写失败测试 → 验证失败 → 编写最小化代码 → 验证通过 → 提交 |
| 6. 代码评审 | 任务间自动触发评审 | 按严重程度标记问题，关键问题会阻塞流程 |
| 7. 分支收尾 | 任务完成后验证测试 | 提供合并 / PR / 保留 / 丢弃分支的选项，清理工作树 |

## 技能库详解

项目包含 14 个核心技能，分为几大类别：

{==测试类==}

- test-driven-development：强制执行 RED-GREEN-REFACTOR 循环。核心规则是"先写测试失败的代码？删掉，重新来"，包含详细的反模式参考

{==调试类==}

- systematic-debugging：四阶段根因定位流程，整合了 root-cause-tracing（逆向追踪调用栈）、defense-in-depth（多层验证）、condition-based-waiting（基于条件的等待替代任意超时）等技术

- verification-before-completion：确保问题真正被修复

{==协作类==}

- brainstorming：苏格拉底式设计提炼

- writing-plans：详细实现计划

- executing-plans：批量执行与检查点

- dispatching-parallel-agents：并发子代理工作流

- requesting-code-review / receiving-code-review：代码审查的请求与响应

- using-git-worktrees：并行开发分支

- finishing-a-development-branch：合并/PR 决策工作流

- subagent-driven-development：两阶段审查的快速迭代

{==元技能==}

- using-superpowers：技能系统入门

- writing-skills：如何创建新技能（包含测试方法论）


## 安装与使用

**Claude Code 安装**

```shell
# 1. 添加插件市场
/plugin marketplace add obra/superpowers-marketplace

# 2. 从市场安装插件
/plugin install superpowers@superpowers-marketplace

# 3. 验证安装是否成功
/help
# 成功将会看到:
# /superpowers:brainstorm - Interactive design refinement
# /superpowers:write-plan - Create implementation plan
# /superpowers:execute-plan - Execute plan in batches
```

**Codex/OpenCode 安装**

加载对应平台的远程安装指南，复用核心技能逻辑。

## 总结

Superpowers 本质是「编码代理的开发流程操作系统（Development Process OS for Coding Agents）」，通过标准化技能和工作流，让代理能够像专业工程师一样遵循规范完成开发，大幅提升编码代理的输出质量和开发效率。

## Reference

[Superpowers GitHub](https://github.com/obra/superpowers)

[告别Vibe Coding！用Superpowers让Claude Code写出工程级代码，一次通过零报错！遵循TDD最佳实践！支持Codex和OpenCode](https://juejin.cn/post/7593573617648123956)
