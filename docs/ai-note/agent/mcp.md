## 什么是MCP

MCP(Model Context Protocol，模型上下文协议)定义了应用程序和AI模型之间交换上下文信息的方式。这使得开发者能够以一致的方式将各种数据源、工具和功能连接到AI模型，就像USB-C让不同设备能够通过相同的接口连接一样。MCP的目标是创建一个通用标准，使AI应用程序的开发和集成变得更加简单和统一。

![mcp](image/mcp.png)

可以看出，MCP就是以更标准的方式让LLM Chat使用不同工具，Anthropic旨在实现LLM Tool Call的标准。

## 为什么是MCP

- 手工prompt的局限性：许多LLM平台（如OpenAI、Google）引入了function call功能，这一机制允许模型在需要时调用预定义的函数来获取数据或执行操作，显著提升了自动化水平。

- function call的局限性：其平台依赖性强，不同LLM平台的function call API实现差异较大。开发者在切换模型时需要重写代码，增加了适配成本，还有安全性，交互性等问题。

- 痛点所在：数据和工具本身是客观存在的，我们希望将数据连接到模型的这个环节可以更智能更统一。

MCP的优势：

1. 生态：MCP提供很多现成的插件

2. 统一性：不限制于特点的AI模型，任何支持MCP的模型都可以灵活切换

3. 数据安全：敏感数据保留本地不必上传

## MCP Architecture

![mcp architecture](image//model-context-protocol-architecture.png)

MCP由三个核心组件构成：Host、Client和Server。

假设你正在使用 Claude Desktop (Host) 询问："我桌面上有哪些文档？"

1. Host：Claude Desktop 作为 Host，负责接收你的提问并与 Claude 模型交互。

2. Client：当 Claude 模型决定需要访问你的文件系统时，Host 中内置的 MCP Client 会被激活。这个 Client 负责与适当的 MCP Server 建立连接。

3. Server：在这个例子中，文件系统 MCP Server 会被调用。它负责执行实际的文件扫描操作，访问你的桌面目录，并返回找到的文档列表。

整个流程是这样的：你的问题 → Claude Desktop(Host) → Claude 模型 → 需要文件信息 → MCP Client 连接 → 文件系统 MCP Server → 执行操作 → 返回结果 → Claude 生成回答 → 显示在 Claude Desktop 上。

这种架构设计使得 Claude 可以在不同场景下灵活调用各种工具和数据源，而开发者只需专注于开发对应的 MCP Server，无需关心 Host 和 Client 的实现细节。

## 原理：模型如何确定工具的选用

Anthropic的解释：

当用户提出一个问题时：

- 客户端（Claude Desktop / Cursor）将你的问题发送给 Claude。

- Claude 分析可用的工具，并决定使用哪一个（或多个）。

- 客户端通过 MCP Server 执行所选的工具。

- 工具的执行结果被送回给 Claude。

- Claude 结合执行结果构造最终的 prompt 并生成自然语言的回应。

- 回应最终展示给用户！

这个调用过程可以分为两步：

1. 由LLM确定使用哪些MCP Server

2. 执行对应的MCP Server并对执行结果进行重新处理

![mcp-call](image/mcp-call.png)

模型是通过prompt，即提供所有工具的结构化描述和few-shot的example来确定该使用哪些工具。

## 一些观点

1. function call中，每个工具接口的调用方式不一样，固然可以让LLM读取接口文档调用，但是还是很麻烦。MCP就是让所有的工具接口都用同一个方式调用，相当于用MCP协议包装了一层。这样，LLM就不需要每调用一个接口就要学习一次接口文档了。

2. MCP的能力核心在于 Claude 的function call能力。如果你的tools是上百个的，那么上下文就会超，而且部分大模型在如此多的tools选择和parameter列表生成中一定会有鲁棒性差的问题，所以当你tools上百个时不适合用MCP，或者不该直接使用MCP一次性对全部tools进行选择（可以像N叉树那样，将一次分类转移成多层级多次的分类，减小每次分类的选择空间）；而当你的tools只有十几个的时候，其实压根不用这么复杂——你大可以自己实现一下OpenAI或某个大模型API的function call代码，然后自己开一个tools.py文件，将所有tools按顺序描述好。

## 如何使用

[For Claude Desktop Users](https://modelcontextprotocol.io/docs/develop/connect-local-servers)

## Reference

[Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)

[MCP (Model Context Protocol)，一篇就够了](https://zhuanlan.zhihu.com/p/29001189476)

