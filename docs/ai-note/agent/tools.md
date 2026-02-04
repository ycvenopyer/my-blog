## 什么是 Tool Use

大模型的 Tool（也称Function Calling）是指允许大模型调用外部工具或函数的机制。通过定义工具，模型可以决定何时调用、传递什么参数，并将结果整合到回复中。

![tool-use](image/tool-use.png)

一次完整的tool-use流程包含以下步骤：

- step1：定义tools，并将其包含在prompt中，发送给LLM；
- step2：拿到LLM的tool-use请求，调用tool；
- step3：把tool调用结果作为新的context，再发送给LLM。

## Tool 定义

对于每一个tool，需提供三类信息：

- name。工具的名称；
- description。工具的功能、限制、使用指南等；
- input schema。工具调用的入参规范，包括每个字段的名称、类型（int、bool、string、list等）、取值范围、描述等；通常采取json schema格式。

实际上，定义tools也是一次提示词工程，好的tool定义能帮助模型更好理解tools、从而带来更好的tool-use效果。

Tool-Use为LLM提供了More Context + Take Actions的组合拳，帮助LLM解决knowledge gap问题、自身能力限制问题以及无法执行外部动作的问题，大大拓宽了LLM的应用场景；其实现方式也比较简单，但如果遇到效果不及预期的情况，就得注意tools定义、tools数量及相关性、基座模型能力等因素。

## Tool 的常见格式

### OpenAI Function Calling 格式

```json
{
  "type": "function",
  "function": {
    "name": "get_current_weather",
    "description": "获取指定城市的当前天气",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "城市名称，如：'北京'"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "温度单位"
        }
      },
      "required": ["location"]
    }
  }
}
```

### LangChain Tool 格式

LangChain 自定义 tools有三种方式：

1.@tool装饰器

```python
search_wrapper = GoogleSearchClient()

@tool("my_search_tool")
def search(query: str) -> list[str]:
    """通过搜索引擎查询"""
    result = search_wrapper.search(query)
    return [res["snippet"] for res in result]

print(search.name)
print(search.description)
print(search.args)
```

用装饰器来定义Tool是最简单的方式，会默认函数名作为Tool的名称。也可以多传一个string类型的参数来覆盖名称。此外，装饰器会使用函数的注释作为tool的描述，所以函数必须有注释。

2.继承 BaseTool 类

```python
class SearchQuery(BaseModel):
    query: str = Field(..., description="要查询的query")

class CustomSearchTool(BaseTool):
    name = "my_search_tool_class"
    description = "通过搜索引擎来查询信息"
    args_schema: Type[BaseModel] = SearchQuery

    def _run(self, query: str) -> list[str]:
        """调用工具"""
        result = search_wrapper.search(query)
        return [res["snippet"] for res in result]


search = CustomSearchTool()
print(search.name)
print(search.description)
print(search.args)
```

3.用 StructuredTool 类提供的函数

你也可以用dataclass：StructuredTool。这种方法有点类似于上面两种方法的混合，比继承类方便，比用decorator的功能多。

```python
def search(query: str) -> list[str]:
    """通过搜索引擎查询"""
    result = search_wrapper.search(query)
    return [res["snippet"] for res in result]


search_tool = StructuredTool.from_function(
    func=search,
    name="我的搜索方法",
    description="通过搜索引擎查询，方便又强大",
    # coroutine= ... <- 如果需要，可以指定一个异步方法
)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args)
```

## Reference
[Langchain自定义Tool的三种方式](https://blog.csdn.net/weixin_48707135/article/details/137740363)

[Agent基础篇：Tool-Use的定义、实现方式和效果优化](https://zhuanlan.zhihu.com/p/1921263213938443768)

[OpenAI Platform Using Tools](https://platform.openai.com/docs/guides/tools)

[详解LLM大模型是如何理解并使用 tools](https://blog.csdn.net/2401_82469710/article/details/139984847)

