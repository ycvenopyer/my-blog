# 提示词工程教学

## 提示词定义

提示词是人类与AI沟通的{==指令系统==}，它通过文字定义AI的身份、任务、行为边界和执行逻辑。

核心价值：提示词不是零散的指令堆砌，而是通过结构化设计，让AI理解“我是谁、该做什么、怎么做、不能做什么”，从而将人类意图转化为AI的稳定行为。

## 提示词的核心要素

1. 身份与目标：AI的角色定位和存在意义，解决“为什么做”；

2. 交互逻辑：AI接收信息（输入）和输出结果的规范，解决“如何沟通”；

3. 执行流程：AI完成任务的步骤和技能拆解，解决“怎么做”；

4. 行为边界：AI绝对不能触碰的红线，解决“什么不能做”。

## 提示词策略——ChatGPT官方推荐

### 1.指令要清晰

要清晰地表达你想要什么，不要让GPT猜你想要什么。

- 问题里包含更多细节。

- 让模型角色扮演（[GitHub人设大全](https://github.com/f/prompts.chat)）。

- 充当ChatGPT提升生成器。

!!! note "prompt generator"
    适用场景：当不知道如何写提示词时，可以尝试使用这种方法。（当然，这种方法依旧不完美，需要不断优化，最终达到自己的目的）

    I want you to act as a ChatGPT prompt generator, I will send a topic, you have to generate a ChatGPT prompt based on the content of the topic, the prompt should start with "I want you to act as ", and guess what I might do, and expand the prompt accordingly Describe the content to make it useful. response need in simplified Chinese. My query is: 英语听力老师。

- 使用分隔符：使用三重引号、XML标签、章节标题等分隔符可以帮助划分文本的不同部分。

- 指定完成任务所需的步骤。

- 提供示例（few-shot）。

- 设定回答的长度。

- 指定段落数量。

- 指定要点数量。

### 2.提供参考文本

- 命令模型根据参考文本回答问题：就像我们在写作业时，如果有了老师给的参考资料，就可以利用这些资料来写答案，这个模型也一样，有了相关的、可信的信息，就可以用这些信息来回答问题。

- 让ChatGPT用引用参考文本的方式回答问题：我们可以给ChatGPT提供材料，并让它回复的时候，标明是根据材料的哪一部分做出的回答。这就好像，我们在写论文的时候，要标注信息来源一样。这样做帮助我们在材料里找到引用的文字，来确认这些引用的文字是否真的存在。

### 3.将复杂任务拆分为更简单的子任务

- 问题分类：根据不同的任务类型进行分类，每一个任务可能都需要不同的步骤或者指令。

- 长对话处理：总结或过滤

- 分段总结长文并递归构建完整摘要

### 4.给GPT时间“思考”

- 在下结论之前，先引导GPT生成自己的答案：有时候，如果我们明确地告诉模型，在得出结论之前，先按照基本原理进行推理，可能会得到更好的结果。

- 隐藏推理过程

!!! note "prompt template"
    Follow these steps to answer the user queries. 
    Step 1 - First work out your own solution to the problem. Don't rely on the student's solution since it may be incorrect. Enclose all your work for this step within triple quotes ("""). 
    Step 2 - Compare your solution to the student's solution and evaluate if the student's solution is correct or not. Enclose all your work for this step within triple quotes ("""). 
    Step 3 - If the student made a mistake, determine what hint you could give the student without giving away the answer. Enclose all your work for this step within triple quotes ("""). 
    Step 4 - If the student made a mistake, provide the hint from the previous step to the student (outside of triple quotes). Instead of writing "Step 4 - ..." write "Hint:".  
    按照以下步骤回答用户查询。 
    第 1 步 - 首先找出您自己的问题解决方案。不要依赖学生的解决方案，因为它可能不正确。将您为此步骤所做的所有工作用三重引号 (""") 括起来。 
    第 2 步 - 将您的解决方案与学生的解决方案进行比较，并评估学生的解决方案是否正确。将您为此步骤所做的所有工作用三重引号 (""") 括起来。 
    第 3 步 - 如果学生犯了错误，请确定您可以在不给出答案的情况下给学生什么提示。将您为此步骤所做的所有工作用三重引号 (""") 括起来。 
    第 4 步 - 如果学生犯了错误，请向学生提供上一步的提示（三重引号外）。不要写“第 4 步 - ...”，而写“提示：”。

    Problem Statement: <insert problem statement> 
    Student Solution: <insert student solution>  
    问题陈述：<插入问题陈述> 
    学生解决方案：<插入学生解决方案>

- 答案不全的问题：我们正在用一个ChatGPT模型从一堆资料中找出跟我们提出的问题有关的内容。每找到一个内容，模型就要决定是不是要继续找下一个，还是停下来不找了。如果那堆资料特别大，模型有时候会停得太早，没能把所有跟问题有关的内容都找出来。这个时候，如果我们再向模型提出一些新的问题：让它再去找找看有没有之前漏掉的内容，往往能让模型的输出结果，变得更好。

### 5.使用外部工具

- 嵌入：使用基于嵌入的搜索来实现高效的知识检索。一个模型可以利用外部信息作为其输入的一部分。比如RAG。

- 计算：使用代码或者调用外部的API，来进行更精确的计算。

- 让模型使用特定功能：Chat Completions API允许在请求中传递一系列功能描述。这让模型能根据提供的方案生成功能参数。通过API以JSON格式返回生成的功能参数，可以用来执行功能调用。功能调用提供的输出可以在下一个请求中反馈给模型，从而形成一个完整的循环。这是使用GPT模型调用外部功能的推荐方式。

### 6.系统地测试变化

- 参考黄金标准答案评估模型输出：假设已知问题的正确答案应该参考一组特定的已知事实。然后我们可以使用模型查询来计算答案中包含了多少所需事实。

## Reference

[六种方法，写出好的提示词——ChatGPT官方推荐](https://zhuanlan.zhihu.com/p/648018011)
