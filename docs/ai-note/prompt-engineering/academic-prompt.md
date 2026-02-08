# 学术提示词总结

## 1.Preparation

首先我们要知道学术提示词仍然是提示词，因此提示词的技巧仍然适用，可以给定大模型一个角色定位和行为边界，参考系统提示词的写法，然后再引导其进行论文翻译和润色。

比如：
```text
假设你是一名资深的中英文翻译大师和英语学术作者，请帮我翻译一些中文段落成英文，确保你的语言符合专业学术期刊的语言风格。对翻译的要求有以下几点:1.确保翻译成英文的语言和原文意义一致，不可篡改原文意思；2.提供确切定义，确保你的术语和定义准确无误，特别是对于领域的专有名词和术语；2. 确保语言的逻辑关系准确，条理清晰；3. 采用简洁明确的表达方式，避免使用模糊或不必要的词汇、术语或句子；4. 注意英语语法表达的准确性，确保句子结构正确。你能做到吗？
```

## 2.[chatgpt-prompts-for-academic-writing](https://github.com/ahmetbersoz/chatgpt-prompts-for-academic-writing/blob/main/README.md)

### 2.1 BRAINSTORMING

```text
Find a research topic for a PhD in the area of [TOPIC]
```

```text
Write a detailed proposal on the following research topic. Make Sure it is free from plagiarism. [PARAGRAPH]
```

```text
Identify gaps in the literature on [TOPIC SENTENCE]
```

```text
Generate 10 academic research questions about [PARAGRAPHS]
```

```text
Generate a list of research hypotheses related to [TOPIC SENTENCE]
```

```text
Identify potential areas for future research in the context of this [TOPIC SENTENCE]
```

```text
Suggest novel applications of [TOPIC SENTENCE] within [RESEARCH DOMAIN]
```

### 2.2 ARTICLE SECTIONS

#### Title/Topic Sentence

```text
Suggest 5 titles for the following abstract: [ABSTRACT PARAGRAPH]
```

```text
Write a topic sentence for this paragraph: [PARAGRAPH] 
```

#### Keywords

```text
Provide 5 keywords for this: [PARAGRAPHS]
```

#### Abstract

```text
Generate an abstract for a scientific paper based on this information for: [PARAGRAPHS]
```

#### Outline

```text
Generate an outline for [TOPIC SENTENCE]
```

```text
I want to write a journal article about [TOPIC SENTENCE]. Give me an outline for the article that I can use as a starting point.
```

#### Introduction

```text
Come up with an introduction for the following research topic: [TOPIC SENTENCE]
```

#### Literature Review

```text
Conduct a literature review on [TOPIC SENTENCE] and provide review paper references
```

```text
Provide me with references and links to papers in [PARAPGRAPH]
```

> **NOTE:** Be careful and double-check article existence. ChatGPT may generate fake references

```text
Summarize the scholarly literature, including in text citations on [PARAGRAPHS]
```

```text
Write this in standard Harvard referencing [PARAGRAPH]
```

```text
Convert this [BIBLIOGRAPHY] from MLA to APA style.
```

```text
Compare and contrast [THEORY1] and [THEORY2] in the context of [RESEARCH DOMAIN]:
```

#### Methodology

```text
Create objectives and methodology for [TOPIC SENTENCE]
```

```text
Write a detailed methodology for the topic: [TOPIC SENTENCE]
```

```text
Analyze the strengths and weaknesses of this methodology: [PARAGRAPHS]
```

```text
Write objectives for this study: [TOPIC SENTENCE]
```

```text
What are the limitations of using [TOPIC SENTENCE] in [RESEARCH DOMAIN]?
```

```text
Create a recipe for the methods used in this [PARAGRAPHS]
```

```text
Suggest interdisciplinary approaches to [TOPIC SENTENCE]
```

```text
Explain how qualitative/quantitative research methods can be used to address [RESEARCH QUESTIONS]
```

```text
Recommend best practices for data collection and analysis in [TOPIC SENTENCE]
```

#### Experiments

```text
Design an experiment that [ACTION]
```

#### Results

```text
Write a result section for the following paragraphs. Please write this in the third person. [PARAGRAPHS]
```

#### Discussion

```text
Discuss these results: [RESULT PARAGRAPHS]
```

#### Conclusion

```text
Generate a conclusion for this: [PARAGRAPHS]
```

```text
Give recommendations and conclusion for: [PARAGRAPHS]
```

#### Future Works

```text
Can you suggest 3 directions for future research on this topic: [PARAGRAPH]]?
```

### 2.3 IMPROVING LANGUAGE

```text
Rewrite this paragraph in an academic language: [PARAGRAPH]
```

```text
Paraphrase the text using more academic and scientific language. Use a neutral tone and avoid repetitions of words and phrases. [PARAGRAPH]
```

```text
Correct the grammar: [PARAGRAPH]
```

```text
What do you think of how this paragraph is written?:  [PARAGRAPH]
```

```text
What 3 points would you suggest to improve this paragraph?: [PARAGRAPH]
```

```text
Improve the style of my writing? [PARAGRAPHS]
```

```text
Improve the clarity and coherence of my writing [PARAGRAPHS]
```

```text
Improve the organization and structure of my paper [PARAGRAPHS]
```

```text
Provide feedback on this text and suggest areas for improvement [PARAGRAPHS]
```

```text
Can you improve this paragraph using passive voice: [PARAGRAPH]
```

```text
Can you improve this paragraph to make it more cohesive? [PARAGRAPH]
```

```text
Analyze the text below for style, voice, and tone. Using NLP, create a prompt to write a new article in the same style, voice, and tone: [PARAGRAPHS]
```

```text
Please write a few paragraphs using the following list of points [LIST] 
```

```text
Give three variations of this sentence: [SENTENCE] 
```

```text
Write a transition sentence to connect the following two paragraphs: [PARAGRAPH1] [PARAPGRAPH2]
```

```text
Provide effective transitions between paragraphs [PARAGRAPH1] [PARAGRAPH2]
```

```text
Rewrite this paragraph as an introduction: [PARAGRAPH]
```

```text
Rewrite this paragraph as a conclusion: [PARAGRAPH]
```

```text
Write a counterargument to the following claim: [PARAGRAPH]
```

```text
Rewrite this in an academic voice: [PARAGRAPH]
```

```text
Expand these notes: [PARAGRAPH]
```

```text
Provide me a list of words and phrases which were repeatedly / more than 3 times used: [PARAGRAPHS]
```

```text
Provide me a list of synonyms for [PARAGRAPH] and evaluate them in the context of [PARAGRAPH]
```

```text
Act as a language expert, proofread my paper on [TOPIC SENTENCE] while putting a focus on grammar and punctuation.
```

```text
In the context of [RESEARCH DOMAIN] translate [PARAPGRAPH] into the [LANGUAGE] language.
```

```text
Proofread the following text for spelling and grammatical errors and rewrite it with corrections. [PARAGRAPHS] 
```

### 2.4 SUMMARIZATION

```text
Summarize the following content: [PARAPGRAPHS]
```

```text
Summarize the text in simpler and easier-to-understand terms. [PARAGRAPHS]
```

```text
Come up with a summary that is exactly [NUMBER OF WORDS] words: [PARAPGRAPHS]
```

```text
Reduce the following to [NUMBER OF WORDS] words: [PARAPGRAPHS]
```

```text
Shorten to [NUMBER OF CHARACTERS] characters: [PARAPGRAPHS]
```

```text
Give me a bullet point summary for [PARAPGRAPHS]
```

```text
Extract the important key points of this: [PARAPGRAPHS]
```

```text
Summarize the text by extracting the most important information in the form of bullet points [PARAGRAPHS]
```

```text
Explain this again but simpler: [PARAGRAPHS]
```

```text
Explain this research to a 12 year old: [PARAGRAPHS]
```

```text
Identify the key findings and implications of this: [PARAGRAPHS]
```

```text
Remove the throat-clearing sentence from this paragraph: [PARAGRAPH]
```

```text
Frontload the argument in the following paragraph: [PARAGRAPH]
```

```text
Explain [TOPIC] as an analogy
```

### 2.5 PLAN/PRESENTATION

```text
Develop a research plan for: [TOPIC SENTENCE]
```

```text
Write a schedule for completion in [TOPIC SENTENCE] in [NUMBER OF DAYS/MONTHS/YEARS]
```

```text
The deadline for the submission of the first draft is [DATE]. give me a week-by-week breakdown so I can plan my writing better.
```

```text
Write a sensational press release for this research: [PARAGRAPHS]
```

```text
Make this more persuasive: [PARAGRAPH]
```

```text
Write 3 tweets about this research. [PARAGRAPHS]
```

## 3.[Academic Prompts Collection](https://github.com/Kiteflyingee/academic_prompts?tab=readme-ov-file#1-%E6%B6%A6%E8%89%B2%E4%B8%8E%E7%BA%A0%E9%94%99-polishing--correction)

### 3.1 润色与纠错

#### 中文学术润色
> 作为一名中文学术论文写作改进助理，你的任务是改进所提供文本的拼写、语法、清晰、简洁和整体可读性，同时分解长句，减少重复，并提供改进建议。请只提供文本的更正版本，避免包括解释。请编辑以下文本

#### 英语学术润色
```text
Below is a paragraph from an academic paper. Polish the writing to meet the academic style, improve the spelling, grammar, clarity, concision and overall readability. When necessary, rewrite the whole sentence. Furthermore, list all modification and explain the reasons to do so in markdown table.
```

#### 查找语法错误
```text
Can you help me ensure that the grammar and the spelling is correct? Do not try to polish the text, if no mistake is found, tell me that this paragraph is good. If you find grammar or spelling mistakes, please list mistakes you find in a two-column markdown table, put the original text the first column, put the corrected text in the second column and highlight the key words you fixed.

Example:
Paragraph: How is you? Do you knows what is it?
| Original sentence | Corrected sentence |
| :--- | :--- |
| How **is** you? | How **are** you? |
| Do you **knows** what **is** **it**? | Do you **know** what **it** **is** ? |

Below is a paragraph from an academic paper. You need to report all grammar and spelling mistakes as the example before.
```

#### Latex 英文润色
```text
Below is a section from an academic paper, polish this section to meet the academic standard, improve the grammar, clarity and overall readability, do not modify any latex command such as \section, \cite and equations.
```

#### Latex 中文润色
> 以下是一篇学术论文中的一段内容，请将此部分润色以满足学术标准，提高语法、清晰度和整体可读性，不要修改任何LaTeX命令，例如\section，\cite和方程式。

#### AIGC内容“降AI味”
```text
你是一个伪装成人类科研学者的AI，能够将AI生成的文本改写成人类学者常用的表达方式。请对以下由AI生成的段落进行深度改写，在保持原意的基础上，使其风格更自然、更符合学术写作的习惯，让人难以分辨其是由AI生成。待改写段落：
```

---

### 3.2 翻译

#### 中译英
```text
Please translate following sentence to English, making it more accureate and academic:
```

#### 英译中
> 把下面的句子翻译成地道的中文，使它更加学术化：

#### 学术中英互译
```text
I want you to act as a scientific English-Chinese translator, I will provide you with some paragraphs in one language and your task is to accurately and academically translate the paragraphs only into the other language. Do not repeat the original provided paragraphs after translation. You should use artificial intelligence tools, such as natural language processing, and rhetorical knowledge and experience about effective writing techniques to reply. I'll give you my paragraphs as follows, tell me what language it is written in, and then translate:
```

---

### 3.3 阅读与写作辅助

#### 高效阅读论文
> 你是一位精通各领域前沿研究的学术文献解读专家，面对一篇给定的论文，请你高效阅读并迅速提取出其核心内容。要求在解读过程中，先对文献的背景、研究目的和问题进行简明概述，再详细梳理研究方法、关键数据、主要发现及结论，同时对新颖概念进行通俗易懂的解释，帮助读者理解论文的逻辑与创新点；最后，请对文献的优缺点进行客观评价，并指出可能的后续研究方向。整体报告结构清晰、逻辑严谨。

#### 优化文章结构
> 你是一位资深的文章优化专家，请你对给定的文章进行结构优化。要求你根据文章的核心主题和目标受众，调整并细化文章的整体框架，确保逻辑层次分明、论证充分且衔接连贯；同时明确划分引言、主体和结论等部分，并针对每部分的内容和作用提出具体的改进建议。请输出一个优化后的文章结构大纲，并用严谨、学术的语言详细说明各部分的功能和优化方案。

#### 解释代码
```text
请解释以下代码：```代码块```
```

---

### 3.4 研究与选题 

#### 论文选题
> 根据【研究方向】发展趋势、研究热点与已有文献，推荐一个创新性强且具有研究价值的研究论文选题。结合现有研究中的空白，提出一个具有填补空白潜力的问题，确保该选题能够推动学科的理论发展或实践应用。

#### 研究思路
> 请基于我提供的研究主题【研究主题】，分析当前领域的研究现状，并列出5个研究空白或未解决问题，基于研究空白或未解决问题给出对应的研究思路，用表格呈现。

#### 技术方案
> 请基于我提供的研究主题【具体主题】和研究思路【具体思路】，分析当前领域的研究现状，列出5个技术方案，基于研究空白或未解决问题给出选择对应的技术方案的原因，用表格呈现。

---

### 3.5 代码辅助 

#### 代码架构分析
> 你是一个专业的软件架构师。请分析以下多个代码文件的集合，并以清晰的语言总结这个项目的整体架构、核心功能、主要模块以及它们之间的相互关系。不要深入每一行代码的细节，重点在于宏观结构。

#### 代码自动注释
> 你是一个遵守代码最佳实践的程序员。请为以下代码的每一部分（函数、类、复杂逻辑块）添加简洁明了的注释，解释其功能、参数和返回值。


## Reference

[GPT 学术优化 (GPT Academic)](https://github.com/binary-husky/gpt_academic)

[导师推荐：12个论文翻译润色场景，专业版ChatGPT指令合集【附使用教程】](https://cloud.tencent.com/developer/article/2502595)

[50个顶级的ChatGPT学术论文指令！](https://zhuanlan.zhihu.com/p/688171911)

[黑科技！6个AI读文献指令，让你效率提高100倍！](https://zhuanlan.zhihu.com/p/691642735)

[10个顶级的论文降重指令！](https://zhuanlan.zhihu.com/p/688842912)

