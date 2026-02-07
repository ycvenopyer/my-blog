# MkDocs 个人博客

一个基于 MkDocs 的轻量级个人博客，用于存储学习资料和个人信息。基本框架由Claude Code生成。欢迎使用此框架生成您自己的个人博客网站！

## 项目结构

```
my-blog/
├── mkdocs.yml          # MkDocs 配置文件
├── README.md           # 项目说明
├── requirements.txt    # Python 依赖
└── docs/               # 源文件目录
    ├── index.md        # 首页
    ├── about.md        # 关于我
    ├── ai.md           # AI相关
    ├── ai-note/        # AI学习笔记
    │   ├── a2a.md
    │   └── agent.md
    ├── stylesheets/    # 自定义样式
    │   └── extra.css
    └── javascripts/    # 自定义脚本
        └── extra.js
```

## 安装与使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 本地预览

```bash
mkdocs serve
```

访问 http://127.0.0.1:8000 查看博客

### 3. 构建静态网站

```bash
mkdocs build
```

生成的静态文件在 `site/` 目录下

### 4. 部署到 GitHub Pages

```bash
mkdocs gh-deploy
```

## 自定义内容

### 修改个人信息

编辑 `docs/about.md` 文件，填写你的个人信息。

### 添加新文章

在 `docs/../` 目录下创建新的 `.md` 文件，然后在 `mkdocs.yml` 的 `nav` 部分添加链接。

### 更新配置

编辑 `mkdocs.yml` 文件自定义博客名称、主题、导航等。

## 常用命令

| 命令 | 说明 |
|------|------|
| `mkdocs serve` | 启动本地预览服务器 |
| `mkdocs build` | 构建静态网站 |
| `mkdocs gh-deploy` | 部署到 GitHub Pages |
| `mkdocs --help` | 查看帮助信息 |

## 资源链接

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Markdown 语法指南](https://www.markdownguide.org/)
- [MkDocs-Material](https://squidfunk.github.io/mkdocs-material/)
