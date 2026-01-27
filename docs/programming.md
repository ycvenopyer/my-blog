# 编程笔记

这里记录我的编程学习笔记和代码片段。

## Python

### 列表推导式

```python
# 基础用法
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
# 结果: [1, 4, 9, 16, 25]

# 带条件
even_squares = [x**2 for x in numbers if x % 2 == 0]
# 结果: [4, 16]
```

### 装饰器

```python
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.2f}秒")
        return result
    return wrapper

@timer
def my_function():
    time.sleep(1)
    return "完成"
```

## JavaScript

### Promise 基础

```javascript
const fetchData = () => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve("数据获取成功");
    }, 1000);
  });
};

fetchData().then(data => {
  console.log(data);
});
```

### 数组方法

```javascript
const arr = [1, 2, 3, 4, 5];

// map: 转换数组
const doubled = arr.map(x => x * 2);
// [2, 4, 6, 8, 10]

// filter: 过滤数组
const evens = arr.filter(x => x % 2 === 0);
// [2, 4]

// reduce: 累加
const sum = arr.reduce((acc, x) => acc + x, 0);
// 15
```

## Git 常用命令

| 命令 | 说明 |
|------|------|
| `git init` | 初始化仓库 |
| `git clone <url>` | 克隆远程仓库 |
| `git add .` | 添加所有变更 |
| `git commit -m "msg"` | 提交变更 |
| `git push origin main` | 推送到远程 |
| `git pull origin main` | 拉取远程变更 |

## 数据库

### SQL 基础查询

```sql
-- 基础查询
SELECT * FROM users WHERE age > 18;

-- 连接查询
SELECT u.name, o.order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- 聚合
SELECT COUNT(*) as total FROM orders;
```

---

!!! note "持续更新"
    更多笔记内容将持续添加...
