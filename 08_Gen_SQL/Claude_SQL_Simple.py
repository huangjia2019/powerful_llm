from dotenv import load_dotenv
load_dotenv()

# 导入必要的库
from anthropic import Anthropic
import sqlite3

# 设置Anthropic API客户端
client = Anthropic()
MODEL_NAME = "claude-3-opus-20240229"

# 连接到测试数据库(如果不存在则创建)
conn = sqlite3.connect("test_db.db")
cursor = conn.cursor()

# 创建示例表
cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary INTEGER
    )
""")

# 插入示例数据
sample_data = [
    (6, "黄佳", "销售", 50000),
    (7, "赵宁", "工程", 75000),
    (8, "黄训谦", "销售", 60000),
    (9, "黄海悦", "工程", 80000),
    (10, "黄仁勋", "市场", 55000)
]
cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", sample_data)
conn.commit()

# 获取数据库模式
schema = cursor.execute("PRAGMA table_info(employees)").fetchall()
schema_str = "CREATE TABLE EMPLOYEES (\n" + "\n".join([f"{col[1]} {col[2]}" for col in schema]) + "\n)"
print("数据库模式:")
print(schema_str)

# 定义一个函数,将查询发送给Claude并获取响应
def ask_claude(query, schema):
    prompt = f"""这是一个数据库的模式:

{schema}

根据这个模式,你能输出一个SQL查询来回答以下问题吗?只输出SQL查询,不要输出其他任何内容。

问题:{query}
"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        messages=[{
            "role": 'user', "content":  prompt
        }]
    )
    return response.content[0].text

# 示例自然语言问题
question = "工程部门员工的姓名和工资是多少?"

# 将问题发送给Claude并获取SQL查询
sql_query = ask_claude(question, schema_str)
print("生成的SQL查询:")
print(sql_query)

# 执行SQL查询并打印结果
print("查询结果:")
results = cursor.execute(sql_query).fetchall()

for row in results:
    print(row)

# 关闭数据库连接
conn.close()