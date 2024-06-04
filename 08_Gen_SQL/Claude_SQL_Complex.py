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


# 创建部门表
cursor.execute("""
    CREATE TABLE IF NOT EXISTS departments (
        id INTEGER PRIMARY KEY,
        name TEXT,
        manager TEXT
    )
""")

# 插入示例数据
sample_departments = [
    (1, "销售", "王经理"),
    (2, "工程", "李经理"),
    (3, "市场", "张经理")
]
cursor.executemany("INSERT INTO departments VALUES (?, ?, ?)", sample_departments)
conn.commit()

# 获取完整的数据库模式
tables = ["employees", "departments"]
schema_str = ""
for table in tables:
    schema = cursor.execute(f"PRAGMA table_info({table})").fetchall()
    schema_str += f"CREATE TABLE {table} (\n" + "\n".join([f"{col[1]} {col[2]}" for col in schema]) + "\n);\n\n"

print("完整的数据库模式:")
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

# 查询每个部门的员工人数和平均工资
question = "根据两个表之间的关系,列出每个部门的员工人数和平均工资"

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


import matplotlib.pyplot as plt
plt.rcParams["font.family"]=['SimHei'] # 用来设定字体样式
plt.rcParams['font.sans-serif']=['SimHei'] # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

# 计算平均工资
departments = [row[0] for row in results]
avg_salaries = [row[2] for row in results]

# 生成条形图
plt.figure(figsize=(8, 5))
bars = plt.bar(departments, avg_salaries, color=['#1f77b4', '#ff7f0e', '#2ca02c'])  # 为每个部门设置不同的颜色

# 设置图表标题和标签
plt.xlabel("部门")
plt.ylabel("平均工资")
plt.title("各部门平均工资")

# 添加网格线，提高图表的可读性
plt.grid(True, linestyle='--', alpha=0.6)

# 在每个条形图上方显示具体数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', va='bottom', ha='center', color='black')

# 保存图表到文件
plt.savefig('Average_Salary_by_Department.png')


