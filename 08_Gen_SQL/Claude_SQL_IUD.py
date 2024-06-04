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

# 插入新员工
question = "在销售部门增加一个新员工,姓名为张三,工资为45000"  
sql_query = ask_claude(question, schema_str)
print(sql_query)

# 更新员工信息
question = "将黄佳的工资调整为55000"
sql_query = ask_claude(question, schema_str)
print(sql_query)

# 删除员工
question = "删除市场部门的黄仁勋"
sql_query = ask_claude(question, schema_str)
print(sql_query)

# 关闭数据库连接
conn.close()