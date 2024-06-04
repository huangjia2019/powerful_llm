import sqlite3

# 连接到已存在的数据库
conn = sqlite3.connect('test_db.db')
cursor = conn.cursor()

# 删除 'employees' 表
cursor.execute("DROP TABLE IF EXISTS employees")

# 删除 'departments' 表
cursor.execute("DROP TABLE IF EXISTS departments")

# 提交更改
conn.commit()

# 关闭数据库连接
conn.close()

print("已删除 'employees' 和 'departments' 表。")
