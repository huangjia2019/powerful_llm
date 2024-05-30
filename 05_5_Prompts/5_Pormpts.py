# 导入所需的库
import re
import requests
from anthropic import Anthropic

client = Anthropic()
MODEL_NAME = "claude-3-sonnet-20240229"  # 指定要使用的Claude模型的名称

# 定义获取天气信息的函数
def get_weather(city):
   api_key = "213745ddc9d6130ff1335e7b92b93294"  # 替换为你自己的OpenWeatherMap API密钥
   url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
   
   response = requests.get(url)
   if response.status_code == 200:
       data = response.json()
       weather = data["weather"][0]["description"]
       temp = data["main"]["temp"]
       return f"The weather in {city} is {weather} with a temperature of {temp}°C."
   else:
       return f"Unable to fetch weather data for {city}."

# 构建格式化的工具描述字符串
def construct_format_tool_for_claude_prompt(name, description, parameters):
   constructed_prompt = (
       "<tool_description>\n"
       f"<tool_name>{name}</tool_name>\n"
       "<description>\n"
       f"{description}\n"
       "</description>\n"
       "<parameters>\n"
       f"{construct_format_parameters_prompt(parameters)}\n"
       "</parameters>\n"
       "</tool_description>"
   )
   return constructed_prompt

# 构建格式化的工具参数描述字符串
def construct_format_parameters_prompt(parameters):
   constructed_prompt = "\n".join(f"<parameter>\n<name>{parameter['name']}</name>\n<type>{parameter['type']}</type>\n<description>{parameter['description']}</description>\n</parameter>" for parameter in parameters)

   return constructed_prompt

# 构建系统提示,告诉Claude如何使用可用的工具
def construct_tool_use_system_prompt(tools):
   tool_use_system_prompt = (
       "In this environment you have access to a set of tools you can use to answer the user's question.\n"
       "\n"
       "You may call them like this:\n"
       "<function_calls>\n"
       "<invoke>\n"
       "<tool_name>$TOOL_NAME</tool_name>\n"
       "<parameters>\n"
       "<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
       "...\n"
       "</parameters>\n"
       "</invoke>\n"
       "</function_calls>\n"
       "\n"
       "Here are the tools available:\n"
       "<tools>\n"
       + '\n'.join([tool for tool in tools]) +
       "\n</tools>"
   )
   return tool_use_system_prompt

# 从给定的字符串中提取指定标签之间的内容
def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
   ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
   if strip:
       ext_list = [e.strip() for e in ext_list]
   return ext_list

# 构建格式化的函数调用结果字符串
def construct_successful_function_run_injection_prompt(invoke_results):
   constructed_prompt = (
       "<function_results>\n"
       + '\n'.join(
           f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>" 
           for res in invoke_results
       ) + "\n</function_results>"
   )
   
   return constructed_prompt

# 定义天气查询工具的名称、描述和参数
tool_name = "weather"
tool_description = "A tool to get the current weather for a given city."

parameters = [
   {
       "name": "city",
       "type": "str", 
       "description": "The name of the city to get the weather for."
   }
]

# 构建天气查询工具的描述字符串和系统提示
tool = construct_format_tool_for_claude_prompt(tool_name, tool_description, parameters)
system_prompt = construct_tool_use_system_prompt([tool])

# 定义用户消息,询问伦敦的天气
weather_message = {
   "role": "user", 
   "content": "What's the weather like in London?"
}

# 发送用户消息给Claude,获取Claude的部分返回,其中包含对天气查询工具的调用
function_calling_message = client.messages.create(
   model=MODEL_NAME,
   max_tokens=1024,
   messages=[weather_message],
   system=system_prompt,
   stop_sequences=["\n\nHuman:", "\n\nAssistant", "</function_calls>"]
).content[0].text

# 从Claude的部分返回中提取城市名称,并调用get_weather函数获取天气信息
city = extract_between_tags("city", function_calling_message)[0]
result = get_weather(city)

# 将get_weather函数的返回值格式化为Claude期望的格式
formatted_results = [{
   'tool_name': 'get_weather',
   'tool_result': result
}]
function_results = construct_successful_function_run_injection_prompt(formatted_results)

# 将原始消息、Claude的部分返回和格式化的函数调用结果组合成最终的提示
partial_assistant_message = function_calling_message + "</function_calls>" + function_results

# 将最终的提示发送给Claude,获取并打印出包含实际天气信息的完整回复
final_message = client.messages.create(
   model=MODEL_NAME,
   max_tokens=1024,
   messages=[
       weather_message,
       {
           "role": "assistant",
           "content": partial_assistant_message
       }
   ],
   system=system_prompt
).content[0].text

print(partial_assistant_message + final_message)