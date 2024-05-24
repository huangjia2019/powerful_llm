import openai
import json
import requests

# 设置OpenAI API Key
# openai.api_key = "YOUR_API_KEY"  

# 定义获取天气信息的函数
def get_weather(city):
    api_key = "213745ddc9d6130ff1335e7b92b93294"  # 替换为你自己的OpenWeatherMap API密钥，用我的也无所谓啦，反正免费。
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return json.dumps({"city": city, "weather": weather, "temperature": temp})
    else:
        return json.dumps({"city": city, "error": "Unable to fetch weather data"})

def run_conversation():
    # 第一步:发送对话内容和可用函数给模型
    messages = [{"role": "user", "content": "Beijing的气温如何?"}]
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name",
                    }
                },
                "required": ["city"],
            }
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    
    # 第二步:检查模型是否想要调用函数
    response_message = response.choices[0].message  
    if response_message.function_call:
        function_name = response_message.function_call.name
        function_args = json.loads(response_message.function_call.arguments)
        
        # 第三步:调用函数
        if function_name == 'get_weather':
            function_response = get_weather(city=function_args["city"])
        else:
            function_response = f"Unknown function: {function_name}"
        
        # 第四步:将函数的响应添加到对话中,发送给模型    
        messages.append(response_message)
        messages.append({"role": "function", "name": function_name, "content": function_response}) 
        second_response = openai.chat.completions.create( 
            model="gpt-3.5-turbo-0613",
            messages=messages
        )
        return second_response.choices[0].message.content
    else:
        # 如果模型没有调用函数,直接返回模型的响应
        return response_message.content

# 运行对话
print(run_conversation())