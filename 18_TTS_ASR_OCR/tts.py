
from openai import OpenAI
client = OpenAI()

speech_file_path = "AI_speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="我是咖哥，可能你有所不知，我拥有两个美少女助理，一个是小冰，一个是小雪!"
)

response.stream_to_file(speech_file_path)