from openai import OpenAI

def download_file(filename, file_id):
    file_content = client.files.content(file_id)
    with open(filename, 'wb') as file:
        file.write(file_content.read())

def get_assistant_messages(thread_id, assistant_id):
    response = client.beta.threads.messages.list(thread_id=thread_id)
    messages = [message for message in response.data if message.role == 'assistant' and message.assistant_id == assistant_id]
    return messages

def process_message(message):
    for content_block in message.content:
        if content_block.type == 'image_file':
            file_id = content_block.image_file.file_id
            filename = f"image_{file_id}.png"
            download_file(filename, file_id)
            print(f"Downloaded image: {filename}")
        elif content_block.type == 'text':
            text = content_block.text.value
            annotations = content_block.text.annotations
            for annotation in annotations:
                if annotation.type == 'file_path':
                    file_id = annotation.file_path.file_id
                    filename = annotation.text.split('/')[-1]
                    download_file(filename, file_id)
                    print(f"Downloaded file: {filename}")

def main():
    thread_id = 'thread_D3JUkq05bfvtmBw7QoNSuTy7'
    assistant_id = 'asst_ZzFnvavBoYZ59NDLnLinNC99'

    messages = get_assistant_messages(thread_id, assistant_id)
    for message in messages:
        process_message(message)

if __name__ == "__main__":
    client = OpenAI()
    main()