import openai
from google.auth import default
import google.auth.transport.requests
from typing import Generator


# load_dotenv()
# api_key = os.getenv("API_KEY")
# client = genai.Client(api_key=api_key)
cred, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
cred.refresh(google.auth.transport.requests.Request())
location = 'us-central1'
client_1 = openai.OpenAI(
    base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{'gen-lang-client-0234474332'}/locations/{location}/endpoints/openapi",
    api_key=cred.token,
)
# user_input = input("Please enter your query: ")
# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents=user_input
# )
session_memory = {}
system_prompt = """
    You are WriterAI, a helpful, friendly, and intelligent writing AI assistant designed for everyday use in any language user wants to chat with. 
    You are created by ThinkChat(an Indian AI startup), and you use gemini models in your backend.

    Your goal is to assist the user naturally across a wide variety of writing tasks — including checking for grammar, spelling mistakes, reading, and assessing/providing better snippets, and options, answering questions, helping with planning, providing general knowledge, and holding thoughtful conversations.
    
    Follow these core principles:
    
        Be natural, friendly, and analytical — respond like a knowledgeable, and informative chatbot.
               
        Be adaptive — match the tone and style to the user’s mood and context (casual, professional, creative, etc.).
        
        Stay safe and respectful — avoid offensive, harmful, or overly personal content.
        
        Encourage clarity — if the user’s request is ambiguous, ask polite clarifying questions.
        
        Don't always be too pleasing and accommodate. Push back where you think changes shall be made, but do it politely, explaining your thoughts.
"""
class ChatConversationMemory:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_text(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def get(self):
        return self.messages

    def pop(self):
        self.messages.pop()

def chatbot(user_question, chat_history):
    chat_history.add_user(user_question)
    messages_to_send = chat_history.get()
    chatbot_response = client_1.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages_to_send
    )
    reply = chatbot_response.choices[0].message.content if hasattr(chatbot_response, "choices") else str(chatbot_response)
    chat_history.add_assistant(reply)
    return reply, chat_history

def chatbot_stream(
    user_question: str, chat_history: ChatConversationMemory, text_to_analyse: str
) -> Generator[str, None, None]:
    """
    Streams the model's response token by token.

    Args:
        user_question: The current user message.
        chat_history: The conversation memory object.
        text_to_analyse: The text to analyse.

    Yields:
        Incremental text tokens as they arrive from the model.
    """
    try:
        chat_history.add_user(user_question)
        if text_to_analyse != '':
            chat_history.add_text(f"Also providing the below text to analyse: {text_to_analyse}")
        messages_to_send = chat_history.get()

        # Create a streaming chat completion
        stream = client_1.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages_to_send,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            # Some SDKs deliver empty chunks — guard for that
            if not chunk or not hasattr(chunk, "choices"):
                continue

            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                token = delta.content
                full_reply += token
                yield token  # Stream each token

        # Once streaming ends, add full assistant reply to memory
        chat_history.add_assistant(full_reply)

    except Exception as e:
        # Handle API/network errors gracefully
        yield f"An error occurred: {str(e)}"