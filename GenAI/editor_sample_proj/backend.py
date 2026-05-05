import time
import openai
from openai.types.chat.completion_create_params import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from google.auth import default
import google.auth.transport.requests
from typing import Generator
# from langchain_core.output_parsers import JsonOutputParser
from prompts import proofreading_prompt, summarizer_prompt, \
    style_detector_prompt, rewrite_prompt, reader_prompt, expert_sel_prompt, intermediate_system_prompt, \
    main_prompt, discriminator_prompt
import json
import re
import streamlit as st

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
#
# client = openai.OpenAI(
#     base_url='https://api.deepseek.com',
#     api_key='sk-2bbbdafda56b4162b8af1c4d554a1f6a'
# )
#
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )
#
# print(response.choices[0].message.content)
# user_input = input("Please enter your query: ")
# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents=user_input
# )
session_memory = {}
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

class IntermediateConversationMemory:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tool_calls = []

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_call(self, tool_name: str, args:dict, tool_call_id: str):
        self.messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call_id,  # unique id for tracking
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args
                }
            }]
        })

        self.tool_calls.append(tool_name)

    def add_tool_result(self, content: str, tool_name: str, tool_call_id: str):
        self.messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "content": content
        })

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

def discriminator(full_text, rewritten_text):
    messages = [{"role": "system", "content": discriminator_prompt},{"role":"user", "content": f"Here's the original text: {full_text}."}, {"role":"user", "content": f"Here's the rewritten text: {rewritten_text}."}]
    try:
        response = client_1.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
        ).choices[0].message.content
    except Exception as e:
        return {"error": str(e)}

    return response

def chatbot_stream(
    user_question: str, chat_history: ChatConversationMemory
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

def extract_json_from_text(raw_text: str):
    """
    Extracts a JSON object from a text that may contain extra commentary or formatting.
    Returns:
        parsed_json (dict or list): The loaded JSON if found, else None.
        json_str (str): The raw JSON substring extracted.
    """
    try:
        # Try a simple JSON parse first (if it's pure JSON)
        return json.loads(raw_text), raw_text.strip()
    except Exception:
        pass

    # Otherwise, search for JSON boundaries
    match = re.search(r'\{[\s\S]*\}', raw_text)
    if not match:
        print("⚠️ No JSON object found in text.")
        return None, None

    json_str = match.group(0)

    # Attempt to clean invalid trailing commas or escape sequences
    cleaned_json = re.sub(r',\s*}', '}', json_str)
    cleaned_json = re.sub(r',\s*\]', ']', cleaned_json)

    try:
        parsed = json.loads(cleaned_json)
        return parsed, cleaned_json
    except json.JSONDecodeError as e:
        print("⚠️ Failed to decode JSON:", e)
        return None, cleaned_json

def proofreader(full_text):
    messages = [{"role": "system", "content": proofreading_prompt}, {"role":"user", "content": f"Here's the text I want you to examine: {full_text}."}]
    # parser=JsonOutputParser()
    try:
        proofreading_response = client_1.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
        ).choices[0].message.content
    except Exception as e:
        return {"error": str(e)}
    print(proofreading_response)
    json_res, cleaned_json = extract_json_from_text(proofreading_response)
    return json_res

def summarizer(full_text):
    messages = [{"role":"system", "content": summarizer_prompt}, {"role":"user", "content": f"Here's the text I want you to examine: {full_text}."}]
    try:
        summary_response = client_1.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
        ).choices[0].message.content
    except Exception as e:
        return {"error": str(e)}
    print(summary_response)
    json_res, cleaned_json = extract_json_from_text(summary_response)
    return json_res

def reader_reactions_func(full_text):
    messages = [{"role": "system", "content": reader_prompt},
                {"role": "user", "content": f"Here's the text I want you to examine: {full_text}."}]
    try:
        read_response = client_1.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
        ).choices[0].message.content
    except Exception as e:
        return {"error": str(e)}

    print(read_response)
    json_res, cleaned_json = extract_json_from_text(read_response)
    return json_res

def expert_review(full_text):
    messages = [{"role": "system", "content": expert_sel_prompt},
                {"role": "user", "content": f"Here's the text I want you to examine: {full_text}."}]

    try:
        expert_response = client_1.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
        ).choices[0].message.content
    except Exception as e:
        print(e)
        return

    print(expert_response)
    json_res, cleaned_json = extract_json_from_text(expert_response)

    for expert in json_res.values():  # ← iterate values, not keys
        for correction in expert["corrections"]:
            original = correction["original_section"]
            corrected = correction["corrected_section"]

            # Apply discriminator
            discriminated_text = discriminator(original, corrected)

            # Update the value in place
            correction["corrected_section"] = discriminated_text

    # response={}
    # if "experts" not in response:
    #     response["experts"] = json_res

    # messages_rewrite = [{"role": "system", "content": rewrite_expert(expert_response)},
    #             {"role": "user", "content": f"Here's the text I want you to examine: {full_text}."}]
    # try:
    #     rewrite = client_1.chat.completions.create(
    #         model="google/gemini-2.5-flash",
    #         messages=messages_rewrite,
    #     ).choices[0].message.content
    # except Exception as e:
    #     print(e)
    #     return
    #
    # if "rewrite" not in response:
    #     response['rewrite'] = discriminator(full_text, rewrite)
    #
    # print(response)
    return json_res

def detect_styles(full_text):
    messages = [{"role": "system", "content": style_detector_prompt},
                {"role": "user", "content": f"Here's the text I want you to examine: {full_text}."}]
    try:
        style_response = client_1.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
        ).choices[0].message.content
    except Exception as e:
        return {"error": str(e)}
    print(style_response)
    json_res, cleaned_json = extract_json_from_text(style_response)
    return json_res

def rewrite_in_style(full_text, style_description):
    messages = [{"role": "system", "content": rewrite_prompt(style_description)},
                {"role": "user", "content": f"Here's the text I want you to rewrite: {full_text}."}]
    try:
        rewrite_response = client_1.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
        ).choices[0].message.content
    except Exception as e:
        return {"error": str(e)}

    return discriminator(full_text, rewrite_response)

def count_characters(text: str) -> int:
    return len(text.replace(" ", ""))

def on_text_change(CHAR_LIMIT):
    """Syncs text and count when user modifies the editor."""
    text = st.session_state.editor_text
    char_count = count_characters(text)
    if char_count > CHAR_LIMIT:
        text = text[:CHAR_LIMIT]
        st.session_state.editor_text = text
        char_count = CHAR_LIMIT
    st.session_state.full_text = text
    st.session_state.char_count = char_count

proofreader_tool = ChatCompletionToolParam(
    type="function",
    function=FunctionDefinition(
        name="proofreader",
        description="Proofreads and corrects text for grammar, spelling, punctuation, and readability improvements.",
        parameters={
            "type": "object",
            "properties": {
                "full_text": {
                    "type": "string",
                    "description": "The complete text to proofread."
                }
            },
            "required": ["full_text"]
        }
    )
)

summarizer_tool = ChatCompletionToolParam(
    type="function",
    function=FunctionDefinition(
        name="summarizer",
        description="Summarizes the provided text into a concise version capturing key ideas and tone.",
        parameters={
            "type": "object",
            "properties": {
                "full_text": {
                    "type": "string",
                    "description": "The text that needs to be summarized."
                }
            },
            "required": ["full_text"]
        }
    )
)

reader_reactions_tool = ChatCompletionToolParam(
    type="function",
    function=FunctionDefinition(
        name="reader_reactions_func",
        description="Analyzes how readers might emotionally respond to a given text and what questions or reactions they might have.",
        parameters={
            "type": "object",
            "properties": {
                "full_text": {
                    "type": "string",
                    "description": "The full text for which to analyze reader reactions."
                }
            },
            "required": ["full_text"]
        }
    )
)

rewrite_in_style_tool = ChatCompletionToolParam(
    type="function",
    function=FunctionDefinition(
        name="rewrite_in_style",
        description="Rewrites the given full text in any style mentioned",
        parameters={
            "type": "object",
            "properties": {
                "full_text": {
                    "type": "string",
                    "description": "The text to be rewritten."
                },
                "style_description": {
                    "type": "string",
                    "description": "The description of the style to be rewritten in."
                }
            },
            "required": ["full_text", "style_description"]
        }
    )
)
expert_review_tool = ChatCompletionToolParam(
    type="function",
    function=FunctionDefinition(
        name="expert_review",
        description="Simulates expert critique of the text, provides feedback, and suggests a rewritten version based on expert insights.",
        parameters={
            "type": "object",
            "properties": {
                "full_text": {
                    "type": "string",
                    "description": "The text to be reviewed and rewritten by experts."
                }
            },
            "required": ["full_text"]
        }
    )
)

tool_schema = [proofreader_tool, summarizer_tool, reader_reactions_tool, expert_review_tool, rewrite_in_style_tool]

TOOLS = {
    "proofreader" : proofreader,
    "summarizer": summarizer,
    "expert_review": expert_review,
    "reader_reactions_func": reader_reactions_func,
    "rewrite_in_style": rewrite_in_style,
}

def writing_agent(chat_history):
    memory = IntermediateConversationMemory(intermediate_system_prompt)
    i = 0

    while True:
        if i >= 15:
            return "The system tried multiple times but couldn’t complete the task. Please try again or rephrase your question."
        i += 1

        messages_to_send = chat_history.get() + memory.get()

        try:
            response = client_1.chat.completions.create(
                model="google/gemini-2.5-flash",  # ✅ add a model name!
                messages=messages_to_send,
                tools=tool_schema,
                tool_choice="auto",
                temperature=0
            )
        except Exception as e:
            memory.pop()
            memory.add_assistant(content=str(e))  # ✅ fixed: no e.error.message
            continue

        message = response.choices[0].message  # ✅ message is now an object, not dict

        # Log reasoning if content and tool calls both exist
        if message.content and message.tool_calls:
            print("Assistant reasoning:", message.content)
            memory.add_assistant(content=message.content)

        if message.tool_calls:
            # If the assistant calls a tool, we execute it
            for call in message.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)
                print("args generated by decider llm", args)
                tool_call_id = call.id

                if fn_name in TOOLS:
                    try:
                        result = TOOLS[fn_name](**args)
                    except Exception as e:
                        result = f"Tool Execution Failed: {fn_name} ({e})"
                    finally:
                        memory.add_assistant_tool_call(fn_name, json.dumps(args), tool_call_id)
                        memory.add_tool_result(str(result), fn_name, tool_call_id)
                        print(f"Tool called: {fn_name} with result: {result}")

                else:
                    memory.add_assistant_tool_call(fn_name, json.dumps(args), tool_call_id)
                    memory.add_tool_result(fn_name, tool_call_id, f"Tool '{fn_name}' not implemented.")

            continue  # Keep looping if tools were called

        else:
            # No tool calls → final response
            memory.add_assistant(content=message.content)
            print(f"Assistant's final response: {message.content}")
            break

    return message.content

def stream_text_streamlit(text: str, delay: float = 0.01):
    """
    Stream text in Streamlit character-by-character for a typing effect.
    """
    placeholder = st.empty()
    streamed = ""
    for char in text:
        streamed += char
        placeholder.markdown(streamed)
        time.sleep(delay)

