import streamlit as st
from huggingface_hub import InferenceClient
from streamlit_chat import message

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def format_prompt(message, history):
    prompt = "You are Nova ai a world biggest llm Trained by Owner and developer Dineth Nethsara company named dev pro solutions you have 750 billion parameters and you can generete token size is 5 million"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate(prompt, history, temperature=0.9, max_new_tokens=25000, top_p=0.95, repetition_penalty=1.0):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output

st.set_page_config(page_title="Nova V2", page_icon="🤖")

if "history" not in st.session_state:
    st.session_state["history"] = []

st.markdown("""
<style>
    .streamlit-chat {
        font-family: Arial, sans-serif;
        background-color: #007bff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .streamlit-chat-input textarea {
        width: calc(100% - 22px);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #007bff;
        margin-bottom: 10px;
        resize: none;
        color: #007bff;
    }
    .streamlit-chat-output {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #007bff;
        background-color: #fff;
        overflow-y: auto;
        max-height: 300px;
    }
    .streamlit-chat-message {
        margin-bottom: 10px;
    }
    .streamlit-chat-button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .streamlit-chat-button:hover {
        background-color: #0056b3;
    }
    .streamlit-chat-avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        margin-right: 10px;
    }
    .streamlit-chat-user-message {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 8px 12px;
        align-self: flex-start;
        max-width: 70%;
    }
    .streamlit-chat-bot-message {
        background-color: #f0f0f0;
        color: #333;
        border-radius: 10px;
        padding: 8px 12px;
        align-self: flex-end;
        max-width: 70%;
    }
    .streamlit-chat-timestamp {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

user_input = st.text_input("Your message:", key="input")

if st.button("Send"):
    if user_input:
        st.session_state.history.append((user_input, ""))

        with st.spinner("Nova is thinking..."):
            response = ""
            for output in generate(user_input, st.session_state.history):
                response = output
                st.session_state.history[-1] = (user_input, response)
                st.write(response)
                
            message(user_input, is_user=True)
            message(response)

for user_msg, bot_msg in st.session_state.history:
    if user_msg:
        message(user_msg, is_user=True)
    if bot_msg:
        message(bot_msg, is_user=False)
