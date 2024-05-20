from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def format_prompt(message, history):
    prompt = "You are Nova ai a world biggest llm Trained by Owner and developer Dineth Nethsara company named dev pro solutions you have 750 billion parameters and you can generete token size is 5 million"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate(
    prompt, history, temperature=0.9, max_new_tokens=25000, top_p=0.95, repetition_penalty=1.0,
):
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
    
# Expanded CSS styles
custom_css = """
/* Chat UI styles */
.gradio-chat-interface {
    font-family: Arial, sans-serif;
    background-color: #007bff; /* Blue background */
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* Add shadow */
}
.gradio-chat-input textarea {
    width: calc(100% - 22px);
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #007bff; /* Blue border */
    margin-bottom: 10px;
    resize: none;
    color: #007bff; /* Blue text color */
}
.gradio-chat-output {
    margin-top: 10px;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #007bff; /* Blue border */
    background-color: #fff;
    overflow-y: auto;
    max-height: 300px;
}
.gradio-chat-message {
    margin-bottom: 10px;
}
.gradio-chat-button {
    background-color: #007bff; /* Blue button */
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
}
.gradio-chat-button:hover {
    background-color: #0056b3; /* Darker blue on hover */
}
.gradio-chat-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    margin-right: 10px;
}
.gradio-chat-user-message {
    background-color: #007bff; /* Blue user message */
    color: white;
    border-radius: 10px;
    padding: 8px 12px;
    align-self: flex-start;
    max-width: 70%;
}
.gradio-chat-bot-message {
    background-color: #f0f0f0;
    color: #333;
    border-radius: 10px;
    padding: 8px 12px;
    align-self: flex-end;
    max-width: 70%;
}
.gradio-chat-timestamp {
    font-size: 12px;
    color: #888;
    margin-top: 5px;
}
"""

mychatbot = gr.Chatbot(
    avatar_images=["./user.png", "./botm.png"], bubble_full_width=False, show_label=False, show_copy_button=True, likeable=True,)

demo = gr.ChatInterface(fn=generate, 
                        chatbot=mychatbot,
                        title="Nova V2",
                        retry_btn=None,
                        undo_btn=None,
                        css=custom_css  # Use the modified CSS styles
                       )

demo.queue().launch() 
