import os
import logging
from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from chat_with_documents import configure_retrieval_chain
from prompt import user_prompt
from langchain_core.output_parsers import StrOutputParser
from flask_cors import CORS,cross_origin


parser = StrOutputParser()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the HuggingFaceHub LLM
llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    model_kwargs={
        "max_new_tokens": 2000,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

# Initialize the memory buffer
MEMORY = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    input_key='input',
    output_key='output'
)

# Flask app setup
app = Flask(__name__)
CORS(app)



# Function to detect greetings and expressions of gratitude
def handle_message(message):
    if "thank you" in message.lower():
        return "You're welcome! If you have any more questions, feel free to ask."
    elif is_greeting(message):
        return "Hello! How can I assist you today?"
    else:
        return None

# Function to detect greetings
def is_greeting(message):
    greetings = ["hi", "hello", "hey"]
    return any(greeting in message.lower() for greeting in greetings)

# Function to check if the response is incomplete
def is_response_incomplete(response):
    return not response.strip().endswith(('.', '!', '?')) and len(response.split()) < 100

# Function to get a complete response from the model
def get_complete_response(question, chat_history, max_iterations=2):
    response = ""
    iterations = 0

    while True:
        # Configure the retrieval chain with the prompt
        CONV_CHAIN = configure_retrieval_chain()
        
        # Generate response
        result = CONV_CHAIN.invoke({
            "question": question,
            "chat_history": chat_history
        })
        part_response = str(result['answer'])
        response += part_response

        # Increment iterations counter
        iterations += 1

        if not is_response_incomplete(part_response) or iterations >= max_iterations:
            if iterations >= max_iterations and is_response_incomplete(part_response):
                response += " This concludes the explanation based on the information provided."
            break

        # Update the question and chat_history for the next iteration
        chat_history.append({"input": question, "output": part_response})
        question = "Please continue."

    return response

@app.route("/api/chat", methods=["GET", "POST"])
@cross_origin()
def qa():
    if request.method == "POST":
        question = request.json.get("msg")
        if not question:
            return jsonify({"error": "No question provided"}), 400
        print(question)

        # Check if the message is a greeting or expression of gratitude
        response = handle_message(question)
        if response:
            return jsonify({"answer": response})
        
        # Initialize an empty chat history
        chat_history = []

        # Get a complete response
        response = get_complete_response(question, chat_history)
        return jsonify({"answer": response})
    
    # Default response for GET requests
    data = {"result": "Thank you! I'm just a machine learning model designed to respond to questions and generate text based on my training data. Is there anything specific you'd like to ask or discuss?"}
    return print(parser.invoke(data))

if __name__ == "__main__":
    # os.environ['FLASK_ENV'] = 'production'
    app.run(debug=True, host='0.0.0.0', port=8000,use_reloader=False)
