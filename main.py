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
        return None  # No specific action for the message

# Function to detect greetings
def is_greeting(message):
    greetings = ["hi", "hello", "hey"]
    return any(greeting in message.lower() for greeting in greetings)

# Route to render index.html
@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

# Updated qa function to handle POST requests
@app.route("/qa", methods=["POST"])
@cross_origin()
def qa():
    if request.method == "POST":
        question = request.json.get("msg")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Check if the message is a greeting or expression of gratitude
        response = handle_message(question)
        if response:
            return jsonify({"answer": response})

        # Use memory to keep track of conversation history
        MEMORY.save_context({"input": question}, {"output": ""})
        chat_history = MEMORY.load_memory_variables({})

        # Configure the retrieval chain with the prompt
        CONV_CHAIN = configure_retrieval_chain()

        # Generate response
        response = CONV_CHAIN.invoke({
            "question": question,
            "chat_history": chat_history
        })

        response_text = str(response['answer'])

        # Update memory with the response
        MEMORY.save_context({"input": question}, {"output": response_text})

        return jsonify({"answer": response_text})

    # Default response for GET requests
    return jsonify({"result": "Thank you! I'm just a machine learning model designed to respond to questions and generate text based on my training data. Is there anything specific you'd like to ask or discuss?"})

if __name__ == '__main__':
    app.run(debug=True)