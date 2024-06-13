# Legal Chatbot Application

This repository contains a Flask-based web application for a legal conversational chatbot specialized in legal terms and topics. The chatbot leverages various libraries and services to provide accurate and relevant legal information.

## Features

- **Legal-Specific Responses:** The chatbot is designed to respond strictly with court and legal terminology and concepts.
- **Greeting and Gratitude Detection:** Automatically detects greetings and expressions of gratitude, providing appropriate responses.
- **Memory Management:** Utilizes memory to maintain the context of the conversation.
- **Text Generation and Embeddings:** Uses Hugging Face models for text generation and embeddings.
- **Document Retrieval:** Configured with Pinecone for efficient document retrieval and search.


### Prerequisites

1. **Python**: Make sure Python is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
   
2. **Virtual Environment (optional but recommended)**: It's good practice to use a virtual environment to manage dependencies for your project. If you don't have `virtualenv` installed, you can install it using pip:

   ```bash
   pip install virtualenv
   ```

### Setup Instructions

1. **Clone the Repository**: First, clone the repository containing the chatbot application:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment**: Create a virtual environment for this project to isolate dependencies:

   ```bash
   # If using virtualenv
   virtualenv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**: Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**: Create a `.env` file in the root directory of your project and add the following variables (replace `<your_huggingface_api_key>` and `<your_pinecone_api_key>` with your actual API keys):

   ```plaintext
   HUGGINGFACE_API_KEY=<your_huggingface_api_key>
   PINECONE_API_KEY=<your_pinecone_api_key>
   ```


### Pinecone Index Setup

Ensure you have a Pinecone index created and set up correctly. This application assumes an index name of `legal-bot`.

## File Structure

- **main.py:** The main application file containing Flask routes and logic for handling chat interactions.
- **chat_with_doc.py:** Contains the configuration for the retrieval chain and document search.
- **prompt.py:** Defines the user prompt for the chatbot.
- **embeddings.py:** Handles reading PDF documents, generating embeddings, and storing them in Pinecone.
- **utils.py:** Contains utility functions, including memory initialization.
- **templates/index.html:** The front-end HTML for the chat interface.

## Usage

### Running the Application

To run the application, execute the following command:

```bash
python main.py
```

The application will start and be accessible at `http://127.0.0.1:8000`.

### Chat Interface

Open `templates/index.html` in your browser to interact with the chatbot. Type your questions in the input box and click "Send" to receive responses.

## Example Interaction

1. **Greeting:**
   - **User:** Hi
   - **AI:** Hello! How can I assist you with legal information today?

2. **Legal Question:**
   - **User:** What is a contract?
   - **AI:** A contract is a legally binding agreement between two or more parties that is enforceable by law. It typically involves an offer, acceptance, consideration, and mutual intent to be bound.

3. **Non-Legal Question:**
   - **User:** What's the weather like today?
   - **AI:** I don't have information on non-legal topics.

## Development

### Adding New Features

To add new features or modify existing ones, update the corresponding files and ensure you maintain the structure and logic of the application.

### Debugging

To run the application in debug mode, set the `FLASK_ENV` environment variable to `development` and use the `debug=True` option in the `app.run` method.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to contribute to this project by opening issues or submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.