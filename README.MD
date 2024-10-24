# Text Generation Application Using vLLM

## Description
This is a simple web application built using FastAPI and Hugging Face's **transformers** library, leveraging a pre-trained language model to generate text based on user input. The application offers a conversational interface with a loading indicator to enhance user experience.

## Features
- Generate text: Uses NLP models to generate text based on user prompts.
- Clean and preprocess input: Input text is preprocessed to enhance output quality.
- Interactive UI: Responsive interface with loading indicators and continuous conversation flow.
- Error handling: Logs errors and provides detailed error messages for debugging.


## Technologies Used
- **FastAPI**: A modern web framework for building APIs
- **Hugging Face Transformers**: A library for natural language processing (NLP) tasks
- **vLLM**: Efficient language model inference library.
- **HTML/CSS**: For the frontend UI.

## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/AH-bi/text-generation-vllm.git
   cd text-generation-vllm
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application: `uvicorn main:app --reload`
5. Open your browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000).


## Usage 
- Enter a prompt in the input field and click "Generate" to see the generated text.
- A loading message will appear while the text is being processed.
- The conversation will continue as responses are generated and displayed.

## Example
- **Input:** "Once upon a time"
- **Output:** "Once upon a time, in a land far, far away..."

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.