# Text Generation Application Using vLLM

## Description
This is a simple web application that uses FastAPI and vLLM to generate text based on user input.

## Features
- User can input a prompt to generate text.
- Built with FastAPI and vLLM.

## Setup Instructions
1. Clone this repository.
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


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.