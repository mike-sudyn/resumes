## Practical task for learning LlamaIndex

1. Install dependancies from `requirements.txt`

2. To ingest documents run `python ingest.py`

3. To run application:
    - This project now uses OpenAI's `gpt-4o-mini`.
    Set your OpenAI API key in the environment in .env file:
    
        OPENAI_API_KEY=sk-...

    - Install dependencies and run:

        pip install -r requirements.txt
        streamlit run app.py