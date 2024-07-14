# Research Paper Summariser and ChatApp

It is designed to basically get summary of any pdf and ask questions to it in that context.
This uses Gemini as an LLM model , FAISS to help create embeddings and vector stores and in turn calculate semantic similarity for the queries.

Live link- https://research-paper-summariser-and-chatapp.streamlit.app/

To run it locally-

When into the root directory(containing app.py) right click and open VS code or Windows Powershell and execute these commands-
1. Create a virtual env using
    python -m venv .venv
2. Install required dependencies
   pip install -r requirements.txt
3. Create a .env file it the root directory and put your own GOOGLE_API_KEY
4. Run Streamlit to view the app
   streamlit run app.py\
5.You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501 
