# policySynthesisToolkitApp


## Running the app locally:
Steps for running the app locally:

1) make sure you have your own OLLAMA_API_KEY added to this file
.streamlit/secret.toml

On macOS, folders beginning with `.` may be hidden in Finder.
To show hidden files: Press `Cmd + Shift + .` in Finder. 
Use can rename this example file if needed: 
.streamlit/secret_example.toml

Ollama api key can be generated from https://ollama.com/settings/keys, if you have an ollama account. 

2) on terminal go to the folder where the code is, and create a virtual enviornment
python3 -m venv venv 

3) activate the virtual enviornment: 
source venv/bin/activate 

4) install project dependencies:
pip install -r requirements.txt

5) once successful, run code:
streamlit run app.py
this will let you see the app on localhost in your browser: http://localhost:8501