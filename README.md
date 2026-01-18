# policySynthesisToolkitApp


## Running the app locally:
Steps for running the app locally:

1. make sure you have your own OLLAMA_API_KEY added to this file
`.streamlit/secret.toml`

On macOS, folders beginning with `.` may be hidden in Finder.
To show hidden files: Press `Cmd + Shift + .` in Finder. 
Use can rename this example file if needed: 
`.streamlit/secret_example.toml`

Ollama api key can be generated from https://ollama.com/settings/keys, if you have an ollama account. 

2. on terminal go to the folder where the code is, and create a virtual enviornment
`python3 -m venv venv` 

3. activate the virtual enviornment: 
`source venv/bin/activate `

4. install project dependencies:
`pip install -r requirements.txt`

5. once successful, run code:
`streamlit run app.py`
this will let you see the app on localhost in your browser: http://localhost:8501


## Running the app on streamlit cloud:

1. Make a  copy of the repo
On GitHub, click Fork on theBRAAINproject/policySynthesisToolkitApp to create your-username/policySynthesisToolkitApp (or a renamed fork).
​
Alternatively, git clone the main repo locally, change code, then push to a new GitHub repository under your own account.
​

2. Set up secrets in your own app
In local development they can keep using .streamlit/secrets.toml or similar, but this file should not be committed. On Streamlit Cloud, they must open Settings → Secrets for your deployed app and paste TOML like:

`OLLAMA_API_KEY = "their_own_key_here"`
Then access it in code using `st.secrets["OLLAMA_API_KEY"]` (adjust to whatever key name the app expects).
​

4. Deploy from your copy on Streamlit Cloud
Go to https://streamlit.io/cloud and click New app.
​
Choose the following settings:
GitHub repo: your fork or copy (e.g. their-username/policySynthesisToolkitApp)
Branch: typically main
Main file path: app.py 
​

Click Deploy; Streamlit Cloud will install dependencies and run the app from that personal repo.
​
## View the live Streamlit app
Click below for a live version of the app:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://braain-pst.streamlit.app/)
