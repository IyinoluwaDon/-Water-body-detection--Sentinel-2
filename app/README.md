# Streamlit app

1. Run the notebook to Step 8. It writes `outputs/streamlit_data/`.
2. Copy the **contents** of `outputs/streamlit_data/` into `app/data/` (so `app/data/summary.json` exists).
3. Run locally:

```powershell
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

4. Deploy free: push the repo to GitHub, open <https://share.streamlit.io>, choose the repo, set the main file to
   `app/streamlit_app.py`. Streamlit installs `app/requirements.txt` by itself.
