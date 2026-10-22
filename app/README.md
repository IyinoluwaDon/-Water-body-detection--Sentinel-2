# Streamlit app

1. Run the pipeline with `--presentation`. It writes `outputs/streamlit_data/`.
2. Copy the **contents** of that run's `streamlit_data/` into `app/data/` (so `app/data/summary.json` exists). Keep assets from one run together.
3. Run locally:

```powershell
python -m pip install -e ".[app]"
python -m streamlit run app/streamlit_app.py
```

4. Deploy free: push the repo to GitHub, open <https://share.streamlit.io>, choose the repo, set the main file to
   `app/streamlit_app.py`. Streamlit installs `app/requirements.txt` by itself.
