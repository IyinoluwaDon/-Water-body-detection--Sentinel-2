# Sentinel-2 Water Mapping with a Decision Tree

A Python geospatial workflow that derives NDWI/MNDWI features from Sentinel-2 imagery and trains a spatially evaluated decision tree against historical surface-water labels for a Lagos-area study.

## What is included

- Offline synthetic demo and Streamlit viewer
- Optional Earth Engine export workflow
- Spatial-block evaluation, threshold baselines and documented map class codes

## Run locally

```sh
python -m venv .venv
# Activate .venv, then:
python -m pip install -e ".[gee,dev,app]"
python -m waterdet.demo --out-dir outputs/demo
python -m streamlit run app/streamlit_app.py
```

## Verify

python -m pytest -ra

## Repository map

`src/waterdet/` contains processing and evaluation code. `app/` contains the Streamlit app. `config.yaml` configures data/export paths; `docs/` explains methodology and use; `tests/` covers core contracts.

## Evidence and limitations

Offline tests and synthetic demo checks pass. They validate pipeline behavior on generated data, not the accuracy of a real-world water map. Earth Engine workflows require an authenticated Google project.

Independent manual reference checks on real imagery remain outstanding. Historical label agreement is not current ground truth. Cloud masking, seasonal coverage, boundary choice and source licensing affect results. Earth Engine export cannot be run without appropriate credentials/project access.

## License

See [LICENSE](LICENSE) for the repository license. Dataset and pretrained-model licenses may be separate.
