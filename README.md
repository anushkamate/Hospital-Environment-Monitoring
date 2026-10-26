# Hospital Environment Monitoring

A machine-learning and edge-inference demonstration for reviewing indoor environmental readings assigned to simulated hospital rooms.

## Features

- Upload an indoor air-pollution CSV dataset in the Streamlit app.
- Clean and prepare eight numeric measurements: `NH3`, `NO2`, `CO`, `PM2.5`, `Temp`, `Pressure`, `Humidity`, and `O3`.
- Generate demonstration labels (`Normal`, `Warning`, `Critical`) using explicit configurable rules.
- Train and evaluate a Random Forest classifier.
- Enter measurements for a room and run local inference in the app.
- View simulated room summaries and download CSV reports.

## Important limitations

This is an educational prototype, not a certified hospital monitoring system. The provided dataset does not supply verified hospital-room labels. The app therefore creates labels using percentile-based pollution thresholds and illustrative temperature/humidity limits. Test accuracy measures agreement with those generated rules; it does **not** establish real-world hospital safety or clinical reliability.

The app's room overview assigns fictional room names to sampled dataset rows. These are not live sensor readings. The app does not connect to physical sensors or an actual cloud database.

Before real-world use, obtain appropriate domain-approved thresholds, independently labelled measurements, and validation by qualified facility/environmental-safety professionals.

## Dataset requirements

Upload a CSV containing these columns (common spelling variants are supported):

- `NH3`
- `NO2`
- `CO`
- `PM2.5`
- `Temp`
- `Pressure`
- `Humidity`
- `O3`

Check the original dataset's units and licence. Do not commit a dataset to GitHub unless you have permission to redistribute it. If redistribution is not permitted, download it from the original source and upload it into the app when prompted.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install packages and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload the CSV using the sidebar.

## Train from the command line (optional)

The training script saves a model bundle and a labelled demonstration CSV.

```bash
pip install -r requirements.txt
python train_model.py "Indoor Air Pollution Data(1).csv"
```

The Streamlit app trains its own cached demonstration model from the uploaded CSV; it does not require the saved model bundle.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository named `hospital-environment-monitoring`.
2. Upload `app.py`, `requirements.txt`, `README.md`, `train_model.py`, and `.gitignore`.
3. Keep the dataset out of the repository unless its licence explicitly permits redistribution.
4. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
5. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
6. Deploy the app.
7. Open the deployed URL and upload your CSV through the sidebar.

If deployment fails, open the app's logs and check that the requirements install successfully and that the CSV has all required columns.

## Repository structure

```text
hospital-environment-monitoring/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## License

Choose a licence only after checking the dataset's terms and deciding how you want others to reuse your code. No licence is included by default.
