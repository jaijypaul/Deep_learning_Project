# A-eye: Real vs AI-Generated Video Detector

A web interface for a frequency-domain classifier that decides whether a video is **real** or **AI-generated**.
You upload a video in the browser, and the app extracts a 3068-dimensional frequency fingerprint from it,
runs a trained MLP, and shows the verdict with its probabilities.

Built for the IE 643 Deep Learning course project.

---



## Repository contents

```
aeye_webapp.zip
└── aeye_webapp/
    ├── app.py              FastAPI server: serves the page and the /api/analyze endpoint
    ├── detector.py         Feature extraction (identical to the training notebook) + MLP + inference
    ├── static/index.html   The user interface
    ├── models/             Place mlp_detector_v4.pt (and optionally metrics_v4.json) here
    ├── run_in_colab.py     Optional launcher for Google Colab
    └── requirements.txt
```

---

## Running the interface

### 1. Install Python and unzip
Install **Python 3.10 or newer** from [python.org](https://www.python.org/downloads/).
On Windows, tick **"Add Python to PATH"** during installation.

Download `aeye_webapp.zip` from this repository and unzip it.

### 2. Add the trained model
Download the checkpoint and place it in `aeye_webapp/models/`:
📁 **Data folder:** [`Data_1` on Google Drive](https://drive.google.com/drive/folders/1ZaGTlcDO5vOlV2Bs5sPupkf4yGiWgrMd?usp=sharing)
- `mlp_detector_v4.pt` (required) from `MyDrive/Data_1/models_v4/`
- `metrics_v4.json` (optional, lets the page show the test accuracy) from `MyDrive/Data_1/results_v4/`

### 3. Open a terminal in the project folder
```
cd path/to/aeye_webapp
```
This is the folder that contains `app.py`.

### 4. Create and activate a virtual environment

Windows (PowerShell):
```
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:
```
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of the terminal line.

### 5. Install the packages
Install the CPU build of PyTorch first, then the remaining packages:
```
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```
Using `python -m pip` ensures the packages go into the virtual environment.

### 6. Start the server
```
python -m uvicorn app:app --port 8000
```
The log should show:
```
[A-eye] model loaded from ...mlp_detector_v4.pt | threshold = 0.500
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 7. Use the interface
Open **http://localhost:8000** in your browser, drop in a video (`.mp4`, `.mov`, `.avi`, `.mkv` or `.webm`)
and click **Analyse video**.

Stop the server with **Ctrl + C** in the terminal.

**Next time:** Only steps 3, 4 (just the `activate` line) and 6 are needed.

---

## Output

For each video the page shows:

- The verdict (**Real** or **AI-generated**)
- P(real) and P(AI-generated), with the decision threshold marked on a probability bar
- Video resolution, frame rate, duration, number of frames and analysis time
- A sample of the frequency data: one centre crop, its 2-D log-power spectrum and radial profile.
  This is an illustration only; the decision uses the full 3068-value feature vector from all 20 crops.

Uploaded videos are saved to a temporary file only during analysis and deleted immediately afterwards.


---

