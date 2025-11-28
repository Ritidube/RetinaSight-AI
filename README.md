# DR Detection + Localization Flask App


## Setup
1. Copy this project folder to your machine.
2. Put your trained unified model at `models/unified_xception_dr_loc.h5` (or change path in `app.py`).
3. Create a virtualenv and install dependencies:


python -m venv venv
source venv/bin/activate # (Windows: venv\Scripts\activate)
pip install -r requirements.txt


4. Run the app:


python app.py


5. Open http://127.0.0.1:5000 in your browser.




## Notes
- The model must output a vector with values: `[class_prob, od_cx, od_cy, od_w, od_h, ma_cx, ma_cy, ma_w, ma_h]` or similar. Update `utils.postprocess_pred()` to match your model's output format.
- Grad-CAM: a simple implementation is included. It may need tweaks depending on your model's architecture.