# import os
# from flask import Flask, render_template, request, redirect, url_for
# import numpy as np
# import tensorflow as tf
# from utils import (
#     load_and_preprocess,
#     postprocess_pred,
#     draw_boxes_on_image,
#     make_gradcam_heatmap,
#     overlay_heatmap_on_image
# )
# import base64

# # ----------------------------
# # PATH CONFIG
# # ----------------------------

# MODEL_PATH = os.path.join("models", "unified_dr_localization_model.h5")
# ##MODEL_PATH = "models\unified_dr_localization_model.h5"
# UPLOAD_FOLDER = os.path.join("uploads")

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ----------------------------
# # LOAD MODEL
# # ----------------------------

# print("Loading model:", MODEL_PATH)
# model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# print("Model loaded successfully!")

# # Find last conv layer for Grad-CAM
# def find_last_conv_layer(m):
#     for layer in reversed(m.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name
#     return None

# last_conv_layer = find_last_conv_layer(model)
# print("Grad-CAM using layer:", last_conv_layer)

# # ----------------------------
# # FLASK APP
# # ----------------------------

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return redirect(url_for("index"))

#     file = request.files["image"]
#     if file.filename == "":
#         return redirect(url_for("index"))

#     filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(filepath)

#     # Preprocess
#     img = load_and_preprocess(filepath)

#     # Predict
#     raw_pred = model.predict(np.expand_dims(img, axis=0))[0]

#     # Decode output
#     class_prob, od_box, ma_box = postprocess_pred(raw_pred)
#     label = "DR" if class_prob >= 0.5 else "No DR"

#     # Grad-CAM
#     gradcam_image = None
#     if last_conv_layer:
#         heatmap = make_gradcam_heatmap(img, model, last_conv_layer, pred_index=0)
#         gradcam_bytes = overlay_heatmap_on_image(filepath, heatmap)
#         gradcam_image = base64.b64encode(gradcam_bytes).decode("utf-8")

#     # Bounding Box Overlay
#     box_bytes = draw_boxes_on_image(filepath, od_box, ma_box, label, class_prob)
#     boxed_image = base64.b64encode(box_bytes).decode("utf-8")

#     return render_template(
#         "result.html",
#         label=label,
#         prob=f"{class_prob:.4f}",
#         boxed_image=boxed_image,
#         gradcam_image=gradcam_image
#     )


# if __name__ == "__main__":
#     app.run(debug=True)
import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import base64

from utils import (
    load_and_preprocess,
    postprocess_pred,
    draw_boxes_on_image,
    make_gradcam_heatmap,
    overlay_heatmap_on_image
)

# ----------------------------
# PATHS
# ----------------------------

MODEL_PATH = os.path.join("models", "unified_dr_localization_model.h5")
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# Find last conv layer for Grad-CAM
def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

last_conv_layer = find_last_conv_layer(model)
print("Using Grad-CAM layer:", last_conv_layer)

# ----------------------------
# FLASK APP
# ----------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Preprocess
    img = load_and_preprocess(filepath)

    # Predict (unified model gives 2 outputs)
    pred_class, pred_loc = model.predict(np.expand_dims(img, axis=0))
    pred_class = pred_class[0]
    pred_loc = pred_loc[0]

    # Decode output into DR class + boxes
    class_prob, od_box, ma_box = postprocess_pred(pred_class, pred_loc)
    label = "DR" if class_prob >= 0.5 else "No DR"

    # Grad-CAM
    gradcam_image = None
    if last_conv_layer:
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer)
        gradcam_bytes = overlay_heatmap_on_image(filepath, heatmap)
        gradcam_image = base64.b64encode(gradcam_bytes).decode("utf-8")

    # Bounding box overlay
    boxed_bytes = draw_boxes_on_image(filepath, od_box, ma_box, label, class_prob)
    boxed_image = base64.b64encode(boxed_bytes).decode("utf-8")

    return render_template(
        "result.html",
        label=label,
        prob=f"{class_prob:.4f}",
        boxed_image=boxed_image,
        gradcam_image=gradcam_image
    )


if __name__ == "__main__":
    app.run(debug=True)
