# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
# from io import BytesIO
# from PIL import Image

# IMG_SIZE = (299, 299)

# # ----------------------------
# # IMAGE PREPROCESSING
# # ----------------------------

# def load_and_preprocess(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, IMG_SIZE)

#     # CLAHE
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
#     cl = clahe.apply(l)
#     lab = cv2.merge((cl,a,b))
#     img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)

#     img = xcep_preprocess(img)
#     return img


# # ----------------------------
# # DECODE MODEL PREDICTION
# # ----------------------------

# def postprocess_pred(pred):
#     p = np.array(pred).reshape(-1)

#     # EXPECTED FORMAT = 9 VALUES
#     # [class_prob, od_x, od_y, od_w, od_h, ma_x, ma_y, ma_w, ma_h]
#     class_prob = float(p[0])
#     od = [float(x) for x in p[1:5]]
#     ma = [float(x) for x in p[5:9]]

#     return class_prob, od, ma


# # ----------------------------
# # DRAW BOXES ON IMAGE
# # ----------------------------

# # def draw_boxes_on_image(img_path, od_box, ma_box, label, prob):
# #     img = cv2.imread(img_path)
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     h, w = img.shape[:2]

# #     def to_xyxy(box):
# #         cx, cy, bw, bh = box
# #         x1 = int((cx - bw/2) * w)
# #         x2 = int((cx + bw/2) * w)
# #         y1 = int((cy - bh/2) * h)
# #         y2 = int((cy + bh/2) * h)
# #         return x1, y1, x2, y2

# #     od = to_xyxy(od_box)
# #     ma = to_xyxy(ma_box)

# #     # Draw OD (red)
# #     cv2.rectangle(img, (od[0], od[1]), (od[2], od[3]), (255,0,0), 3)
# #     cv2.putText(img, "Optic Disc", (od[0], od[1]-10),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

# #     # Draw MA (green)
# #     cv2.rectangle(img, (ma[0], ma[1]), (ma[2], ma[3]), (0,255,0), 3)
# #     cv2.putText(img, "Macula", (ma[0], ma[1]-10),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# #     # Label (top left)
# #     cv2.putText(img,
# #                 f"{label} ({prob*100:.1f}%)",
# #                 (10,30),
# #                 cv2.FONT_HERSHEY_SIMPLEX,
# #                 1.0,
# #                 (255,255,255),
# #                 2)

# #     # Convert to bytes
# #     pil_img = Image.fromarray(img)
# #     buffer = BytesIO()
# #     pil_img.save(buffer, format="JPEG")
# #     return buffer.getvalue()


# # ----------------------------
# # GRAD-CAM
# # ----------------------------

# def make_gradcam_heatmap(img_array, model, last_conv_name, pred_index=None):
#     img_tensor = tf.expand_dims(img_array, axis=0)

#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [
#             model.get_layer(last_conv_name).output,
#             model.output
#         ]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_tensor)
#         class_channel = predictions[:, 0] if pred_index is None else predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled = tf.reduce_mean(grads, axis=(0,1,2))
#     conv_outputs = conv_outputs[0]

#     heatmap = conv_outputs @ pooled[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap)+1e-6)
#     heatmap = heatmap.numpy()
#     heatmap = cv2.resize(heatmap, IMG_SIZE)

#     return heatmap


# def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

#     overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

#     pil_img = Image.fromarray(overlay)
#     buffer = BytesIO()
#     pil_img.save(buffer, format="JPEG")
#     return buffer.getvalue()
# def to_xyxy(box):
#     """
#     Convert center-x,center-y,width,height → x1,y1,x2,y2
#     Safe: returns None if box is empty.
#     """
#     if box is None or len(box) != 4:
#         return None

#     cx, cy, bw, bh = box
#     x1 = int((cx - bw/2) * 299)
#     y1 = int((cy - bh/2) * 299)
#     x2 = int((cx + bw/2) * 299)
#     y2 = int((cy + bh/2) * 299)
#     return [x1, y1, x2, y2]


# def draw_boxes_on_image(image_path, od_box, ma_box, label, prob):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # convert boxes
#     od = to_xyxy(od_box)
#     ma = to_xyxy(ma_box)

#     # draw optic disc
#     if od is not None:
#         x1, y1, x2, y2 = od
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
#         cv2.putText(img, "Optic Disc", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     # draw macula
#     if ma is not None:
#         x1, y1, x2, y2 = ma
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#         cv2.putText(img, "Macula", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     # classification text
#     cv2.putText(img, f"{label} ({prob:.2f})", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#                 (255, 255, 0), 2)

#     # Convert to bytes for Flask
#     _, buffer = cv2.imencode('.jpg', img)
#     return buffer.tobytes()


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input as x_pre
from io import BytesIO
from PIL import Image

IMG_SIZE = (299, 299)

# ----------------------------
# IMAGE PREPROCESS
# ----------------------------
def load_and_preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)

    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = img.astype(np.float32)

    img = x_pre(img)
    return img

# ----------------------------
# DECODE MODEL OUTPUTS
# ----------------------------
def postprocess_pred(pred_class, pred_loc):
    class_prob = float(pred_class[0])  # scalar

    od_box = [float(x) for x in pred_loc[0:4]]
    ma_box = [float(x) for x in pred_loc[4:8]]

    return class_prob, od_box, ma_box

# ----------------------------
# DRAW BOXES
# ----------------------------
def draw_boxes_on_image(img_path, od_box, ma_box, label=None, prob=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    def to_xyxy(box):
        cx, cy, bw, bh = box
        x1 = int((cx - bw/2) * w)
        x2 = int((cx + bw/2) * w)
        y1 = int((cy - bh/2) * h)
        y2 = int((cy + bh/2) * h)
        return max(0,x1), max(0,y1), min(w,x2), min(h,y2)

    od = to_xyxy(od_box)
    ma = to_xyxy(ma_box)

    # OD - red
    cv2.rectangle(img, (od[0],od[1]), (od[2],od[3]), (255,0,0), 3)
    cv2.putText(img, "Optic Disc", (od[0], od[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # MA - green
    cv2.rectangle(img, (ma[0],ma[1]), (ma[2],ma[3]), (0,255,0), 3)
    cv2.putText(img, "Macula", (ma[0], ma[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if label:
        txt = f"{label} ({prob*100:.1f}%)"
        cv2.putText(img, txt, (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)

    im = Image.fromarray(img)
    buf = BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()

# ----------------------------
# GRAD-CAM
# ----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    img_tensor = tf.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output[0]]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_tensor)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap.numpy(), IMG_SIZE)
    return heatmap

def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

    im = Image.fromarray(overlay)
    buf = BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()
