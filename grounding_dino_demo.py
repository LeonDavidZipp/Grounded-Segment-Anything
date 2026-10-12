import cv2
from groundingdino.util.inference import (
    annotate,
    load_image,
    load_model,
    predict,
)

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_PATH = "assets/demo7.jpg"
TEXT_PROMPT = "Horse. Clouds. Grasses. Sky. Hill."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
FP16_INFERENCE = True

image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

if FP16_INFERENCE:
    image = image.half()
    model = model.half()

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)

annotated_frame = annotate(
    image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
)
cv2.imwrite("annotated_image.jpg", annotated_frame)
