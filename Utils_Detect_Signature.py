import numpy as np
import tensorflow as tf

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

def img_to_tensor(img_np):
  img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32 )
  img_tf = tf.expand_dims(img_tf, axis=0)
  return img_tf

def img_proccess(img):
  img = Image.open(img)

  detection_img = img.resize((640,640))
  detection_img = np.array(detection_img)

  img = np.array(img)
  origin_img = np.zeros(img.shape)

  np.copyto(origin_img ,img)

  detection_img = img_to_tensor(detection_img)

  return {'origin_img_np':origin_img , 'detection_img_tensor':detection_img}

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin

# from IPython.display import display as ds
def draw_boxes_s(
        image,
        class_names,
        boxes,
        scores,
        score_limit, path_saved, label_map):
    now_image_np = np.zeros((image.shape))
    np.copyto(now_image_np, image)

    for i in range(0, len(boxes)):
        if (float(scores[i]) > score_limit):
            box, score, class_name = boxes[i], scores[i], class_names[i]
            class_name = label_map[class_name.numpy()+1] #LUIS label_map[class_name]
            ymin, xmin, ymax, xmax = tuple(box)
            colors = list(ImageColor.colormap.values())

            font = ImageFont.load_default()
            display_str = (str(class_name) + ":" + str(score))
            color = colors[hash(str(class_name)) % len(colors)]

            image_pil = Image.fromarray(np.uint8(now_image_np)).convert("RGB")

            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])

            np.copyto(now_image_np, np.array(image_pil))

    image_arry = Image.fromarray(np.uint8(np.array(now_image_np))).convert("RGB")
    print("draw_boxes_s saved with detection Path: ", path_saved)
    image_arry.save(path_saved)