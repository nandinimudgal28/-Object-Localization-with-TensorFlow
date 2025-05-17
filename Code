!wget https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip
!mkdir emojis
!unzip -q openmoji-72x72-color.zip -d ./emojis
!pip install tensorflow==2.4

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageDraw
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

print('Check if we are using TensorFlow 2.4')
print('Using TensorFlow version', tf.__version__)

emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

plt.figure(figsize=(9, 9))

for i, (j, e) in enumerate(emojis.items()):
    plt.subplot(3, 3, i + 1)
    plt.imshow(plt.imread(os.path.join('emojis', e['file'])))
    plt.xlabel(e['name'])
    plt.xticks([])
    plt.yticks([])
plt.show()

for class_id, values in emojis.items():
    png_file = Image.open(os.path.join('emojis', values['file'])).convert('RGBA')
    png_file.load()
    new_file = Image.new("RGB", png_file.size, (255, 255, 255))
    new_file.paste(png_file, mask=png_file.split()[3])
    emojis[class_id]['image'] = new_file

emojis

def create_example():
  class_id = np.random.randint(0,9)
  image = np.ones((144,144,3), dtype=np.uint8) * 255
  row = np.random.randint(0,72)
  col = np.random.randint(0,72)
  emoji_image_array = np.asarray(emojis[class_id]['image']).astype(np.uint8)
  image[row:row+72,col:col+72,:] = emoji_image_array[:, :, :3] # Paste only RGB channels
  image_pil = Image.fromarray(image, 'RGB')
  draw = ImageDraw.Draw(image_pil)

  return image, class_id, (row + 10), (col + 10)/144

image, class_id, row, col = create_example()
plt.imshow(image)
plt.show()

def plot_bounding_box(image, gt_coords, pred_coords=None, norm=False):
  if norm:
    image*= 255.
    image = image.astype(np.uint8)
  image_pil = Image.fromarray(image, 'RGB')
  draw = ImageDraw.Draw(image_pil)

  row, col = gt_coords
  row *= 144
  col *= 144
  draw.rectangle([col, row, col + 52, row + 52], outline='red', width=3)

  if pred_coords:
    row, col = pred_coords
  row *= 144
  col *= 144
  draw.rectangle([col, row, col + 52, row + 52], outline='red', width=3)

  return Image

image_np, class_id, row, col = create_example()

def plot_bounding_box(image, gt_coords, pred_coords=[], norm=False):
  if norm:
    image*= 255.
    image = image.astype(np.uint8)
  image_pil = Image.fromarray(image, 'RGB')
  draw = ImageDraw.Draw(image_pil)

  row, col = gt_coords
  row_px = row * 144
  col_px = col * 144

  draw.rectangle([col_px, row_px, col_px + 72, row_px + 72], outline='red', width=3)

  if len(pred_coords) == 2:
    row_pred, col_pred = pred_coords
    row_pred_px = row_pred * 144
    col_pred_px = col_pred * 144
    draw.rectangle([col_pred_px, row_pred_px, col_pred_px + 72, row_pred_px + 72], outline='blue', width=3) # Changed color for predicted box

  return image_pil

image_with_box = plot_bounding_box(image_np, gt_coords=(row, col))

plt.imshow(image_with_box)
plt.title(emojis[class_id]['name'])
plt.show()

def data_generator(batch_size):
  while True:
    x_batch = np.zeros((batch_size, 144, 144, 3))
    y_batch = np.zeros((batch_size, 9))
    bbox_batch = np.zeros((batch_size, 2))

    for i in range(0, batch_size):
      image, class_id, row, col = create_example()
      x_batch[i] = image / 255.
      y_batch[i, class_id] = 1.0
      bbox_batch[i] = np.array([row, col])

    yield {'image': x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}

example, label = next(data_generator(1))
image = example['image'][0]
class_id = np.argmax(label['class_out'][0])
coords = label['box_out'][0]

image = plot_bounding_box(image, gt_coords=(coords[0], coords[1]), norm=True)

plt.imshow(image)
plt.title(emojis[class_id]['name'])
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageDraw
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

print('Check if we are using TensorFlow 2.4')
print('Using TensorFlow version', tf.__version__)

emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

# Load emoji images into the dictionary
for class_id, values in emojis.items():
    png_file = Image.open(os.path.join('emojis', values['file'])).convert('RGBA')
    png_file.load()
    new_file = Image.new("RGB", png_file.size, (255, 255, 255))
    new_file.paste(png_file, mask=png_file.split()[3])
    emojis[class_id]['image'] = new_file


def create_example():
  class_id = np.random.randint(0,9)
  image = np.ones((144,144,3), dtype=np.uint8) * 255
  row = np.random.randint(0,72)
  col = np.random.randint(0,72)
  emoji_image_array = np.asarray(emojis[class_id]['image']).astype(np.uint8)
  image[row:row+72,col:col+72,:] = emoji_image_array[:, :, :3] # Paste only RGB channels
  image_pil = Image.fromarray(image, 'RGB')
  draw = ImageDraw.Draw(image_pil)

  # Return the top-left pixel coordinates of the emoji
  return image, class_id, row, col

def plot_bounding_box(image, gt_coords, pred_coords=None, norm=False):
  if norm:
    image*= 255.
    image = image.astype(np.uint8)
  image_pil = Image.fromarray(image, 'RGB')
  draw = ImageDraw.Draw(image_pil)

  row, col = gt_coords
  # gt_coords are already pixel coordinates in create_example
  top_left_x = col
  top_left_y = row
  bottom_right_x = col + 72  # Add the width of the emoji
  bottom_right_y = row + 72  # Add the height of the emoji

  # Draw the ground truth bounding box (red)
  draw.rectangle([(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)], outline='red', width=3)

  # If predicted coordinates are provided, draw the predicted bounding box (blue)
  if pred_coords:
    row_pred, col_pred = pred_coords
    # Assuming predicted coordinates are normalized (0 to 1), convert to pixels
    top_left_x_pred = col_pred * 144
    top_left_y_pred = row_pred * 144
    bottom_right_x_pred = top_left_x_pred + 72
    bottom_right_y_pred = top_left_y_pred + 72
    draw.rectangle([(top_left_x_pred, top_left_y_pred), (bottom_right_x_pred, bottom_right_y_pred)], outline='blue', width=3)


  return image_pil


def data_generator(batch_size):
  while True:
    x_batch = np.zeros((batch_size, 144, 144, 3))
    y_batch = np.zeros((batch_size, 9))
    bbox_batch = np.zeros((batch_size, 2))

    for i in range(0, batch_size):
      image, class_id, row, col = create_example()
      x_batch[i] = image / 255.
      y_batch[i, class_id] = 1.0
      bbox_batch[i] = np.array([row, col])

    yield {'image': x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}


example, label = next(data_generator(1))
image = example['image'][0]
class_id = np.argmax(label['class_out'][0])
coords = label['box_out'][0]

image = plot_bounding_box(image, gt_coords=(coords[0], coords[1]), norm=True)

plt.imshow(image)
plt.title(emojis[class_id]['name'])
plt.show()

input_ = Input(shape=(144, 144, 3,), name='image')

x= input_

for i in range(0,5):
 n_filters=2**(4+i)
 x=Conv2D(n_filters, 3, activation='relu')(x)
 x=BatchNormalization()(x)
 x=MaxPool2D()(x)

X=Flatten()(x)
X=Dense(128, activation='relu')(X)
class_out=Dense(9, activation='softmax', name='class_out')(X)
box_out=Dense(2, activation='sigmoid', name='box_out')(X)

model = tf.keras.Model(inputs=[input_], outputs=[class_out, box_out])
model.summary()

class iou(tf.keras.metrics.Metric):
  def __init__(self, name='iou', **kwargs):
    super(iou, self).__init__(name=name, **kwargs)

    # These weights should ideally be initialized in the __init__ method
    # and not re-initialized in reset_state.
    self.iou = self.add_weight(name='iou' , initializer='zeros')
    self.total_iou = self.add_weight(name='total_iou' , initializer='zeros')
    self.num_ex = self.add_weight(name='count' , initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    def get_box(y):
      # Corrected 'v' to 'y' to match the input parameter name
      rows, cols = y[:, 0], y[:, 1]
      rows, cols = rows * 144, cols * 144
      # Corrected variable names y1 and y2
      y1, y2 = rows, rows + 72 # Assuming emoji height is 72
      x1, x2 = cols, cols + 72 # Assuming emoji width is 72
      return y1, x1, y2, x2

    def get_area(x1, y1, x2, y2):
      # Use tf.maximum to handle cases where coordinates are swapped or box has zero size
      return tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    gt_y1, gt_x1, gt_y2, gt_x2 = get_box(y_true) # Swapped x1 and y1 to match function return
    pred_y1, pred_x1, pred_y2, pred_x2 = get_box(y_pred) # Swapped x1 and y1 to match function return

    i_x1 = tf.maximum(gt_x1, pred_x1)
    i_y1 = tf.maximum(gt_y1, pred_y1)
    i_x2 = tf.minimum(gt_x2, pred_x2)
    i_y2 = tf.minimum(gt_y2, pred_y2)

    # Calculate intersection area. Ensure area is non-negative.
    intersection_area = tf.maximum(0.0, i_x2 - i_x1) * tf.maximum(0.0, i_y2 - i_y1)

    gt_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2)
    pred_area = get_area(pred_x1, pred_y1, pred_x2, pred_y2)

    # Calculate union area
    union_area = gt_area + pred_area - intersection_area

    # Avoid division by zero
    iou = tf.where(tf.equal(union_area, 0), 0.0, intersection_area / union_area)

    self.num_ex.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32)) # Increment by batch size
    self.total_iou.assign_add(tf.reduce_sum(iou)) # Accumulate sum of iou for batch


  def result(self):
    # Return the mean IOU over all examples seen so far
    return self.total_iou / self.num_ex

  def reset_state(self):
    # Reset weights to zero
    self.iou.assign(0.)
    self.total_iou.assign(0.)
    self.num_ex.assign(0.)

model.compile(
    loss={
        'class_out': 'categorical_crossentropy',
        'box_out': 'mse'
    },
    optimizer=tf.keras.optimizers.Adam(learning_rate= 1e-3),
    metrics={
        'class_out': 'accuracy',
        'box_out': iou(name='iou')
    }
)

def test_model(model, test_datagen):
  example, label = next(test_datagen)
  x = example['image']
  y = np.argmax(label['class_out'])
  box = label['box_out']

  pred_y, pred_box = model.predict(x)

  pred_coords = pred_box[0]
  gt_coords = box[0]
  pred_class = np.argmax(pred_y[0])
  image = x[0]

  gt = emojis[np.argmax(y[0])]['name']
  pred_class_name = emojis[pred_class]['name']

  image = plot_bounding_box(image, gt_coords=gt_coords, pred_coords=pred_coords, norm=True)
  color = 'green' if gt == pred_class_name else 'red'

  plt.imshow(image)
  plt.xlabel(f'Pred: {pred_class_name}', color=color)
  plt.ylabel(f'GT: {gt}', color=color)
  plt.xticks([])
  plt.yticks([])

def test(model):
  test_datagen = data_generator(1)

  plt.figure(figsize=(16, 4))

  for i in range(0, 6):
    plt.subplot(1, 6, i + 1)
    test_model(model, test_datagen)
  plt.show()

def test_model(model, test_datagen):
  example, label = next(test_datagen)
  x = example['image']
  y_class_out = label['class_out'][0]
  box = label['box_out']

  pred_y, pred_box = model.predict(x)

  pred_coords = pred_box[0]
  gt_coords = box[0]
  pred_class = np.argmax(pred_y[0])

  gt_class_id = np.argmax(y_class_out)
  image = x[0]

  # Use the ground truth class ID to get the name
  gt = emojis[gt_class_id]['name']
  pred_class_name = emojis[pred_class]['name']

  image = plot_bounding_box(image, gt_coords=gt_coords, pred_coords=pred_coords, norm=True)
  color = 'green' if gt == pred_class_name else 'red'

  plt.imshow(image)
  plt.xlabel(f'Pred: {pred_class_name}', color=color)
  plt.ylabel(f'GT: {gt}', color=color)
  plt.xticks([])
  plt.yticks([])

test(model)

class ShowTestImages(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    test(self.model)

