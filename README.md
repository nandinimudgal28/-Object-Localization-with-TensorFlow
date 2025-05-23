# Object-Localization-with-TensorFlow

This project demonstrates a simple implementation of **object localization** using **TensorFlow 2.4**. The task is to detect and classify emoji faces randomly placed within synthetic 144x144 images. The model is trained to predict both the **class of the emoji** and its **top-left coordinates** in the image.

## Key Features

* **Custom Dataset Generation**: Synthetic images generated by pasting emojis at random locations.
* **Multi-output Model**:

  * Classification: Predicts which emoji is present.
  * Localization: Predicts the (x, y) coordinates of the emoji's position.
* **Visualization**: Ground truth and predicted bounding boxes are displayed for qualitative analysis.
* **End-to-End Training Pipeline**: Includes data generation, model creation, training, and evaluation.

## Architecture

* **Convolutional Neural Network (CNN)** with shared layers for feature extraction.
* **Two output heads**:

  * One for class probabilities (softmax).
  * One for bounding box regression (linear activation).

## Dataset

Emojis are downloaded from the [OpenMoji project](https://openmoji.org/):

* Downloaded using `wget` and unzipped into the `./emojis/` folder.
* A white canvas is used, and emojis are pasted at random positions.
* Bounding box coordinates (top-left corner) are recorded and normalized.
  
## Dependencies

* `TensorFlow 2.4`
* `NumPy`
* `Pillow`
* `Matplotlib`

Install with:

```bash
pip install tensorflow==2.4 matplotlib pillow numpy
```
## How to Use

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/object-localization-tensorflow.git
   cd object-localization-tensorflow
   ```

2. **Run the notebook** or script:

   * Open the Jupyter/Colab notebook and execute all cells.
   * Or run the `.py` file if exported.

3. **Train the Model**:

   * The data generator creates batches of images with labels on-the-fly.
   * The model is trained with both classification and localization losses.

4. **Visualize Predictions**:

   * Ground truth is shown in red.
  

