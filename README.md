# License Plate Detection and OCR

This project demonstrates a simple pipeline for detecting a license plate from an image and extracting text using Optical Character Recognition (OCR). The process is carried out using popular libraries like OpenCV, EasyOCR, and Matplotlib for visualization. The code is designed to preprocess the image, detect license plate-like regions, extract the text using OCR, and overlay the recognized text on the original image.

## Features
- **Preprocessing**: Adaptive thresholding and morphological transformations for robust edge detection.
- **Contour Detection**: Locates potential license plates based on shape and aspect ratio filtering.
- **OCR (Optical Character Recognition)**: Extracts text from the detected license plate region using EasyOCR.
- **Overlay**: Displays the detected text and outlines the license plate on the original image.
- **Visualization**: Step-by-step visualization of intermediate results (e.g., edges, contours, cropped license plate).

## Prerequisites

To run this code, you need to have the following libraries installed:
- `OpenCV` for image processing and contour detection
- `Matplotlib` for visualizing the images
- `imutils` for contour operations
- `numpy` for handling arrays and mathematical operations
- `EasyOCR` for performing OCR on the license plate

You can install the required dependencies using pip:

```bash
pip install opencv-python-headless matplotlib numpy imutils easyocr
```

## How It Works

1. **Image Preprocessing**:
   - Convert the image to grayscale.
   - Apply adaptive thresholding to enhance contrast.
   - Perform morphological transformations to fill in gaps and enhance contours.

2. **Contour Detection**:
   - Find contours in the preprocessed image and sort them based on area.
   - Filter contours to identify potential license plate candidates based on their rectangular shape and aspect ratio.

3. **Masking and Cropping**:
   - Once a valid contour is found, mask the image to isolate the region containing the license plate.
   - Crop the masked area for further OCR processing.

4. **OCR (Optical Character Recognition)**:
   - Use EasyOCR to read the text from the cropped license plate region.
   - Extract the text with the highest confidence.

5. **Overlay Text**:
   - Overlay the detected text onto the original image, highlighting the license plate region with a rectangle.

6. **Visualization**:
   - At each step, the code displays the current image state (e.g., edge detection, detected contours, cropped license plate, final result) for easier debugging and understanding of the pipeline.

## File Structure

- `license_plate_detection.py`: The main Python script containing the code to detect and recognize text from a license plate.

## Usage

1. Clone the repository or download the script.
2. Ensure the required dependencies are installed.
3. Place your input image (containing a license plate) in the same directory as the script or specify the correct path.
4. Call the main function in the script:

```python
extract_license_plate('image4.jpg')
```

This function takes an image file path as input and processes it to detect and extract text from the license plate.

### Example

```bash
python license_plate_detection.py
```

### Input Image:
Ensure the image contains a clear view of a license plate.

### Output:
The code will display the processed image at each stage and output the final image with the detected license plate and recognized text overlaid on it.

## Notes

- The code assumes that the license plate is rectangular and has an aspect ratio between 2 and 5.
- Depending on image quality (e.g., noise, lighting), you may need to adjust certain parameters like threshold values or morphological kernel size for optimal results.
- The OCR works best with high-quality images where the license plate text is clear and legible.

## Future Improvements

- Add support for multiple languages in OCR.
- Improve detection for skewed or tilted license plates using perspective transforms.
- Extend support for recognizing license plates with more complex backgrounds or distortions.
