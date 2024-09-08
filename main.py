import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

# Function to display images using Matplotlib
def display_image(image, title="Image", cmap=None):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if cmap is None else image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to preprocess the image (grayscale, threshold, morphology)
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding to enhance contours in different lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Morphological transformations to enhance and close small gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    display_image(morph, "Morphological Transformation", cmap='gray')
    return morph

# Function to find and filter contours based on aspect ratio
def find_license_plate_contour(morph, img):
    keypoints = cv2.findContours(morph.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        # Dynamic precision based on image size
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # Looking for quadrilateral shapes
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 <= aspect_ratio <= 5:  # Check aspect ratio range for license plate-like contours
                location = approx
                break

    if location is None:
        print("No valid license plate contour found!")
    else:
        # Draw contours for visual confirmation
        contour_image = cv2.drawContours(img.copy(), [location], 0, (0, 255, 0), 3)
        display_image(contour_image, "Detected Contour")
    
    return location

# Function to mask and crop the license plate region
def crop_license_plate(img, location):
    mask = np.zeros(img.shape[:2], np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    display_image(new_image, "Masked License Plate Region")

    # Crop the plate region
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = img[x1:x2+1, y1:y2+1]
    display_image(cropped_image, "Cropped License Plate")
    
    return cropped_image

# Function to perform OCR and get the highest confidence result
def perform_ocr(cropped_image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    if len(result) == 0:
        print("No text detected by OCR.")
        return None, None

    # Extract the text with the highest confidence
    highest_confidence_text = max(result, key=lambda x: x[-1])
    text, confidence = highest_confidence_text[-2], highest_confidence_text[-1]
    print(f"Detected Text: {text} (Confidence: {confidence})")
    return text, confidence

# Function to overlay text on the original image
def overlay_text_on_image(img, location, text):
    if text is None:
        print("No text to overlay!")
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define the bottom-left corner of the text on the image
    res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60), 
                      fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    display_image(res, "Final Image with Detected Text")
    return res

# Main function that combines the steps
def extract_license_plate(image_path):
    # Step 1: Load the image
    img = cv2.imread(image_path)

    # Step 2: Preprocess the image (thresholding and morphological transformation)
    morph = preprocess_image(img)

    # Step 3: Find and filter contours to locate license plate
    location = find_license_plate_contour(morph, img)

    if location is not None:
        # Step 4: Mask and crop the license plate region
        cropped_image = crop_license_plate(img, location)

        # Step 5: Perform OCR on the cropped license plate
        text, confidence = perform_ocr(cropped_image)

        # Step 6: Overlay the text back on the original image
        final_image = overlay_text_on_image(img, location, text)

# Call the main function with your image
extract_license_plate('image4.jpg')
