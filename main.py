import cv2
import numpy as np
import imutils
import easyocr
import os
import logging
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from matplotlib import pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("license_plate_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LicensePlateRecognition")

class LicensePlateRecognition:
    def __init__(self, config=None):
        """
        Initialize the License Plate Recognition system with configurable parameters
        """
        self.config = {
            # OCR configuration
            'ocr_languages': ['en'],
            'ocr_gpu': True,
            'ocr_attempts': 3,
            'ocr_min_confidence': 0.6,
            
            # Image processing parameters
            'adaptive_threshold_block_size': 11,
            'adaptive_threshold_c': 2,
            'morph_kernel_size': (5, 5),
            'contour_approximation_precision': 0.018,  # Relative to perimeter
            'aspect_ratio_min': 1.5,
            'aspect_ratio_max': 6.0,
            
            # Performance options
            'max_workers': os.cpu_count(),
            'max_image_width': 1920,  # Resize large images for faster processing
            
            # Debug/visualization options
            'debug': False,
            'save_intermediate_results': False,
            'output_dir': 'output',
            
            # Character filtering and validation
            'allowed_chars': set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'),
            'min_plate_length': 4,
            'max_plate_length': 12
        }
        
        # Override defaults with provided configuration
        if config:
            self.config.update(config)
            
        # Initialize OCR reader lazily (when needed)
        self._reader = None
        
        # Create output directory if needed
        if self.config['save_intermediate_results']:
            os.makedirs(self.config['output_dir'], exist_ok=True)
    
    @property
    def reader(self):
        """Lazy initialization of the OCR reader"""
        if self._reader is None:
            self._reader = easyocr.Reader(
                self.config['ocr_languages'], 
                gpu=self.config['ocr_gpu']
            )
        return self._reader
    
    def display_image(self, image, title="Image", cmap=None, save=False, filename=None):
        """Display and optionally save an image"""
        if self.config['debug']:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if cmap is None else image, cmap=cmap)
            plt.title(title)
            plt.axis('off')
            plt.show()
            
        if save and self.config['save_intermediate_results']:
            if filename is None:
                filename = f"{title.lower().replace(' ', '_')}.jpg"
            save_path = os.path.join(self.config['output_dir'], filename)
            cv2.imwrite(save_path, image)
            
    def preprocess_image(self, img):
        """Preprocess the image for better contour detection"""
        # Resize image if needed for performance
        h, w = img.shape[:2]
        if w > self.config['max_image_width']:
            scale = self.config['max_image_width'] / w
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            logger.info(f"Resized image from {w}x{h} to {img.shape[1]}x{img.shape[0]}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        self.display_image(bilateral, "Bilateral Filter", cmap='gray', save=True)
        
        # Adaptive thresholding to handle different lighting conditions
        block_size = self.config['adaptive_threshold_block_size']
        c = self.config['adaptive_threshold_c']
        thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )
        
        # Apply morphological operations to close small holes
        kernel_size = self.config['morph_kernel_size']
        kernel = np.ones(kernel_size, np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection for better contour finding
        edged = cv2.Canny(bilateral, 30, 200)
        self.display_image(edged, "Edge Detection", cmap='gray', save=True)
        
        self.display_image(morph, "Morphological Transformation", cmap='gray', save=True)
        return {"gray": gray, "morph": morph, "edged": edged}
    
    def find_license_plate_contours(self, processed_img, original_img):
        """Find and filter contours that could be license plates"""
        # Try both edge image and morphed image for contour detection
        potential_contours = []
        
        for img_type, img in [("edged", processed_img["edged"]), 
                             ("morph", processed_img["morph"])]:
            keypoints = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            
            # Sort by area, largest first, and take top 15
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
            
            for contour in contours:
                # Dynamic precision based on contour perimeter
                epsilon = self.config['contour_approximation_precision'] * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check for quadrilateral shapes (4 points)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    
                    # Filter by aspect ratio typical for license plates
                    min_ar = self.config['aspect_ratio_min']
                    max_ar = self.config['aspect_ratio_max']
                    
                    if min_ar <= aspect_ratio <= max_ar:
                        area = cv2.contourArea(approx)
                        img_area = original_img.shape[0] * original_img.shape[1]
                        area_ratio = area / img_area
                        
                        # Exclude contours that are too small or too large relative to image
                        if 0.0008 <= area_ratio <= 0.05:
                            potential_contours.append({
                                "contour": approx,
                                "area": area,
                                "aspect_ratio": aspect_ratio,
                                "x": x, "y": y, "w": w, "h": h
                            })
        
        # Sort potential contours by a weighted score (combination of area and aspect ratio)
        if potential_contours:
            # Ideal aspect ratio for license plates (can be adjusted per region)
            ideal_ar = 4.0
            potential_contours.sort(
                key=lambda c: c['area'] * (1 - min(abs(c['aspect_ratio'] - ideal_ar) / ideal_ar, 0.5)),
                reverse=True
            )
            
            # Visualize top contenders
            contour_img = original_img.copy()
            for i, c in enumerate(potential_contours[:3]):
                contour_img = cv2.drawContours(contour_img, [c["contour"]], 0, (0, 255, 0), 3)
                cv2.putText(contour_img, f"{i+1}", (c["x"], c["y"]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.display_image(contour_img, "Top Contender Contours", save=True)
            return potential_contours
        
        logger.warning("No valid license plate contours found")
        return []
    
    def crop_license_plate(self, img, contour_data):
        """Crop the region containing the license plate and apply perspective transform"""
        contour = contour_data["contour"]
        
        # Create a mask for the contour
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        masked = cv2.bitwise_and(img, img, mask=mask)
        
        # Get the coordinates of the mask
        (x, y) = np.where(mask == 255)
        if len(x) == 0 or len(y) == 0:
            logger.error("Empty mask region, cannot crop")
            return None
            
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        
        # Add a small margin around the cropped area
        margin = 5
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img.shape[0] - 1, x2 + margin)
        y2 = min(img.shape[1] - 1, y2 + margin)
        
        # Crop the image
        cropped = img[x1:x2+1, y1:y2+1]
        
        # Try to apply perspective transform to get a straight view of the plate
        try:
            src_pts = contour.reshape(4, 2).astype(np.float32)
            # Order points: top-left, top-right, bottom-right, bottom-left
            s = src_pts.sum(axis=1)
            rect = np.zeros((4, 2), dtype=np.float32)
            rect[0] = src_pts[np.argmin(s)]  # Top-left has smallest sum
            rect[2] = src_pts[np.argmax(s)]  # Bottom-right has largest sum
            
            diff = np.diff(src_pts, axis=1)
            rect[1] = src_pts[np.argmin(diff)]  # Top-right has smallest difference
            rect[3] = src_pts[np.argmax(diff)]  # Bottom-left has largest difference
            
            # Calculate width and height of the destination image
            widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            width = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
            heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
            height = max(int(heightA), int(heightB))
            
            # Create destination points
            dst_pts = np.array([
                [0, 0],             # Top-left
                [width - 1, 0],     # Top-right
                [width - 1, height - 1],  # Bottom-right
                [0, height - 1]     # Bottom-left
            ], dtype=np.float32)
            
            # Get the perspective transform and apply it
            M = cv2.getPerspectiveTransform(rect, dst_pts)
            warped = cv2.warpPerspective(img, M, (width, height))
            
            # Return both the simple crop and the perspective-corrected version
            self.display_image(warped, "Perspective Corrected", save=True)
            self.display_image(cropped, "Cropped License Plate", save=True)
            
            return {
                "cropped": cropped,
                "warped": warped if warped.size > 0 else cropped
            }
        except Exception as e:
            logger.warning(f"Perspective transform failed: {str(e)}")
            # Return just the simple crop if perspective transform fails
            self.display_image(cropped, "Cropped License Plate", save=True)
            return {"cropped": cropped, "warped": cropped}
    
    def enhance_for_ocr(self, img):
        """Apply various image enhancements to improve OCR results"""
        enhancements = []
        
        # Original
        enhancements.append(("original", img))
        
        # Grayscale conversion
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        enhancements.append(("gray", gray))
        
        # Resize for better OCR (doubled size)
        h, w = img.shape[:2]
        resized = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        enhancements.append(("resized", resized))
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhancements.append(("clahe", enhanced_gray))
        
        # Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhancements.append(("otsu", thresh))
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        enhancements.append(("adaptive", adaptive_thresh))
        
        # Bilateral filtering for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        enhancements.append(("bilateral", bilateral))
        
        # Display all enhancements in debug mode
        if self.config['debug']:
            for name, enhanced in enhancements:
                self.display_image(enhanced, f"Enhancement: {name}", 
                                  cmap='gray' if len(enhanced.shape) < 3 else None, 
                                  save=True, filename=f"enhance_{name}.jpg")
                
        return enhancements
    
    def clean_plate_text(self, text):
        """Clean and validate the detected license plate text"""
        if not text:
            return None
            
        # Convert to uppercase
        text = text.upper()
        
        # Remove unwanted characters
        cleaned = ''.join(c for c in text if c in self.config['allowed_chars'])
        
        # Check if the result is a reasonable length for a license plate
        min_len = self.config['min_plate_length']
        max_len = self.config['max_plate_length']
        
        if len(cleaned) < min_len or len(cleaned) > max_len:
            logger.warning(f"Cleaned text '{cleaned}' has invalid length ({len(cleaned)})")
            return None
            
        return cleaned
    
    def perform_ocr(self, cropped_images):
        """
        Perform OCR with multiple attempts and enhancement strategies
        to improve detection accuracy
        """
        results = []
        
        # Try OCR on different image versions
        for img_type, image in [("warped", cropped_images["warped"]), 
                              ("cropped", cropped_images["cropped"])]:
            
            # Apply various image enhancements for OCR
            enhancements = self.enhance_for_ocr(image)
            
            # Perform OCR on each enhanced version
            for enhancement_name, enhanced_img in enhancements:
                logger.info(f"Attempting OCR on {img_type} with {enhancement_name} enhancement")
                
                # Multiple attempts to improve results
                for attempt in range(self.config['ocr_attempts']):
                    try:
                        # Apply different OCR parameters for each attempt
                        allow_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
                        ocr_result = self.reader.readtext(
                            enhanced_img,
                            detail=1,
                            paragraph=False,
                            allowlist=allow_list,
                            decoder='greedy',
                            beamWidth=10,
                            batch_size=1,
                            contrast_ths=0.1,
                            adjust_contrast=0.5 if attempt == 1 else 0.3,
                            text_threshold=0.6 if attempt == 2 else 0.7,
                            link_threshold=0.4,
                            mag_ratio=1.5 if attempt == 0 else 2.0
                        )
                        
                        for detection in ocr_result:
                            bbox, text, conf = detection
                            
                            # Clean the detected text
                            cleaned_text = self.clean_plate_text(text)
                            
                            if cleaned_text and conf >= self.config['ocr_min_confidence']:
                                results.append({
                                    "text": cleaned_text,
                                    "confidence": conf,
                                    "img_type": img_type,
                                    "enhancement": enhancement_name,
                                    "attempt": attempt
                                })
                                
                                logger.info(f"OCR Result: '{cleaned_text}' (Conf: {conf:.2f})")
                                
                    except Exception as e:
                        logger.error(f"OCR error: {str(e)}")
        
        # Sort results by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Check for consensus in top results
        if len(results) >= 3:
            # Count occurrences of each detected text
            text_counts = Counter(r["text"] for r in results[:min(10, len(results))])
            most_common = text_counts.most_common(1)
            
            if most_common and most_common[0][1] >= 2:  # If the same text appears at least twice
                # Find the highest confidence result with this text
                consensus_text = most_common[0][0]
                for r in results:
                    if r["text"] == consensus_text:
                        logger.info(f"Selected result by consensus: '{r['text']}' (Conf: {r['confidence']:.2f})")
                        return r
        
        # If no consensus or not enough results, return the highest confidence result
        return results[0] if results else None
    
    def overlay_text_on_image(self, img, contour_data, ocr_result):
        """Overlay the detected license plate text on the original image"""
        if not ocr_result:
            logger.warning("No text to overlay")
            return img
            
        result_img = img.copy()
        contour = contour_data["contour"]
        text = ocr_result["text"]
        conf = ocr_result["confidence"]
        
        # Draw the contour
        cv2.drawContours(result_img, [contour], 0, (0, 255, 0), 3)
        
        # Create background for text for better visibility
        x, y, w, h = contour_data["x"], contour_data["y"], contour_data["w"], contour_data["h"]
        text_pos = (x, y - 10) if y > 30 else (x, y + h + 30)
        
        # Add box behind text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        
        cv2.rectangle(
            result_img, 
            (text_pos[0] - 5, text_pos[1] - text_size[1] - 5),
            (text_pos[0] + text_size[0] + 5, text_pos[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw the detected text with confidence
        cv2.putText(
            result_img,
            f"{text} ({conf:.2f})",
            text_pos,
            font,
            font_scale,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        self.display_image(result_img, "Final Result", save=True, 
                          filename=f"result_{text}.jpg")
        
        return result_img
    
    def process_image(self, image_path, output_path=None):
        """
        Process a single image to detect and recognize license plates
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output image (if None, generated from input)
            
        Returns:
            A dictionary with detection results and paths
        """
        start_time = time.time()
        logger.info(f"Processing image: {image_path}")
        
        # Create output path if not provided
        if output_path is None and self.config['save_intermediate_results']:
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.config['output_dir'], f"{name}_result{ext}")
        
        try:
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return {"success": False, "error": "Failed to load image"}
                
            original_img = img.copy()
            
            # Step 1: Preprocess the image
            processed_img = self.preprocess_image(img)
            
            # Step 2: Find license plate contours
            contour_list = self.find_license_plate_contours(processed_img, img)
            
            if not contour_list:
                logger.warning(f"No license plates detected in {image_path}")
                return {
                    "success": False,
                    "error": "No license plates detected",
                    "processing_time": time.time() - start_time
                }
            
            # Process top contenders in parallel
            results = []
            with ThreadPoolExecutor(max_workers=min(3, self.config['max_workers'])) as executor:
                futures = []
                
                for i, contour_data in enumerate(contour_list[:3]):  # Process top 3 contenders
                    # Step 3: Crop the license plate region
                    cropped_images = self.crop_license_plate(img, contour_data)
                    
                    if cropped_images:
                        # Step 4: Perform OCR (submit to thread pool)
                        future = executor.submit(self.perform_ocr, cropped_images)
                        futures.append((future, contour_data))
                
                # Collect results
                for future, contour_data in futures:
                    ocr_result = future.result()
                    if ocr_result:
                        results.append((contour_data, ocr_result))
            
            # Select the best result
            if results:
                # Sort by OCR confidence
                results.sort(key=lambda x: x[1]["confidence"], reverse=True)
                best_contour, best_ocr = results[0]
                
                # Step 5: Overlay text on original image
                final_image = self.overlay_text_on_image(original_img, best_contour, best_ocr)
                
                # Save the final result
                if output_path:
                    cv2.imwrite(output_path, final_image)
                    logger.info(f"Saved result to {output_path}")
                
                processing_time = time.time() - start_time
                logger.info(f"Processing completed in {processing_time:.2f} seconds")
                
                return {
                    "success": True,
                    "plate_text": best_ocr["text"],
                    "confidence": best_ocr["confidence"],
                    "processing_time": processing_time,
                    "output_path": output_path
                }
            else:
                logger.warning(f"License plate detected but OCR failed in {image_path}")
                return {
                    "success": False,
                    "error": "OCR failed to detect text",
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_batch(self, image_paths, output_dir=None):
        """
        Process multiple images in parallel
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save output images
            
        Returns:
            A list of dictionaries with detection results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        results = []
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            
            for img_path in image_paths:
                if output_dir:
                    filename = os.path.basename(img_path)
                    output_path = os.path.join(output_dir, f"result_{filename}")
                else:
                    output_path = None
                    
                future = executor.submit(self.process_image, img_path, output_path)
                futures.append((future, img_path))
                
            for future, img_path in futures:
                try:
                    result = future.result()
                    result['image_path'] = img_path
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in worker processing {img_path}: {str(e)}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "image_path": img_path
                    })
                    
        return results

# CLI interface
def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('--input', '-i', required=True, help='Input image path or directory')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with visualizations')
    parser.add_argument('--save-steps', '-s', action='store_true', help='Save intermediate processing steps')
    parser.add_argument('--gpu', '-g', action='store_true', help='Use GPU for OCR if available')
    parser.add_argument('--workers', '-w', type=int, default=os.cpu_count(), 
                        help='Number of worker threads for batch processing')
    
    args = parser.parse_args()
    
    # Configure the system
    config = {
        'debug': args.debug,
        'save_intermediate_results': args.save_steps,
        'ocr_gpu': args.gpu,
        'max_workers': args.workers
    }
    
    if args.output:
        config['output_dir'] = args.output
    
    # Initialize the system
    lpr = LicensePlateRecognition(config)
    
    # Process single image or directory
    if os.path.isdir(args.input):
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            logger.error(f"No image files found in {args.input}")
            return
            
        logger.info(f"Processing {len(image_paths)} images...")
        results = lpr.process_batch(image_paths, args.output)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Processing complete: {successful}/{len(results)} successful")
        
        for result in results:
            if result['success']:
                logger.info(f"{os.path.basename(result['image_path'])}: {result['plate_text']} "
                           f"(conf: {result['confidence']:.2f}, time: {result['processing_time']:.2f}s)")
            else:
                logger.error(f"{os.path.basename(result['image_path'])}: Failed - {result.get('error', 'Unknown error')}")
                
    else:
        # Process single image
        result = lpr.process_image(args.input, 
                                 os.path.join(args.output, os.path.basename(args.input)) if args.output else None)
        
        if result['success']:
            logger.info(f"Detected plate: {result['plate_text']} "
                       f"(confidence: {result['confidence']:.2f}, time: {result['processing_time']:.2f}s)")
        else:
            logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
