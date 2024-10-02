import cv2
import numpy as np

def detect_ats_score(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to segment out the score
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the score
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and find the one with the largest area
    max_area = 0
    score_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            score_contour = contour

    # Draw a bounding rectangle around the score
    x, y, w, h = cv2.boundingRect(score_contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Extract the score region of interest (ROI)
    score_roi = gray[y:y+h, x:x+w]

    # Apply OCR to extract the score text
    # For simplicity, we'll use a basic OCR approach using OpenCV's template matching
    # In a real-world application, you'd use a more robust OCR library like Tesseract
    score_text = ""
    for i in range(10):
        template = cv2.imread(f"template_{i}.png")
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(score_roi, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > 0.8:
            score_text += str(i)

    # Display the detected score
    cv2.putText(image, score_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Detected Score", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return score_text