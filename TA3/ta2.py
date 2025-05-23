# Work developed by: [Izalorran Oliveira Santos Bonaldi (GRR20210582) & Yuri Junqueira Tobias (GRR20211767)]
# Deadline: 2025-05-23
# Description: This script allows the user to correct the perspective of an image by clicking on four points.

# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image path from user input
imagePath = input("Enter the path of the input image (ex: /path/to/image.jpg):").strip()

# Check if the image path is valid loading the image
image = cv2.imread(imagePath)
if image is None:
    print("Error loading image.")
    exit()

points = []
# Function to set points on the image
def set_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        # Draw a circle on the clicked point
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Click on the 4 points (order: TL, TR, BR, BL)", image)

# Display the image and set mouse callback
print("Click on the 4 points of the image (order: top-left, top-right, bottom-right, bottom-left).")
cv2.namedWindow("Click on the 4 points (order: TL, TR, BR, BL)", cv2.WINDOW_NORMAL)
cv2.imshow("Click on the 4 points (order: TL, TR, BR, BL)", image)
cv2.setMouseCallback("Click on the 4 points (order: TL, TR, BR, BL)", set_point)

# Wait for the user to click on 4 points
while True:
    if len(points) == 4:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        print("Closed by user.")
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# Get the width and height of the new image
keepDimensions = input("Do you want to keep the original image dimensions? (y/n): ").strip().lower()
if keepDimensions == 'y':
    height, width = image.shape[:2]
else:
    width = int(input("Enter the width of the output image (e.g. 600): "))
    height = int(input("Enter the height of the output image (e.g. 800): "))

oldPoints = np.float32(points)
newPoints = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# Calculate the perspective transformation matrix and apply it
matrix = cv2.getPerspectiveTransform(oldPoints, newPoints)
newImage = cv2.warpPerspective(image, matrix, (width, height))

# Display the corrected image
cv2.namedWindow("Corrected Image", cv2.WINDOW_NORMAL)
cv2.imshow("Corrected Image", newImage)

print("Press any key or close the window to continue...")
while True:
    key = cv2.waitKey(100) 
    if key != -1:
        break
    if cv2.getWindowProperty("Corrected Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()

# Save the corrected image
newImagePath = input("Enter the output file path (e.g. /path/output.jpg): ").strip()
cv2.imwrite(newImagePath, newImage)
print(f"Corrected image saved in: {newImagePath}")
