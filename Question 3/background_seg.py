import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os

matplotlib.use('TkAgg')
frames = []
folder_path = r"C:/Users/ekcgi/PycharmProjects/DSP Assignments/Question 3/Frames Folder"

# part a
for i in range(40):
    image_path = os.path.join(folder_path, f"Frame{i}.jpg")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        frames.append(image)
    else:
        print(f"Warning: Unable to read image at {image_path}")

if len(frames) > 0:
    frames_stack = np.stack(frames, axis=0)  # Shape: (num_frames, height, width)
    background = np.median(frames_stack, axis=0).astype(np.uint8)
    plt.imshow(background, cmap='gray',vmin=0, vmax=255)
    plt.title("Background Image via Median")
    plt.axis('off')
    plt.show()
else:
    print("No valid images found.")

# part b
image_2_path = r"C:/Users/ekcgi/PycharmProjects/DSP Assignments/Question 3/Frames Folder/Frame5.jpg"
image_2 = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)
Absolute_diff_image = np.abs(image_2.astype(np.int16) - background.astype(np.int16)).astype(np.uint8)
plt.imshow(Absolute_diff_image, cmap='gray',vmin=0, vmax=255)
plt.title("Absolute Difference Image")
plt.axis('off')
plt.show()

# part c
image_3 = Absolute_diff_image
Threshold = 10
N=8

for i in range(N):
    G1 = image_3[image_3 > Threshold]
    G2 = image_3[image_3 <= Threshold]

    m1 = np.mean(G1) if G1.size > 0 else 0
    m2 = np.mean(G2) if G2.size > 0 else 0

    Threshold = (m1 + m2)/2

image_3_final = (image_3 > Threshold).astype(np.uint8)*255
plt.figure()
plt.imshow(image_3_final, cmap='gray',vmin=0, vmax=255)
plt.title('Adaptive segmentation (T=10 N=8)')
plt.axis('off')
plt.show()