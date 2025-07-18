import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# Image size
imageSizeX, imageSizeY = 200, 200
X, Y = np.meshgrid(np.arange(imageSizeX), np.arange(imageSizeY))

# Circle parameters
centerX, centerY = 100, 100
radius = 50

# Create binary circle mask
circle_binary = ((Y - centerY)**2 + (X - centerX)**2) <= radius**2

# Create grayscale circle image
circle_gray = 100 * circle_binary + 100  # Background = 100, Circle = 200

# Display circle
plt.figure()
plt.imshow(circle_gray, cmap='gray')
plt.title('Image of Circle')
plt.axis('off')
plt.show()

# Add noise
noise = 20*np.random.randn(*circle_gray.shape)
noisy_image = circle_gray.astype(float) + noise
noisy_image = np.clip(noisy_image, 0, 255) # Clip values between 0 to 255
plt.figure()
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')
plt.show()

# Initial Segmentation
initial_T = 100
segmented_initial = (noisy_image > initial_T).astype(np.uint8) * 255

plt.figure()
plt.imshow(segmented_initial, cmap='gray')
plt.title('Initial Segmentation (T=100)')
plt.axis('off')
plt.show()

# Adaptive Segmentation
Threshold = initial_T
N = 10

for i in range(N):
    G1 = noisy_image[noisy_image > Threshold]
    G2 = noisy_image[noisy_image <= Threshold]

    m1 = np.mean(G1) if G1.size > 0 else 0
    m2 = np.mean(G2) if G2.size > 0 else 0

    Threshold = (m1 + m2)/2

print(Threshold)

# Final Segmentation After Adaptive Segmentatiıon
segmented_final = (noisy_image> Threshold).astype(np.uint8)*255
plt.figure()
plt.imshow(segmented_final, cmap='gray')
plt.title(' Adaptive Segmented Image')
plt.axis('off')
plt.show()


