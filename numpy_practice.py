import numpy as np

image = np.array([
    [[250, 100,  30], [120, 200, 240]],
    [[ 20,  50, 100], [230, 250,  10]]
])

perturbation = np.array([10, -20, 30])

adversarial_image = image + perturbation
clipped_image = np.clip(adversarial_image, 0, 255)
normalized_image = clipped_image / 255.0
transposed_image = np.transpose(normalized_image, (2, 0, 1))
normalized_image_mean = np.mean(normalized_image, axis=(0, 1))
transposed_image_mean = np.mean(transposed_image, axis=(1, 2))
print("Original Shape:", normalized_image.shape)
print("CHW Shape:", transposed_image.shape)
print("Mean before Transpose:", normalized_image_mean)
print("Mean after Transpose:", transposed_image_mean)
print("Final CHW array:", transposed_image)