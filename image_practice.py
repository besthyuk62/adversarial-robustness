from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_score(img):
    return np.mean(img[:, :, 0]) - np.mean(img[:, :, 2])

image = Image.open("sample.png").convert("RGB")
image_array = np.array(image)

# crop image making
h, w, c = image_array.shape
center_y = h // 2
center_x = w // 2
crop=image_array[center_y-128:center_y+128, center_x-128:center_x+128, :]


crop_float = crop.astype(np.float32) / 255.0
epsilon = 8/ 255.0

#perturbation
direction = np.array([1, 0, -1])
directed_perturbation = epsilon * direction
noised_crop = crop_float + directed_perturbation
clipped_noised_crop = np.clip(noised_crop, 0, 1)

original_score = get_score(crop_float)
directed_score = get_score(clipped_noised_crop)

random_changes = []
for i in range(1000):
    random_direction = np.random.choice([-1, 1], size=crop_float.shape)
    random_perturbation = epsilon * random_direction
    random_noised_crop = crop_float + random_perturbation
    clipped_random_noised_crop = np.clip(random_noised_crop, 0, 1)
    random_score = get_score(clipped_random_noised_crop)
    change = random_score - original_score
    random_changes.append(change)

directed_change = directed_score - original_score

print("Random Average Changes:", np.mean(random_changes))
print("Random Max Changes:", np.max(random_changes))
print("Random Min Changes:", np.min(random_changes))
print("Directed Change:", directed_change)


plt.hist(random_changes, bins=20)
#plt.axvline(directed_change)
plt.xlabel("Score Change")
plt.ylabel("Count")
plt.title("Random vs Directed Perturbation")
plt.show()