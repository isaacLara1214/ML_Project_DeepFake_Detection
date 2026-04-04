import cv2, os, random
import matplotlib.pyplot as plt

def get_all_images(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith((".jpg", ".png")):
                images.append(os.path.join(root, f))
    return images

def show_grid(folder, label, n=20):
    os.makedirs("eda", exist_ok=True)  # creates eda/ if it doesn't exist
    all_files = get_all_images(folder)
    files = random.sample(all_files, n)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(f"Crop check — {label}", fontsize=14)
    for ax, fpath in zip(axes.flat, files):
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"eda/script2_sample_{label}.png", dpi=150)
    plt.show()

show_grid("data/faces/real", "real")
show_grid("data/faces/fake", "fake")