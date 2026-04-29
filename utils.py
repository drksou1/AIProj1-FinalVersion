import matplotlib.pyplot as plt  # New
import matplotlib.patches as patches  # New
import math


def resize_box_xyxy(box, old_w, old_h, new_w, new_h):
    x1, y1, x2, y2 = box

    scale_x = new_w / old_w
    scale_y = new_h / old_h

    x1 *= scale_x
    y1 *= scale_y
    x2 *= scale_x
    y2 *= scale_y

    return x1, y1, x2, y2


def show_batch(images, targets):
    batch_size = len(images)
    if batch_size == 0:
        return

    cols = min(4, batch_size)
    rows = math.ceil(batch_size / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for i in range(batch_size):
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        image = image.clip(0.0, 1.0)
        boxes = targets[i]["boxes"].detach().cpu().numpy()
        labels = targets[i]["labels"].detach().cpu().numpy()
        ax = axes[i]
        ax.imshow(image)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"class {label}",
                fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

        ax.set_title(f"Sample {i + 1}")
        ax.axis("off")

    for j in range(batch_size, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1.0)
    plt.close(fig)
