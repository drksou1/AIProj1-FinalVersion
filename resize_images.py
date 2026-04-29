import os
from PIL import Image

def resize_with_white_padding(input_folder, output_folder, size=(512, 512)):
    os.makedirs(output_folder, exist_ok=True)
    target_w, target_h = size

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if not os.path.isfile(input_path):
            continue

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                w, h = img.size

                # Scale while preserving aspect ratio
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = img.resize((new_w, new_h), Image.LANCZOS)

                # Create white padded canvas
                canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
                offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
                canvas.paste(resized, offset)

                output_path = os.path.join(output_folder, filename)
                canvas.save(output_path)
                print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Skipping {filename}: {e}")

# Example usage
resize_with_white_padding("F:\Chairs\Chairs", "F:\Chairs\Chairs_resized")