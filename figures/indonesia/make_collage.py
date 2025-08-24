import zipfile
from PIL import Image
import os

def create_collage_from_zip(zip_path, output_path, grid=(1, 3), image_size=None):
    """
    Ekstrak semua PNG dari ZIP, susun dalam grid, dan simpan sebagai satu gambar kolase.
    
    Args:
        zip_path (str): Path ke file .zip yang berisi PNG.
        output_path (str): Path untuk menyimpan hasil kolase (PNG).
        grid (tuple): (rows, cols), misal (1,3) untuk satu baris tiga kolom.
        image_size (tuple | None): (width, height) untuk resize tiap gambar; None = original size.
    """
    # 1. Buka ZIP dan kumpulkan nama-nama file PNG
    with zipfile.ZipFile(zip_path, 'r') as z:
        png_files = sorted([f for f in z.namelist() if f.lower().endswith('.png')])
        images = [Image.open(z.open(f)).convert("RGBA") for f in png_files]

    # 2. Optional: resize semua gambar ke ukuran sama
    if image_size:
        images = [img.resize(image_size, Image.ANTIALIAS) for img in images]

    rows, cols = grid
    if len(images) != rows * cols:
        # jika jumlah gambar berbeda, hitung ulang rows/cols otomatis
        cols = len(images) if rows == 1 else cols
        rows = (len(images) + cols - 1) // cols

    # 3. Hitung ukuran kanvas
    widths, heights = zip(*(img.size for img in images))
    max_w, max_h = max(widths), max(heights)
    canvas_w, canvas_h = cols * max_w, rows * max_h

    # 4. Buat kanvas putih transparan dan paste masing-masing gambar
    collage = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 0))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        collage.paste(img, (c * max_w, r * max_h))

    # 5. Simpan hasil
    collage.save(output_path)
    print(f"Kolase dibuat: {output_path}")

# Contoh penggunaan untuk S0_baseline_indonesia.zip
if __name__ == "__main__":
    zip_file = "figures\indonesia\S0_baseline_indonesia.zip"
    output_file = "S0_baseline_indonesia_collage.png"
    create_collage_from_zip(zip_file, output_file, grid=(1, 3))
