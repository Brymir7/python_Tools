from wand.image import Image
import io
import pytesseract


lang = 'deu'

import os

pdf_path = '/home/brymir/Downloads/kapitel3.pdf'

png_dir = 'png_images'
txt_dir = 'txt_output'
os.makedirs(png_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)

# Convert each page of the PDF to a PNG image
with Image(filename=pdf_path, resolution=300) as pdf:
    for i, page in enumerate(pdf.sequence):
        with Image(page) as img:
            img.format = 'png'
            img.save(filename=os.path.join(png_dir, f'page_{i+1}.png'))



from PIL import Image #overwrite definition of image from wand.image

res = []
for file in os.listdir(png_dir):
    with open(f"png_images/{file}", 'rb') as png:
        image = Image.open(io.BytesIO(png.read()))
        text = pytesseract.image_to_string(image, lang=lang)
        print(f"{file} finished!")
        res.append(text)

a = "hello"
with open(f"txt_output/{a}.txt", 'w') as f:
    for string in res:
        f.write(string)