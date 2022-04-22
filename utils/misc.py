import gzip
import json

from PIL import Image

def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)
