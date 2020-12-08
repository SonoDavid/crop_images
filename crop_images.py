from os import listdir, mkdir
from os.path import isfile, join, splitext, isdir, split
from shutil import copyfile
import logging
import argparse
from typing import Tuple

from PIL import Image
import numpy as np
from pdf2image import convert_from_path

def check_document_white(image: np.array) -> bool:
    # Check if 75 percent of the image is white
    return np.count_nonzero(image > 245) >= int(0.75 * image.size)

def is_white_line(vertical_line: np.array) -> bool:
    return np.all(vertical_line > 200) and np.mean(vertical_line) > 245

def convert_to_bw(line: np.array) -> np.array:
    if line.ndim == 2:
        bw_line = np.mean(line, axis=1)
    elif line.ndim == 1:
        bw_line = line
    else:
        raise Exception('Unknown number of dimensions')
    return bw_line

def get_white_borders(image: np.array) -> Tuple[int, int, int, int]:
    # go down while all pixels are white
    vertic, horiz, colors = image.shape

    for i in range(horiz):
        vertical_line = image[:,i]
        bw_line = convert_to_bw(vertical_line)
        if not is_white_line(bw_line):
            break
    left_point = i

    for i in range(horiz-1, 0, -1):
        vertical_line = image[:,i]
        bw_line = convert_to_bw(vertical_line)
        if not is_white_line(bw_line):
            break
    right_point = i

    for i in range(vertic):
        horiz_line = image[i,:]
        bw_line = convert_to_bw(horiz_line)
        if not is_white_line(bw_line):
            break
    top_point = i

    for i in range(vertic-1, 0, -1):
        horiz_line = image[i,:]
        bw_line = convert_to_bw(horiz_line)
        if not is_white_line(bw_line):
            break
    bottom_point = i

    return left_point, right_point, top_point, bottom_point

def output_processed_unprocessed(output_dir: str, processed_dir: str, unprocessed_dir: str, file: str) -> None:
    
    # Make folder if they don't exist yet
    if not isdir(join(output_dir, processed_dir)):
        mkdir(join(output_dir, processed_dir))
    if not isdir(join(output_dir, unprocessed_dir)):
        mkdir(join(output_dir, unprocessed_dir))

    pages = convert_from_path(file)

    if len(pages) > 1:
        copyfile(file, join(output_dir, unprocessed_dir, split(file)[-1]))
    elif len(pages) == 1:
        page = pages[0]
        image = np.asarray(page)

        left_point, right_point, top_point, bottom_point = get_white_borders(image)

        cropped = image[top_point:bottom_point, left_point:right_point]

        if cropped.ndim == 3:
            cropped_bw = np.mean(cropped, axis=2)
        elif cropped.ndim == 2:
            cropped_bw = cropped
        else:
            raise Exception('Unexpected number of dimensions for image')

        if check_document_white(cropped_bw):
            cropped = image

        im = Image.fromarray(cropped)
        im.save(join(output_dir, processed_dir, split(splitext(file)[0])[-1] + '.jpeg'), quality=100, subsampling=0)


def main(input_folder: str, output_folder: str) -> None:
    for file in [join(input_folder, f) for f in listdir(input_folder)
                 if isfile(join(input_folder, f)) and splitext(f)[-1].lower() == '.pdf']:
        if not isdir(output_folder):
            logging.warning(f"The output folder '{output_folder}' does not exist")
            mkdir(output_folder)

        output_processed_unprocessed(output_folder, 'processed', 'unprocessed', file)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    parser = argparse.ArgumentParser(description='Crop images white borders')
    parser.add_argument('--pdf_input_folder', type=str, default='input_pdfs')
    parser.add_argument('--output_folder', type=str, default='output')
    args = parser.parse_args()

    main(args.pdf_input_folder, args.output_folder)
