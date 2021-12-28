#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Cut lines from IAM dataset."""
import cv2
import argparse

import numpy as np

from tqdm import tqdm
from pathlib import Path
from xml.etree import ElementTree as et


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--img-dir', type=Path, required=True,
                        help='Path to the IAM directory with images.')
    parser.add_argument('--xml-dir', type=Path, required=True,
                        help='Path to the directory with XML markup files.')
    parser.add_argument('--save-to', type=Path, required=True,
                        help='Path to save dir.')
    parser.add_argument('--border-expand', type=int, default=35,
                        help='Border expansion.')
    
    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()
    
    xmls = list(args.xml_dir.glob('*.xml'))
    for xml_path in tqdm(xmls):
        # read image
        img_name = xml_path.stem + '.png'
        img_path = str(args.img_dir.joinpath(img_name))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # read xml
        xml_root = et.parse(xml_path).getroot()
        for xml_line in list(list(xml_root)[1]):
            # search sentence start\end
            coords = []
            for word in list(xml_line):
                for pt in list(word):
                    coords.append(int(pt.attrib['x']))
            
            left = min(coords)
            right = max(coords)
            
            # cut line
            xml_attrib = xml_line.attrib
            upper_border = int(xml_attrib['uby']) - args.border_expand
            lower_border = int(xml_attrib['lby']) + args.border_expand
            left_border = np.clip(left - args.border_expand,
                                  0, img.shape[-1])
            right_border = np.clip(right + args.border_expand,
                                   args.border_expand, img.shape[-1])
            
            line = img[upper_border:lower_border, left_border:right_border]
            
            # check aspect ratio
            diff = line.shape[0] - line.shape[1]
            if diff >= 0:
                cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                cv2.imshow('img', line)
                cv2.waitKey()
                
                choice = None
                while choice not in ['1', '2']:
                    print('1 - cut')
                    print('2 - pad reflective')
                    choice = input()
                
                print('Shape before:', line.shape)
                if choice == '1':
                    cut_size = (diff + 10)
                    line = line[cut_size:]
                else:
                    line = cv2.copyMakeBorder(line, 0, 0, 0, diff + 1,
                                              cv2.BORDER_REPLICATE)
                
                print('Shape after:', line.shape)
                cv2.imshow('img', line)
                cv2.waitKey()
                cv2.destroyWindow('img')
            
            # saving
            dir_name = xml_path.stem.split('-')[0]
            save_dir = args.save_to.joinpath(dir_name).joinpath(xml_path.stem)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(save_dir.joinpath(xml_attrib['id'] + '.png')), line)


if __name__ == '__main__':
    main()
