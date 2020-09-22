import selective_search as ss
import edge_boxes as es
import utils as utils
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default=None)
    args = parser.parse_args()

    utils.parse_xmlfile(args.filename)
    # ss.run_selective_search(args.filename)
    es.run_edge_boxes(args.filename)