import selective_search as ss
import edge_boxes as es
import utils as utils
import cv2

if __name__ == '__main__':
    file_number = '000009'
    utils.parse_xmlfile(file_number)
    ss.run_selective_search(file_number)
    es.run_edge_boxes(file_number)