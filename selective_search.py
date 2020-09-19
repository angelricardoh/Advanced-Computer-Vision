from cv2 import cv2

def run_selective_search():
    img = cv2.imread('./JPEGImages/000009.jpg')
    # newHeight = 200
    # newWidth = int(img.shape[1]*200/img.shape[0])
    # img = cv2.resize(img, (newWidth, newHeight)) 

    # cv2.imshow('0009.img', img); 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    ss.setBaseImage(img)

    ss.addStrategy(strategy)

    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()

    bboxes = ss.process()

    imOut = img.copy()

    # number of region proposals to show
    numShowRects = 100

    # iterate over all the region proposals
    for i, rect in enumerate(bboxes):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    cv2.imshow("Output", imOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()