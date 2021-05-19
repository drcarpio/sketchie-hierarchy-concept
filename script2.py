import cv2
import numpy as np

# IMPORTANT! during classification, do not draw bounding boxes or write text on the image


class HtmlNode:
    def __init__(self):
        self.children = set()
        self.isDiv = False


def getHierarchy(boxes):
    # map coordinates to node
    node_handles = dict()

    # boxes need to be sorted for loop to work
    boxes.sort()
    for i in range(len(boxes)-1):
        x1, y1, w1, h1 = boxes[i]
        node_handles[boxes[i]] = HtmlNode()

        for j in range(i+1, len(boxes)):
            x2, y2, w2, h2 = boxes[j]
            node_handles[boxes[j]] = HtmlNode()

            if (y1 < y2) and (x1 + w1 > x2 + w2) and (y1 + h1 > y2 + h2):
                node_handles[boxes[i]].isDiv = True
                node_handles[boxes[i]].children.add(node_handles[boxes[j]])

    return node_handles


# number of test images
tests = 5

for i in range(1, tests + 1):
    which = i
    # Normal routines
    img = cv2.imread(f'input/test{which}.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, 1)

    # Remove some small noise if any.
    dilate = cv2.dilate(thresh, None)
    erode = cv2.erode(dilate, None)

    cv2.imwrite(f'inter/eroded{which}.jpeg', erode)

    # Find contours with cv2.RETR_CCOMP
    contours, hierarchy = cv2.findContours(
        erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    boxes_to_check = []

    for i, cnt in enumerate(contours):
        # Check if it is an external contour and its area is more than X% of img area
        if hierarchy[0, i, 3] == -1 and cv2.contourArea(cnt) > (img.shape[0] * img.shape[1] / 3000):
            x, y, w, h = cv2.boundingRect(cnt)

            if h > img.shape[0] / 10 and w > img.shape[1] / 10:
                boxes_to_check.append((x, y, w, h))

    node_handles = getHierarchy(boxes_to_check)
    need_to_classify = []

    for box in boxes_to_check:
        x, y, w, h = box

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if node_handles[box].isDiv:
            cv2.putText(img, 'div', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (36, 255, 12), 2)
        else:
            need_to_classify.append(box)

        m = cv2.moments(cnt)
        cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        cv2.circle(img, (int(cx), int(cy)), 3, 255, -1)

    for i in range(len(need_to_classify)):
        x, y, w, h = need_to_classify[i]
        path = f'slices/slices{which}/slice{i}.jpeg'

        cv2.imwrite(path, img[y:y+h, x:x+w])
        cv2.putText(img, 'CLASSIFY ME', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (36, 255, 12), 2)

    cv2.imwrite(f'output/output{which}.jpeg', img)
