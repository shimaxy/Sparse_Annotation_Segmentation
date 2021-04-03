from PIL import Image
from skimage.filters import threshold_multiotsu
from skimage.color import label2rgb
from scipy.spatial import distance as dist
import numpy as np
import os
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import cv2
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
import glob
import pdb
import util
import filter
from imutils import perspective


def parseXml(Xmlpath):
    # load
    doctree = ET.ElementTree(file=Xmlpath)
    Region = []
    for elem in doctree.iter(tag='graphic'):
        points = elem.findall('./point-list/')
        ptlist = [point.text.split(',') for point in points]
        ptlist.append(ptlist[0])
        ptlist = np.array([list(map(float, ptlist[i])) for i in np.arange(0, len(ptlist))], dtype=np.int32)
        Region.append(ptlist)
    return np.array(Region)



def threshold_entropy_canny(image):
    
    gray = filter.filter_rgb_to_grayscale(image)
    canny = filter.filter_canny(gray, output_type="bool")
    entropy = filter.filter_entropy(gray, output_type="bool")
    mask = canny + entropy
    return mask


def threshold_otsu(image):
    # return labels [0, 1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholds = threshold_multiotsu(gray, classes=2)
    regions = np.digitize(gray, bins=thresholds)
    assert len(np.unique(regions)) == 2, "number of labels not right"
    #print(len(np.unique(regions)))
    return regions==0

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def area(box1, box2):
    (x1, y1, w1, h1) = box1
    (x2, y2, w2, h2) = box2
    dx = min(x1+w1, x2+w2) - max(x1, x2)
    dy = min(y1+h1, y2+h2) - max(y1, y2)
    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0

def find_if_overlap(cnt1, cnt2):
    box1 = cv2.boundingRect(cnt1)
    box2 = cv2.boundingRect(cnt2)

    box1_area = box1[2] * box1[3]
    box2_area = box1[2] * box2[3]
    # if the boxes overlapped heavily, they should be merged
    overlapped_area = area(box1, box2)

    if (overlapped_area / box1_area) >= 0.9 or (overlapped_area /
       box2_area)>=0.9:
        return True
    return False


def find_if_close(cnt1,cnt2, threshold=100):
    # M1 = cv2.moments(cnt1)
    # M2 = cv2.moments(cnt2)
    # cX1 = int(M1['m10'] /M1['m00'])
    # cY1 = int(M1['m01'] /M1['m00'])
    # cX2 = int(M2['m10'] /M2['m00'])
    # cY2 = int(M2['m01'] /M2['m00'])
    # dist = np.linalg.norm((cX2-cX1, cY2-cY1))
    # if abs(dist) < threshold:
    #     return True
    # return False

    # compute the rotated bounding box of the contour
    box1 = cv2.minAreaRect(cnt1)
    box2 = cv2.minAreaRect(cnt2)
    box1 = cv2.boxPoints(box1)
    box2 = cv2.boxPoints(box2)
    box1 = np.array(box1, dtype="int")
    box2 = np.array(box2, dtype="int")
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box1 = perspective.order_points(box1)
    box2 = perspective.order_points(box2)
    # compute the center of the bounding box
    cX1 = np.average(box1[:, 0])
    cY1 = np.average(box1[:, 1])
    cX2 = np.average(box2[:, 0])
    cY2 = np.average(box2[:, 1])

    (tl, tr, br, bl) = box1
    (tlblX, tlblY) = midpoint(tl, bl)
    (tltrX, tltrY) = midpoint(tl, tr)
    (trbrX, trbrY) = midpoint(tr, br)
    (blbrX, blbrY) = midpoint(bl, br)

    D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    refObj = (box1, (cX1, cY1), D)

    # stack the reference coordinates and the object coordinates
    # to include the object center
    refCoords = np.vstack([refObj[0], refObj[1], (tlblX, tlblY),
                           (trbrX, trbrY), (tltrX, tltrY),
                           (blbrX, blbrY)])
    (tl, tr, br, bl) = box2
    (tlblX, tlblY) = midpoint(tl, bl)
    (tltrX, tltrY) = midpoint(tl, tr)
    (trbrX, trbrY) = midpoint(tr, br)
    (blbrX, blbrY) = midpoint(bl, br)
    objCoords = np.vstack([box2, (cX2, cY2), (tlblX, tlblY),
                           (trbrX, trbrY), (tltrX, tltrY),
                           (blbrX, blbrY)])

    min_D = float('Inf')
    for (xA, yA) in refCoords:
        for (xB, yB) in objCoords:
    # for ((xA, yA), (xB, yB)) in zip(refCoords, objCoords):
            D = dist.euclidean((xA, yA), (xB, yB))
            if D < min_D:
                min_D = D
    if min_D < threshold:
        return True
    return False



def merge_contours(contours, area_threshold=6000,
                   factor=1.5, distance_threshold=100,
                   max_slices=7, patience=3):
# merge small contours
# we want to recursively merge small contours until we reach the max_slices number
    p = 0
    num_slices = len(contours)
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))
    while ((num_slices > max_slices) and (p < patience)) or (p == 0):
        LENGTH = len(contours)
        status = np.zeros((LENGTH, 1))
        for i, cnt1 in enumerate(contours):
            if cv2.contourArea(cnt1) < 1000:
                status[i] = -1
                continue
            if i != LENGTH - 1:
                for x, cnt2 in enumerate(contours[i + 1:]):
                    if (area_threshold > cv2.contourArea(cnt1)):
                        dist = find_if_close(cnt1, cnt2,
                                             threshold=distance_threshold)
                        close = find_if_overlap(cnt1, cnt2)
                        if dist or close:
                            if status[i + x + 1] in [-1, 0] and status[i] in [-1, 0]:
                                val = i + 1
                            else:
                                val = max(status[i], status[i + x + 1])
                            assert val > 0
                            status[i + x + 1] = status[i] = val
                        else:
                            if status[i + x + 1] == 0:
                                status[i + x + 1] = -1
                    else:
                        if find_if_overlap(cnt1, cnt2):
                            if status[i + x + 1] in [-1, 0] and status[i] in [-1, 0]:
                                val = i + 1
                            else:
                                val = max(status[i], status[i + x + 1])
                            assert val > 0
                            status[i + x + 1] = status[i] = val
                if status[i] in [0, -1]:
                    status[i] = i + 1

        unified = []
        maximum = int(status.max())+1
        for i in range(maximum):
            pos = np.where(status==i)[0]
            if pos.size > 1:
                cont = np.vstack(contours[a] for a in pos)
            elif pos.size == 1:
                cont = contours[pos[0]]
            else:
                continue
            hull = cv2.convexHull(cont)
            if area_threshold <= cv2.contourArea(hull):
                unified.append(hull)
        contours = unified
        num_slices = len(unified)
        area_threshold *= factor
        p += 1
    return unified

def get_rotation_matrix_cnts(cnt):
    rect = cv2.minAreaRect(cnt)
    # the order of the box points: bottom left, top left,
    # top right, bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box
    # points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]],
                       dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return M

def get_bg_value_from_crop(x, y, w, h, mask, img):
    crop = img[y:y+h, x:x+w,:]
    m = mask[y:y+h, x:x+w]
    res = cv2.bitwise_and(crop,crop,mask = 1-m)
    res = np.array(res, dtype=np.float)
    res[res == 0] = np.nan
    rgb = np.nanmean(res, axis=(0,1))
    return rgb


def write_to_csv(bn, ind, rect, filename):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([bn, int(ind),
                         int(rect[0]),
                         int(rect[1]),
                         int(rect[2]),
                         int(rect[3])])
    return

def read_ind_from_csv(filename):
    indices = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row:
                if row[1].isdigit():
                    indices.append(int(row[1]))

    if len(indices) == 0:
        return 0
    return max(indices)


def slide_segment(args):
    if args.input_image is not None:
        assert os.path.exists(args.input_image), "Invalid Path for Input Image"
        im_list = [args.input_image]
    else:
        assert os.path.exists(args.input_folder), "Invalid Path for Input Folder"
        im_list = glob.glob(os.path.join(args.input_folder, '*.tif'))
    for im_p in im_list:
        args.input_image = im_p
        bn = os.path.splitext(os.path.basename(args.input_image))[0]
        if bn not in args.output_path:
            output_path = os.path.join(args.output_path, bn)
        else:
            output_path = args.output_path
        if not os.path.exists(output_path):
            Path(output_path).mkdir(parents=True, exist_ok=True)
        import pdb
        csv_filename = os.path.join(output_path, 'segments.csv')
        if args.xml is not None:
            assert os.path.exists(args.xml), "Invalid Path for Input XML file"
            final_contours = parseXml(args.xml)
            base = read_ind_from_csv(csv_filename) + 1
        else:
            base = 0
            # create csv file to write to
            with open(csv_filename, 'w') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(['Filename/CaseID', 'Slide Number', 'x', 'y', 'w', 'h'])
        image = cv2.imread(args.input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.uint8)

        pdb.set_trace()
        mask = get_tissue_mask(image)

        if args.xml is None:
            contours, hierarchy = cv2.findContours(mask,
                                                      cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
            final_contours = merge_contours(contours, args.area_threshold,
                                            factor=args.factor,
                                            distance_threshold=args.distance_threshold,
                                            max_slices=args.max_num_slices,
                                            patience=args.patience)

        #assert len(final_contours) <= args.max_num_slices, "too many slices"
          # generate cropped boxes
        for i in range(len(final_contours)):
            cnt = final_contours[i]
            #pdb.set_trace()
            (x,y,w,h) = cv2.boundingRect(cnt)
            M = get_rotation_matrix_cnts(cnt)
            # ((x, y), (w, h), angle)
            rect = cv2.minAreaRect(cnt)
            write_to_csv(bn, base+i, (x,y,w,h), csv_filename)
            width = int(rect[1][0])
            height = int(rect[1][1])
            bg_rgb = get_bg_value_from_crop(x, y, w, h, mask, image)
            warped_crop = cv2.warpPerspective(image, M, (width, height),
                                              borderValue=(int(bg_rgb[2]),
                                                           int(bg_rgb[1]),
                                                           int(bg_rgb[0])))
            warped_crop = cv2.cvtColor(warped_crop, cv2.COLOR_RGB2BGR)
            crop_outname = os.path.join(output_path,
                                        '{}_{}.jpg'.format(bn, i+base))
            cv2.imwrite(crop_outname, warped_crop)

        if args.create_overlay:
            vis_outname = os.path.join(output_path, '{}_outlined.jpg'.format(bn))
            vis = visualization(image, final_contours)
            vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_outname, vis)


def visualization(image, contours):
    cmap=plt.get_cmap('Set1')
    colors = cmap.colors
    colors_rgb = [[i* 255 for i in c] for c in colors]
    for i in range(len(contours)):
        cv2.drawContours(image,[contours[i]],-1, colors_rgb[i%(len(colors_rgb))], 2)
    return image


def get_tissue_mask(image):
    mask1 = np.array(threshold_entropy_canny(image), dtype=bool)
    mask2 = np.array(threshold_otsu(image), dtype=bool)
    mask = np.bitwise_or(mask1, mask2)
    h, w = image.shape[:2]

    # hole filling
    im_out = filter.filter_binary_fill_holes(mask)
    mask = np.array(mask, dtype=np.uint8)

    dilate = filter.filter_binary_dilation(mask, disk_size=5)
    opening = filter.filter_binary_opening(dilate, disk_size=3)
    return opening



if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Script for segmenting whole slide into individual slices and save box coordinates')
        parser.add_argument('--input_image', default=None, type=str)
        parser.add_argument('--input_folder', type=str, default=None)
        parser.add_argument('--output_path', default=None, type=str)
        parser.add_argument('--create_overlay', default=False, type=bool)
        parser.add_argument('--distance_threshold', default=100, type=int)
        parser.add_argument('--area_threshold', default=6000, type=int)
        parser.add_argument('--factor', default = 1.5, type=float)
        parser.add_argument('--max_num_slices', default=7, type=int)
        parser.add_argument('--xml', default=None, type=str)
        parser.add_argument('--patience', default=5, type=int)
        args = parser.parse_args()
        slide_segment(args)
