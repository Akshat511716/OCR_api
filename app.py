from collections import Counter
from difflib import SequenceMatcher
import sys
import easyocr
import keras_ocr
import language_tool_python
import numpy as np
import pandas as pd
from PIL import Image

file = sys.argv[1]
def get_image(file):
    # OCR using Image Processing
    # Image Preprocessing

    # 1. Perspective Correction
    # 2. Image Upscale
    # 3. Deskew
    # 4. Invert
    # 5. Noise Removal
    # 6. Binarization

    # 1. Perspective Correction
    import cv2
    from imgscan.cvtools import resize
    from imgscan.cvtools import perspective_transform
    from imgscan.cvtools import getoutlines
    from imgscan.cvtools import simple_erode
    from imgscan.cvtools import simple_dilate
    from imgscan.cvtools import brightness_contrast
    from imgscan.cvtools import blank

    # READ INPUT IMAGE
    # img = cv2.imread("C:/Users/aksha/IdeaProjects/try/OCR_api/temp/image.jpg")
    # img = file

    pil_image = Image.open(file)

    # Convert the PIL image to a NumPy array
    np_array = np.array(pil_image)

    # Convert the NumPy array to a cv2 image
    img = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

    try:
        # if input image is empty, notify the user and quit program
        if img is None:
            print()
            print("The file does not exist or is empty!")
            print("Please select a valid image file!")
            print()
            exit(0)  # exit code zero means a clean exit with no output/errors etc.

        """
        Primary Functions
        """

        def preprocess(img):
            """
            BASIC PRE-PROCESSING TO OBTAIN A CANNY EDGE IMAGE
            """

            # increase contrast between paper and background
            img_adj = brightness_contrast(img, 1.56, -60)

            # calculate the ratio of the image to the new height (500px) so we
            # can scale the manipulated image back to the original size later
            scale = img_adj.shape[0] / 500.0

            # scale the image down to 500px in height;
            img_scaled = resize(img_adj, height=500)

            # convert image to grayscale
            img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

            # apply gaussian blur with a 11x11 kernel
            img_gray = cv2.GaussianBlur(img_gray, (11, 11), 0)

            # apply canny edge detection
            img_edge = cv2.Canny(img_gray, 60, 245)

            # dilate the edge image to connect any small gaps
            img_edge = simple_dilate(img_edge)

            return img_adj, scale, img_scaled, img_edge

        def gethull(img_edge):
            """
            1st ROUND OF OUTLINE FINDING, + CONVEX HULL
            """

            # make a copy of the edge image because the following function manipulates
            # the input
            img_prehull = img_edge.copy()

            # find outlines in the (newly copied) edge image
            outlines = getoutlines(img_prehull)

            # create a blank image for convex hull operation
            img_hull = blank(img_prehull.shape, img_prehull.dtype, "0")

            # draw convex hulls (fit polygon) for all outlines detected to 'img_contour'
            for outline in range(len(outlines)):
                hull = cv2.convexHull(outlines[outline])

                # parameters: source image, outlines (contours),
                #             contour index (-1 for all), color, thickness
                cv2.drawContours(img_hull, [hull], 0, 255, 3)

            # erode the hull image to make the outline closer to paper
            img_hull = simple_erode(img_hull)

            return img_hull

        def getcorners(img_hull):
            """
            2nd ROUND OF OUTLINE FINDING, + SORTING & APPROXIMATION
            """

            # make a copy of the edge image because the following function manipulates
            # the input
            img_outlines = img_hull.copy()

            # find outlines in the convex hull image
            outlines = getoutlines(img_outlines)

            # sort the outlines by area from large to small, and only take the largest 4
            # outlines in order to speed up the process and not waste time
            outlines = sorted(outlines, key=cv2.contourArea, reverse=True)[:4]

            # loop over outlines
            for outline in outlines:

                # find the perimeter of each outline for use in approximation
                perimeter = cv2.arcLength(outline, True)

                # > approximate a rough contour for each outline found, with (hopefully)
                #   4 points (rectangular sheet of paper); [Douglas-Peuker Algorithm]
                # > FIRST OPTION is the input outline;
                # > SECOND OPTION is the accuracy of approximation (epsilon), here it
                #   is set to a percentage of the perimeter of the outline
                # > THIRD OPTION is whether to assume an outline
                #   is closed, which in this case is yes (sheet of paper)
                approx = cv2.approxPolyDP(outline, 0.02 * perimeter, True)

                # if the approximation has 4 points, then assume it is correct, and
                # assign these points to the 'corners' variable
                if len(approx) == 4:
                    corners = approx
                    break

            return corners

        """
        Main Process of the Program
        """

        # obtain the adjusted image, scaled image along with its scale factor, and the
        # Canny edge image
        img_adj, scale, img_scaled, img_edge = preprocess(img)

        # perform convex hull on edge image to prevent incomplete outline
        img_hull = gethull(img_edge)

        # obtain 4 corner points of the convex hull image
        corners = getcorners(img_hull)

        # scale the corner points back to the original size of the image using the scale
        # calculated previously
        corners = corners.reshape(4, 2) * scale

        # finally correct the perspective of the image by applying four-point
        # perspective transform
        img_corrected = perspective_transform(img_adj, corners)

        # write corrected image to file
        image = img_corrected
    except:
        image = img

    # 2. Upscale image

    try:
        width = image.shape[1]
        height = image.shape[0]

        if height and width < 700:
            super_res = cv2.dnn_superres.DnnSuperResImpl_create()

            super_res.readModel('imgscan/EDSR_x4.pb')
            super_res.setModel('edsr', 4)
            edsr_image = super_res.upsample(image)
            image = edsr_image

    except:
        pass

    # 3. Deskew

    image_file = image
    from scipy.ndimage import rotate

    def correct_skew(image, delta=1, limit=5):
        def determine_score(arr, angle):
            data = rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return histogram, score

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

        return best_angle, corrected

    _, corrected = correct_skew(image_file)
    image_file = corrected
    # cv2.imwrite("temp/corrected.jpg", corrected)

    # 4. Inverted Images

    inverted_image = cv2.bitwise_not(corrected)

    # 5. Noise Removal

    def noise_removal(image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)

    no_noise = noise_removal(inverted_image)

    # cv2.imwrite("temp/no_noise.jpg", no_noise)

    # 6. Binarization
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = grayscale(img)

    _, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    # cv2.imwrite("temp/binarization.jpg", im_bw)
    # OCR on Book Covers
    tool = language_tool_python.LanguageTool('en-US')

    # Get the list of all files
    # image_file = [corrected, inverted_image, no_noise, im_bw, img]
    image_file = [corrected]
    keras_images = [keras_ocr.tools.read(i) for i in image_file]

    # Easy OCR
    texts = []
    text_reader = easyocr.Reader(['en'])  # Initializing the ocr
    for i in keras_images:
        results = text_reader.readtext(i)
        text_result = ""
        for (bbox, text, prob) in results:
            text_result += text + " "
        texts.append(tool.correct(text_result.strip()))

    # Train keras ocr on the images
    # pipline = keras_ocr.pipeline.Pipeline()  # Creating a pipline
    # kerasocr_preds = pipline.recognize(keras_images)

    # Store the results of OCR
    # texts = []
    # for pred in kerasocr_preds:
    #     text = ""
    #     for info in pred:
    #         text += info[0] + " "
    #     texts.append(tool.correct(text.strip()))

    # Remove empty elements from the results
    new_list = [x for x in texts if x != '']
    texts = new_list

    # Use Spell check on the OCR results
    from spellchecker import SpellChecker
    spell = SpellChecker()
    temp_list = []
    for i in texts:
        temp = ""
        for j in i.split(" "):
            if spell.correction(j) is None:
                temp += j + " "
            else:
                temp += str(spell.correction(j)) + " "
        temp_list.append(temp)
    texts = temp_list

    # Remove meaningless word using english words dataset
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    temp_list = []
    word_data = pd.read_csv('C:/Users/aksha/IdeaProjects/try/OCR_api/unigram_freq.csv')
    word_data = list(word_data['word'])
    for i in texts:
        temp = ""
        for j in i.split():
            for k in word_data:
                if similar(j, str(k)) > 0.6 and len(j) > 1:
                    temp += j + " "
                    break
        temp_list.append(temp.strip())
    texts = temp_list

    # Correct the grammar of the results
    temp_list = []
    from gingerit.gingerit import GingerIt
    for i in texts:
        corrected_text = GingerIt().parse(i)
        temp_list.append(corrected_text['result'])
    texts = temp_list

    # Remove empty elements
    temp_list = []
    for i in texts:
        temp = ""
        for j in i:
            if j.isalnum() or j == ' ':
                temp += j
        temp_list.append(temp.strip())
    texts = temp_list

    # Max of 6 words in OCR results
    # Number of words to consider in results of OCR
    words = 6
    ocr_result = []
    for i in texts:
        book = ""
        for i in i.split(' ')[:words]:
            book += i + " "
        book = book.strip()
        ocr_result.append(book)

    for i in ocr_result:
        if i == "" or i == " ":
            ocr_result.remove(i)

    def max_occurring_element(lst):
        counts = Counter(lst)
        max_count = max(counts.values())
        max_elements = [k for k, v in counts.items() if v == max_count]
        if max_elements:
            return max_elements[0]
        else:
            return lst[0]

    # store(username, str(most_frequent(ocr_result)))
    return f"{max_occurring_element(ocr_result)}"

print(get_image(file))