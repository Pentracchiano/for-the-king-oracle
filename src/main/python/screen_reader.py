import math
import mss
import cv2
import numpy as np
import os


tokens_rect = {"top": 480, "left": 700, "width": 550, "height": 100}
damage_rect = {"top": 725, "left": 810, "width": 100, "height": 45}
accuracy_rect = {"top": 725, "left": 960, "width": 130, "height": 45}


class ScreenReader:
    def __init__(self, application_context):
        self.sct = mss.mss()
        self.application_context = application_context

        self.media_root_path = application_context.get_resource('')
        self.numbers_root_path = os.path.join(self.media_root_path, 'numbers')
        self.tokens_root_path = os.path.join(self.media_root_path, 'tokens')

    def get_accuracy(self):
        accuracy_image = np.array(self.sct.grab(accuracy_rect))
        roi_list = self.extract_digits(accuracy_image, additional_preprocessing=self.remove_percentage)
        accuracy = self.read_number_from_digit_images(roi_list)
        return accuracy / 100 if accuracy is not None else -1

    def get_damage(self):
        damage_image = np.array(self.sct.grab(damage_rect))
        damage_rois = self.extract_digits(damage_image)
        damage_num = self.read_number_from_digit_images(damage_rois)
        return damage_num if damage_num is not None else -1

    def get_tokens(self):
        tokens_image = np.array(self.sct.grab(tokens_rect))
        non_focused_tokens = self.count_tokens(tokens_image)
        return non_focused_tokens

    # UTILITY METHODS

    def check_similarity(self, image1, image2):
        return cv2.matchTemplate(image1, image2, cv2.TM_CCORR)[0][0]

    def classify_digit(self, image):
        max_confidence = 0
        number = -1
        for filename in os.listdir(self.numbers_root_path):
            confidence = self.check_similarity(image, cv2.imread(os.path.join(self.numbers_root_path, filename), cv2.IMREAD_GRAYSCALE))
            if confidence > max_confidence:
                max_confidence = confidence
                try:
                    filename = filename.split('_')[0].split('.')[0]
                    number = int(filename)
                except ValueError:
                    number = -1

        return number

    def find_template_in_image(self, image, template):
        processed = cv2.matchTemplate(image, template, cv2.TM_CCORR)
        max_loc = cv2.minMaxLoc(processed)[3]
        return max_loc

    def remove_template_from_image(self, image, template):
        template_position = self.find_template_in_image(image, template)
        x = template_position[0]
        return image[:, :x]

    def preprocess_digit_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image

    def resize_roi(self, roi):
        roi = cv2.resize(roi, (22, 22))
        roi = 255 - roi
        roi = cv2.copyMakeBorder(roi, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
        roi = np.reshape(roi, (28, 28, 1))
        return roi

    def extract_components(self, image):
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        roi_list = []
        for label in range(1, n_labels):
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            roi = labels[y - 1:y + height + 1,
                  x - 1:x + width + 1].copy()  # create a copy of the interest region from the labeled image
            roi[roi != label] = 255  # set the other labels to white to eliminate intersections with other labels
            roi[roi == label] = 0  # set the interest region to black

            roi = np.array(roi, dtype=np.uint8)
            if roi.size > 0:
                roi = self.resize_roi(roi)
                roi_list.append(roi)
        return roi_list

    def remove_percentage(self, image):
        percentage_image = cv2.imread(os.path.join(self.media_root_path, 'percentage.png'), cv2.IMREAD_GRAYSCALE)
        image = self.remove_template_from_image(image, percentage_image)
        return image

    def extract_digits(self, image, additional_preprocessing=None):
        image = self.preprocess_digit_image(image)

        if additional_preprocessing:
            image = additional_preprocessing(image)

        return self.extract_components(image)

    def read_number_from_digit_images(self, digit_list):
        if len(digit_list) > 0:
            num = 0
            i = 0
            for digit_image in digit_list[::-1]:
                digit = self.classify_digit(digit_image)
                if digit != -1:
                    num += digit * 10 ** i
                    i += 1
            return num

        return None

    count = 0

    def debug_view_rois(self, rois, num, screen_name, save_character):
        global count

        white = np.full((28, 28, 1), 255, dtype=np.uint8)
        img = cv2.vconcat([white] + rois)
        img = cv2.putText(img, str(num), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow(screen_name, img)
        if cv2.waitKey(25) & 0xFF == ord(save_character):
            for roi in rois:
                cv2.imwrite(f"number_{count}.png", roi)
                count += 1

    def preprocess_token_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        return image

    def distance(self, first_point, second_point):
        diff_x = first_point[0] - second_point[0]
        diff_y = first_point[1] - second_point[1]
        return math.sqrt(diff_x ** 2 + diff_y ** 2)

    def count_tokens(self, image):
        preprocessed_image = self.preprocess_token_image(image)

        token = cv2.imread(os.path.join(self.tokens_root_path, 'intelligence.png'))
        token = self.preprocess_token_image(token)

        w, h = token.shape[::-1]
        res = cv2.matchTemplate(preprocessed_image, token, cv2.TM_CCOEFF_NORMED)
        res = np.where(res >= 0.25)  # a low threshold helps finding tokens of a different kind
        points = list(zip(*res[::-1]))

        # it's needed to filter out the false positives, which are immediately near pixels

        filtered_points = []
        if len(points) > 0:
            filtered_points.append(points[0])
            for point1 in points[1:]:
                to_add = True
                for point2 in filtered_points[:]:
                    dist = self.distance(point1, point2)
                    if dist <= 20:
                        to_add = False
                        break
                if to_add:
                    filtered_points.append(point1)

        # now it's needed to check for the focus in the interesting regions.
        focused = 0
        for pt in filtered_points:
            roi = image[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
            # the focus is just presence of yellow:
            gr_sum = np.sum(roi, (0, 1))[1:3]
            yellow = gr_sum[0] + gr_sum[1]
            if yellow >= 875000:
                focused += 1

        return len(filtered_points) - focused
