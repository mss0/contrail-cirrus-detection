import cv2
import numpy as np

from math import radians

# all earth parameters are to be given in km and all photo parameters in px


def transform_to_photo(earth_a, earth_b, earth_x, earth_y, photo_a, photo_b):

    denom = earth_a**2 + earth_b**2
    interm1, interm2 = (earth_x * earth_a + earth_y * earth_b) / denom, (earth_y * earth_a - earth_x * earth_b) / denom

    print(interm1, interm2)

    return photo_a * interm1 - photo_b * interm2, photo_b * interm1 + photo_a * interm2


# assume the earth is a sphere with the radius of 6371 km

earth_radius = 6371


def convert_to_km(lat1, long1, lat2, long2):

    return radians(long2 - long1) * earth_radius, radians(lat2 - lat1) * earth_radius


def photo_coordinates(lat1, long1, lat2, long2, plane_lat, plane_long, photo_x, photo_y):

    to_km1 = convert_to_km(lat1, long1, lat2, long2)
    to_km2 = convert_to_km(lat1, long1, plane_lat, plane_long)

    print(to_km1, to_km2)

    return transform_to_photo(to_km1[0], to_km1[1], to_km2[0], to_km2[1], photo_x, photo_y)


def find_template(image, template):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    confidence = result.max()

    return np.where(result == confidence), confidence


def calculate_displacement(photo1, photo2):

    (height, width) = photo1.shape[:2]
    (center_x, center_y) = (width // 2, height // 2)

    sections = [photo1[0:center_y, 0:center_x], photo1[0:center_y, center_x:width],
                photo1[center_y:height, 0:center_x], photo1[center_y:height, center_x:width]]

    top_left = [(0, 0), (center_x, 0), (0, center_y), (center_x, center_y)]

    best, best_index = 0, 0
    box = []

    for index, section in enumerate(sections):

        result = find_template(photo2, section)

        if result[1] < best:
            continue

        best, best_index = result[1], index
        box = result[0]

    return box[1][0] - top_left[best_index][0], box[0][0] - top_left[best_index][1]
