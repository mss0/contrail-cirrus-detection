"""Transforms Earth coordinates to pixel coordinates for orbital (ISS) photography.

The module contains a number of functions defined in order to solve the following problem:
    Given two overlapping photos of the surface of the Earth and the position at which they
    were taken, as well as the geographical location of a certain point, determine where
    that point would appear in the image plane.

The solution is based on the fact that the displacement between the positions at which the two
images were taken can be trivially expressed in geographical coordinates, as well as image coordinates.
The latter is achieved through pattern matching, as we can use the overlapping section to determine the
direction and magnitude of the offset in the image plane.

Since the map between the two coordinate systems is the composition of a rotation and a uniform scaling,
the two measurements offer enough information to fully define this map, and we can then use it to convert
Earth coordinates to image coordinates.

In terms of the functions of this module, the position of `p` in the plane defined by `photo1` and
`photo2`, taken at `pos1` and `pos2`, can be determined as follows:

    ref_x, ref_y, confidence = calculate_displacement(photo1, photo2)  # the px displacement
    px, py = photo_coordinates(pos1_lat, pos1_long, pos2_lat, pos2_long,
                               target_lat, target_long, ref_x, ref_y)

These (px, py) are offsets in pixels relative to a coordinate system in the mathematical convention, with
the origin at the center of photo1. To express them as indices of photo1's array (although they need not be
contained in it), one can use:

    (height, width) = photo1.shape[:2]
    column, row = round(width / 2 + px), round(height / 2 - py)

Conventions:
    Ground displacements are (east, north). Image displacements are (x, y) with the positive direction of the
    y axis set upwards (the mathematical convention, right-handed cartesian basis). This is deliberately not
    the screen convention used by OpenCV arrays, where y is a row index growing downwards. Conversion between
    the two is to be handled outside of this module.

Units:
    Ground values are in km, image values in px.

Assumptions:
    - The Earth is a sphere of radius 6371 km.
    - The camera is always pointing towards the center of the Earth. The position recorded for a photo should
      therefore be that of the point depicted at its center.
    - The field of view of the camera is sufficiently small for the Earth area captured to be reasonably
      approximated as a plane.
    - There are no perspective distortions of any kind.
    - The camera does not rotate between the two photos (cv2.matchTemplate is not invariant to rotation).
    - The altitude is the same for both photos, since it determines the scale factor of the map.
    - The two photos overlap by enough that at least one quadrant of the first lies wholly within the second.
"""

from math import cos, radians

import cv2
import numpy as np
import numpy.typing as npt

# 3-channel BGR image, as returned by cv2.imread.
Image = npt.NDArray[np.uint8]

EARTH_RADIUS_KM = 6371


def transform_to_photo(g_ref_x: float, g_ref_y: float, g_target_x: float, g_target_y: float,
                       i_ref_x: float, i_ref_y: float
                       ) -> tuple[float, float]:
    """Transforms a km offset in the ground plane to a pixel offset in the image plane.

    We use a reference vector expressed both in ground coordinates (g_ref) and image coordinates (i_ref).
    These coordinate systems both use the mathematical convention, i.e. they use a right-handed cartesian basis.
    Therefore, the vector g_ref and its image i_ref determine a similarity transformation from the ground plane
    to the image plane, that is the composition of a rotation and a uniform scaling.

    This can be interpreted in standard fashion using complex multiplication:
        Let u = g_ref_x + j*g_ref_y, w = i_ref_x + j*i_ref_y, a = g_target_x + j*g_target_y, where j denotes the
        imaginary unit. The similarity is then seen as a multiplication by w/u, so the desired offset is obtained
        as b = a * (w/u), where Re(b) is the x coordinate and Im(b) is the y coordinate.

    Args:
        g_ref_x, g_ref_y (float): x and y coordinates of reference ground displacement, in km.
        g_target_x, g_target_y (float): x and y coordinates of ground displacement to convert, in km.
        i_ref_x, i_ref_y (float): x and y coordinates of reference image displacement, in px.

    Returns:
        tuple[float, float]: x and y coordinates of converted offset, in px.

    Raises:
        ZeroDivisionError: if g_ref is the zero vector, i.e. g_ref_x == 0 and g_ref_y == 0.
            Occurs when the two ground reference points used coincide.

    .. note::
        Image coordinates are neither integers nor given in screen coordinates. Conversion is to be handled
        externally.
    """
    g_ref_norm_sq = g_ref_x**2 + g_ref_y**2

    real = (g_target_x * g_ref_x + g_target_y * g_ref_y) / g_ref_norm_sq
    imag = (g_target_y * g_ref_x - g_target_x * g_ref_y) / g_ref_norm_sq

    return i_ref_x * real - i_ref_y * imag, i_ref_y * real + i_ref_x * imag


def convert_to_km(lat1: float, long1: float, lat2: float, long2: float) -> tuple[float, float]:
    """Converts a lat/long displacement into an (east, north) displacement in the ground plane.

    We use the equirectangular projection since the lat/long span of the camera's field of view is small
    enough for it to be accurate. Since we use overlapping photos, the camera's FOV is also the upper bound
    for the distance between successive images.

    The projection uses cos(lat1) since we use the first point as the reference across the project. This is
    particularly relevant for the use of this function in photo_coordinates, where we use this point as the
    origin of the ground plane, which we require to be the same both for the reference and target vectors in
    transform_to_photo. This would have otherwise not been true had we used the standard convention,
    cos((lat1 + lat2) / 2), since calculating the reference and target would call convert_to_km with
    different values for lat2 (and long2).

    .. note::
        Since the reference is the first point, the order of arguments matters. Contrary to what one might
        expect, convert_to_km is therefore not antisymmetric under swapping the two points
        (i.e. convert_to_km(a, b) != -convert_to_km(b, a)).

    Args:
        lat1, long1 (float): lat/long coordinates of the first point.
        lat2, long2 (float): lat/long coordinates of the second point.

    Returns:
        tuple[float, float]: eastward and northward displacement between the two points, in km.
    """
    east = radians(long2 - long1) * EARTH_RADIUS_KM * cos(radians(lat1))
    north = radians(lat2 - lat1) * EARTH_RADIUS_KM

    return east, north


def photo_coordinates(lat1: float, long1: float, lat2: float, long2: float,
                      target_lat: float, target_long: float, i_ref_x: float, i_ref_y: float
                      ) -> tuple[float, float]:
    """Calculates the image plane offset of a point given its latitude and longitude.

    Functions as a wrapper of transform_to_photo, with lat/long inputs.

    Using point 1 as origin, we calculate a reference and target displacement in km, according to the
    specifications of transform_to_photo.

    Args:
        lat1, long1 (float): lat/long coordinates of the first reference point (origin).
        lat2, long2 (float): lat/long coordinates of the second reference point.
        target_lat, target_long (float): lat/long coordinates of the target.
        i_ref_x, i_ref_y (float): pixel displacement of the second reference point from the first.

    Returns:
        tuple[float, float]: x and y coordinates of the target's offset from point 1's image position, in px.

    Raises:
        ZeroDivisionError: if ref_km is the zero vector, i.e. lat1 == lat2 and long1 == long2 (or when
            lat1 == lat2 == +/-90, where longitude lines converge, so cos(lat1) == 0).
            Occurs when the two ground reference points used coincide.

    .. note::
        Reference points that are not well separated may lead to catastrophic cancellation when their
        lat/long difference is taken, and consequently to large errors in the result, which grow in
        inverse proportion to the norm of the reference.

    .. note::
        Image coordinates are neither integers nor given in screen coordinates, on input or on output. The
        supplied i_ref displacement must use the mathematical convention, i.e. a right-handed cartesian basis,
        matching the requirement of transform_to_photo. Conversion is to be handled externally.
    """
    ref_km = convert_to_km(lat1, long1, lat2, long2)
    target_km = convert_to_km(lat1, long1, target_lat, target_long)

    return transform_to_photo(ref_km[0], ref_km[1], target_km[0], target_km[1], i_ref_x, i_ref_y)


def find_template(image: Image, template: Image) -> tuple[tuple[int, int], float]:
    """Locates a template inside an image using ZNCC.

    The images are converted to grayscale before matching to improve efficiency and account for variation
    in color balance.

    Matching is done using OpenCV's cv2.matchTemplate, with cv2.TM_CCOEFF_NORMED chosen as the method.
    This is the zero-mean normalized cross correlation (ZNCC), which guarantees both brightness (zero-mean)
    and contrast invariance (normalized). Since it mathematically is the Pearson correlation coefficient, it
    also returns a confidence score in [-1, 1].

    .. note::
        A high score indicates a high similarity, but it may not necessarily be that the template was found
        within the image. Requires additional attention if the template has no distinguishing features.

    Args:
        image (Image): the image to search within, 3-channel BGR.
        template (Image): the pattern to search for, 3-channel BGR, no larger than image in either dimension.

    Returns:
        tuple[tuple[int, int], float]: the (x, y) position of the top left corner of the best match, and the
            correlation score in [-1, 1] at that position. Note that x indexes columns and y indexes rows.

    Raises:
        cv2.error: if input is not 3-channel (cvtColor converts from BGR2GRAY)
            or if template exceeds image in any dimension.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    _, confidence, _, position = cv2.minMaxLoc(result)

    return position, confidence


def calculate_displacement(photo1: Image, photo2: Image) -> tuple[int, int, float]:
    """Measures how much the camera has moved relative to the scene in the image plane.

    The size of the template is ideally maximal with respect to the area of overlap between
    two consecutive photos in our dataset. Given this, and for ease of implementation, the division
    of photo1 into quarters was chosen to create the templates. These are then searched for in
    photo2. The best matching template is chosen, so as to reduce the risk of false positives (e.g.
    a match due to a featureless template).

    Args:
        photo1, photo2 (Image): the two successive images, as returned by cv2.imread (3-channel BGR).
            It is implied by our use case that their dimensions are identical and that at least one
            quadrant of photo1 lies inside photo2.

    Returns:
        tuple[int, int, float]: the displacement in the image plane between the two positions the photos were
            taken at, in px, in the mathematical convention, followed by the correlation score.
    """
    (height, width) = photo1.shape[:2]
    (center_x, center_y) = (width // 2, height // 2)

    sections = [
        photo1[0:center_y, 0:center_x],
        photo1[0:center_y, center_x:width],
        photo1[center_y:height, 0:center_x],
        photo1[center_y:height, center_x:width],
    ]

    # Top left corner of every section, in the same order
    top_left = [(0, 0), (center_x, 0), (0, center_y), (center_x, center_y)]

    best, best_index, position = float("-inf"), 0, (0, 0)

    for index, section in enumerate(sections):

        result = find_template(photo2, section)

        if result[1] < best:
            continue

        best, best_index = result[1], index
        position = result[0]

    # First term returns the x component of the camera's displacement in the image plane, in both
    # the screen and mathematical convention. Second term is reversed in order to return the y component
    # of the camera's displacement in the mathematical convention.
    return top_left[best_index][0] - position[0], position[1] - top_left[best_index][1], best
