import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import dask

from shiftscope.cellfaceMultistreamSupport.cellface.storage import default_datasets
#from cellface.storage.utils import column_dict_to_dtype_list
import dask.array as da
#dtype_list = column_dict_to_dtype_list(default_datasets['morphological_features']['columns'])

@dask.delayed
def get_mf(image, mask, min_size=30):
    # convert to float32
    img = np.float32(image)
    mask = np.uint8(mask)  #added
    # physical parameters
    wave_length = 0.53
    pixel_size = 3.45 / 40 * 4
    #output array
    #array = np.array(0, dtype=dtype_list)
    dtype_list = [
        ('mass_center_shift', float),
        ('optical_height_max', float),
        ('optical_height_min', float),
        ('optical_height_mean', float),
        ('optical_height_var', float),
        ('volume', float),
        ('area', float),
        ('radius_mean', float),
        ('radius_var', float),
        ('perimeter', float),
        ('circularity', float),
        ('solidity', float),
        ('equivalent_diameter', float),
        ('aspect_ratio', float),
        ('height', float),
        ('width', float),
        ('biconcavity', float),
        ('sphericity', float),
        ('contrast', float),
        ('dissimilarity', float),
        ('homogeneity', float),
        ('correlation', float),
        ('energy', float),
        ('entropy', float),
    ]

    array = np.zeros(1, dtype=dtype_list)

    # Base variables
    contour, duplet = get_contour(mask, min_size)
    masked_image = img * mask

    # weighted image moments
    weighted_moments = cv2.moments(masked_image)
    # image moments (unweighted)
    moments = get_moments(contour)
    # center of mass (in pixels)
    mass_center = get_center(weighted_moments)
    # geometric center (center of contour) (in pixels)
    center = get_center(moments)

    # mass center shift (Euclidian distance between geometric centroid and mass centroid) (in µm)
    array['mass_center_shift'] = get_mass_center_shift(center, mass_center, pixel_size)
    # optical height max, min, mean and variance (the maximum, minimum, mean and variance of phase value inside the contour) (in µm)
    array['optical_height_max'] = np.max(masked_image) * wave_length / (2 * np.pi)
    array['optical_height_min'] = np.min(masked_image) * wave_length / (2 * np.pi)
    array['optical_height_mean'] = np.mean(masked_image) * wave_length / (2 * np.pi)
    array['optical_height_var'] = np.var(masked_image) * wave_length / (2 * np.pi)
    # optical volume (in µm^3)
    array['volume'] = get_volume(masked_image, wave_length, pixel_size)
    # area of cell (in pixels and in µm)
    area_pixels = moments['m00']
    array['area'] = area_pixels * pixel_size * pixel_size
    # mean and variance of cells radius (in µm)
    array['radius_mean'], array['radius_var'] = get_radius(center, contour, pixel_size)
    # cell perimeter (in pixels and in µm)
    perimeter_pixels,  array['perimeter'] = get_perimeter(contour, pixel_size)
    # circularity of cell (circularity of a circle is 1; the closer it gets to 0, the less circular the contour is)
    array['circularity'] = 4.0 * np.pi * area_pixels / perimeter_pixels**2
    # solidity (the ratio of contour area to its convex hull area)
    array['solidity'] = get_solidity(area_pixels, contour)
    # equivalent diameter (calculated from cell area) (in µm)
    array['equivalent_diameter'] = get_equivalent_diameter(area_pixels, pixel_size)
    # height and width of cell (in µm), and aspect ratio
    array['aspect_ratio'] = get_aspect_ratio(contour, pixel_size)
    array['height'], array['width'] = get_height_and_width(contour, pixel_size)
    # biconcavity and sphericity (how biconcave and spherical a cell is)
    array['biconcavity'],  array['sphericity'] = get_biconcavity_and_sphericity(masked_image, center)
    # textural features based on grey-level co-occurence matrix
    # (commented out, since computationally expensive)
    array['contrast'], array['dissimilarity'], array['homogeneity'], array['correlation'], array['energy'], array['entropy'] = \
    get_glcm_features((np.clip(masked_image,0,4)/4*255).astype(np.uint8), mask)
    
    return array[np.newaxis]


@dask.delayed
def check_duplet(mask, min_size=30):
    _, duplet = get_contour(mask, min_size)
    return 1 if duplet else 0

def get_contour(mask, min_size):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_size]
    duplet = False
    # If there is more then one contour pick the biggest one and set duplet flag
    if len(contours) > 1:
        contours.sort(key=cv2.contourArea)
        contours = contours[:1]
        duplet = True
    return contours[0], duplet



# From RetroFace    
def get_volume(masked_image, wave_length, pixelSize):
    volume = np.sum(masked_image) * wave_length / (2 * np.pi) * pixelSize * pixelSize
    return volume


def get_moments(contour):
    return cv2.moments(contour)


def get_center(moments):
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return np.array((cx, cy))


def get_perimeter(contour, pixel_size):
    perimeter_pixels = cv2.arcLength(contour, True)
    perimeter = perimeter_pixels * pixel_size
    return perimeter_pixels, perimeter


def get_height_and_width(contour, pixelSize):
    (_, (width, height), _) = cv2.minAreaRect(contour)
    width = width * pixelSize
    height = height * pixelSize
    return width, height


def get_aspect_ratio(contour, pixel_size):
    height, width = get_height_and_width(contour, pixel_size)
    return max(width, height) / min(width, height)


def get_mass_center_shift(center, mass_center, pixel_size):
    return np.linalg.norm(center - mass_center) * pixel_size


def get_equivalent_diameter(area_pixels, pixel_size):
    return np.sqrt((4.0 * area_pixels)/np.pi) * pixel_size


def get_solidity(cell_area, contour):
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return float(cell_area) / hull_area


def get_radius(center, contour, pixelSize):
    distances_from_center = np.sqrt(((contour[:, 0] - center) ** 2).sum(1))
    radius_mean = np.mean(distances_from_center) * pixelSize
    radius_var = np.var(distances_from_center) * pixelSize
    return radius_mean, radius_var


def get_biconcavity_and_sphericity(masked_image, center):
    mid_column = masked_image[:, center[0]]
    mid_column = mid_column[mid_column != 0]
    mid_row = masked_image[center[1], :]
    mid_row = mid_row[mid_row != 0]
    x = np.linspace(-1, 1, len(mid_row))
    y = np.linspace(-1, 1, len(mid_column))

    with np.errstate(divide='ignore', invalid='ignore'):
        ##biconcavity
        poly_x = -4 * x ** 4 + 4 * x ** 2 + 0.5
        poly_y = -4 * y ** 4 + 4 * y ** 2 + 0.5
        biconcavity = min(np.corrcoef(mid_row, poly_x)[0, 1], np.corrcoef(mid_column, poly_y)[0, 1])

        ##sphericity
        poly_x = -x ** 2 + 1
        poly_y = -y ** 2 + 1
        sphericity = min(np.corrcoef(mid_row, poly_x)[0, 1], np.corrcoef(mid_column, poly_y)[0, 1])

    return biconcavity, sphericity


def get_glcm_features(masked_image, mask):
    # grey level co-occurence matrix (mean of 4 directions [0, np.pi/4, np.pi/2, 3*np.pi/4])
    glcm = np.expand_dims(greycomatrix(masked_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True,
                                       normed=True).mean(3), 3)

    # textural faetures from glcm
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    entropy = - np.sqrt(energy) * np.log(np.sqrt(energy))

    return contrast, dissimilarity, homogeneity, correlation, energy, entropy
