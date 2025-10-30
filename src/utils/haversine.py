import math


def haversine(lat1, len1, lat2, len2):
    """
    The Haversine function calculates the shortest (greatest circle) distance between two points on the surface of a sphere, such as the Earth, from their latitude and longitude coordinates.
    """

    # Radius of the Earth in kilometers
    R = 6371

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    len1_rad = math.radians(len1)
    lat2_rad = math.radians(lat2)
    len2_rad = math.radians(len2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlen = len2_rad - len1_rad

    # Haversine Formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlen / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance
