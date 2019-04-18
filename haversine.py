from math import radians, cos, sin, asin, sqrt
def distance_from_flatiron(coords_tuple):
    #'''47.610361, -122.336107'''
    '''COPIED FROM STACKOVERFLOW MICHAEL DUNN
        INSPIRED BY DAVID KASPAR THE RAINBOW UNICORN'''
    lon1, lat1, lon2, lat2 = coords_tuple[1], coords_tuple[0], -122.336107, 47.610361
    
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r