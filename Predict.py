import math
import pickle
import geopandas as gpd
import pandas as pd
from math import radians
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_parquet('./../data/BMTC.parquet.gzip', engine='pyarrow')
dfInput = pd.read_csv('./../data/Input.csv')
dfGroundTruth = pd.read_csv('./../data/GroundTruth.csv')


def EstimatedTravelTime(df, dfInput):  # The output of this function will be evaluated
    # Function body - Begins
    model1 = pickle.load(open('Loc_Speed.sav', 'rb'))
    model2 = pickle.load(open('Speeds_Dist_ETT.sav', 'rb'))
    ETT, distances, speeds = prediction(dfInput, model1, model2)
    dfOutput = pd.DataFrame(ETT, columns=['ETT'])
    # Function body - Ends
    return dfOutput


"""
Other function definitions here: BEGINS
"""


def getPathLength(lat1, lng1, lat2, lng2):
    """calculates the distance between two lat, long coordinate pairs"""

    R = 6371000  # radius of earth in m
    lat1rads = math.radians(lat1)
    lat2rads = math.radians(lat2)
    deltaLat = math.radians((lat2 - lat1))
    deltaLng = math.radians((lng2 - lng1))
    a = math.sin(deltaLat / 2) * math.sin(deltaLat / 2) + math.cos(lat1rads) * math.cos(lat2rads) * math.sin(
        deltaLng / 2) * math.sin(deltaLng / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def getDestinationLatLong(lat, lng, azimuth, distance):
    """returns the lat an long of destination point
    given the start lat, long, azimuth, and distance"""

    R = 6378.1  # Radius of the Earth in km
    bearing = math.radians(azimuth)  # Bearing is degrees converted to radians.
    d = distance / 1000  # Distance m converted to km
    lat1 = math.radians(lat)  # Current dd lat point converted to radians
    lon1 = math.radians(lng)  # Current dd long point converted to radians
    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) + math.cos(lat1) * math.sin(d / R) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d / R) * math.cos(lat1),
                             math.cos(d / R) - math.sin(lat1) * math.sin(lat2))
    # convert back to degrees
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return [lat2, lon2]


def calculateBearing(lat1, lng1, lat2, lng2):
    """calculates the azimuth in degrees from start point to end point"""

    startLat = math.radians(lat1)
    startLong = math.radians(lng1)
    endLat = math.radians(lat2)
    endLong = math.radians(lng2)
    dLong = endLong - startLong
    dPhi = math.log(math.tan(endLat / 2.0 + math.pi / 4.0) / math.tan(startLat / 2.0 + math.pi / 4.0))
    if abs(dLong) > math.pi:
        if dLong > 0.0:
            dLong = -(2.0 * math.pi - dLong)
        else:
            dLong = (2.0 * math.pi + dLong)
    bearing = (math.degrees(math.atan2(dLong, dPhi)) + 360.0) % 360.0;
    return bearing


def coords_src_dest(interval, azimuth, lat1, lng1, lat2, lng2):
    """returns every coordinate pair inbetween two coordinate
    pairs given the desired interval"""

    d = getPathLength(lat1, lng1, lat2, lng2)
    remainder, dist = math.modf((d / interval))
    counter = float(interval)
    coords = [[lat1, lng1]]
    for distance in range(0, int(dist)):
        coord = getDestinationLatLong(lat1, lng1, azimuth, counter)
        counter = counter + float(interval)
        coords.append(coord)
    coords.append([lat2, lng2])
    return coords


def locs_to_avgSpeed(lat1, lng1, lat2, lng2, model):
    """calculates the mean speed between 2 co-ordinates"""

    interval = 25
    azimuth = calculateBearing(lat1, lng1, lat2, lng2)
    coords = coords_src_dest(interval, azimuth, lat1, lng1, lat2, lng2)
    speeds = float()
    for lat, lng in coords:
        speeds = speeds + model.predict([[math.radians(lat), math.radians(lng)]])
    return float(speeds / len(coords))


def prediction(dfInput, model1, model2):
    """combines model1 and model2 and predicting ETT"""

    ETT = []
    distances = []
    speeds = []
    Numpy_Vectorized_dfInput = dfInput.to_numpy()
    for row in Numpy_Vectorized_dfInput:
        speed = locs_to_avgSpeed(row[0], row[1], row[2], row[3], model1)
        distance = getPathLength(row[0], row[1], row[2], row[3]) / 1000
        time = distance * 60 / speed
        time_updated = model2.predict([[speed, distance, time]])[0]
        ETT.append(time_updated)
        distances.append(distance)
        speeds.append(speed)
    return ETT, distances, speeds


def clean(df):
    """converts coordinates from degrees to
       radians for normalization purposes"""

    df['Latitude'] = df['Latitude'].apply(radians)
    df['Longitude'] = df['Longitude'].apply(radians)
    return df


def train_regression_model():
    """training the model using KNN Regressor"""

    X = df[['Latitude', 'Longitude']].values
    y = df['Speed'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = KNeighborsRegressor(weights='distance', metric='haversine', n_neighbors=21)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred


"""
Other function definitions here: ENDS
"""

# Remove Unnamed column in pandas dataframe
dfInput = dfInput.loc[:, ~dfInput.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# pandas dataframe that contains the output with column name 'ETT'
dfOutput = EstimatedTravelTime(df, dfInput)
