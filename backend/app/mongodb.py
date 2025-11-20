# Not completed yet, for future use next semester
# Do NOT use yet 

import os 
import certifi

from pymongo import MongoClient
from dotenv import load_dotenv

ca = certifi.where()

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

# Accesses the database from MongoDB cluster
def get_database():
    client = MongoClient(MONGODB_URI, tlsCAFile=ca)

    return client ['aquaticSustainability']

if __name__ == "__main__":
    dbname = get_database()

collection_name = dbname["modelOutputs"]

# Checks if coordinates already exist or in certain distance (to be determined)
# If they do, updates that document with model outputs
# If not, creates new document
# def checkDocuments():

# Creates new document and stores model outputs: longitude, latitude, flood predictability
def createDocument():
    location = {
        "longitude": -122.4194,
        "latitude": 37.7749,
        # "flood_predictability": Array[0.8, 0.9, 0.95], 
        }
    collection_name.insert_one(location)
