import os 

from pymongo import MongoClient
from dotenv import load_dotenv
from . import mongoDB_uri


load_dotenv()

# Accesses the database from MongoDB cluster
MONGODB_URI = mongoDB_uri()
#Creates connection using MongoClient
client = MongoClient(MONGODB_URI)
#Import the database from MongoDB; need to create first, will do later
db = client['Test']

#Imports a collection from in the database
collection_name = db['TestCollection']

#Creates a new document in the database
#This will be the location the user queries, and where the model outputs will be stored based on location
def newDocument(long, lat, flood_predictability):
    loc = {
        "longitutde": long,
        "latitude": lat,
        "flood_predictability": flood_predictability
        }
    collection_name.insert_one(loc)

def getDocument(long, lat):
    item = db.collection_name.find({
        "longtitude": long,
        "latitude": lat
        })
    return item['flood_predictability']