// Javascript version of MongoDB implementation
// For future use (js integrates smoother than python)
// Do NOT use this, use mongodb.py

import { MongoClient } from 'mongodb';
 
// Load URI from environmental variable
const uri = process.env.MONGODB_URI;
const options = {
    useUnifiedTopology: true,
    useNewUrlParser: true,
}

let client
let clientPromise

// Throw an error if URI is not set
if(!process.env.MONGODB_URI){
    throw new Error("Please add your Mongo URI to .env.local")
}

// Global variable for development to prevent multiple connections
if(process.env.NODE_ENV === 'development'){
    if(!global._mongoClientPromise){
        client = new MongoClient(uri, options)
        global._mongoClientPromise = client.connect()
    } 
    clientPromise = global._mongoClientPromise
}
// Normal connection for production
else {
    client = new MongoClient(uri, options)
    clientPromise = client.connect()
}

// Export the client promise to be used across functions
export default clientPromise;
