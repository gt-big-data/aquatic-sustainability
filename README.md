# Watershed Monitoring for Improved Freshwater Access and Incident Prevention
Motivation
Water has always been one of the most vital, yet alarmingly vulnerable, resources for human survival and development. In many less developed countries (LDCs), effective water management remains a significant challenge due to limitations in infrastructure and data as well as constantly changing environments. These issues affect billions worldwide and make it difficult to detect water contamination, effectively predict hydrological disasters, and ensure water remains available and affordable for those who need it.

The primary goal of this multi-faceted project is to aggregate data from a combination of satellite imagery, historical datasets, and comma-separated value (CSV) files in order to build an interface that geolocates water-related vulnerabilities and inequities. It will be easy for global citizens and governments alike to mark an area with this tool and view watershed perturbations, extreme weather risks, and discrepancies in water access. As the world is forced to accommodate more people and expend increased energy on technology, this initiative will be crucial for enabling community-level and large-scale awareness of aquatic affairs.

Methodology

To achieve the project's objectives, we will utilize advanced computer vision and machine learning techniques. The key components of our methodology include:
Data Collection and Preprocessing:
Acquire satellite imagery and street view data from sources like Google Maps and other satellite services.
Gather historical flood and drought data from the EPA and related agencies.
Preprocess the data to ensure consistency and quality for training machine learning models.
Model Development:
Computer Vision Techniques:
Implement segmentation techniques to identify and delineate regions of oil spills or algal blooms from pure water.
Use large convolution-based models (CNNs) to classify floods and droughts based on image data.
Time Series Analysis:
Apply Recurrent Neural Networks (RNNs) or Transformers to analyze timewise relationships in flood/drought data to predict future hydrological events and enable early flood/drought warning systems.
System Integration:
Develop an intuitive user interface allowing users to examine future flood or drought risk in a specific geographical region.
Integrate the ML models to provide real-time annotations and yield predictions for the flood/drought prediction.
Implement the tool as a Chrome Extension or a standalone website for ease of access.
Use Cases

Fishermen:
Obtain insights into water pollution and marine health to optimize where to fish.
Access to algal bloom heatmaps will ease understanding of ecological imbalances and thus mitigate overfishing.
Policy Makers/Government:
Utilize accurate data on water affordability and potability to alleviate local water shortage crises.
Monitor local precipitation and provide early-stage intervention prior to experiencing a major flood or drought.
Enforce regulations and new policies to promote sustainable water practices.
Researchers:
Analyze trends in coastline erosion and precipitation disasters exacerbated by climate change.
Develop correlational studies between flooding, algal blooms, oil spills, and overall shifts in global climate.
Environmental Agencies:
Identify regions facing water scarcity, pollution, and overuse to implement regulatory actions and provide aid.
Monitor and report on the efficiency of environmental policies and adjust as needed.
Enforce clean and sustainable water practices at both the regional and national level.

Benefits

Enhanced Disaster Planning: Provides precise LSTM data on flood and drought likelihoods, helping people in critical areas to plan ahead or evacuate.
Policy Development: Supports policymakers with geographic water affordability data to address regional pipe quality and lack of access to clean water.
Watershed Protection: Facilitates the discovery of both man-made and organic accidents like oil spills and algal blooms, leading to faster removal of contaminants.
Environmental Monitoring: Helps record the onset of rising sea levels from global warming by using flood and drought frequency as key indices.

Datasets
EPA Waters Geospatial Data
https://www.epa.gov/waterdata/waters-geospatial-data-downloads 
Applicable to Water Affordability
GIS (Geographic Information Systems) water quality assessment dataset
Archive of historical water quality data with metadata
Water Quality Portal (WQP)
https://www.waterqualitydata.us
Applicable to Water Affordability
Premiere source of discrete water-quality data filtered by precise location and site type
Aggregates data from the USGS, EPA, and state/local water agencies
EPA ECHO Database
https://echo.epa.gov/tools/data-downloads#downloads
Applicable to Water Affordability
ECHO Exporter provides CSV data on violations by water system, including “failure to afford”
FEMAFlood Map 
https://msc.fema.gov/portal/search?AddressQuery=united%20states 
Flood Map data specifically for the United States
Applicable to Flood/Drought Prediction
Not Satellite or CSV format. FIRMette(PDF/JPG map of flood zones). Tabular/spatial, can overlay over satellite or extract csv from a shapefile.
Public EM-DAT
https://public.emdat.be/ 
Data on all natural/industrial disasters from 2000-2025
Applicable to oil spill detection and flood/drought prediction.
CSV format
Contains year, location, and if available death toll and approximate damages.
Copernicus (Satellite)
Copernicus Browser
Wide range of satellite data for monitoring ocean bodies and flood/droughts.
Flood/Drought satellite imagery available
Applicable to Flood/Drought detection
Satellite imagery
FAO AQUASTAT 
https://www.fao.org/aquastat/en/ 
Applicable to Water Quality and flood/drought prediction
CSV format
Contains industrial, general, agricultural, etc water efficiency data.
Harmful Algal BloomS Observing System (HABSOS)
https://habsos.noaa.gov
Applicable to Algal Blooms & Oil Spills
Data files and interactive ArcGIS map
Can be used to verify real-time algal blooms by retrieving from the latest sample data

Previous work

Machine Learning for water quality prediction
https://www.sciencedirect.com/science/article/abs/pii/S0022169419308194 
Wavelet De-noising (WDT) Techniques for water quality prediction.
Identifying Oil Spills 
https://ieeexplore.ieee.org/abstract/document/9529039 
Four types of machine learning models are applied in this article: random forest; support vector machine (SVM); and deep neural network (DNN); and DNN with differential pooling (DP-DNN).
Drought Prediction
https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data/data 
Using datasets with weather/soil data from the harmonized world soil database
https://github.com/rksneha/Floods-and-Drought-prediction-using-LSTM
LSTM Model using datasets from NASA’s Global Modelling and Assimilation Office
Time Series Water Quality Analysis in Delhi
https://www.sciencedirect.com/science/article/pii/S1877050924009827 
Flood Prediction
https://www.sciencedirect.com/science/article/pii/S2666592123000963
Uses time series data mining to forecast four types of floods with a PRISMA review
Algal Bloom Detection
https://www.sciencedirect.com/science/article/pii/S0048969724036933 
Reports the latest techniques in identifying algal blooms using both regression/classification as well as image analysis (satellite imagery, surveillance cameras, microscopic images)



Model Resources
Time series models
Transformers
https://www.datacamp.com/tutorial/how-transformers-work 
RNN and LSTM
https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/#
https://www.kaggle.com/code/kcsener/8-recurrent-neural-network-rnn-tuto rial
Image-based time series
https://www.mdpi.com/2072-4292/13/23/4822
Classification
Intro to CNNs
https://www.youtube.com/watch?v=E5Z7FQp7AQQ&list=PLuhqtP7jdD8C D6rOWy20INGM44kULvrHu
https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn
-tutorial
https://www.tensorflow.org/tutorials/images/transfer_learning
https://github.com/tensorflow/models/blob/master/research/object_detection/g3do c/tf2_detection_zoo.md
Segmentation
https://medium.com/@raj.pulapakura/image-segmentation-a-beginners-guide-0e de91052db7
https://medium.com/@robmarkcole/a-brief-introduction-to-satellite-image-segmen tation-with-neural-networks-33ea732d5bce
CSV Files + Machine Learning
Pandas tutorial
https://www.datacamp.com/tutorial/pandas
Matplotlib tutorial
https://www.w3schools.com/python/matplotlib_intro.asp 
Tree-based models in ML
https://www.stratascratch.com/blog/tree-based-models-in-machine-learning/ 

Overall Goals
Platform: Aid Analysis in working with datasets including automating data preprocessing workflows, storing datasets effectively and to be able to find similarities between them. Use relevant data stores from AWS and similar platforms. Implement a backend system including API endpoints, model integration, cloud hosting, databases, and scalability.

Analysis: Research and analyze datasets and factors for the 3 model areas - Algal Blooms/Oil Spill detection, water affordability, and Flood/Drought prediction. Determine implementation approaches for the models, while making the necessary transformations, merges, and filtering to the usable data. Follow an approach to ensure continuous improvement - Integrate prior research or basic approaches, and iteratively (1) incorporate higher concepts and findings and (2) different data sources. Ultimately, develop models that can integrate into a larger tool providing water sustainability insights.

Visualisation: Create an intuitive user interface to allow users to select areas on a map (both land-based and marine) and view the appropriate annotations: flood/drought vulnerability, water affordability, and accident detection. This includes frontend development, visualizing patterns in our datasets, working with Maps API, and interacting with end user data.
