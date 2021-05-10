#!/usr/bin/env python
# coding: utf-8

# # London houseing price&venues

# ## A. Introduction

# ### A.1. Description & Disscusion of London

# London is the capital and largest city of England and the United Kingdom.london has 32 borough. London is one of the world's most important global cities.London has a diverse range of people and cultures, and more than 300 languages are spoken in the region. London covers an area of 1,579 square kilometres (610 sq mi). The population density is 5,177 inhabitants per square kilometre (13,410/sq mi),more than ten times that of any other British region. In terms of population, London is the 19th largest city and the 18th largest metropolitan region.
# As you can see from the above figures, london is a city with a high population and population density. Being such a crowded city leads the owners of shops and social sharing places in the city where the population is dense. When we think of it by the investor, we expect from them to prefer the districts where there is a lower real estate cost and the type of business they want to install is less intense. If we think of the city residents, they may want to choose the regions where real estate values are lower. At the same time, they may want to choose the district according to the social places density. However, it is difficult to obtain information that will guide investors in this direction.
# 
# When we consider all these problems, we can create a map and information chart where the real estate index is placed on london and each neighborhood is clustered according to the venue diversity.

# ### A.2. Data preperation

# The data that i have gathered are
# 1. The neighborhood data from wikipedia page.
# 2. The house price data is taken from united kingdom goverment website.
# 3. I used Forsquare API to get the most common venues of given neighborhood of london.
# 
# 

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis


import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library

#for elbow method
from scipy.spatial.distance import cdist


# ## B. Methodology

# ### B.1 Website scraping

# In[ ]:


url='https://en.wikipedia.org/wiki/List_of_areas_of_London'


# In[ ]:


data=requests.get(url).text
soup=BeautifulSoup(data,'html5lib')
tables =soup.find_all('table')


# In[ ]:


len(tables)


# In[ ]:


l_data=pd.DataFrame(columns=['Area','Borough','Latitude','Longitude'])


# In[ ]:


table_index=1
tables[table_index]
t =tables[table_index]


# In[ ]:


for row in t.tbody.find_all("tr"):
    col = row.find_all("td")
    if (col != []):
        area = col[0].text
        Borough = col[1].text.split('[')[0]
        l_data =l_data.append({"Area":area, "Borough":Borough}, ignore_index=True)


# In[ ]:


l_data['Area']=l_data['Area'].replace({'Barnet (also Chipping Barnet, High Barnet)':'High Barnet',
                                   'Bexley (also Old Bexley, Bexley Village)':'Bexley',
                                   'Bexleyheath (also Bexley New Town)':'Bexleyheath',
                                   'Bromley (also Bromley-by-Bow)':'Bromley',
                                   'Marylebone (also St Marylebone)':'Marylebone',
                                   'Sydenham (also Lower Sydenham,Upper Sydenham)':'Sydenham',
                                   'Widmore (also Widmore Green)':'Widmore',
                                   'Aldborough Hatch':'Ilford'})



ld=pd.DataFrame(columns=['Latitude','Longitude'])

for area in l_data['Area']:
    geolocator = Nominatim(user_agent="ny_explorer")
    location = geolocator.geocode(area)
    latitude = location.latitude
    longitude = location.longitude
    ld=ld.append({"Latitude":latitude,"Longitude":longitude},ignore_index=True)
l_data['Longitude']=ld['Longitude']
l_data['Latitude']=ld['Latitude']


# ### B.2 Using the price data from united kingdom goverment and merge both dataframe.

# In[ ]:


price_data=pd.read_excel(r"C:\Users\harsh\Downloads\london_house_xls.csv",sheet_name='Mean')
price_d = price_data[['Area',"Year ending Dec 2017"]]
price_d.rename({"Area": "Borough","Year ending Dec 2017":"Price"},axis=1,inplace=True)
data = pd.merge(l_data,price_d, on = "Borough", how = "inner")


# In[5]:


data.dtypes


# In[6]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(data['Borough'].unique()),
        data.shape[0]
    ))


# In[7]:


address = 'london,england'
geolocator = Nominatim(user_agent="my_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of London are {}, {}.'.format(latitude, longitude))


# ### B.3 exploring venues from all neighborhood

# In[8]:


# create map of london using latitude and longitude values
map_london = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, area in zip(data['Latitude'], data['Longitude'], data['Borough'], data['Neighborhood']):
    label = '{}, {}'.format(area, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_london)  
    
map_london


# In[9]:


CLIENT_ID = '' # my Foursquare ID
CLIENT_SECRET = '' # my Foursquare Secret
VERSION = '' # Foursquare API version
LIMIT = 500 # A default Foursquare API limit value
ACCESS_TOKEN = '' 
#print('Your credentails:')
#print('CLIENT_ID: ' + CLIENT_ID)
#print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[8]:


data.loc[0, 'Neighborhood']

neighborhood_latitude = data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# for addignton neighborhood searching venues using foresquare

# In[9]:


radius = 2000
LIMIT = 100
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[10]:


results = requests.get(url).json()
results


# In[11]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[12]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[13]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# By using below function we are going to extract venues from all the neighborhood.

# In[14]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[16]:


# type your answer here
london_venues = getNearbyVenues(names=data['Neighborhood'],
                                   latitudes=data['Latitude'],
                                   longitudes=data['Longitude']
                                  )


# In[5]:


london_venues.dtypes


# In[24]:


print(london_venues.shape)


# In[6]:


london_venues.groupby('Neighborhood').count()


# In[13]:


print('There are {} uniques categories.'.format(len(london_venues['Venue Category'].unique())))


# In[14]:


l=london_venues.groupby('Neighborhood').count()
v= l[["Venue"]]
import plotly.express as px
df = px.data.tips()
fig = px.bar(v,y="Venue",height=800)
fig.show()


# In[18]:


fig = px.bar(data, x="Price", title='Total number')
# 'bin_edges' is a list of bin intervals
count, bin_edges = np.histogram(data['Price'])

data['Price'].plot(kind='hist',edgecolor='black', figsize=(8, 5), xticks=bin_edges, color='skyblue')

plt.title('Housing price in Range') # add a title to the histogram
plt.ylabel('Frequecy') # add y-label
plt.xlabel('Housing Price(In Millions)') # add x-label

plt.tight_layout()
plt.show()


# we define the range for value of house
# less than  481000                   “Low Level HSP”
# Between 481000 to 839000            “Mid-1 Level HSP”
# Between 839000 to 1197000           “Mid-2 Level HSP”
# Between 1197000 to 1734000          “High-1 Level HSP”
# more than 1734000                   “High-2 Level HSP”

# In[30]:


prist =[]
for price in data["Price"]:
    if price<481000:
        prist.append("Low_Level_HSP")
    elif 481000>=price<839000:
        prist.append("Mid-1_Level_HSP")
    elif 839000>=price<1197000:
        prist.append("Mid-2_Level_HSP")
    elif 1197000>=price<1734000:
        prist.append("High-1_Level_HSP")
    else:
        prist.append("High-2_Level_HSP")
        
data['Category']=prist


# ### B.4. Analyze Each Neighborhood

# In[7]:


# one hot encoding
london_onehot = pd.get_dummies(london_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
london_onehot['Neighborhood'] = london_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [london_onehot.columns[-1]] + list(london_onehot.columns[:-1])
london_onehot = london_onehot[fixed_columns]

london_onehot.head()
london_onehot.shape


#  let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[8]:


london_grouped = london_onehot.groupby('Neighborhood').mean().reset_index()
london_grouped


# Let's print each neighborhood along with the top 5 most common venues

# In[9]:


num_top_venues = 5

for hood in london_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = london_grouped[london_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# from these make a pandas dataframe

# In[10]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[11]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = london_grouped['Neighborhood']

for ind in np.arange(london_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(london_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.shape


# ### B.5. Cluster Neighborhoods

# In[13]:


# set number of clusters
kclusters = 4

london_grouped_clustering = london_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(london_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[14]:


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(london_grouped_clustering)
    kmeanModel.fit(london_grouped_clustering)
 
    distortions.append(sum(np.min(cdist(london_grouped_clustering, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / london_grouped_clustering.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(london_grouped_clustering, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / london_grouped_clustering.shape[0]
    mapping2[k] = kmeanModel.inertia_


# In[15]:


plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


# By this method we can take the value of k as 4

# In[25]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

london_merged = data

# merge london_grouped with data to add latitude/longitude for each neighborhood
london_merged = london_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

london_merged.head()


# Finally, let's visualize the resulting clusters

# In[142]:


count_venue = london_merged
count_venue = count_venue.drop(['Borough','Price', 'Latitude', 'Longitude'], axis=1)
count_venue = count_venue.groupby(['Cluster Labels','1st Most Common Venue']).size().reset_index(name='Counts')

#we can transpose it to plot bar chart
cv_cluster = count_venue.pivot(index='Cluster Labels', columns='1st Most Common Venue', values='Counts')
cv_cluster = cv_cluster.fillna(0).astype(int).reset_index(drop=True)
cv_cluster


# In[147]:


#creating a bar chart of "Number of Venues in Each Cluster"
frame=cv_cluster.plot(kind='bar',figsize=(20,10),width = 0.8)

plt.legend(labels=cv_cluster.columns,fontsize= 8)
plt.title("Number of Venues in Each Cluster",fontsize= 16)
plt.xticks(fontsize=14)
plt.xticks(rotation=0)
plt.xlabel('Number of Venue', fontsize=14)
plt.ylabel('Clusters', fontsize=14)


# # C. Results

# ### C.1 Main table with list

# In[28]:


#cluster 0 =pub venues
#cluster1 = park venues
#cluster2 = grocery & pub
#cluster3 = all social venues

Cluster_list =[]
for c in london_merged["Cluster Labels"]:
    if c== 0:
        Cluster_list.append("Pub Venues")
    elif c== 1:
        Cluster_list.append("Park Venues")
    elif c== 2:
        Cluster_list.append("Grocery & Pub")
    else:
        Cluster_list.append("All Social Venues")
        
london_merged['Cluster Type']=Cluster_list
#london_merged.drop(['Cluster Label'],axis=1,inplace=True)
london_merged


# ### C.2 map with all the labels

# In[173]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for a,b,c,lat, lon, poi,d, cluster in zip(london_merged['1st Most Common Venue'],london_merged['2nd Most Common Venue'],london_merged['Cluster Type'],london_merged['Latitude'], london_merged['Longitude'], london_merged['Neighborhood'],london_merged['Category'], london_merged['Cluster Labels']):
    label = folium.Popup(str(poi)+ ',' + ' Cluster ' + str(cluster) + ', '+ str(d) +' ,' + str(a)+  ', '  + str(b)+','+ str(c),parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=['green','red','blue','orange'][cluster-1],
        fill=True,
        fill_color=['green','red','blue','orange'][cluster-1],
        fill_opacity=0.9).add_to(map_clusters)
       
map_clusters


# ### C.3 Examine Clusters

# Now, you can examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, you can then assign a name to each cluster.

# In[ ]:


london_merged.loc[london_merged['Cluster Labels'] == 0, london_merged.columns[[1] + list(range(4, london_merged.shape[1]))]]


# In[ ]:


london_merged.loc[london_merged['Cluster Labels'] == 1, london_merged.columns[[1] + list(range(4, london_merged.shape[1]))]]


# In[ ]:


london_merged.loc[london_merged['Cluster Labels'] == 2, london_merged.columns[[1] + list(range(4, london_merged.shape[1]))]]


# In[49]:


london_merged.loc[london_merged['Cluster Labels'] == 3, london_merged.columns[[1] + list(range(4, london_merged.shape[1]))]]


# ## D. Discussion

# As I mentioned before, london is a big city with a high population density . As there is such a complexity, very different approaches can be tried in clustering and classification studies. Moreover, it is obvious that not every classification method can yield the same high quality results for this metropol.
# 
# I used the Kmeans algorithm as part of this clustering study. When I tested the Elbow method, I set the optimum k value to 4.  For more detailed and accurate guidance, the data set can be expanded and the details of the neighborhood  can also be drilled.
# 
# I ended the study by visualizing the data and clustering information on the london map. In future studies, web or telephone applications can be carried out to direct investors.

# ## F. Conclusion

# As a result, people are turning to big cities to start a business or work. For this reason, people can achieve better outcomes through their access to the platforms where such information is provided.
# 
# Not only for investors but also city managers can manage the city more regularly by using similar data analysis types or platforms.

# ## G. Reference

# 1. List of areas of London   https://en.wikipedia.org/wiki/List_of_areas_of_London
# 2. Average House Prices by Borough, Ward, MSOA & LSOA by united kingdom https://data.london.gov.uk/dataset/average-house-prices
# 3. Forsquare API  https://developer.foursquare.com/ 
