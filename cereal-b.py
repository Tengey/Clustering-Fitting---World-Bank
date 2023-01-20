import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt
import seaborn as sns
import warnings
import err_ranges as err
from sklearn.cluster import KMeans 

warnings.filterwarnings('ignore')


def csv_reader(file_path, ind_name): 
    '''
    This function is created to read a csv file, then create a specfic 
    indicator table and then transforms it. 
    
    Agrs : 
        File path : The dataset(csv) file path. 
        ind_name : The indicator table to be created. 
    
    Return :
        It returns dataframe, transformed dataframe and the indicator name. 
    '''
    df = pd.read_csv(file_path, skiprows = 3)
    df.set_index('Country Name', inplace = True)
    df_2 = df[df['Indicator Name']== ind_name]
    df_2.drop(['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'], 1, inplace = True )
    df_2.fillna(0, inplace = True)
    df_t =df_2.T
    return df_2, df_t, ind_name 

data, data_t, ind_title = csv_reader('API_19_DS2_en_csv_v2_4773766.csv', 
                                     'Cereal yield (kg per hectare)')
data_t.head()

data_G = data_t.loc[:, 'World':'Zimbabwe']
plt.figure (figsize = (10, 10))
pd.plotting.scatter_matrix(data_G, figsize=(10,10), s=5, alpha=0.8);

def plots (df, df_t, ind_name, camp="crest" ):
    '''
    This functions creates an heatmap of country and years with relation to the indication.
    
    Parameter : 
        df : This is the normal dataframe 
        df_t : This is the transform dataframe 
        Ind_name : Name of the selected indicator
        cmap ('crest' - default):  the color of the various heatmap 
    Return: 
        It returns two heatmap plot with respect to the indicator 
    '''
    plt.figure(figsize = (5, 5))
    sns.heatmap(df.corr(), cmap =camp, xticklabels= False)
    plt.title ('Year correlation w.r.t {}'.format(ind_name))
    plt.show()
    
    plt.figure(figsize = (5, 5))
    sns.heatmap(df_t.corr(), cmap =camp, xticklabels= False)
    plt.title ('Countries correlation w.r.t {}'.format(ind_name))
    plt.show()
    
    
plots(data, data_t, ind_title)

def MinMaxScaler (data, yr_1, yr_2):
    '''
    This function is a normalizer. It splits the dataframe and scales the dataframe.
    
    Parameters: 
        data : dataframe to splits and scaled 
        yr_1 : First column
        yr_2 : Second Column
        
    Return : 
        it returns a scaled dataframe and the respective columns
    '''
    data_2 = data.loc[:, yr_1:yr_2]
    df_min = data_2 - data_2.min()
    df_ran = data_2.max() - data_2.min()
    scaler = (df_min  / df_ran) 
    df_scaler = scaler.values 
    return df_scaler, yr_1, yr_2

df_scale, year_1, year_2 = MinMaxScaler(data, '2017', '2019')

def cluster_(data_scaler):
    '''
    This functions performs kmeans clustering with 3 number dof clusters. 
    Paramater:
        data_scaler : The scaled dataframe
    Return:
        It returns a lablels and cen    
    '''
    num_cluster = 3
    kmeans = KMeans(n_clusters = num_cluster)
    kmeans.fit(data_scaler)
    labels = kmeans.labels_
    cen = np.array(kmeans.cluster_centers_)
    return labels, cen

labels, cen = cluster_(df_scale)

def kmeans_plot(scale, labels, cen): 
    '''
    The function creates kmeans plot of the clustered dataframe
    
    Parameter: 
        scale : The scaled dataframe
        labels : The clustered lablels 
        cen : The clustered centroid
    
    Return : 
        A plot that shows the cluster membership and cluster centres
    '''
    col1 = scale[labels ==0]
    col2 = scale[labels ==1]
    col3 = scale[labels ==2]
    
    labels=['1', '2', '3']
    sns.scatterplot(x = col1[:, 0], y = col1[:, 1], label = [0])
    sns.scatterplot(x = col2[:, 0], y= col2[:, 1], label = [1])
    sns.scatterplot(x = col3[:, 0], y = col3[:, 1], label = [2])
    plt.legend()
    
    plt.scatter(cen[:, 0], cen[:, 1], s =5, marker = '+', color = 'black')
    plt.title('Cereal yield (kg per hectare)')
    

data = pd.read_csv('API_19_DS2_en_csv_v2_4773766.csv', skiprows=4)
print(data)

# Set indicator
data = data.loc[data['Indicator Name'] == 'Cereal yield (kg per hectare)']

# Clean data
hec_world = data[data['Country Name'] == 'World']
hec_world = hec_world.dropna(axis=1)
hec_world=hec_world.T
hec_world.columns = ['Cereal yield (kg per hectare)']
hec_world = hec_world.iloc[4:]
hec_world = hec_world.reset_index()
hec_world = hec_world.rename(columns={"index": "Year"})
print(hec_world.head())

# Plot the graph
hec_world.plot("Year", 'Cereal yield (kg per hectare)')
plt.title('Cereal yield (kg per hectare) vs Year')
plt.xlabel("Year")
plt.ylabel('Cereal yield (kg per hectare)')
plt.tight_layout;

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f

hec_world.info()

hec_world["Year"] = pd.to_numeric(hec_world["Year"])

param, covar = opt.curve_fit(exponential, hec_world["Year"], 
                             hec_world['Cereal yield (kg per hectare)'],
                             p0=(1482.05609, 0.03))

hec_world["fit"] = exponential(hec_world["Year"], *param)
hec_world.plot("Year", ['Cereal yield (kg per hectare)', "fit"])
plt.title('Cereal yield (kg per hectare) vs Year')
plt.xlabel("Year")
plt.ylabel('Cereal yield (kg per hectare)')
plt.show();

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f


hec_world['Cereal yield (kg per hectare)'] = pd.to_numeric(hec_world['Cereal yield (kg per hectare)'])


param, covar = opt.curve_fit(logistic, hec_world["Year"], hec_world['Cereal yield (kg per hectare)'],
                             p0=(1482.05609, 0.05, hec_world["Year"].min()))

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
hec_world["fit"] = logistic(hec_world["Year"], *param)
hec_world.plot("Year", ['Cereal yield (kg per hectare)', "fit"])
plt.title('Cereal yield (kg per hectare) vs Year')
plt.ylabel('Cereal yield (kg per hectare)')
plt.show();

year = np.arange(hec_world["Year"].min(), 2036)
print(year)

forecast = logistic(year, *param)

plt.figure()
plt.plot(hec_world["Year"], hec_world['Cereal yield (kg per hectare)'], label='Cereal yield (kg per hectare)')
plt.plot(year, forecast, label="forecast")

plt.title('Cereal yield (kg per hectare) vs Year')
plt.xlabel("Year")
plt.ylabel('Cereal yield (kg per hectare)')
plt.legend()
plt.show();

low, up = err.err_ranges(year, logistic, param, sigma)

plt.figure()
plt.plot(hec_world["Year"], hec_world['Cereal yield (kg per hectare)'], label='Cereal yield (kg per hectare)')
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.title('Cereal yield (kg per hectare) vs Year')
plt.xlabel("Year")
plt.ylabel('Cereal yield (kg per hectare)')
plt.legend()
plt.show;

print(err.err_ranges(2035, logistic, param, sigma))