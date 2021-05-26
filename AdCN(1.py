"""
This implementation is for the classifier based on K-means + Flow Duration
"""

import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math


pl= {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]} #, '6':[], '7':[],'8':[],'9':[]}   #packet lengths
pl_centroid={'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]} #, '6':[], '7':[],'8':[],'9':[]}  #for computing centroid of the lateset flow
#pl_test = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[], '6':[], '7':[],'8':[],'9':[]}
pl_test=[]
duration= { '48-62':[], '8-12':[], '0-4':[], '65-117':[]}
        #Youtube  ,   GSearch ,  GMusic,   #GDrive#,  GDoc



packet_lengths=dict()
global centroids; global tt; tt=[]; centroids=[]
global online; online=0
global centroidss

def set_bandwidth(label):
    #print(" in set bandwidth", len(label))
    i=-30
    for k in list(duration):
        i+=30
        for j in range(30):
            duration[k].append(label[i+j])

def parse(path):
    f=open(path, "r")
    for i in range(6):
        t=f.readline()
        t=t.split()
        pl["%s" %i].append(t[2])
        pl_centroid["%s" %i].append(t[2])
    f.close()
    #print("pl for centroid" ,pl_centroid)

def parse2(path):
    f=open(path, "r")
    for i in range(6):
        t=f.readline()
        t=t.split()
        pl_test.append(int(t[2]))
    t=f.readlines()
    f.close()
    #print(t[len(t)-1])
    t=t[len(t)-1].split()
    duratn= int(float(t[1]))
    return duratn

def generate_centroids():
    #Mean of the given packet lengths of an app
    for k in sorted(pl_centroid):
        t=0
        for i in range(len(pl_centroid[k])):
            t+= int(pl_centroid[k][int(i)])
        t=t//30
        tt.append(t)
    centroids.append(tt)

def cluster():
        df = pd.DataFrame(pl)
        #centroids.append([895,85,304,193,1412,100,1412,100,1412,100]) #GDoc
        #centroids.append([632,112,74,74,781,112,74,374,74, 1162])   #GDrive, GSearch
        #centroids.append([])

        centr=np.array(centroids, np.float64)
        kmeans = KMeans(n_clusters= 4, init=centr, n_init=1, max_iter=1, algorithm='elkan')
        kmeans.fit(df)
        label= kmeans.labels_

        print(kmeans.labels_)
        print(kmeans.cluster_centers_)
        centroidss=np.array(kmeans.cluster_centers_)

        set_bandwidth(label)   #adds clutsr to the bandwodth
         #print(df.shape)
        #print(label)
        #u_labels = np.unique(label)
        #print(u_labels)
        # plotting the results:
        #for i in u_labels:
            #plt.scatter(df[0, label == i], df[1, label == i], label=i)
        #plt.legend()
        #plt.show()

        #plt.scatter(df['x'].values, df['y'].values, c=kmeans.labels_.astype(float), s=50, alpha=0.5)
        #plt.scatter(centroidss[:, 0], centroidss[:, 1],marker='s', color ='red', s=50)
        #plt.show()
        return kmeans, centroidss

def classify(kmeans, centroidss):



        l=kmeans.predict([pl_test,[1, 1, 1, 1, 1, 1]])
        ED=[]
        if(len(l)>1):
            #compute eucledian dist from the predicted centroids
            for i in l:
                ED.append(math.sqrt(
                   (pow((pl_test[0] - centroidss[i][0]), 2)) + pow((pl_test[1] - centroidss[i][1]), 2) +
                    pow((pl_test[2] - centroidss[i][2]), 2) + pow((pl_test[3] - centroidss[i][3]), 2) +
                    pow((pl_test[4] - centroidss[i][4]), 2) + pow((pl_test[5] - centroidss[i][5]), 2)))
                    #pow((pl_test[6] - centroidss[i][6]), 2) + pow((pl_test[7] - centroidss[i][7]), 2) +
                    #pow((pl_test[8] - centroidss[i][8]), 2) + pow((pl_test[9] - centroidss[i][9]), 2))
        #print(l, ED)
        cl= ED.index(min(ED))
        if( 48 <= duratn <= 62):
            ll= duration['48-62']
            if l[cl] in ll:
                print("Application is Youtube in cluster ", l[cl])
        elif( 8 <= duratn <= 12):
            ll=duration['8-12']
            if l[cl] in ll:
                print("Application is Google Search in cluster ", l[cl])

        elif( 0 <= duratn <= 4):
            ll=duration['0-4']
            if l[cl] in ll:
                print("Applicaiton is Google Music in cluster ", l[cl])
        elif(65 <= duratn <= 117):
            ll=duration['65-117']
            if l[cl] in ll:
                print("Application is Google Doc in culuster ", l[cl])


if __name__ == '__main__':
    #Read
    #Parse
    #Cluster
    #Classify
    s='/home/zilmarij/Downloads/Dataset/'
    for i in range(30):   #15 files to take data from
        path= s + "Youtube-" + "%s" %i +'.txt'
        parse(path)
    generate_centroids()   #call after each apps parsing
    pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []} # '6': [], '7': [], '8': [], '9': []}

    tt=[]
    for i in range(30):
        path= s + "GoogleSearch-" + "%s" %i + '.txt'
        parse(path)
    generate_centroids()
    pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []} #'6': [], '7': [], '8': [], '9': []}

    tt=[]
    for i in range(30):
        path= s+ "GoogleMusic-" + "%s" %i + ".txt"    #shows fluctuating behav
        parse(path)
    generate_centroids()
    pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []} # '6': [], '7': [], '8': [], '9': []}

    #tt = []
    #for i in range(15):
     #   path = s + "GoogleSearch-" + "%s" % i + '.txt'
      #  parse(path)
    #generate_centroids()
    #pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    #tt = []
    #for i in range(20):
     #   path = s + "GoogleDrive-" + "%s" % i + '.txt'
      #  parse(path)
    #generate_centroids()
    #pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    tt = []
    for i in range(30):
        path = s + "GoogleDoc-" + "%s" % i + '.txt'
        parse(path)
    generate_centroids()
    pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []} #, '6': [], '7': [], '8': [], '9': []}

    kmeans, centroidss=cluster()

    online=1

    path= "/home/zilmarij/Downloads/Testing_Data"
    for i in range(11):
        pl_test=[]
        path = s + "Youtube-" + "%s" %i+ ".txt"
        duratn = parse2(path)
        #duratn = int(float(duratn))
        classify(kmeans, centroidss)

    for i in range(11):
        pl_test = []
        path = s + "GoogleSearch-" + "%s" %i+ ".txt"
        duratn = parse2(path)
        #duratn = int(float(duratn))
        #print(pl_test, duratn)
        classify(kmeans, centroidss)

    pl_test = []
    for i in range(11):
        pl_test = []
        path = s + "GoogleMusic-" + "%s" %i+ ".txt"
        duratn = parse2(path)
        duratn = int(float(duratn))
        #print(" pl test and duration ", pl_test, duratn)
        classify(kmeans, centroidss)

    pl_test = []
    for i in range(11):
        pl_test = []
        path = s + "GoogleDoc-" + "%s" %i+ ".txt"
        duratn = parse2(path)
        duratn = int(float(duratn))
        #print(pl_test, duratn)
        classify(kmeans, centroidss)
