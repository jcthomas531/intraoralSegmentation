
#exploring centroids and measures between teeth
#i am going to work using mostly the annotated training data just to get an idea
#of how these methods work. Once we make better predictions, these things will
#translate

import os
from  plyfile import PlyData
import pandas as pd
import numpy as np
toolsPath = "H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools\\"
dataPath = "H:\\schoolFiles\\dissertation\\intraoralSegmentation\\fastTgcn\\data\\"
os.chdir(toolsPath)
import plyFunctions as pf


#train patient 76 has all colors
dataPath = "H:\\schoolFiles\\dissertation\\intraoralSegmentation\\fastTgcn\\data\\"
#lower
os.chdir(dataPath+"train-Lall")
l76 = PlyData.read("076_L.ply")
l76Vert = pd.DataFrame(l76["vertex"].data)
l76Face = pd.DataFrame(l76["face"].data)
# pf.plotPly(face = l76Face, vertex = l76Vert)
#upper
os.chdir(dataPath+"Train-U")
u76 = PlyData.read("076_U.ply")
u76Vert = pd.DataFrame(u76["vertex"].data)
u76Face = pd.DataFrame(u76["face"].data)
# pf.plotPly(face = u76Face, vertex = u76Vert)
#to make centroids, we must uniquely identify each tooth, this is done in the 
#data by color which is stored across 3 variables
l76Face["color"] = (l76Face["red"].astype(str).str.zfill(3) + "-" +
                     l76Face["green"].astype(str).str.zfill(3) + "-" +
                     l76Face["blue"].astype(str).str.zfill(3))
u76Face["color"] = (u76Face["red"].astype(str).str.zfill(3) + "-" +
                     u76Face["green"].astype(str).str.zfill(3) + "-" +
                     u76Face["blue"].astype(str).str.zfill(3))

#now lets go through each unique color and make all other colors than it white
#so that we can see which tooth its identifying
#function to do so
def standOut(face, vertex, toothCol):
    #make copies of the dataframes so you dont edit in place
    faceC = face.copy()
    vertexC = vertex.copy()
    #make all colors besides the one we are looking at white
    faceC["red"] = np.where(faceC["color"] == toothCol, faceC["red"], 255)
    faceC["green"] = np.where(faceC["color"] == toothCol, faceC["green"], 255)
    faceC["blue"] = np.where(faceC["color"] == toothCol, faceC["blue"], 255)
    return pf.plotPly(face = faceC, vertex = vertexC)
#starting with lower jaw
#checking all the colors
lCols = l76Face["color"].unique()
standOut(l76Face, l76Vert, lCols[16])
lCols[16]
#dictonary of colors
#'139-000-000' = 25
#'255-048-048' = 24
#'144-238-144' = 26
# '000-191-255'= 23
# '000-139-139'= 27
# '255-165-000'= 22
# '000-000-139'= 28
# '202-255-112' = 21
# '139-000-139'= 29
# '200-255-255'= 20
# '255-105-180'= 30
# '255-228-255' = 19
#  '230-230-250' = 31
# '255-155-255' = 18
# '255-228-181' = 32
# '255-069-000' = 17


#upper jaw
uCols = u76Face["color"].unique()
standOut(u76Face, u76Vert, uCols[16])
uCols[16]
#16 = '155-048-255'
#15 = '255-099-071'
#14 = '255-211-155'
#13='131-111-255'
#12='255-106-106'
#11='060-179-113'
#10='255-246-143'
#3='255-000-255'
#2='030-144-255'
#1='000-255-127'
#4='000-255-255'
#9='127-255-000'
#5='255-255-000'
#6='000-255-000'
#8='255-000-000'
#7='000-000-255'






#data frames associating tooth number with color
uNumCol = pd.DataFrame(
    {
     "toothNum": ["16","15","14","13","12","11","10","3","2","1","4","9","5","6","8","4","gum"],
     "color": ['155-048-255', '255-099-071', '255-211-155','131-111-255','255-106-106',
               '060-179-113', '255-246-143', '255-000-255', '030-144-255', '000-255-127',
               '000-255-255', '127-255-000', '255-255-000', '000-255-000', '255-000-000',
               '000-000-255', '255-255-255']
    }
    )

lNumCol = pd.DataFrame(
    {
    "toothNum": ["25","24","26","23","27","22","28","21","29","20","30","19","31","18","32","17","gum"],
    "color": ['139-000-000', '255-048-048', '144-238-144', '000-191-255', '000-139-139',
              '255-165-000', '000-000-139','202-255-112', '139-000-139', '200-255-255',
              '255-105-180', '255-228-255',  '230-230-250', '255-155-255', '255-228-181',
              '255-069-000', '255-255-255']
    }
    )


#as a double check, lets make sure that none of the colors are the same in the
a = uNumCol["color"].isin(lNumCol["color"])
#only the gums color is in both

aa = a.sum()
aa



#create a function that reads in a ply file and formats in how I like
#this will require PlyData and pandas package
def plyRead(file):
    #read in the object
    plyObject = PlyData.read(file)
    #get the vertex and face data
    plyVert = pd.DataFrame(plyObject["vertex"].data)
    plyFace = pd.DataFrame(plyObject["face"].data)
    #create new variable in face data for tooth color that concats RGB vals
    plyFace["color"] = (plyFace["red"].astype(str).str.zfill(3) + "-" +
                         plyFace["green"].astype(str).str.zfill(3) + "-" +
                         plyFace["blue"].astype(str).str.zfill(3))
    #return a dictionary of the vertex and face information
    #this dictionary object seems like a named list in R
    return {"vert": plyVert, "face": plyFace}




os.chdir(dataPath+"Train-U")
t1 = plyRead("076_U.ply")
t1["vert"]
t1["face"]
pf.plotPly(face = t1["face"], vertex = t1["vert"])
standOut(vertex=t1["vert"], face = t1["face"], toothCol="155-048-255")





#function that adds tooth number to the face data frame
#this takes a face data frame that has been set up using plyRead
def toothNum(face):
    #make copies of the dataframes so you dont edit in place
    faceC = face.copy()
    #color and tooth number associations
    uNumCol = pd.DataFrame(
        {
         "toothNum": ["16","15","14","13","12","11","10","3","2","1","4","9","5","6","8","4","gum"],
         "color": ['155-048-255', '255-099-071', '255-211-155','131-111-255','255-106-106',
                   '060-179-113', '255-246-143', '255-000-255', '030-144-255', '000-255-127',
                   '000-255-255', '127-255-000', '255-255-000', '000-255-000', '255-000-000',
                   '000-000-255', '255-255-255']
        }
        )
    lNumCol = pd.DataFrame(
        {
        "toothNum": ["25","24","26","23","27","22","28","21","29","20","30","19","31","18","32","17","gum"],
        "color": ['139-000-000', '255-048-048', '144-238-144', '000-191-255', '000-139-139',
                  '255-165-000', '000-000-139','202-255-112', '139-000-139', '200-255-255',
                  '255-105-180', '255-228-255',  '230-230-250', '255-155-255', '255-228-181',
                  '255-069-000', '255-255-255']
        }
        )
    #identify if we have an upper or lower arch
    #get the count of how many colors in this face data are in the colors of
    #the upper face data frame. This will always be at least 1 bc of the gums
    #if this value is above 1, then we have an upper arch, if this value is 1
    #then we have a lower arch
    upperColCount = uNumCol["color"].isin(faceC["color"]).sum()
    if upperColCount > 1:
        faceC = faceC.merge(uNumCol, on="color", how = "left", validate = "many_to_one")
        faceC["arch"] = "upper"
    elif upperColCount == 1:
        faceC = faceC.merge(lNumCol, on="color", how = "left", validate = "many_to_one")
        faceC["arch"] = "lower"
    else:
        ValueError("could not identify input as upper or lower arch")
    #return updated dataframe
    return faceC


toothNum(t1["face"])["toothNum"].unique()

os.chdir(dataPath+"train-Lall")
t2 = plyRead("076_L.ply")
toothNum(t2["face"])["toothNum"].unique()

pf.plotPly(face = t2["face"], vertex = t2["vert"])



#create a function that takes a data frame of faces, vertices, and tooth number
#and shows that specific tooth


#when you functionize the application of these numbers make sure to check every number








