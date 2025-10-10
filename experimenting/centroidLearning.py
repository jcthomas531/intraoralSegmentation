
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
#now lets go through each unique color and make all other colors than it white
#so that we can see which tooth its identifying
#starting with lower jaw
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










