import os
os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools")
import plyFunctions as pf
os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools\\colorClean")
import colorCleanFuns as cc
trainDir = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train"
testDir = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\test"
import pandas as pd


# step 1, find which files need to have color cleaning
#for training data
os.chdir(trainDir)
trainDiscrip = pd.DataFrame([cc.numExtract(i) for i in os.listdir(trainDir)],
             columns=['fileName', 'pat', 'arch', 'numTeeth', 'verts', 'faces', 'anyNaTeeth'])
trainNas = trainDiscrip[trainDiscrip["anyNaTeeth"] == True]

#for test data
os.chdir(testDir)
testDiscrip =  pd.DataFrame([cc.numExtract(i) for i in os.listdir(testDir)],
             columns=['fileName', 'pat', 'arch', 'numTeeth', 'verts', 'faces', 'anyNaTeeth'])
testNas = testDiscrip[testDiscrip["anyNaTeeth"] == True]

# step 2, clean all of those files

def readAndClean(fileName, arch):
    dat = pf.readAndFormat(fileName, arch)
    datClean = cc.colorCleaner(dat)
    return datClean

#dont want to figure out how to do an apply with multiple args rn so using this
def cleanApplyer(dfNas):
    #create dicionary to hold objects
    outHolder = {}
    for i in range(len(dfNas)):
        #name dictionary element and give it the cleaned data
        outHolder[dfNas["fileName"].iloc[i]] = readAndClean(
            dfNas["fileName"].iloc[i],
            dfNas["arch"].iloc[i]
            )
    return outHolder

os.chdir(trainDir)
cleanTrain = cleanApplyer(trainNas)
os.chdir(testDir)
cleanTest = cleanApplyer(testNas)







#
#now just gotta figure out how to export them lol
#also check each of the cleaned files to make sure things actually went right
#









# step 3, export all of those files


#note that this will need to be exported and have the same exact format as the ones
#that are read in. That means getting rid of some of the stuff that we added
#for exporting, see this example from chatgpt
from plyfile import PlyData, PlyElement
import numpy as np

vertex_array = np.array(
    list(points_df[["x","y","z"]].itertuples(index=False, name=None)),
    dtype=[ ("x","f4"), ("y","f4"), ("z","f4") ]
)

# Build faces
faces_list = []
for _, row in faces_df.iterrows():
    faces_list.append( ([int(v) for v in row.tolist()],) )

faces_array = np.array(
    faces_list,
    dtype=[ ('vertex_indices', 'i4', (len(faces_df.columns),)) ]
)

ply = PlyData([PlyElement.describe(vertex_array, "vertex"),
               PlyElement.describe(faces_array, "face")])

ply.write("output.ply")


# step 4, move all of the other files over
