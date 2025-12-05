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

# step 3, check all of these changes
cc.plotIssue(dat36L)
dat36L = cc.colorCleaner(dat36L)
pf.plotPly(dat36L["face"], dat36L["vert"])



list(cleanTrain.values())[1]

cc.colorCleaner(list(cleanTrain.values())[1])

#functions that test for errors in the plotting and cleaning functions on the 
#cleaned data, we are looking for these to throw back errors as this means there is 
#nothing more to fix
def plotError(dat):
    try: 
        cc.plotIssue(dat)
    except ValueError:
        return "proper response"
    return "improper response, still color issues"


def cleanError(dat):
    try:
        cc.colorCleaner(dat)
    except Exception:
        return "proper response"
    return "improper response, still color issues"

plotError(list(cleanTrain.values())[1])
len(cleanTrain.values())
len(cleanTrain)
list(cleanTrain.keys())[1]



#checking train data
trainCheck = pd.DataFrame({
    "fileName": [pd.NA]*len(cleanTrain),
    "plotResponse": [pd.NA]*len(cleanTrain),
    "cleanResponse": [pd.NA]*len(cleanTrain)
    })
for i in range(len(trainNas)):
    trainCheck["fileName"].loc[i] = list(cleanTrain.keys())[i]
    trainCheck["plotResponse"].loc[i] =plotError(list(cleanTrain.values())[i])
    trainCheck["cleanResponse"].loc[i] = cleanError(list(cleanTrain.values())[i])
    
    
#checking the test data
testCheck = pd.DataFrame({
    "fileName": [pd.NA]*len(cleanTest),
    "plotResponse": [pd.NA]*len(cleanTest),
    "cleanResponse": [pd.NA]*len(cleanTest)
    })
for i in range(len(testNas)):
    testCheck["fileName"].loc[i] = list(cleanTest.keys())[i]
    testCheck["plotResponse"].loc[i] =plotError(list(cleanTest.values())[i])
    testCheck["cleanResponse"].loc[i] = cleanError(list(cleanTest.values())[i])
    

trainCheck
testCheck

#perfect, everything is how it should be

#
#now just gotta figure out how to export them lol
#also check each of the cleaned files to make sure things actually went right
#
list(cleanTest.values())[1]["face"][["vertex_indices", "red"]]


def faceFormatter(dat):
    datC = dat.copy()
    datC["face"] = datC["face"][["vertex_indices", "red", "green", "blue", "alpha"]]
    return datC





aaa = faceFormatter(list(cleanTest.values())[1])

# step 3, export all of those files


def write_ply(filename, data):
    """
    data["vert"] : pandas DataFrame with columns:
        x, y, z, nx, ny, nz
        
    data["face"] : pandas DataFrame with columns:
        vertex_indices (list of ints), red, green, blue, alpha
    """

    verts = data["vert"]
    faces = data["face"]

    n_verts = len(verts)
    n_faces = len(faces)

    with open(filename, "w") as f:
        # ----- HEADER -----
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")

        # ----- VERTEX DATA -----
        for _, row in verts.iterrows():
            f.write(f"{row.x} {row.y} {row.z} {row.nx} {row.ny} {row.nz}\n")

        # ----- FACE DATA -----
        for _, row in faces.iterrows():
            indices = row.vertex_indices  # must be list-like
            f.write(
                f"{len(indices)} " +
                " ".join(str(int(i)) for i in indices) + " " +
                f"{int(row.red)} {int(row.green)} {int(row.blue)} {int(row.alpha)}\n"
            )



os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\testDir")
write_ply("testPly.ply", aaa)

#this seems to work, check that it can be read in with your functions



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
