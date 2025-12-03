import os

os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools")
import plyFunctions as pf
import pandas as pd
import re
import pyvista as pv
import numpy as np
trainPath = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train"
os.chdir(trainPath)



def plotIssue(pat):
    #making a copy just for saftey
    patC = pat.copy()
    
    #get original surface
    patSurf = pf.giveSurf(face = patC["face"], vertex = patC["vert"])
    
    #identify problem faces
    patNaFace = patC["face"][patC["face"]["toothNum"].isna()]
    
    #get the points associated with the coloring
    #the explode function essentially flattens this column of lists and set keeps
    #only the unique values. made to a list for easy usage
    patVertInd = list(set(patNaFace["vertex_indices"].explode()))
    
    #get the points in the original scan that correspond to these indicies
    patVert = patC["vert"].iloc[patVertInd,]
    
    #plot these on top of the mesh
    plot1 = pv.Plotter()
    plot1.add_mesh(patSurf, scalars = "rgba", rgb = True)
    plot1.add_points(np.array(patVert.iloc[:,range(0,3)]),
                        color = "black", point_size=10,
                        render_points_as_spheres=True)
    return plot1.show()



def naColorExtract(pat):
    #copy just in case
    patC = pat.copy()
    patCNaFace = patC["face"][patC["face"]["toothNum"].isna()]
    return patCNaFace["color"].unique()

#028L, 035L, 036L, 038L, 045L, 045U, 046L, 053L, 065U, 071U, 087L all have NA teeth categorizations
#list of unique NA colors for lower jaw
lowerNaColors = [naColorExtract(pf.readAndFormat("028_L.ply", "L")),
                 naColorExtract(pf.readAndFormat("035_L.ply", "L")),
                 naColorExtract(pf.readAndFormat("036_L.ply", "L")),
                 naColorExtract(pf.readAndFormat("038_L.ply", "L")),
                 naColorExtract(pf.readAndFormat("045_L.ply", "L")),
                 naColorExtract(pf.readAndFormat("046_L.ply", "L")),
                 naColorExtract(pf.readAndFormat("053_L.ply", "L")),
                 naColorExtract(pf.readAndFormat("087_L.ply", "L"))
                 ]
lowerNaColorsU = list(set(np.concatenate(lowerNaColors)))


#list of unique NA colors for upper jaw
upperNaColors = [naColorExtract(pf.readAndFormat("045_U.ply", "U")),
                 naColorExtract(pf.readAndFormat("065_U.ply", "U")),
                 naColorExtract(pf.readAndFormat("071_U.ply", "U"))
                 ]
upperNaColorsU = list(set(np.concatenate(upperNaColors)))



#pass fileName as a string
def numExtract(fileName):
    #extract patient information
    #the r here means raw, avoids double escaping
    #the function natrually outputs a list, so taking the first (and only) element
    patNum = re.findall(pattern = r"^[0-9]{3}", string = fileName)[0]
    patArch = re.findall(pattern = r"_([A-Z]{1})\.", string = fileName)[0]
    #read in file 
    pat = pf.readAndFormat(fileName, arch = patArch)
    #extract summary information
    numTeeth = len(pat["face"]["toothNum"].unique())-1 #subtract 1 for the gum group
    numVerts = pat["vert"].shape[0]
    numFaces = pat["face"].shape[0] 
    anyNaTeeth = pat["face"]["toothNum"].isna().any()
    return [patNum, patArch, numTeeth, numVerts, numFaces, anyNaTeeth]


#this is like an apply
datList = [numExtract(i) for i in os.listdir(trainPath)]
dat = pd.DataFrame(datList, columns=['pat', 'arch', 'numTeeth', 'verts', 'faces', 'anyNaTeeth'])
print(dat.to_string())


###############################################################################
#lets functionize this a bit at a time so its not impossible at the end
#dat will take in an object created by pf.readAndFormat()
#outputs the same dat with the color and tooth number corrected
def colorCleaner(dat):
    #make a copy of the data for safe keeping
    datC = dat.copy()
    
    #extract if this is an upper or a lower arch
    arch = datC["face"]["arch"].iloc[0] #just taking the first one as they will all be the same
  #extract the faces that have missing toothNum
    naFaces = datC["face"][datC["face"]["toothNum"].isna()]
    
    #if there is no missing toothNum, this function should stop
    if len(naFaces.index) == 0:
        raise Exception("No missing faces missing a toothNum value")
    
    #create a dictionary with datasets for each color that has a missing toothNum
    #this will often be just a single 
    naFacesGrp = {g: df for g, df in naFaces.groupby("color")}
    
    #get the color and tooth number data frames
    if arch == "upper":
        numCol = pf.colorNumFrame("U")
    elif arch == "lower":
        numCol = pf.colorNumFrame("L")
    else:
        raise Exception("The arch column in the face data is not 'lower' or 'upper' and it must be one of these two things")
    
    
    
    #for each of these groups in the dictionary, we want to extract the color values and compare them to 
    #each of the colors for their arch
    for i in range(len(naFacesGrp)):
        
        groupi = list(naFacesGrp.values())[i]
        groupiCol = groupi[["red", "green", "blue"]].iloc[0,:] #just taking the first one as they will all be the same
        groupiCol = pd.to_numeric(groupiCol) #these are strings, switch to numeric
        
        #grab the index for these faces in the face data
        naInd = groupi.index
        
        #check against each color that should be in this arch
        #initialize a norm holder in color dataframe
        numCol["diffNorm"] = pd.NA
        for j in range(len(numCol)):
            #extract color from mapping df
            colj = numCol[["red", "green", "blue"]].iloc[j,:]
            #find the difference vector
            #this does not work...
            diffVec = groupiCol - colj
            #take the l2 norm
            numCol.loc[j, "diffNorm"] = np.linalg.norm(diffVec, ord = 2)
        
        
        #which tooth "should" it be
        corInd = numCol["diffNorm"].idxmin()
        corNum = numCol["toothNum"].iloc[corInd]
        corCol = numCol["color"].iloc[corInd]
        #overwrite the color and the tooth number in the original data and export it
        datC["face"].loc[naInd, "toothNum"] = corNum
        datC["face"].loc[naInd, "color"] = corCol
        
        
    return datC


dat=pf.readAndFormat("036_L.ply", "L")
plotIssue(dat)
dat = colorCleaner(dat)
pf.plotPly(dat["face"], dat["vert"])


#pass fileName as a string
def numExtract(fileName):
    #extract patient information
    #the r here means raw, avoids double escaping
    #the function natrually outputs a list, so taking the first (and only) element
    patNum = re.findall(pattern = r"^[0-9]{3}", string = fileName)[0]
    patArch = re.findall(pattern = r"_([A-Z]{1})\.", string = fileName)[0]
    #read in file 
    pat = pf.readAndFormat(fileName, arch = patArch)
    #extract summary information
    numTeeth = len(pat["face"]["toothNum"].unique())-1 #subtract 1 for the gum group
    numVerts = pat["vert"].shape[0]
    numFaces = pat["face"].shape[0] 
    anyNaTeeth = pat["face"]["toothNum"].isna().any()
    return [patNum, patArch, numTeeth, numVerts, numFaces, anyNaTeeth]


#this is like an apply
datList = [numExtract(i) for i in os.listdir(trainPath)]
dat = pd.DataFrame(datList, columns=['pat', 'arch', 'numTeeth', 'verts', 'faces', 'anyNaTeeth'])
print(dat.to_string())





################################################################################    
    










