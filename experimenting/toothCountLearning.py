import os

os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools")
import plyFunctions as pf
import pandas as pd
import re
import pyvista as pv
import numpy as np
trainPath = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train"
os.chdir(trainPath)


#i think an issue we will run into here is that all of the scans have 16000 faces...
#i think this was made this way intentionally
#lets try this anyway. First step is to make a table of the results by number
#of teeth




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
dat
dat["anyNaTeeth"].mean()
    
dat.groupby("numTeeth").agg(
    mean = ("verts", "mean"),
    sd = ("verts", "std"),
    n = ("verts", "count")
    ).reset_index()
#not sure there is enought here to categorize anything


#pf.readAndPlot("095_U.ply", arch = "U")

print(dat.to_string())
#028L, 035L, 036L, 038L, 045L, 045U, 046L, 053L, 065U, 071U, 087L all have NA teeth categorizations
pf.readAndPlot("028_L.ply", "L") #scan shows 14 colored teeth, report says 15...
pf.readAndPlot("035_L.ply", "L") #scan shows 14 colored teeth, report says 15...
pf.readAndPlot("036_L.ply", "L") #scan shows 14 colored teeth, report says 15...
pf.readAndPlot("045_U.ply", "U") #scan shows 14 colored teeth, report says 15...
pf.readAndPlot("065_U.ply", "U") #scan shows 12 colored teeth, report says 13...
#so what is happening here? there is a mystery tooth being added.
#ideas
#perhaps the gum is slightly different color? perhaps there is a very small portion
#that is accidentally been colored as a not present tooth
#the first thing to do is identify where the color is that is being toothed as NA



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


#first lets look at 028L
pat28L = pf.readAndFormat("028_L.ply", "L")
pat28LNaFace = pat28L["face"][pat28L["face"]["toothNum"].isna()]
pat28LNaFace["color"].unique() #this is like a light green
plotIssue(pat28L)
#lets look at 035L
pat35L = pf.readAndFormat("035_L.ply", "L")
pat35LNaFace = pat35L["face"][pat35L["face"]["toothNum"].isna()]
pat35LNaFace["color"].unique() #this is that same color, effects 6 faces
plotIssue(pat35L)
#look at an upper, 045U
pat45U = pf.readAndFormat("045_U.ply", "U")
pat45UNaFace = pat45U["face"][pat45U["face"]["toothNum"].isna()]
pat45UNaFace["color"].unique() #this is a more standard kelly green, effects 5 faces
plotIssue(pat45U)
#look at another upper, 065U
pat65U = pf.readAndFormat("065_U.ply", "U")
pat65UNaFace = pat65U["face"][pat65U["face"]["toothNum"].isna()]
pat65UNaFace["color"].unique() #this is a sort of orange, affects 1 face
plotIssue(pat65U)
#028L, 035L, 036L, 038L, 045L, 045U, 046L, 053L, 065U, 071U, 087L all have NA teeth categorizations

#quick looks
plotIssue(pf.readAndFormat("036_L.ply", "L"))
plotIssue(pf.readAndFormat("038_L.ply", "L"))
plotIssue(pf.readAndFormat("045_L.ply", "L"))
plotIssue(pf.readAndFormat("046_L.ply", "L"))
plotIssue(pf.readAndFormat("053_L.ply", "L"))#this is  really good example of it being a border issue
plotIssue(pf.readAndFormat("071_U.ply", "U"))
plotIssue(pf.readAndFormat("087_L.ply", "L"))

#one possible thing could be switched wires from the top and bottom. are these colors
#that we are observing as odd ones out actually mislabels as top or bottom teeth?
#perhaps not, most of these sets are very small

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

#color data frames for upper and lower jaw
uNumCol = pd.DataFrame(
    {
     "toothNum": ["16","15","14","13","12","11","10","3","2","1","4","9","5","6","8","7","gum"], 
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




#are the Na colors in the lower jaw actually colors of the upper jaw?
archSwitchL = pd.DataFrame(pd.NA, index=range(len(lowerNaColorsU)), columns=["color", "inOtherArch"])
for i in range(len(lowerNaColorsU)):
    archSwitchL.iloc[i,0] = lowerNaColorsU[i]
    archSwitchL.iloc[i,1] = np.where((lowerNaColorsU[i] == uNumCol["color"]).any(), True, False)
archSwitchL

#are the Na colors in the upper jaw actually colors in the lower jaw?
archSwitchU = pd.DataFrame(pd.NA, index=range(len(upperNaColorsU)), columns=["color", "inOtherArch"])
for i in range(len(upperNaColorsU)):
    archSwitchU.iloc[i,0] = upperNaColorsU[i]
    archSwitchU.iloc[i,1] = np.where((upperNaColorsU[i] == lNumCol["color"]).any(), True, False)
archSwitchU



#another intereseting idea for determining missing teeth would be quantifying
#the distance between teeth. we will see large gaps where a tooth is not there
#probably

#also all the function that are now taking a arch input should just extract that 
#information from the file name



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
dat



#for any file that says there are NA teeth, we want to match that NA color to its
#closest color, not spatially, just in actual color
#we will write a function that subsets the face data to only the faces with the NA
#color and assign it to the color that is closest, since color is given by RGB
#we can probably just L2 norm this

pat36L = pf.readAndFormat("036_L.ply", "L")
pat36LNaFace = pat36L["face"][pat36L["face"]["toothNum"].isna()]

pat36LNaGroups = {g: df for g, df in pat36LNaFace.groupby("color")}
#for each of these groups, we want to extract the color values and compare them to 
#each of the colors for their arch
#lets first write the function that compares an incoming color to all colors of
#its arch



groupi = list(pat36LNaGroups.values())[1]
groupiCol = groupi.iloc[0,range(1,4)]
groupiArch = groupi["arch"].iloc[0]

#color mappings for upper teeth
uNumCol = pd.DataFrame(
    {
     "toothNum": ["16","15","14","13","12","11","10","3","2","1","4","9","5","6","8","7","gum"], 
     "color": ['155-048-255', '255-099-071', '255-211-155','131-111-255','255-106-106',
               '060-179-113', '255-246-143', '255-000-255', '030-144-255', '000-255-127',
               '000-255-255', '127-255-000', '255-255-000', '000-255-000', '255-000-000',
               '000-000-255', '255-255-255']
    }
    )
uNumCol = uNumCol.assign(
    red = uNumCol["color"].str.extract(r"(^[0-9]{3})"),
    green = uNumCol["color"].str.extract(r"^[0-9]{3}-([0-9]{3})-[0-9]{3}$"),
    blue = uNumCol["color"].str.extract(r"([0-9]{3}$)")
    )

#color mappings for lower teeth
lNumCol = pd.DataFrame(
    {
    "toothNum": ["25","24","26","23","27","22","28","21","29","20","30","19","31","18","32","17","gum"],
    "color": ['139-000-000', '255-048-048', '144-238-144', '000-191-255', '000-139-139',
              '255-165-000', '000-000-139','202-255-112', '139-000-139', '200-255-255',
              '255-105-180', '255-228-255',  '230-230-250', '255-155-255', '255-228-181',
              '255-069-000', '255-255-255']
    }
    )
lNumCol = lNumCol.assign(
    red = lNumCol["color"].str.extract(r"(^[0-9]{3})"),
    green = lNumCol["color"].str.extract(r"^[0-9]{3}-([0-9]{3})-[0-9]{3}$"),
    blue = lNumCol["color"].str.extract(r"([0-9]{3}$)")
    )


if groupiArch == "lower":
    #initialize a norm holder in color dataframe
    uNumCol["diffNorm"] = pd.NA
    for j in range(len(uNumCol)):
        #extract color from mapping df
        colj = lNumCol.iloc[j, range(2,5)]
        #these are strings rn, switch to numeric
        groupiCol = pd.to_numeric(groupiCol)
        colj = pd.to_numeric(colj)
        #find the difference vector
        #this does not work...
        diffVec = groupiCol - colj
        #take the l2 norm
        uNumCol["diffNorm"].iloc[j] = np.linalg.norm(diffVec, ord = 2)
elif groupiArch == "upper":
    #code
    1
else:
    ValueError("data does not have an arch column taking values 'lower' and 'upper'")


#which tooth "should" it be
uNumCol["diffNorm"].idxmin()
#then overwrite the color






###############################################################################
#lets functionize this a bit at a time so its not impossible at the end
dat=pf.readAndFormat("036_L.ply", "L")
#dat will take in an object created by pf.readAndFormat()

def colorCleaner(dat):
    #extract if this is an upper or a lower arch
    arch = dat["face"]["arch"].iloc[0] #just taking the first one as they will all be the same
  #extract the faces that have missing toothNum
    naFaces = dat["face"][dat["face"]["toothNum"].isna()]
    
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
        properNum = numCol["diffNorm"].idxmin()
        
        #overwrite the color and the tooth number in the original data and export it
        
        
    return 1
        
    
################################################################################    
    






#checking tpyes
type(groupiCol.iloc[0])
type(colj.iloc[0])
groupiCol.dtype == "object" and all(isinstance(x, str) for x in groupiCol)
pd.api.types.is_bool_dtype(groupiCol)






