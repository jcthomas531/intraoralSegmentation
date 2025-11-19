import os

os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools")
import plyFunctions as pf
import pandas as pd
import re
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
#the first thing to do is identify what the color is that is being toothed as NA
#first lets look at 028L
pat28L = pf.readAndFormat("028_L.ply", "L")
pat28LNaFace = pat28L["face"][pat28L["face"]["toothNum"].isna()]
pat28LNaFace["color"].unique() #this is a lime green color, effects 14 faces

#lets look at 035L
pat35L = pf.readAndFormat("035_L.ply", "L")
pat35LNaFace = pat35L["face"][pat35L["face"]["toothNum"].isna()]
pat35LNaFace["color"].unique() #this is that same color, effects 6 faces


#look at an upper, 045U
pat45U = pf.readAndFormat("045_U.ply", "U")
pat45UNaFace = pat45U["face"][pat45U["face"]["toothNum"].isna()]
pat45UNaFace["color"].unique() #this is a more standard kelly green, effects 5 faces

#look at another upper, 065U
pat65U = pf.readAndFormat("065_U.ply", "U")
pat65UNaFace = pat65U["face"][pat65U["face"]["toothNum"].isna()]
pat65UNaFace["color"].unique() #this is a sort of orange, affects 1 face

#one possible thing could be switched wires from the top and bottom. are these colors
#that we are observing as odd ones out actually mislabels as top or bottom teeth?
#perhaps not, most of these sets are very small



#####################################################################################################
#lets plot the points that at assoiated with these faces to see where they are
#########################################################################################################


#another intereseting idea for determining missing teeth would be quantifying
#the distance between teeth. we will see large gaps where a tooth is not there
#probably

#also all the function that are now taking a arch input should just extract that 
#information from the file name



