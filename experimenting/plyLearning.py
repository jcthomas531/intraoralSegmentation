import os
os.getcwd()
os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\fastTgcn\\data\\train-L\\")

#options for basic visualization
#https://docs.pyvista.org/
import pyvista as pv
#read in
mesh61_3 = pv.read("061_L.ply")
#summary of the object
print(mesh61_3)
#the vertices
mesh61_3.points #the points
mesh61_3.point_data #shows additional information about the points
mesh61_3.point_data["Normals"] #normalized version of the points
#the faces
mesh61_3.faces #these are shown as a 1 dimensional object
mesh61_3.cell_data #additional information about the faces
mesh61_3.cell_data["RGBA"] #this is the RGBA data
mesh61_3.cell_data["RGBA-normed"] #not quite sure what this is
#something else
mesh61_3.field_data
#visualize
mesh61_3.plot()
#another option of visualization allowing for more specification
a = pv.Plotter()
a.add_mesh(mesh61_3)
a.show()



#this is great for plotting but what about extracting things from the .ply file
#i want to be about to look at attributes and mess with them directly as might 
#happen in some sort of computer vision model


#specifically i want to know what color (RGB) represents what tooth number
#i am assuming that this is the same across the files, otherwise how would we 
#be segmeneting
#looking at the dataloader.py file, i bet that these are the color assignments
# labels = (
#     # [255, 0, 0],[255, 255, 0], [0, 255, 0], [0, 255, 255],
#     #       [0, 0, 255], [255, 0, 255], [30, 144, 255], [0, 255, 127], [127, 255, 0],
#     #       [255, 246, 143],[60, 179, 113], [255, 106, 106], [131, 111, 255], [255, 211, 155], [255, 99, 71],[155, 48, 255],
#            [255, 48, 48],[0, 191, 255], [255, 165, 0], [202, 255, 112],
#            [200, 255, 255], [255, 228, 255], [255, 155, 255], [255, 69, 0], [139, 0, 0],
#            [144, 238, 144],[0, 139, 139], [0, 0, 139], [139, 0, 139], [255, 105, 180], [230, 230, 250],[255, 228, 181],
#            [255, 255, 255]
#           )


#additionally, as i think about the format of this data and such, it makes me 
#wonder a bit at how this fastTgcn works? Is it more face classificiation rather
#than point?



#an approach for viewing the information in the file more as data and less
#as just something for visualization. Im sure there is some way to take it from 
#this more raw form into a visualization at a later step
from  plyfile import PlyData
import pandas as pd
#read in
dat = PlyData.read("061_L.ply")
dat.elements
#vertex data and making it into a data frame
dat["vertex"]
dat["vertex"].data
verDat = pd.DataFrame(dat["vertex"].data)
verDat.head()
#face data
dat["face"]
dat["face"].data
faceDat = pd.DataFrame(dat["face"].data)
faceDat.head()
#this is really interesting as it keeps the 3 entries descibing which vertices 
#make up the triangle as a single vector (called a list)
#another interesting thing is that dat["face"].data doees not include the value
#3 at the very start saying that three points are involved in the facec
#perhaps that is treated implicitly as the vector of the vertices is of length 3
#still i would have assumed in a more raw look, i would see that number 3
#if i ever needed it, i could just extract the length of that vector
#now that things are in a numpy data frame, i can work with them in a very detailed
#way



#first thing I want to do (just for fun) is to make that list of vertex indices
#into 3 different columns
#the pd.Series i am not sure I understand. it is taking a column of lists and
#turning it into a data frame. The apply function works similar to the apply function
#in R. It is defined for data frames and allows you to choose an axis like R but
#here we are using  the version of it defined for series, so axis has no bearing.
#the join function defaults to joining on index, which is really handy. You can 
#also rename specific columns in place without reassignement which is handy. You 
#could also just reassign but this is nice
vertDf = faceDat["vertex_indices"].apply(pd.Series)
faceDat2 = faceDat.join(vertDf)
faceDat2.rename(columns = {0:"vertex1", 1: "vertex2", 2: "vertex3"}, inplace=True)
faceDat2
#could also use this approach where we take the column, make it into a list, and
#then make that list into a data frame. This is a more direct data frame creation
#approach
a = pd.DataFrame(faceDat["vertex_indices"].tolist(), 
                        columns=["vertex1", "vertex2", "vertex3"])
aa = faceDat.join(a)
aa


#now that we have that, lets see how many unique colors there are in the dataset
#and find out what they are
#concatenation is done with the + sign for strings
#i am using the str.zfill() function to pad with leading zeros to make all the 
#values into 3 digit numbers. if you want to spread this over multiple lines, it 
#must be included in a parenthesis
faceDat2["color"] = (faceDat2["red"].astype(str).str.zfill(3) + "-" +
                     faceDat2["green"].astype(str).str.zfill(3) + "-" +
                     faceDat2["blue"].astype(str).str.zfill(3))
#produce an array of the unique colors
uniqueColors = faceDat2["color"].unique()
#count of the number of faces with each color
colorCounts = faceDat2["color"].value_counts()


#now lets make sure that the colorings are consistent across the training objects
#by bringing in another training set
#we will also need to determine which color represents which tooth






