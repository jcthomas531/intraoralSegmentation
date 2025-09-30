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
#make up the triangle as a single vector
#another interesting thing is that dat["face"].data doees not include the value
#3 at the very start saying that three points are involved in the facec
#perhaps that is treated implicitly as the vector of the vertices is of length 3
#still i would have assumed in a more raw look, i would see that number 3
#if i ever needed it, i could just extract the length of that vector
#now that things are in a numpy data frame, i can work with them in a very detailed
#way
#first thing I want to do (just for fun) is to make that vector of vertex indices
#into 3 different columsn

