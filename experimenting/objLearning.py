import os
os.getcwd()
os.chdir("H:\\schoolFiles\\dissertation\\teeth3DS\\data_part_1\\lower\\patient2")

import pyvista as pv
#both of the following ways work
#the first one just explicitly declares we are working with an obj file
#were as i believe the second one will just get the appropraite read for the file type
p2Read = pv.OBJReader("patient2_lower.obj")
p2Mesh = p2Read.read()
p2Mesh.plot()
# 


#now a question here is can I get it into the same form that I have been playing
#with for the other scans
#another important aspect to this will be removing the base added onto the scan
#to keep things uniform across different data sources. one idea for this would 
#be to remove points that have "too long" of connection lines. see the image with
#the lines turned on to see what i mean. all of the lines in the artifical base 
#are very long. additionally, another good cleaning step is to take only the larges
#component.


#in an obj file there are many different aspects: https://en.wikipedia.org/wiki/Wavefront_.obj_file
#but all we will really care about is the faces and indicies

import trimesh
import pandas as pd

p2TriMesh = trimesh.load("patient2_lower.obj", process = False)
dir(p2TriMesh)
#most of the time, an edge is in two different triangles (bc its a boundary for both)
#the object.edges_unique only displays an edge once even if it is in two triangles
#object.edges_unique_length are the lengths corresponding to those unique edges
#if we want to remove the artificial base, lets look at a histogram of those lengths 
#to see if we want to impliment a cutoff
p2TriMesh.edges_unique_length
import matplotlib.pyplot as plt
plt.hist(p2TriMesh.edges_unique_length, bins=1000, range=(0,1))
#basesd on just a quick pass, it seems that lengths over 0.8 are overly large
#perhaps this is where the artificial base stuff comes in, this value may take 
#some playing with. additionally, if all of the models are not scaled appropriately,
#this may vary across scan
#make a list of all points that are involved in these edges and exclude them




p2TriMesh.edges_unique
# Convert to pandas DataFrames
p2TriMeshVert = pd.DataFrame(p2TriMesh.vertices, columns=['x', 'y', 'z'])
p2TriMeshFace = pd.DataFrame(p2TriMesh.faces, columns=['v1', 'v2', 'v3'])









#another option
import meshio
pat2Meshio = meshio.read("patient2_lower.obj")
dir(pat2Meshio)
pat2MeshioVert = pd.DataFrame(pat2Meshio.points, columns=['x', 'y', 'z'])
pat2MeshioFace = pd.DataFrame(pat2Meshio.cells_dict["triangle"], columns=['v1', 'v2', 'v3'])