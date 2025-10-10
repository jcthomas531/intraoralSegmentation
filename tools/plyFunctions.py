#helpful functions
import pandas as pd
import pyvista as pv



#plot face and vertex data from PlyData
#takes the ply data as data frames so you can manipulate it beforehand
#faces is a df of the faces
#vertices is a df of the vertices
def plotPly(face, vertex):
    
    #copy the dataframes so it doesnt edit in place
    faceC = face.copy()
    vertexC = vertex.copy()
    
    #faces
    #get the number of vertexes for each shape (3 here bc triangles)
    faceC["nVert"] = faceC["vertex_indices"].apply(len)
    #make the vertices for the shape into columns
    faceCExpand = pd.DataFrame(faceC["vertex_indices"].tolist(),
                 columns=["v1", "v2", "v3"])
    faceC = faceC.join(faceCExpand)
    #order the data in the way that pyvista expects it and remove extra pieces
    #the color codes come at a different step
    faceCPV = faceC[["nVert", "v1", "v2", "v3"]]
    faceCPV = faceCPV.to_numpy()
    
    #vertices
    #extract the relavent columns
    #this could also be done with the normalized coordinates if you want
    vertexC = vertexC[["x", "y", "z"]]
    #make it how pyvista likes
    vertexC = vertexC.to_numpy()
    
    #use the vertex and face information to form the mesh
    surf = pv.PolyData(vertexC, faceCPV)
    
    #colors
    colors_ = faceC[["red", "green", "blue", "alpha"]]
    colors_ = colors_.to_numpy()
    #add the color information to the mesh
    surf.cell_data["rgba"] = colors_
    
    return surf.plot(scalars = "rgba", rgb = True)

#example
# import os
# from plyfile import PlyData
# os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\fastTgcn\\data\\train-Lall")
# l76 = PlyData.read("076_L.ply")
# l76Vert = pd.DataFrame(l76["vertex"].data)
# l76Face = pd.DataFrame(l76["face"].data)
# plotPly(face = l76Face, vertex = l76Vert)








#read in a file with PlyData and get the face and vertex data








#identify colors by tooth number


