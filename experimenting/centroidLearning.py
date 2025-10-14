
#exploring centroids and measures between teeth
#i am going to work using mostly the annotated training data just to get an idea
#of how these methods work. Once we make better predictions, these things will
#translate

import os
toolsPath = "H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools\\"
dataPath = "H:\\schoolFiles\\dissertation\\intraoralSegmentation\\fastTgcn\\data\\"
os.chdir(toolsPath)
import plyFunctions as pf
#train patient 76 has all colors
os.chdir(dataPath+"train-Lall")
l76 = pf.plyRead("076_L.ply")
l76["face"] = pf.toothVars(l76["face"])
os.chdir(dataPath+"Train-U")
u76 = pf.plyRead("076_U.ply")
u76["face"] = pf.toothVars(u76["face"])


#get meshjfor the arch we are working on
l76Surf = giveSurf(face = l76["face"], vertex = l76["vert"])
#get centroids for the arch
l76Cent = toothCentroids(face = l76["face"], vertex = l76["vert"])
#plot
l76Plot = pv.Plotter()
l76Plot.add_mesh(l76Surf, scalars = "rgba", rgb = True)
l76Plot.add_points(np.array(l76Cent.iloc[:,range(1,4)]),
                    color = "black", point_size=10,
                    render_points_as_spheres=True)
l76Plot.show()



