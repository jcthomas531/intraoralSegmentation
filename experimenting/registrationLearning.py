import os
os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools\\")
import plyFunctions as pf 
import pyvista as pv


os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\fastTgcn\\data\\train-Lall")
#l76
l76 = pf.plyRead("076_L.ply")
l76["face"] = pf.toothVars(l76["face"])
l76Surf = pf.giveSurf(face = l76["face"], vertex = l76["vert"])
#l08
l08 = pf.plyRead("008_L.ply")
l08["face"] = pf.toothVars(l08["face"])
l08Surf = pf.giveSurf(face = l08["face"], vertex = l08["vert"])
#l51
l51 = pf.plyRead("051_L.ply")
l51["face"] = pf.toothVars(l51["face"])
l51Surf = pf.giveSurf(face = l51["face"], vertex = l51["vert"])
#try plotting both surfaces together
manyPlot = pv.Plotter()
manyPlot.add_mesh(l76Surf, scalars = "rgba", rgb = True)
manyPlot.add_mesh(l08Surf, scalars = "rgba", rgb = True)
manyPlot.add_mesh(l51Surf, scalars = "rgba", rgb = True)
manyPlot.show()
