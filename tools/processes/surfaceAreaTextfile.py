import sys

import trimesh

#testing
inPath = "K:/iowaExpTest/remeshDescriptorTesting/remesh/pat004Pre_remesh8500.ply"
outPath = "K:/iowaExpTest/testDir/saText.txt"

#arguments from snakemake
inPath = sys.argv[1]
outPath = sys.argv[2]

mesh = trimesh.load(inPath, process = False)

with open(outPath, "w") as f:
    f.write(f"{mesh.area}\n")
