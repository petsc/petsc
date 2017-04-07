from petsc4py import PETSc
import numpy as np


plex = PETSc.DMPlex().createBoxMesh(2,3)
plex.view()

# Create coord Section: 1 field with 2 DoF per vertex, 0 per edge and cell
numComp = 1
numDof = [0] * 3
numDof[0] = 2        # Field defined on vertexes
origSect = plex.createSection(numComp, numDof)
plex.setDefaultSection(origSect)

dim = plex.getDimension()
vStart, vEnd = plex.getDepthStratum(0)
numVertices = vEnd-vStart

# create a metric tensor field corresponding to a uniform mesh size of 0.1
metric_array = np.zeros(dim*dim*numVertices)
for d in range(dim*numVertices): metric_array[d*dim+d%dim] = 100.
print metric_array

# create PETSc Vec to store the metric
metric = plex.createGlobalVec()
metric.createWithArray(metric_array)      

# call adapt routine - boundary label is "" by defaut
newplex = plex.adapt(metric)
newplex.view()

# write to file 
viewer = PETSc.Viewer().createVTK('mesh_adapt.vtk', 'w')
viewer(newplex)