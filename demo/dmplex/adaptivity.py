import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

OptDB = PETSc.Options()

dim = OptDB.getInt('dim', 2)
plex = PETSc.DMPlex().createBoxMesh([4]*dim, simplex=True)
plex.view()

dim = plex.getDimension()
vStart, vEnd = plex.getDepthStratum(0)
numVertices = vEnd-vStart

# Create a metric tensor field corresponding to a uniform mesh size of 0.1
metric_array = np.zeros([numVertices,dim,dim])
for met in metric_array:
    met[:,:] = np.diag([100]*dim)
metric = PETSc.Vec().createWithArray(metric_array)

# Call adapt routine - boundary label None by default
newplex = plex.adapt(metric)
newplex.view()

# Write to VTK file
viewer = PETSc.Viewer().createVTK('mesh_base.vtk', 'w')
viewer(plex)
viewer = PETSc.Viewer().createVTK('mesh_adapt.vtk', 'w')
viewer(newplex)
