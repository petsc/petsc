import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

OptDB = PETSc.Options()

dim = OptDB.getInt('dim', 2)
plex = PETSc.DMPlex().createBoxMesh([4]*dim, simplex=True)
plex.view()

# Create two metric tensor fields corresponding to uniform mesh sizes of 0.1 and 0.2
metric1 = plex.metricCreateUniform(100.0)
metric2 = plex.metricCreateUniform(25.0)

# Ensure that we do ineed have metrics, i.e. they are SPD
plex.metricEnforceSPD(metric1)
plex.metricEnforceSPD(metric2)

# The metrics can be combined using intersection, the result of which corresponds to
# the maximum ellipsoid at each point
metric = plex.metricIntersection2(metric1, metric2)
metric1.axpy(-1, metric)
assert np.isclose(metric1.norm(), 0.0)

# Call adapt routine - boundary label None by default
newplex = plex.adaptMetric(metric)
newplex.view()

# Write to VTK file
viewer = PETSc.Viewer().createVTK('base_mesh.vtk', 'w')
viewer(plex)
viewer = PETSc.Viewer().createVTK('isotropic_adapted_mesh.vtk', 'w')
viewer(newplex)
