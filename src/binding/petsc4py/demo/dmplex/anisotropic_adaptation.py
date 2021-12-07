import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np


def sensor(x, y):
    """
    Classic hyperbolic sensor function for testing
    multi-scale anisotropic mesh adaptation:

    f:[-1, 1]² → R,
        f(x, y) = sin(50xy)/100 if |xy| > 2π/50 else sin(50xy)

    (mapped to have domain [0,1]² in this case).
    """
    xy = (2*x - 1)*(2*y - 1)
    ret = np.sin(50*xy)
    if np.abs(xy) > 2*np.pi/50:
        ret *= 0.01
    return ret


# Set metric parameters
h_min = 1.0e-10             # Minimum tolerated metric magnitude ~ cell size
h_max = 1.0e-01             # Maximum tolerated metric magnitude ~ cell size
a_max = 1.0e+05             # Maximum tolerated anisotropy
targetComplexity = 10000.0  # Analogous to number of vertices in adapted mesh
p = 1.0                     # Lᵖ normalization order

# Create a uniform mesh
OptDB = PETSc.Options()
dim = OptDB.getInt('dim', 2)
numEdges = 10
simplex = True
plex = PETSc.DMPlex().createBoxMesh([numEdges]*dim, simplex=simplex)
plex.distribute()
plex.view()
viewer = PETSc.Viewer().createVTK('base_mesh.vtk', 'w')
viewer(plex)

# Do four mesh adaptation iterations
for i in range(4):
    vStart, vEnd = plex.getDepthStratum(0)

    # Create a P1 sensor function
    comm = plex.getComm()
    fe = PETSc.FE().createLagrange(dim, 1, simplex, 1, -1, comm=comm)
    plex.setField(0, fe)
    plex.createDS()
    f = plex.createLocalVector()
    csec = plex.getCoordinateSection()
    coords = plex.getCoordinatesLocal()
    pf = f.getArray()
    pcoords = coords.getArray()
    for v in range(vStart, vEnd):
        off = csec.getOffset(v)
        x = pcoords[off]
        y = pcoords[off+1]
        pf[off//dim] = sensor(x, y)
    viewer = PETSc.Viewer().createVTK('sensor.vtk', 'w')
    viewer(f)

    # Recover the gradient of the sensor function
    dmGrad = plex.clone()
    fe = PETSc.FE().createLagrange(dim, dim, simplex, 1, -1, comm=comm)
    dmGrad.setField(0, fe)
    dmGrad.createDS()
    g = dmGrad.createLocalVector()
    plex.computeGradientClementInterpolant(f, g)
    viewer = PETSc.Viewer().createVTK('gradient.vtk', 'w')
    viewer(g)

    # Recover the Hessian of the sensor function
    dmHess = plex.clone()
    H = dmHess.metricCreate()
    dmGrad.computeGradientClementInterpolant(g, H)
    viewer = PETSc.Viewer().createVTK('hessian.vtk', 'w')
    viewer(H)

    # Obtain a metric by Lᵖ normalization
    dmHess.metricSetMinimumMagnitude(h_min)
    dmHess.metricSetMaximumMagnitude(h_max)
    dmHess.metricSetMaximumAnisotropy(a_max)
    dmHess.metricSetNormalizationOrder(p)
    dmHess.metricSetTargetComplexity(targetComplexity)
    metric = dmHess.metricNormalize(H)

    # Call adapt routine - boundary label None by default
    plex = plex.adaptMetric(metric)
    plex.distribute()
    plex.view()

# Write to VTK file
viewer = PETSc.Viewer().createVTK('anisotropic_mesh.vtk', 'w')
viewer(plex)
