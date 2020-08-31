# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 17:03:18 2015

@author: ale
"""

import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

dim = 2
if not PETSc.COMM_WORLD.rank:
    coords = np.asarray([[0.0, 0.0],
                         [0.5, 0.0],
                         [1.0, 0.0],
                         [0.0, 0.5],
                         [0.5, 0.5],
                         [1.0, 0.5],
                         [0.0, 1.0],
                         [0.5, 1.0],
                         [1.0, 1.0]], dtype=float)
    cells = np.asarray([[0,1,4,3],
                        [1,2,5,4],
                        [3,4,7,6],
                        [4,5,8,7]], dtype=PETSc.IntType)
else:
    coords = np.zeros((0, 2), dtype=float)
    cells = np.zeros((0, 4), dtype=PETSc.IntType)

plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=PETSc.COMM_WORLD)

pStart, pEnd = plex.getChart()
plex.view()
print("pStart, pEnd: ", pStart, pEnd)

# Create section with 1 field with 1 DoF per vertex, edge amd cell
numComp = 1
# Start with an empty vector
numDof = [0] * 3
# Field defined on vertexes
numDof[0] = 1
# Field defined on edges
numDof[1] = 1
# Field defined on cells
numDof[2] = 1

plex.setNumFields(1)
origSect = plex.createSection(numComp, numDof)
origSect.setFieldName(0, 'TestField')
origSect.setUp()
origSect.view()

plex.setSection(origSect)
origVec = plex.createGlobalVec()
origVec.view()

origVec.setValues(list(range(pStart, pEnd)),list(range(pStart,pEnd)))
origVec.assemblyBegin()
origVec.assemblyEnd()

origVec.view()

if PETSc.COMM_WORLD.size > 1:
    sf = plex.distribute()
    sf.view()        

    newSect, newVec = plex.distributeField(sf, origSect, origVec)

else:
    newSect = origSect
    newVec = origVec

newSect.view()
newVec.view()

