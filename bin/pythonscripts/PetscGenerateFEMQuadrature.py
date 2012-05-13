#!/usr/bin/env python
import os, sys

# Find PETSc/BuildSystem
if 'PETSC_DIR' in os.environ:
  configDir = os.path.join(os.environ['PETSC_DIR'], 'config')
  bsDir     = os.path.join(configDir, 'BuildSystem')
  sys.path.insert(0, bsDir)
  sys.path.insert(0, configDir)

import PETSc.FEM
from FIAT.reference_element import default_simplex
from FIAT.lagrange import Lagrange
from FIAT.discontinuous_lagrange import DiscontinuousLagrange

generator  = PETSc.FEM.QuadratureGenerator()
generator.setup()
elements   = []
if not (len(sys.argv)-2) % 5 == 0:
  sys.exit('Incomplete set of arguments')
for n in range((len(sys.argv)-2) / 5):
  dim        = int(sys.argv[n*5+1])
  order      = int(sys.argv[n*5+2])
  components = int(sys.argv[n*5+3])
  numBlocks  = int(sys.argv[n*5+4])
  operator   = sys.argv[n*5+5]
  if order == 0:
    element  = DiscontinuousLagrange(default_simplex(dim), order)
  else:
    element  = Lagrange(default_simplex(dim), order)
  element.numComponents = components
  elements.append(element)
filename = sys.argv[-1]
generator.quadDegree = max([e.order for e in elements])
generator.run(elements, numBlocks, operator, filename)
