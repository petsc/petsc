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

generator = PETSc.FEM.QuadratureGenerator()
generator.setup()
dim      = int(sys.argv[1])
order    = int(sys.argv[2])
elements = [Lagrange(default_simplex(dim), order)]
generator.run(elements, sys.argv[3])
