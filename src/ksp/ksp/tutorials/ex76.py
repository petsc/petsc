# mpirun -n 4 python3.12 ex76.py -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_1_eps_threshold 0.1 -ksp_monitor

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

rank = PETSc.COMM_WORLD.getRank()
if PETSc.COMM_WORLD.getSize() != 4:
  if rank == 0:
    print("This example requires 4 processes")
  quit()

load_dir = PETSc.Options().getString("load_dir", "${DATAFILESPATH}/matrices/hpddm/GENEO")

sizes = PETSc.IS().load(PETSc.Viewer().createBinary(f"{load_dir}/sizes_{rank}_4.dat", "r", comm = PETSc.COMM_SELF))
idx   = sizes.getIndices()

A = PETSc.Mat().create()
A.setSizes([[idx[0], idx[2]], [idx[1], idx[3]]])
A.setFromOptions()
A = A.load(PETSc.Viewer().createBinary(f"{load_dir}/A.dat", "r", comm = PETSc.COMM_WORLD))

aux_IS  = PETSc.IS().load(PETSc.Viewer().createBinary(f"{load_dir}/is_{rank}_4.dat", "r", comm = PETSc.COMM_SELF))
aux_IS.setBlockSize(2)
aux_Mat = PETSc.Mat().load(PETSc.Viewer().createBinary(f"{load_dir}/Neumann_{rank}_4.dat", "r", comm = PETSc.COMM_SELF))

ksp = PETSc.KSP(PETSc.COMM_WORLD).create()
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.HPDDM)
pc.setHPDDMAuxiliaryMat(aux_IS, aux_Mat)
pc.setHPDDMHasNeumannMat(True)
ksp.setFromOptions()
ksp.setOperators(A)

b, x = A.createVecs()
b.setRandom()
ksp.solve(b, x)
gc, oc = pc.getHPDDMComplexities()
if rank == 0:
  print("grid complexity = ", gc, ", operator complexity = ", oc, sep = "")
