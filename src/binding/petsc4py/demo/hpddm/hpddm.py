# This script demonstrates solving a symmetric positive definite linear system using PETSc and HPDDM preconditioner
# It must be run with exactly 4 MPI processes
# Example run:
#   mpirun -n 4 python3 hpddm.py -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 20 -pc_hpddm_levels_1_eps_threshold_absolute 0.1 -ksp_monitor
# For more options, see ${PETSC_DIR}/src/ksp/ksp/tutorials/ex76.c

import sys
import petsc4py
# Initialize PETSc with command-line arguments
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Get the rank of the current process
rank = PETSc.COMM_WORLD.getRank()
# Ensure that the script is run with exactly 4 processes
if PETSc.COMM_WORLD.getSize() != 4:
  if rank == 0:
    print("This example requires 4 processes")
  quit()

# Load directory for input data
load_dir = PETSc.Options().getString("load_dir", "${DATAFILESPATH}/matrices/hpddm/GENEO")

# Load an index set (IS) from binary file
sizes = PETSc.IS().load(PETSc.Viewer().createBinary(f"{load_dir}/sizes_{rank}_4.dat", "r", comm = PETSc.COMM_SELF))
# Get indices from the loaded IS
idx   = sizes.getIndices()

# Create a PETSc matrix object
A = PETSc.Mat().create()
# Set the local and global sizes of the matrix
A.setSizes([[idx[0], idx[2]], [idx[1], idx[3]]])
# Configure matrix using runtime options
A.setFromOptions()
# Load matrix A from binary file
A = A.load(PETSc.Viewer().createBinary(f"{load_dir}/A.dat", "r", comm = PETSc.COMM_WORLD))

# Load an index set (IS) from binary file
aux_IS  = PETSc.IS().load(PETSc.Viewer().createBinary(f"{load_dir}/is_{rank}_4.dat", "r", comm = PETSc.COMM_SELF))
# Set the block size of the index set
aux_IS.setBlockSize(A.getBlockSize())
# Load the Neumann matrix of the current process
aux_Mat = PETSc.Mat().load(PETSc.Viewer().createBinary(f"{load_dir}/Neumann_{rank}_4.dat", "r", comm = PETSc.COMM_SELF))

# Create and configure the linear solver (KSP) and preconditioner (PC)
ksp = PETSc.KSP(PETSc.COMM_WORLD).create()
pc = ksp.getPC()
# Use HPDDM as the preconditioner
pc.setType(PETSc.PC.Type.HPDDM)
# Set the index set (local-to-global numbering) and auxiliary matrix
pc.setHPDDMAuxiliaryMat(aux_IS, aux_Mat)
# Inform HPDDM that the auxiliary matrix is the local Neumann matrix
pc.setHPDDMHasNeumannMat(True)
# Apply any command-line options (which may override options from the source file)
ksp.setFromOptions()
# Set the system matrix (Amat = Pmat)
ksp.setOperators(A)

# Create RHS (b) and solution (x) vectors, set random values to b, and solve the system
b, x = A.createVecs()
b.setRandom()
ksp.solve(b, x)

# Output grid and operator complexities on rank 0
gc, oc = pc.getHPDDMComplexities()
if rank == 0:
  print("grid complexity = ", gc, ", operator complexity = ", oc, sep = "")
