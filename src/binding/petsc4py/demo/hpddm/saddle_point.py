# This script demonstrates solving a symmetric saddle-point linear system using PETSc and HPDDM preconditioner
# It must be run with exactly 4 MPI processes
# Example run:
#   mpirun -n 4 python3 saddle_point.py -ksp_monitor -ksp_type fgmres -ksp_max_it 10 -ksp_rtol 1e-4 -fieldsplit_ksp_max_it 100 -fieldsplit_pc_hpddm_levels_1_eps_nev 10 -fieldsplit_pc_hpddm_levels_1_st_share_sub_ksp -fieldsplit_pc_hpddm_has_neumann -fieldsplit_pc_hpddm_define_subdomains -fieldsplit_1_pc_hpddm_schur_precondition geneo -fieldsplit_pc_hpddm_coarse_pc_type cholesky -fieldsplit_pc_hpddm_levels_1_sub_pc_type lu -fieldsplit_ksp_type fgmres -fieldsplit_1_pc_hpddm_coarse_correction balanced -fieldsplit_1_pc_hpddm_levels_1_eps_gen_non_hermitian
# For more options, see ${PETSC_DIR}/src/ksp/ksp/tutorials/ex87.c

import sys
import petsc4py
# Initialize PETSc with command-line arguments
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Function to load matrices and index sets from binary files
def mat_and_is_load(prefix, identifier, A, aux_IS, aux_Mat, rank, size):
# Load an index set (IS) from binary file
  sizes = PETSc.IS().load(PETSc.Viewer().createBinary(f"{prefix}{identifier}_sizes_{rank}_{size}.dat", "r", comm = PETSc.COMM_SELF))
# Get indices from the loaded IS
  idx   = sizes.getIndices()
# Set the local and global sizes of the matrix
  A.setSizes([[idx[0], idx[2]], [idx[1], idx[3]]])
# Configure matrix using runtime options
  A.setFromOptions()
# Load matrix A from binary file
  A = A.load(PETSc.Viewer().createBinary(f"{prefix}{identifier}.dat", "r", comm = PETSc.COMM_WORLD))

# Load an index set (IS) from binary file
  aux_IS.load(PETSc.Viewer().createBinary(f"{prefix}{identifier}_is_{rank}_{size}.dat", "r", comm = PETSc.COMM_SELF))
# Load the Neumann matrix of the current process
  aux_Mat.load(PETSc.Viewer().createBinary(f"{prefix}{identifier}_aux_{rank}_{size}.dat", "r", comm = PETSc.COMM_SELF))

# Get the size of the communicator
size = PETSc.COMM_WORLD.getSize()
# Get the rank of the current process
rank = PETSc.COMM_WORLD.getRank()
if size != 4:
  if rank == 0:
    print("This example requires 4 processes")
  quit()

# Problem type (either 'elasticity' or 'stokes')
system_str = PETSc.Options().getString("system", "elasticity")
id_sys = 0 if system_str == "elasticity" else 1
empty_A11 = False
# Lower-left (1,1) block is never zero when problem type is 'elasticity'
if id_sys == 1:
  empty_A11 = PETSc.Options().getBool("empty_A11", False)

# 2-by-2 block structure
A = [None, None, None, None]
# Auxiliary data only for the diagonal blocks
aux_Mat = [None, None]
aux_IS = [None, None]

# Create placeholder objects for the diagonal blocks
A[0] = PETSc.Mat().create(comm = PETSc.COMM_WORLD)
A[0].setFromOptions()
aux_IS[0] = PETSc.IS().create(comm = PETSc.COMM_SELF)
aux_Mat[0] = PETSc.Mat().create(comm = PETSc.COMM_SELF)
A[3] = PETSc.Mat().create(comm = PETSc.COMM_WORLD)
A[3].setFromOptions()
aux_IS[1] = PETSc.IS().create(comm = PETSc.COMM_SELF)
aux_Mat[1] = PETSc.Mat().create(comm = PETSc.COMM_SELF)

# Load directory for input data
load_dir = PETSc.Options().getString("load_dir", "${DATAFILESPATH}/matrices/hpddm/GENEO")
# Specific prefix for each problem
prefix = f"{load_dir}/{ 'B' if id_sys == 1 else 'A' }"

# Diagonal blocks and auxiliary data for PCHPDDM
mat_and_is_load(prefix, "00", A[0], aux_IS[0], aux_Mat[0], rank, size)
mat_and_is_load(prefix, "11", A[3], aux_IS[1], aux_Mat[1], rank, size)

# Coherent off-diagonal (0,1) block
A[2] = PETSc.Mat().create(comm = PETSc.COMM_WORLD)
n, _ = A[0].getLocalSize()
N, _ = A[0].getSize()
m, _ = A[3].getLocalSize()
M, _ = A[3].getSize()
# Set matrix sizes based on the sizes of (0,0) and (1,1) blocks
A[2].setSizes([[m, M], [n, N]])
A[2].setFromOptions()
A[2].load(PETSc.Viewer().createBinary(f"{load_dir}/{ 'B' if id_sys == 1 else 'A' }10.dat", "r", comm = PETSc.COMM_WORLD))
# Create a matrix that behaves likes A[1]' without explicitly assembling it
A[1] = PETSc.Mat().createTranspose(A[2])

# Global MatNest
S = PETSc.Mat().createNest([[A[0], A[1]], [A[2], A[3] if not empty_A11 else None]])

ksp = PETSc.KSP().create(comm = PETSc.COMM_WORLD)
ksp.setOperators(S)
pc = ksp.getPC()

# Use FIELDSPLIT as the outer preconditioner
pc.setType(PETSc.PC.Type.FIELDSPLIT)
# Use SCHUR since there are only two fields
pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
# Use SELF because HPDDM deals with a Schur complement matrix
pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.SELF)
# Apply any command-line options (which may override options from the source file)
pc.setFromOptions()
# Setup the outer preconditioner so that one can query the inner preconditioners
pc.setUp()

# Retrieve the inner solvers (associated to the diagonal blocks)
ksp0, ksp1 = pc.getFieldSplitSubKSP()

# Upper-left (0,0) block
pc0 = ksp0.getPC()
# Use HPDDM as the preconditioner
pc0.setType(PETSc.PC.Type.HPDDM)
# Set the index set (local-to-global numbering) and auxiliary matrix (first diagonal block)
pc0.setHPDDMAuxiliaryMat(aux_IS[0], aux_Mat[0])
pc0.setFromOptions()

# Schur complement
pc1 = ksp1.getPC()
# Use HPDDM as the preconditioner
pc1.setType(PETSc.PC.Type.HPDDM)
if not empty_A11:
# Set the index set (local-to-global numbering) and auxiliary matrix (second diagonal block)
# If there is no block (-empty_A11), then these are computed automatically by HPDDM
  pc1.setHPDDMAuxiliaryMat(aux_IS[1], aux_Mat[1])
pc1.setFromOptions()

# Create RHS (b) and solution (x) vectors, load b from file, and solve the system
b, x = S.createVecs()
b.load(PETSc.Viewer().createBinary(f"{load_dir}/rhs_{ 'B' if id_sys == 1 else 'A' }.dat", "r", comm = PETSc.COMM_WORLD))
ksp.setFromOptions()
ksp.solve(b, x)
