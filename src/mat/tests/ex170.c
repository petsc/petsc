static char help[] = "Scalable algorithm for Connected Components problem.\n\
Entails changing the MatMult() for this matrix.\n\n\n";

#include <petscmat.h>

PETSC_EXTERN PetscErrorCode MatMultMax_SeqAIJ(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatMultAddMax_SeqAIJ(Mat, Vec, Vec, Vec);
#include <../src/mat/impls/aij/mpi/mpiaij.h>

/*
  Paper with Ananth: Frbenius norm of band was good proxy, but really want to know the rank outside

  LU for diagonal blocks must do shifting instead of pivoting, preferably shifting individual rows (like Pardiso)

  Draw picture of flow of reordering

  Measure Forbenius norm of the blocks being dropped by Truncated SPIKE (might be contaminated by pivoting in LU)

  Report on using Florida matrices (Maxim, Murat)
*/

/*
I have thought about how to do this. Here is a prototype algorithm. Let A be
the adjacency matrix (0 or 1), and let each component be identified by the
lowest numbered vertex in it. We initialize a vector c so that each vertex is
a component, c_i = i. Now we act on c with A, using a special product

  c = A * c

where we replace addition with min. The fixed point of this operation is a vector
c which is the component for each vertex. The number of iterates is

  max_{components} depth of BFS tree for component

We can accelerate this algorithm by preprocessing all locals domains using the
same algorithm. Then the number of iterations is bounded the depth of the BFS
tree for the graph on supervertices defined over local components, which is
bounded by p. In practice, this should be very fast.
*/

/* Only isolated vertices get a 1 on the diagonal */
PetscErrorCode CreateGraph(MPI_Comm comm, PetscInt testnum, Mat *A)
{
  Mat G;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, &G));
  /* The identity matrix */
  switch (testnum) {
  case 0: {
    Vec D;

    PetscCall(MatSetSizes(G, PETSC_DETERMINE, PETSC_DETERMINE, 5, 5));
    PetscCall(MatSetUp(G));
    PetscCall(MatCreateVecs(G, &D, NULL));
    PetscCall(VecSet(D, 1.0));
    PetscCall(MatDiagonalSet(G, D, INSERT_VALUES));
    PetscCall(VecDestroy(&D));
  } break;
  case 1: {
    PetscScalar vals[3] = {1.0, 1.0, 1.0};
    PetscInt    cols[3];
    PetscInt    rStart, rEnd, row;

    PetscCall(MatSetSizes(G, PETSC_DETERMINE, PETSC_DETERMINE, 5, 5));
    PetscCall(MatSetFromOptions(G));
    PetscCall(MatSeqAIJSetPreallocation(G, 2, NULL));
    PetscCall(MatSetUp(G));
    PetscCall(MatGetOwnershipRange(G, &rStart, &rEnd));
    row     = 0;
    cols[0] = 0;
    cols[1] = 1;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    row     = 1;
    cols[0] = 0;
    cols[1] = 1;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    row     = 2;
    cols[0] = 2;
    cols[1] = 3;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    row     = 3;
    cols[0] = 3;
    cols[1] = 4;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    row     = 4;
    cols[0] = 4;
    cols[1] = 2;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));
  } break;
  case 2: {
    PetscScalar vals[3] = {1.0, 1.0, 1.0};
    PetscInt    cols[3];
    PetscInt    rStart, rEnd, row;

    PetscCall(MatSetSizes(G, PETSC_DETERMINE, PETSC_DETERMINE, 5, 5));
    PetscCall(MatSetFromOptions(G));
    PetscCall(MatSeqAIJSetPreallocation(G, 2, NULL));
    PetscCall(MatSetUp(G));
    PetscCall(MatGetOwnershipRange(G, &rStart, &rEnd));
    row     = 0;
    cols[0] = 0;
    cols[1] = 4;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    row     = 1;
    cols[0] = 1;
    cols[1] = 2;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    row     = 2;
    cols[0] = 2;
    cols[1] = 3;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    row     = 3;
    cols[0] = 3;
    cols[1] = 1;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    row     = 4;
    cols[0] = 0;
    cols[1] = 4;
    if ((row >= rStart) && (row < rEnd)) PetscCall(MatSetValues(G, 1, &row, 2, cols, vals, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));
  } break;
  default:
    SETERRQ(comm, PETSC_ERR_PLIB, "Unknown test %d", testnum);
  }
  *A = G;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm     comm;
  Mat          A;    /* A graph */
  Vec          c;    /* A vector giving the component of each vertex */
  Vec          cold; /* The vector c from the last iteration */
  PetscScalar *carray;
  PetscInt     testnum = 0;
  PetscInt     V, vStart, vEnd, v, n;
  PetscMPIInt  size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  /* Use matrix to encode a graph */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-testnum", &testnum, NULL));
  PetscCall(CreateGraph(comm, testnum, &A));
  PetscCall(MatGetSize(A, &V, NULL));
  /* Replace matrix-vector multiplication with one that calculates the minimum rather than the sum */
  if (size == 1) {
    PetscCall(MatShellSetOperation(A, MATOP_MULT, (void(*))MatMultMax_SeqAIJ));
  } else {
    Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

    PetscCall(MatShellSetOperation(a->A, MATOP_MULT, (void(*))MatMultMax_SeqAIJ));
    PetscCall(MatShellSetOperation(a->B, MATOP_MULT, (void(*))MatMultMax_SeqAIJ));
    PetscCall(MatShellSetOperation(a->B, MATOP_MULT_ADD, (void(*))MatMultAddMax_SeqAIJ));
  }
  /* Initialize each vertex as a separate component */
  PetscCall(MatCreateVecs(A, &c, NULL));
  PetscCall(MatGetOwnershipRange(A, &vStart, &vEnd));
  PetscCall(VecGetArray(c, &carray));
  for (v = vStart; v < vEnd; ++v) carray[v - vStart] = v;
  PetscCall(VecRestoreArray(c, &carray));
  /* Preprocess in parallel to find local components */
  /* Multiply until c does not change */
  PetscCall(VecDuplicate(c, &cold));
  for (v = 0; v < V; ++v) {
    Vec       cnew = cold;
    PetscBool stop;

    PetscCall(MatMult(A, c, cnew));
    PetscCall(VecEqual(c, cnew, &stop));
    if (stop) break;
    cold = c;
    c    = cnew;
  }
  /* Report */
  PetscCall(VecUniqueEntries(c, &n, NULL));
  PetscCall(PetscPrintf(comm, "Components: %d Iterations: %d\n", n, v));
  PetscCall(VecView(c, PETSC_VIEWER_STDOUT_WORLD));
  /* Cleanup */
  PetscCall(VecDestroy(&c));
  PetscCall(VecDestroy(&cold));
  PetscCall(PetscFinalize());
  return 0;
}
