
static char help[] = "Tests Mathematica I/O of vectors and illustrates the use of user-defined event logging.\n\n";

#include <petscvec.h>

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output. */

int main(int argc, char *argv[])
{
  PetscViewer viewer;
  Vec         u;
  PetscScalar v;
  int         VECTOR_GENERATE, VECTOR_READ;
  int         i, m = 10, rank, size, low, high, ldim, iglobal;
  int         ierr;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));

  /* PART 1:  Generate vector, then write it to Mathematica */

  PetscCall(PetscLogEventRegister("Generate Vector", VEC_CLASSID, &VECTOR_GENERATE));
  PetscCall(PetscLogEventBegin(VECTOR_GENERATE, 0, 0, 0, 0));
  /* Generate vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, m));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecGetOwnershipRange(u, &low, &high));
  PetscCall(VecGetLocalSize(u, &ldim));
  for (i = 0; i < ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + 100 * rank);
    PetscCall(VecSetValues(u, 1, &iglobal, &v, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));
  PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "writing vector to Mathematica...\n"));

#if 0
  PetscCall(PetscViewerMathematicaOpen(PETSC_COMM_WORLD, 8000, "192.168.119.1", "Connect", &viewer));
  PetscCall(VecView(u, viewer));
#else
  PetscCall(VecView(u, PETSC_VIEWER_MATHEMATICA_WORLD));
#endif
  v = 0.0;
  PetscCall(VecSet(u, v));
  PetscCall(PetscLogEventEnd(VECTOR_GENERATE, 0, 0, 0, 0));

  /* All processors wait until test vector has been dumped */
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscSleep(10));

  /* PART 2:  Read in vector in from Mathematica */

  PetscCall(PetscLogEventRegister("Read Vector", VEC_CLASSID, &VECTOR_READ));
  PetscCall(PetscLogEventBegin(VECTOR_READ, 0, 0, 0, 0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "reading vector from Mathematica...\n"));
  /* Read new vector in binary format */
#if 0
  PetscCall(PetscViewerMathematicaGetVector(viewer, u));
  PetscCall(PetscViewerDestroy(&viewer));
#else
  PetscCall(PetscViewerMathematicaGetVector(PETSC_VIEWER_MATHEMATICA_WORLD, u));
#endif
  PetscCall(PetscLogEventEnd(VECTOR_READ, 0, 0, 0, 0));
  PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  PetscCall(VecDestroy(&u));
  PetscCall(PetscFinalize());
  return 0;
}
