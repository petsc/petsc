
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

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-m", &m, NULL));

  /* PART 1:  Generate vector, then write it to Mathematica */

  CHKERRQ(PetscLogEventRegister("Generate Vector", VEC_CLASSID,&VECTOR_GENERATE));
  CHKERRQ(PetscLogEventBegin(VECTOR_GENERATE, 0, 0, 0, 0));
  /* Generate vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &u));
  CHKERRQ(VecSetSizes(u, PETSC_DECIDE, m));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecGetOwnershipRange(u, &low, &high));
  CHKERRQ(VecGetLocalSize(u, &ldim));
  for (i = 0; i < ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar) (i + 100*rank);
    CHKERRQ(VecSetValues(u, 1, &iglobal, &v, INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));
  CHKERRQ(VecView(u, PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "writing vector to Mathematica...\n"));

#if 0
  CHKERRQ(PetscViewerMathematicaOpen(PETSC_COMM_WORLD, 8000, "192.168.119.1", "Connect", &viewer));
  CHKERRQ(VecView(u, viewer));
#else
  CHKERRQ(VecView(u, PETSC_VIEWER_MATHEMATICA_WORLD));
#endif
  v    = 0.0;
  CHKERRQ(VecSet(u,v));
  CHKERRQ(PetscLogEventEnd(VECTOR_GENERATE, 0, 0, 0, 0));

  /* All processors wait until test vector has been dumped */
  CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
  CHKERRQ(PetscSleep(10));

  /* PART 2:  Read in vector in from Mathematica */

  CHKERRQ(PetscLogEventRegister("Read Vector", VEC_CLASSID,&VECTOR_READ));
  CHKERRQ(PetscLogEventBegin(VECTOR_READ, 0, 0, 0, 0));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "reading vector from Mathematica...\n"));
  /* Read new vector in binary format */
#if 0
  CHKERRQ(PetscViewerMathematicaGetVector(viewer, u));
  CHKERRQ(PetscViewerDestroy(&viewer));
#else
  CHKERRQ(PetscViewerMathematicaGetVector(PETSC_VIEWER_MATHEMATICA_WORLD, u));
#endif
  CHKERRQ(PetscLogEventEnd(VECTOR_READ, 0, 0, 0, 0));
  CHKERRQ(VecView(u, PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  CHKERRQ(VecDestroy(&u));
  ierr = PetscFinalize();
  return ierr;
}
