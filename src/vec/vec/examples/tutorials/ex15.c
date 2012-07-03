/*$Id: ex15.c,v 1.50 2002/09/04 07:43:58 knepley Exp $*/

static char help[] = "Tests Mathematica I/O of vectors and illustrates the use of user-defined event logging.\n\n";

#include <petscvec.h>

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output. */

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscViewer  viewer;
  Vec          u;
  PetscScalar  v;
  int          VECTOR_GENERATE, VECTOR_READ;
  int          i, m = 10, rank, size, low, high, ldim, iglobal;
  int          ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);                                                 CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);                                                          CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);                                                          CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-m", &m, PETSC_NULL);                                            CHKERRQ(ierr);

  /* PART 1:  Generate vector, then write it to Mathematica */

  ierr = PetscLogEventRegister("Generate Vector", VEC_CLASSID,&VECTOR_GENERATE);                           CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VECTOR_GENERATE, 0, 0, 0, 0);                                                 CHKERRQ(ierr);
  /* Generate vector */
  ierr = VecCreate(PETSC_COMM_WORLD, &u);                                                                 CHKERRQ(ierr);
  ierr = VecSetSizes(u, PETSC_DECIDE, m);                                                                 CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);                                                                            CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(u, &low, &high);                                                            CHKERRQ(ierr);
  ierr = VecGetLocalSize(u, &ldim);                                                                       CHKERRQ(ierr);
  for(i = 0; i < ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar) (i + 100*rank);
    ierr = VecSetValues(u, 1, &iglobal, &v, INSERT_VALUES);                                               CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);                                                                             CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);                                                                               CHKERRQ(ierr);
  ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);                                                           CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "writing vector to Mathematica...\n");                             CHKERRQ(ierr);

#if 0
  ierr = PetscViewerMathematicaOpen(PETSC_COMM_WORLD, 8000, "192.168.119.1", "Connect", &viewer);         CHKERRQ(ierr);
  ierr = VecView(u, viewer);                                                                              CHKERRQ(ierr);
#else
  ierr = VecView(u, PETSC_VIEWER_MATHEMATICA_WORLD);                                                      CHKERRQ(ierr);
#endif
  v    = 0.0;
  ierr = VecSet(u,v);                                                                                     CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VECTOR_GENERATE, 0, 0, 0, 0);                                                   CHKERRQ(ierr);

  /* All processors wait until test vector has been dumped */
  ierr = MPI_Barrier(PETSC_COMM_WORLD);                                                                   CHKERRQ(ierr);
  ierr = PetscSleep(10);                                                                                  CHKERRQ(ierr);

  /* PART 2:  Read in vector in from Mathematica */

  ierr = PetscLogEventRegister("Read Vector", VEC_CLASSID,&VECTOR_READ);                                   CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VECTOR_READ, 0, 0, 0, 0);                                                     CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "reading vector from Mathematica...\n");                           CHKERRQ(ierr);
  /* Read new vector in binary format */
#if 0
  ierr = PetscViewerMathematicaGetVector(viewer, u);                                                      CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);                                                                      CHKERRQ(ierr);
#else
  ierr = PetscViewerMathematicaGetVector(PETSC_VIEWER_MATHEMATICA_WORLD, u);                              CHKERRQ(ierr);
#endif
  ierr = PetscLogEventEnd(VECTOR_READ, 0, 0, 0, 0);                                                       CHKERRQ(ierr);
  ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);                                                           CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(&u);                                                                                   CHKERRQ(ierr);
  ierr = PetscFinalize();                                                                                 CHKERRQ(ierr);
  return 0;
}
