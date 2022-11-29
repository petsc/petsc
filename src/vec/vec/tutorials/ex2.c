
static char help[] = "Builds a parallel vector with 1 component on the first processor, 2 on the second, etc.\n\
  Then each processor adds one to all elements except the last rank.\n\n";

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscMPIInt rank;
  PetscInt    i, N;
  PetscScalar one = 1.0;
  Vec         x;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /*
     Create a parallel vector.
      - In this case, we specify the size of each processor's local
        portion, and PETSc computes the global size.  Alternatively,
        if we pass the global size and use PETSC_DECIDE for the
        local size PETSc will choose a reasonable partition trying
        to put nearly an equal number of elements on each processor.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, rank + 1, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecGetSize(x, &N));
  PetscCall(VecSet(x, one));

  /*
     Set the vector elements.
      - Always specify global locations of vector entries.
      - Each processor can contribute any vector entries,
        regardless of which processor "owns" them; any nonlocal
        contributions will be transferred to the appropriate processor
        during the assembly process.
      - In this example, the flag ADD_VALUES indicates that all
        contributions will be added together.
  */
  for (i = 0; i < N - rank; i++) PetscCall(VecSetValues(x, 1, &i, &one, ADD_VALUES));

  /*
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /*
      View the vector; then destroy it.
  */
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

TEST*/
