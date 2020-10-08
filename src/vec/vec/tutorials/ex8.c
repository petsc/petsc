
static char help[] = "Demonstrates using a local ordering to set values into a parallel vector.\n\n";

/*T
   Concepts: vectors^assembling vectors with local ordering;
   Processors: n
T*/

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,ng,*gindices,rstart,rend,M;
  PetscScalar    one = 1.0;
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /*
     Create a parallel vector.
      - In this case, we specify the size of each processor's local
        portion, and PETSc computes the global size.  Alternatively,
        PETSc could determine the vector's distribution if we specify
        just the global size.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,rank+1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);

  /*
     Set the local to global ordering for the vector. Each processor
     generates a list of the global indices for each local index. Note that
     the local indices are just whatever is convenient for a particular application.
     In this case we treat the vector as lying on a one dimensional grid and
     have one ghost point on each end of the blocks owned by each processor.
  */

  ierr = VecGetSize(x,&M);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ng   = rend - rstart + 2;
  ierr = PetscMalloc1(ng,&gindices);CHKERRQ(ierr);
  gindices[0] = rstart - 1;
  for (i=0; i<ng-1; i++) gindices[i+1] = gindices[i] + 1;
  /* map the first and last point as periodic */
  if (gindices[0]    == -1) gindices[0]    = M - 1;
  if (gindices[ng-1] == M)  gindices[ng-1] = 0;
  {
    ISLocalToGlobalMapping ltog;
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,ng,gindices,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(x,ltog);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  }
  ierr = PetscFree(gindices);CHKERRQ(ierr);

  /*
     Set the vector elements.
      - In this case set the values using the local ordering
      - Each processor can contribute any vector entries,
        regardless of which processor "owns" them; any nonlocal
        contributions will be transferred to the appropriate processor
        during the assembly process.
      - In this example, the flag ADD_VALUES indicates that all
        contributions will be added together.
  */
  for (i=0; i<ng; i++) {
    ierr = VecSetValuesLocal(x,1,&i,&one,ADD_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /*
      View the vector; then destroy it.
  */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       nsize: 4

TEST*/
