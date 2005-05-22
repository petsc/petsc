
static char help[] = "Tests PetscObjectPublish().\n\n";

/*T
   Concepts: vectors^assembling vectors;
   Processors: n
T*/

/* 
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscis.h     - index sets
     petscsys.h    - system routines       petscviewer.h - viewers
*/
#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int     i,n,ierr,rank;
  PetscScalar  one = 1.0,*array;
  Vec     x,xlocal;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /*
     Create a parallel vector.
      - In this case, we specify the size of each processor's local
        portion, and PETSc computes the global size.  Alternatively,
        if we pass the global size and use PETSC_DECIDE for the 
        local size PETSc will choose a reasonable partition trying 
        to put nearly an equal number of elements on each processor.
  */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,rank+4,PETSC_DECIDE,&x);CHKERRQ(ierr);
  ierr = PetscObjectPublish((PetscObject)x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,rank+4,&xlocal);CHKERRQ(ierr);
  ierr = PetscObjectPublish((PetscObject)xlocal);CHKERRQ(ierr);
  ierr = VecSet(xlocal,one);CHKERRQ(ierr);

  while (1) {

    /*
       Access the vector entries and add to them
    */
    PetscBarrier((PetscObject)x);
    ierr = VecGetArray(x,&array);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      array[i]++;
    }
    ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);

    ierr = VecGetArray(xlocal,&array);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      array[i]++;
    }
    ierr = VecRestoreArray(xlocal,&array);CHKERRQ(ierr);
  }

  /*
        Destroy the vectors
  */
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(xlocal);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
