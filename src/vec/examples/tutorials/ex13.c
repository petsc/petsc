/*$Id: ex13.c,v 1.7 1999/10/24 14:02:04 bsmith Exp bsmith $*/

static char help[] = "Tests PetscObjectPublish().\n\n";

/*T
   Concepts: Vectors^Assembling vectors;
   Routines: VecCreateMPI(); VecGetSize(); VecSet(); VecSetValues();
   Routines: VecView(); VecDestroy();
   Processors: n
T*/

/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   is.h     - index sets
     sys.h    - system routines       viewer.h - viewers
*/
#include "vec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int     i,n,ierr,rank;
  Scalar  one = 1.0,*array;
  Vec     x,xlocal;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  /*
     Create a parallel vector.
      - In this case, we specify the size of each processor's local
        portion, and PETSc computes the global size.  Alternatively,
        if we pass the global size and use PETSC_DECIDE for the 
        local size PETSc will choose a reasonable partition trying 
        to put nearly an equal number of elements on each processor.
  */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,rank+4,PETSC_DECIDE,&x);CHKERRA(ierr);
  ierr = PetscObjectPublish((PetscObject)x);CHKERRA(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRA(ierr);
  ierr = VecSet(&one,x);CHKERRA(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,rank+4,&xlocal);CHKERRA(ierr);
  ierr = PetscObjectPublish((PetscObject)xlocal);CHKERRA(ierr);
  ierr = VecSet(&one,xlocal);CHKERRA(ierr);

  while (1) {

    /*
       Access the vector entries and add to them
    */
    PetscBarrier((PetscObject)x);
    ierr = VecGetArray(x,&array);CHKERRA(ierr);
    for (i=0; i<n; i++) {
      array[i]++;
    }
    ierr = VecRestoreArray(x,&array);CHKERRA(ierr);

    ierr = VecGetArray(xlocal,&array);CHKERRA(ierr);
    for (i=0; i<n; i++) {
      array[i]++;
    }
    ierr = VecRestoreArray(xlocal,&array);CHKERRA(ierr);
  }

  /*
        Destroy the vectors
  */
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(xlocal);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
