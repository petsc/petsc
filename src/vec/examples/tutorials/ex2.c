/*$Id: ex2.c,v 1.32 1999/05/04 20:31:12 balay Exp bsmith $*/

static char help[] = "Builds a parallel vector with 1 component on the first\n\
processor, 2 on the second, etc.  Then each processor adds one to all\n\
elements except the last rank.\n\n";

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
  int     i, N, ierr, rank;
  Scalar  one = 1.0;
  Vec     x;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRA(ierr);

  /*
     Create a parallel vector.
      - In this case, we specify the size of each processor's local
        portion, and PETSc computes the global size.  Alternatively,
        if we pass the global size and use PETSC_DECIDE for the 
        local size PETSc will choose a reasonable partition trying 
        to put nearly an equal number of elements on each processor.
  */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,rank+1,PETSC_DECIDE,&x);CHKERRA(ierr);
  ierr = VecGetSize(x,&N);CHKERRA(ierr);
  ierr = VecSet(&one,x);CHKERRA(ierr);

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
  for ( i=0; i<N-rank; i++ ) {
    ierr = VecSetValues(x,1,&i,&one,ADD_VALUES);CHKERRA(ierr);  
  }

  /* 
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  /*
      View the vector; then destroy it.
  */
  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
