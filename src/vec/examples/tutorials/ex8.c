#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex8.c,v 1.11 1999/06/30 23:50:45 balay Exp bsmith $";
#endif

static char help[] = "Demonstrates using a local ordering to set values into\n\
a parallel vector.\n\n";

/*T
   Concepts: Vectors^Assembling vectors with local ordering;
   Routines: VecCreateMPI(); VecGetSize(); VecSet(); VecSetValuesLocal();
   Routines: VecView(); VecDestroy(); VecSetLocalToGlobalMapping(); 
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
  int     i, N, ierr, rank, ng,*gindices,rstart,rend,M;
  Scalar  one = 1.0;
  Vec     x;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRA(ierr);

  /*
     Create a parallel vector.
      - In this case, we specify the size of each processor's local
        portion, and PETSc computes the global size.  Alternatively,
        PETSc could determine the vector's distribution if we specify
        just the global size.
  */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,rank+1,PETSC_DECIDE,&x);CHKERRA(ierr);
  ierr = VecGetSize(x,&N);CHKERRA(ierr);
  ierr = VecSet(&one,x);CHKERRA(ierr);

  /*
     Set the local to global ordering for the vector. Each processor 
     generates a list of the global indices for each local index. Note that
     the local indices are just whatever is convenient for a particular application.
     In this case we treat the vector as lying on a one dimensional grid and 
     have one ghost point on each end of the blocks owned by each processor. 
  */

  ierr = VecGetSize(x,&M);CHKERRA(ierr);
  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRA(ierr);
  ng   = rend - rstart + 2;
  gindices = (int*) PetscMalloc(ng*sizeof(int));CHKPTRA(gindices);
  gindices[0] = rstart - 1; 
  for (i=0; i<ng-1; i++ ) {
    gindices[i+1] = gindices[i] + 1;
  }
  /* map the first and last point as periodic */
  if (gindices[0]    == -1) gindices[0]    = M - 1;
  if (gindices[ng-1] == M)  gindices[ng-1] = 0;
  {
    ISLocalToGlobalMapping ltog;
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,ng,gindices,&ltog);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(x,ltog);CHKERRA(ierr);
    ierr = ISLocalToGlobalMappingDestroy(ltog);CHKERRA(ierr);
  }
  ierr = PetscFree(gindices);CHKERRA(ierr);

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
  for ( i=0; i<ng; i++ ) {
    ierr = VecSetValuesLocal(x,1,&i,&one,ADD_VALUES);CHKERRA(ierr);  
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
 
