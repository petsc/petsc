#ifndef lint
static char vcid[] = "$Id: ex8.c,v 1.2 1996/11/19 16:29:48 bsmith Exp $";
#endif

static char help[] = "Demonstrates use of VecCreateMPIGhost().\n\n";

/*T
   Concepts: Vectors^Assembling vectors; Ghost padding
   Routines: VecCreateMPIGhost(); VecGetSize(); VecSet(); VecSetValues();
   Routines: VecView(); VecDestroy(); 
   Processors: n

   Comments: Ghost padding is a good way to handle local calculations that
      involve values from other processors. VecCreateMPIGhost() provides
      a way to create vectors with extra room on each processor to contain
      the needed ghost values from other processors, vector computations
      are otherwise unaffected.
T*/

/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   is.h     - index sets
     sys.h    - system routines       viewer.h - viewers
*/
#include "vec.h"
#include <math.h>

int main(int argc,char **argv)
{
  int     i,rank,nlocal,N,nghost;
  Scalar  one = 1.0;
  Vec     x;

  PetscInitialize(&argc,&argv,(char *)0,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); 

  /*
     Construct a two dimensional graph connecting nlocal degrees of 
     freedom per processor. From this we will generate the global
     indices of needed ghost values

     For simplicity we generate the entire graph on each processor:
     in real application the graph would stored in parallel, but this
     example is only to demonstrate the management of ghost padding
     with VecCreateMPIGhost().
  */

  PetscFinalize();
  return 0;
}
 


