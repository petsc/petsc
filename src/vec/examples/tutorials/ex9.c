#ifndef lint
static char vcid[] = "$Id: ex9.c,v 1.4 1997/04/06 14:17:02 balay Exp bsmith $";
#endif

static char help[] = "Demonstrates use of VecCreateMPIGhost().\n\n";

/*T
   Concepts: Vectors^Assembling vectors; Ghost padding
   Routines: VecCreateMPIGhost(); VecGetSize(); VecSet(); VecSetValues();
   Routines: VecView(); VecDestroy(); PetscSynchronizedPrintf();
   Routines: PetscSynchronizedFlush();
   Processors: n

   Comment: Ghost padding is a good way to handle local calculations that
      involve values from other processors. VecCreateMPIGhost() provides
      a way to create vectors with extra room at the end of the vector 
      array to contain the needed ghost values from other processors, 
      vector computations are otherwise unaffected.
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
  int        rank,nlocal = 6,nghost = 2,ito[2],ifrom[2],size,ierr,i,rstart,rend;
  Scalar     value,*array;
  Vec        lx,gx;
  IS         isfrom,isto;
  VecScatter scatter;

  PetscInitialize(&argc,&argv,(char *)0,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
  MPI_Comm_size(MPI_COMM_WORLD,&size); 
  if (size != 2) SETERRA(1,1,"Must run example with two processors\n");

  /*
     Construct a two dimensional graph connecting nlocal degrees of 
     freedom per processor. From this we will generate the global
     indices of needed ghost values

     For simplicity we generate the entire graph on each processor:
     in real application the graph would stored in parallel, but this
     example is only to demonstrate the management of ghost padding
     with VecCreateMPIGhost().

     In this example we consider the vector as representing 
     degrees of freedom in a one dimensional grid with periodic 
     boundary conditions.

        ----Processor  1---------  ----Processor 2 --------
         0    1   2   3   4    5    6    7   8   9   10   11
                               |----| 
         |-------------------------------------------------|

  */

  /*
     Create the vector with two slots for ghost points. Note that both 
     the local vector (lx) and the global vector (gx) share the same 
     array for storing vector values.
  */
  ierr = VecCreateGhost(PETSC_COMM_WORLD,nlocal,nlocal+nghost,PETSC_DECIDE,&lx,&gx);

  /*
     Create a scatter context to move over the two ghost values
  */
  if (rank == 0) {
    ifrom[0] = 11; ifrom[1] = 6; 
  } else {
    ifrom[0] = 0;  ifrom[1] = 5; 
  }
  ito[0] = 6; ito[1] = 7;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,ifrom,&isfrom);CHKERRA(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,ito,&isto);CHKERRA(ierr);
  ierr = VecScatterCreate(gx,isfrom,lx,isto,&scatter);CHKERRA(ierr);
  ierr = ISDestroy(isfrom); CHKERRA(ierr);
  ierr = ISDestroy(isto); CHKERRA(ierr);

  /*
     Set the values from 0 to 12 into the "global" vector 
  */
  ierr = VecGetOwnershipRange(gx,&rstart,&rend);CHKERRA(ierr);
  for ( i=rstart; i<rend; i++ ) {
    value = (Scalar) i;
    ierr  = VecSetValues(gx,1,&i,&value,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(gx); CHKERRA(ierr);
  ierr = VecAssemblyEnd(gx); CHKERRA(ierr);

  ierr = VecScatterBegin(gx,lx,INSERT_VALUES,SCATTER_FORWARD,scatter); CHKERRA(ierr);
  ierr = VecScatterEnd(gx,lx,INSERT_VALUES,SCATTER_FORWARD,scatter); CHKERRA(ierr);


  ierr = VecScatterDestroy(scatter); CHKERRA(ierr);
  /*
     Print out each vector, including the ghost padding region. 
  */
  ierr = VecGetArray(lx,&array);CHKERRA(ierr);
  for ( i=0; i<nlocal+nghost; i++ ) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%d %g\n",i,PetscReal(array[i]));
  }
  ierr = VecRestoreArray(lx,&array);CHKERRA(ierr);
  PetscSynchronizedFlush(PETSC_COMM_WORLD);

  ierr = VecDestroy(gx);CHKERRA(ierr);
  ierr = VecDestroy(lx);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 


