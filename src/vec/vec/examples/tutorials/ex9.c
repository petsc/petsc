
static char help[] = "Demonstrates use of VecCreateGhost().\n\n";

/*T
   Concepts: vectors^assembling vectors;
   Concepts: vectors^ghost padding;
   Processors: n

   Description: Ghost padding is one way to handle local calculations that
      involve values from other processors. VecCreateGhost() provides
      a way to create vectors with extra room at the end of the vector 
      array to contain the needed ghost values from other processors, 
      vector computations are otherwise unaffected.
T*/

/* 
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/
#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       nlocal = 6,nghost = 2,ifrom[2],i,rstart,rend;
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscScalar    value,*array,*tarray=0;
  Vec            lx,gx,gxs;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 2) SETERRQ(1,"Must run example with two processors\n");

  /*
     Construct a two dimensional graph connecting nlocal degrees of 
     freedom per processor. From this we will generate the global
     indices of needed ghost values

     For simplicity we generate the entire graph on each processor:
     in real application the graph would stored in parallel, but this
     example is only to demonstrate the management of ghost padding
     with VecCreateGhost().

     In this example we consider the vector as representing 
     degrees of freedom in a one dimensional grid with periodic 
     boundary conditions.

        ----Processor  1---------  ----Processor 2 --------
         0    1   2   3   4    5    6    7   8   9   10   11
                               |----| 
         |-------------------------------------------------|

  */

  if (!rank) {
    ifrom[0] = 11; ifrom[1] = 6; 
  } else {
    ifrom[0] = 0;  ifrom[1] = 5; 
  }

  /*
     Create the vector with two slots for ghost points. Note that both 
     the local vector (lx) and the global vector (gx) share the same 
     array for storing vector values.
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-allocate",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscMalloc((nlocal+nghost)*sizeof(PetscScalar),&tarray);CHKERRQ(ierr);
    ierr = VecCreateGhostWithArray(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,nghost,ifrom,tarray,&gxs);CHKERRQ(ierr);
  } else {
    ierr = VecCreateGhost(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,nghost,ifrom,&gxs);CHKERRQ(ierr);
  }

  /*
      Test VecDuplicate()
  */
  ierr = VecDuplicate(gxs,&gx);CHKERRQ(ierr);
  ierr = VecDestroy(gxs);CHKERRQ(ierr);

  /*
     Access the local representation
  */
  ierr = VecGhostGetLocalForm(gx,&lx);CHKERRQ(ierr);

  /*
     Set the values from 0 to 12 into the "global" vector 
  */
  ierr = VecGetOwnershipRange(gx,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    value = (PetscScalar) i;
    ierr  = VecSetValues(gx,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(gx);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(gx);CHKERRQ(ierr);

  ierr = VecGhostUpdateBegin(gx,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(gx,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /*
     Print out each vector, including the ghost padding region. 
  */
  ierr = VecGetArray(lx,&array);CHKERRQ(ierr);
  for (i=0; i<nlocal+nghost; i++) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%D %G\n",i,PetscRealPart(array[i]));CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(lx,&array);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = VecGhostRestoreLocalForm(gx,&lx);CHKERRQ(ierr); 
  ierr = VecDestroy(gx);CHKERRQ(ierr);
  if (flg) {ierr = PetscFree(tarray);CHKERRQ(ierr);}
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 


