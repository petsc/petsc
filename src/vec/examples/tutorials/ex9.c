#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex9.c,v 1.14 1999/04/21 18:16:01 bsmith Exp balay $";
#endif

static char help[] = "Demonstrates use of VecCreateGhost().\n\n";

/*T
   Concepts: Vectors^Assembling vectors; Ghost padding
   Routines: VecCreateGhost(); VecGetSize(); VecSet(); VecSetValues();
   Routines: VecView(); VecDestroy(); PetscSynchronizedPrintf();
   Routines: VecGhostGetLocalForm(); VecGhostUpdateEnd();
   Routines: VecGhostRestoreLocalForm(); VecGhostUpdateBegin();
   Routines: PetscSynchronizedFlush();
   Processors: n

   Comment: Ghost padding is one way to handle local calculations that
      involve values from other processors. VecCreateGhost() provides
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

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        rank,nlocal = 6,nghost = 2,ifrom[2],size,ierr,i,rstart,rend,flag;
  Scalar     value,*array,*tarray=0;
  Vec        lx,gx,gxs;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 2) SETERRA(1,1,"Must run example with two processors\n");

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

  if (rank == 0) {
    ifrom[0] = 11; ifrom[1] = 6; 
  } else {
    ifrom[0] = 0;  ifrom[1] = 5; 
  }

  /*
     Create the vector with two slots for ghost points. Note that both 
     the local vector (lx) and the global vector (gx) share the same 
     array for storing vector values.
  */
  ierr = OptionsHasName(PETSC_NULL,"-allocate",&flag);CHKERRA(ierr);
  if (flag) {
    tarray = (Scalar *) PetscMalloc( (nlocal+nghost)*sizeof(Scalar));CHKPTRA(tarray);
    ierr = VecCreateGhostWithArray(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,nghost,ifrom,tarray,&gxs);CHKERRA(ierr);
  } else {
    ierr = VecCreateGhost(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,nghost,ifrom,&gxs);CHKERRA(ierr);
  }

  /*
      Test VecDuplicate()
  */
  ierr = VecDuplicate(gxs,&gx);CHKERRA(ierr);
  ierr = VecDestroy(gxs);CHKERRA(ierr);

  /*
     Access the local representation
  */
  ierr = VecGhostGetLocalForm(gx,&lx);CHKERRA(ierr);

  /*
     Set the values from 0 to 12 into the "global" vector 
  */
  ierr = VecGetOwnershipRange(gx,&rstart,&rend);CHKERRA(ierr);
  for ( i=rstart; i<rend; i++ ) {
    value = (Scalar) i;
    ierr  = VecSetValues(gx,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(gx);CHKERRA(ierr);
  ierr = VecAssemblyEnd(gx);CHKERRA(ierr);

  ierr = VecGhostUpdateBegin(gx,INSERT_VALUES,SCATTER_FORWARD);CHKERRA(ierr);
  ierr = VecGhostUpdateEnd(gx,INSERT_VALUES,SCATTER_FORWARD);CHKERRA(ierr);

  /*
     Print out each vector, including the ghost padding region. 
  */
  ierr = VecGetArray(lx,&array);CHKERRA(ierr);
  for ( i=0; i<nlocal+nghost; i++ ) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%d %g\n",i,PetscReal(array[i]));
  }
  ierr = VecRestoreArray(lx,&array);CHKERRA(ierr);
  PetscSynchronizedFlush(PETSC_COMM_WORLD);

  ierr = VecGhostRestoreLocalForm(gx,&lx);CHKERRA(ierr); 
  ierr = VecDestroy(gx);CHKERRA(ierr);
  if (flag) {PetscFree(tarray); }
  PetscFinalize();
  return 0;
}
 


