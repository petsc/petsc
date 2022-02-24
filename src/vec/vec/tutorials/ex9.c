
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
#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       nlocal = 6,nghost = 2,ifrom[2],i,rstart,rend;
  PetscErrorCode ierr;
  PetscBool      flg,flg2,flg3;
  PetscScalar    value,*array,*tarray=0;
  Vec            lx,gx,gxs;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run example with two processors");

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
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-allocate",&flg));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-vecmpisetghost",&flg2));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-minvalues",&flg3));
  if (flg) {
    CHKERRQ(PetscMalloc1(nlocal+nghost,&tarray));
    CHKERRQ(VecCreateGhostWithArray(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,nghost,ifrom,tarray,&gxs));
  } else if (flg2) {
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&gxs));
    CHKERRQ(VecSetType(gxs,VECMPI));
    CHKERRQ(VecSetSizes(gxs,nlocal,PETSC_DECIDE));
    CHKERRQ(VecMPISetGhost(gxs,nghost,ifrom));
  } else {
    CHKERRQ(VecCreateGhost(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,nghost,ifrom,&gxs));
  }

  /*
      Test VecDuplicate()
  */
  CHKERRQ(VecDuplicate(gxs,&gx));
  CHKERRQ(VecDestroy(&gxs));

  /*
     Access the local representation
  */
  CHKERRQ(VecGhostGetLocalForm(gx,&lx));

  /*
     Set the values from 0 to 12 into the "global" vector
  */
  CHKERRQ(VecGetOwnershipRange(gx,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    value = (PetscScalar) i;
    CHKERRQ(VecSetValues(gx,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(gx));
  CHKERRQ(VecAssemblyEnd(gx));

  CHKERRQ(VecGhostUpdateBegin(gx,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecGhostUpdateEnd(gx,INSERT_VALUES,SCATTER_FORWARD));

  /*
     Print out each vector, including the ghost padding region.
  */
  CHKERRQ(VecGetArray(lx,&array));
  for (i=0; i<nlocal+nghost; i++) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " %g\n",i,(double)PetscRealPart(array[i])));
  }
  CHKERRQ(VecRestoreArray(lx,&array));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  CHKERRQ(VecGhostRestoreLocalForm(gx,&lx));

  /* Another test that sets ghost values and then accumulates onto the owning processors using MIN_VALUES */
  if (flg3) {
    if (rank == 0)CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nTesting VecGhostUpdate with MIN_VALUES\n"));
    CHKERRQ(VecGhostGetLocalForm(gx,&lx));
    CHKERRQ(VecGetArray(lx,&array));
    for (i=0; i<nghost; i++) array[nlocal+i] = rank ? (PetscScalar)4 : (PetscScalar)8;
    CHKERRQ(VecRestoreArray(lx,&array));
    CHKERRQ(VecGhostRestoreLocalForm(gx,&lx));

    CHKERRQ(VecGhostUpdateBegin(gx,MIN_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecGhostUpdateEnd(gx,MIN_VALUES,SCATTER_REVERSE));

    CHKERRQ(VecGhostGetLocalForm(gx,&lx));
    CHKERRQ(VecGetArray(lx,&array));

    for (i=0; i<nlocal+nghost; i++) {
      CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT " %g\n",i,(double)PetscRealPart(array[i])));
    }
    CHKERRQ(VecRestoreArray(lx,&array));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    CHKERRQ(VecGhostRestoreLocalForm(gx,&lx));
  }

  CHKERRQ(VecDestroy(&gx));

  if (flg) CHKERRQ(PetscFree(tarray));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       nsize: 2

     test:
       suffix: 2
       nsize: 2
       args: -allocate
       output_file: output/ex9_1.out

     test:
       suffix: 3
       nsize: 2
       args: -vecmpisetghost
       output_file: output/ex9_1.out

     test:
       suffix: 4
       nsize: 2
       args: -minvalues
       output_file: output/ex9_2.out
       requires: !complex

TEST*/
