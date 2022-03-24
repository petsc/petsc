
/*
   Simple example to show how PETSc programs can be run from MATLAB.
  See ex12.m.
*/

static char help[] = "Solves the one dimensional heat equation.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       M = 14,time_steps = 20,w=1,s=1,localsize,j,i,mybase,myend,globalsize;
  DM             da;
  Vec            global,local;
  PetscScalar    *globalptr,*localptr;
  PetscReal      h,k;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-time",&time_steps,NULL));

  /* Set up the array */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,M,w,s,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Make copy of local array for doing updates */
  CHKERRQ(DMCreateLocalVector(da,&local));

  /* determine starting point of each processor */
  CHKERRQ(VecGetOwnershipRange(global,&mybase,&myend));

  /* Initialize the Array */
  CHKERRQ(VecGetLocalSize (global,&globalsize));
  CHKERRQ(VecGetArray (global,&globalptr));

  for (i=0; i<globalsize; i++) {
    j = i + mybase;

    globalptr[i] = PetscSinReal((PETSC_PI*j*6)/((PetscReal)M) + 1.2 * PetscSinReal((PETSC_PI*j*2)/((PetscReal)M))) * 4+4;
  }

  CHKERRQ(VecRestoreArray(global,&localptr));

  /* Assign Parameters */
  h= 1.0/M;
  k= h*h/2.2;
  CHKERRQ(VecGetLocalSize(local,&localsize));

  for (j=0; j<time_steps; j++) {

    /* Global to Local */
    CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
    CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

    /*Extract local array */
    CHKERRQ(VecGetArray(local,&localptr));
    CHKERRQ(VecGetArray (global,&globalptr));

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      globalptr[i-1] = localptr[i] + (k/(h*h)) * (localptr[i+1]-2.0*localptr[i]+localptr[i-1]);
    }

    CHKERRQ(VecRestoreArray (global,&globalptr));
    CHKERRQ(VecRestoreArray(local,&localptr));

    /* View Wave */
    /* Set Up Display to Show Heat Graph */
#if defined(PETSC_USE_SOCKET_VIEWER)
    CHKERRQ(VecView(global,PETSC_VIEWER_SOCKET_WORLD));
#endif
  }

  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}
