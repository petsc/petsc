
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-time",&time_steps,NULL));

  /* Set up the array */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,M,w,s,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Make copy of local array for doing updates */
  PetscCall(DMCreateLocalVector(da,&local));

  /* determine starting point of each processor */
  PetscCall(VecGetOwnershipRange(global,&mybase,&myend));

  /* Initialize the Array */
  PetscCall(VecGetLocalSize (global,&globalsize));
  PetscCall(VecGetArray (global,&globalptr));

  for (i=0; i<globalsize; i++) {
    j = i + mybase;

    globalptr[i] = PetscSinReal((PETSC_PI*j*6)/((PetscReal)M) + 1.2 * PetscSinReal((PETSC_PI*j*2)/((PetscReal)M))) * 4+4;
  }

  PetscCall(VecRestoreArray(global,&localptr));

  /* Assign Parameters */
  h= 1.0/M;
  k= h*h/2.2;
  PetscCall(VecGetLocalSize(local,&localsize));

  for (j=0; j<time_steps; j++) {

    /* Global to Local */
    PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
    PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

    /*Extract local array */
    PetscCall(VecGetArray(local,&localptr));
    PetscCall(VecGetArray (global,&globalptr));

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      globalptr[i-1] = localptr[i] + (k/(h*h)) * (localptr[i+1]-2.0*localptr[i]+localptr[i-1]);
    }

    PetscCall(VecRestoreArray (global,&globalptr));
    PetscCall(VecRestoreArray(local,&localptr));

    /* View Wave */
    /* Set Up Display to Show Heat Graph */
#if defined(PETSC_USE_SOCKET_VIEWER)
    PetscCall(VecView(global,PETSC_VIEWER_SOCKET_WORLD));
#endif
  }

  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
