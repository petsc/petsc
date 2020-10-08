
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
  PetscErrorCode ierr;
  DM             da;
  Vec            global,local;
  PetscScalar    *globalptr,*localptr;
  PetscReal      h,k;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-time",&time_steps,NULL);CHKERRQ(ierr);

  /* Set up the array */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,M,w,s,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Make copy of local array for doing updates */
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);


  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRQ(ierr);

  /* Initialize the Array */
  ierr = VecGetLocalSize (global,&globalsize);CHKERRQ(ierr);
  ierr = VecGetArray (global,&globalptr);CHKERRQ(ierr);


  for (i=0; i<globalsize; i++) {
    j = i + mybase;

    globalptr[i] = PetscSinReal((PETSC_PI*j*6)/((PetscReal)M) + 1.2 * PetscSinReal((PETSC_PI*j*2)/((PetscReal)M))) * 4+4;
  }

  ierr = VecRestoreArray(global,&localptr);CHKERRQ(ierr);

  /* Assign Parameters */
  h= 1.0/M;
  k= h*h/2.2;
  ierr = VecGetLocalSize(local,&localsize);CHKERRQ(ierr);

  for (j=0; j<time_steps; j++) {

    /* Global to Local */
    ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

    /*Extract local array */
    ierr = VecGetArray(local,&localptr);CHKERRQ(ierr);
    ierr = VecGetArray (global,&globalptr);CHKERRQ(ierr);

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      globalptr[i-1] = localptr[i] + (k/(h*h)) * (localptr[i+1]-2.0*localptr[i]+localptr[i-1]);
    }

    ierr = VecRestoreArray (global,&globalptr);CHKERRQ(ierr);
    ierr = VecRestoreArray(local,&localptr);CHKERRQ(ierr);

    /* View Wave */
    /* Set Up Display to Show Heat Graph */
#if defined(PETSC_USE_SOCKET_VIEWER)
    ierr = VecView(global,PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr);
#endif
  }

  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

