
/*
   Simple example to show how PETSc programs can be run from MATLAB. 
  See ex12.m.
*/

static char help[] = "Solves the one dimensional heat equation.\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       M = 14,time_steps = 20,w=1,s=1,localsize,j,i,mybase,myend;
  PetscErrorCode ierr;
  DM             da;
  Vec            local,global,copy;
  PetscScalar    *localptr,*copyptr;
  PetscReal      h,k;
 
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-time",&time_steps,PETSC_NULL);CHKERRQ(ierr);
    
  /* Set up the array */ 
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,M,w,s,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy);CHKERRQ(ierr);
  ierr = VecGetArray (copy,&copyptr);CHKERRQ(ierr);


  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRQ(ierr);

  /* Initialize the Array */
  ierr = VecGetLocalSize (local,&localsize);CHKERRQ(ierr);
  ierr = VecGetArray (local,&localptr);CHKERRQ(ierr);
  localptr[0] = copyptr[0] = 0.0;
  localptr[localsize-1] = copyptr[localsize-1] = 1.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin((PETSC_PI*j*6)/((PetscReal)M) 
                        + 1.2 * sin((PETSC_PI*j*2)/((PetscReal)M))) * 4+4;
  }

  ierr = VecRestoreArray (copy,&copyptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(local,&localptr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,INSERT_VALUES,global);CHKERRQ(ierr);

  /* Assign Parameters */
  h= 1.0/M; 
  k= h*h/2.2;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

    /*Extract local array */ 
    ierr = VecGetArray(local,&localptr);CHKERRQ(ierr);
    ierr = VecGetArray (copy,&copyptr);CHKERRQ(ierr);

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      copyptr[i] = localptr[i] + (k/(h*h)) *
                           (localptr[i+1]-2.0*localptr[i]+localptr[i-1]);
    }
  
    ierr = VecRestoreArray (copy,&copyptr);CHKERRQ(ierr);
    ierr = VecRestoreArray(local,&localptr);CHKERRQ(ierr);

    /* Local to Global */
    ierr = DMLocalToGlobalBegin(da,copy,INSERT_VALUES,global);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da,copy,INSERT_VALUES,global);CHKERRQ(ierr);
  
    /* View Wave */ 
  /* Set Up Display to Show Heat Graph */
#if defined(PETSC_USE_SOCKET_VIEWER)
    ierr = VecView(global,PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr);
#endif
  }

  ierr = VecDestroy(&copy);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
