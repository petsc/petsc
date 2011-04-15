
static char help[] = "Solves the 1-dimensional wave equation.\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscInt       M = 60,time_steps = 100, localsize,j,i,mybase,myend,width,xbase,*localnodes = PETSC_NULL;
  DM             da;
  PetscViewer    viewer,viewer_private;
  PetscDraw      draw;
  Vec            local,global,copy;
  PetscScalar    *localptr,*copyptr;
  PetscReal      a,h,k;
  PetscBool      flg = PETSC_FALSE;
 
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-time",&time_steps,PETSC_NULL);CHKERRQ(ierr);
  /*
      Test putting two nodes on each processor, exact last processor gets the rest
  */
  ierr = PetscOptionsGetBool(PETSC_NULL,"-distribute",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscMalloc(size*sizeof(PetscInt),&localnodes);CHKERRQ(ierr);
    for (i=0; i<size-1; i++) { localnodes[i] = 2;}
    localnodes[size-1] = M - 2*(size-1);
  }
    
  /* Set up the array */ 
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,M,1,1,localnodes,&da);CHKERRQ(ierr);
  ierr = PetscFree(localnodes);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  /* Set up display to show combined wave graph */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"Entire Solution",20,480,800,200,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);

  /* determine starting point of each processor */
  ierr = VecGetOwnershipRange(global,&mybase,&myend);CHKERRQ(ierr);

  /* set up display to show my portion of the wave */
  xbase = (int)((mybase)*((800.0 - 4.0*size)/M) + 4.0*rank);
  width = (int)((myend-mybase)*800./M);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Local Portion of Solution",xbase,200,
                         width,200,&viewer_private);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer_private,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);



  /* Initialize the array */
  ierr = VecGetLocalSize(local,&localsize);CHKERRQ(ierr);
  ierr = VecGetArray(local,&localptr);CHKERRQ(ierr);
  localptr[0] = 0.0;
  localptr[localsize-1] = 0.0;
  for (i=1; i<localsize-1; i++) {
    j=(i-1)+mybase; 
    localptr[i] = sin((PETSC_PI*j*6)/((PetscReal)M) 
                        + 1.2 * sin((PETSC_PI*j*2)/((PetscReal)M))) * 2;
  }

  ierr = VecRestoreArray(local,&localptr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,INSERT_VALUES,global);CHKERRQ(ierr);

  /* Make copy of local array for doing updates */
  ierr = VecDuplicate(local,&copy);CHKERRQ(ierr);

  /* Assign Parameters */
  a= 1.0;
  h= 1.0/M;
  k= h;

  for (j=0; j<time_steps; j++) {  

    /* Global to Local */
    ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

    /*Extract local array */ 
    ierr = VecGetArray(local,&localptr);CHKERRQ(ierr);
    ierr = VecGetArray(copy,&copyptr);CHKERRQ(ierr);

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      copyptr[i] = .5*(localptr[i+1]+localptr[i-1]) - 
                    (k / (2.0*a*h)) * (localptr[i+1] - localptr[i-1]);
    }
    ierr = VecRestoreArray(copy,&copyptr);CHKERRQ(ierr);
    ierr = VecRestoreArray(local,&localptr);CHKERRQ(ierr);

    /* Local to Global */
    ierr = DMLocalToGlobalBegin(da,copy,INSERT_VALUES,global);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da,copy,INSERT_VALUES,global);CHKERRQ(ierr);
  
    /* View my part of Wave */ 
    ierr = VecView(copy,viewer_private);CHKERRQ(ierr);

    /* View global Wave */ 
    ierr = VecView(global,viewer);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_private);CHKERRQ(ierr);
  ierr = VecDestroy(&copy);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
 




