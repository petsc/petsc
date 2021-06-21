
static char help[] = "Solves the 1-dimensional wave equation.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscInt       M = 60,time_steps = 100, localsize,j,i,mybase,myend,width,xbase,*localnodes = NULL;
  DM             da;
  PetscViewer    viewer,viewer_private;
  PetscDraw      draw;
  Vec            local,global;
  PetscScalar    *localptr,*globalptr;
  PetscReal      a,h,k;
  PetscBool      flg = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-time",&time_steps,NULL);CHKERRQ(ierr);
  /*
      Test putting two nodes on each processor, exact last processor gets the rest
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-distribute",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscMalloc1(size,&localnodes);CHKERRQ(ierr);
    for (i=0; i<size-1; i++) localnodes[i] = 2;
    localnodes[size-1] = M - 2*(size-1);
  }

  /* Set up the array */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,M,1,1,localnodes,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
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
  ierr  = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Local Portion of Solution",xbase,200,width,200,&viewer_private);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer_private,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);

  /* Initialize the array */
  ierr = VecGetLocalSize(local,&localsize);CHKERRQ(ierr);
  ierr = VecGetArray(global,&globalptr);CHKERRQ(ierr);

  for (i=1; i<localsize-1; i++) {
    j           = (i-1)+mybase;
    globalptr[i-1] = PetscSinReal((PETSC_PI*j*6)/((PetscReal)M) + 1.2 * PetscSinReal((PETSC_PI*j*2)/((PetscReal)M))) * 2;
  }

  ierr = VecRestoreArray(global,&globalptr);CHKERRQ(ierr);

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
    ierr = VecGetArray(global,&globalptr);CHKERRQ(ierr);

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i=1; i< localsize-1; i++) {
      globalptr[i-1] = .5*(localptr[i+1]+localptr[i-1]) - (k / (2.0*a*h)) * (localptr[i+1] - localptr[i-1]);
    }
    ierr = VecRestoreArray(global,&globalptr);CHKERRQ(ierr);
    ierr = VecRestoreArray(local,&localptr);CHKERRQ(ierr);

    /* View my part of Wave */
    ierr = VecView(global,viewer_private);CHKERRQ(ierr);

    /* View global Wave */
    ierr = VecView(global,viewer);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_private);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      nsize: 3
      args: -time 50 -nox
      requires: x

TEST*/
