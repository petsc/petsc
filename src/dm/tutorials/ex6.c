
static char help[] = "\n\n";

/*
     Demonstrates using DM_BOUNDARY_GHOSTED how to handle a rotated boundary conditions where one edge
    is connected to its immediate neighbor

    Consider the domain (with natural numbering)

     6   7   8
     3   4   5
     0   1   2

    The ghost points along the bottom (directly below the three columns above) should be 0 3 and 6
    while the ghost points along the left side should be 0 1 2

    Note that the ghosted local vectors extend in both the x and y directions so, for example if we have a
    single MPI process the ghosted vector has (in the original natural numbering)

     x  x  x  x  x
     2  6  7  8  x
     1  3  4  5  x
     0  0  1  2  x
     x  0  3  6  x

    where x indicates a location that is not updated by the communication and should be used.

    For this to make sense the number of grid points in the x and y directions must be the same

    This ghost point mapping was suggested by: Wenbo Zhao <zhaowenbo.npic@gmail.com>
*/

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt         M = 6;
  PetscErrorCode   ierr;
  DM               da;
  Vec              local,global,natural;
  PetscInt         i,start,end,*ifrom,x,y,xm,ym;
  PetscScalar      *xnatural;
  IS               from,to;
  AO               ao;
  VecScatter       scatter1,scatter2;
  PetscViewer      subviewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_STAR,M,M,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));

  /* construct global to local scatter for the left side of the domain to the ghost on the bottom */
  CHKERRQ(DMDAGetCorners(da,&x,&y,NULL,&xm,&ym,NULL));
  if (!y) { /* only processes on the bottom of the domain fill up the ghost locations */
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,xm,1,1,&to));
  } else {
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,0,&to));
  }
  CHKERRQ(PetscMalloc1(xm,&ifrom));
  for (i=x;i<x+xm;i++) {
    ifrom[i-x] = M*i;
  }
  CHKERRQ(DMDAGetAO(da,&ao));
  CHKERRQ(AOApplicationToPetsc(ao,xm,ifrom));
  if (!y) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,xm,ifrom,PETSC_OWN_POINTER,&from));
  } else {
    CHKERRQ(PetscFree(ifrom));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,0,NULL,PETSC_COPY_VALUES,&from));
  }
  CHKERRQ(VecScatterCreate(global,from,local,to,&scatter1));
  CHKERRQ(ISDestroy(&to));
  CHKERRQ(ISDestroy(&from));

  /* construct global to local scatter for the bottom side of the domain to the ghost on the right */
  if (!x) { /* only processes on the left side of the domain fill up the ghost locations */
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,ym,xm+2,xm+2,&to));
  } else {
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,0,&to));
  }
  CHKERRQ(PetscMalloc1(ym,&ifrom));
  for (i=y;i<y+ym;i++) {
    ifrom[i-y] = i;
  }
  CHKERRQ(DMDAGetAO(da,&ao));
  CHKERRQ(AOApplicationToPetsc(ao,ym,ifrom));
  if (!x) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,ym,ifrom,PETSC_OWN_POINTER,&from));
  } else {
    CHKERRQ(PetscFree(ifrom));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,0,NULL,PETSC_COPY_VALUES,&from));
  }
  CHKERRQ(VecScatterCreate(global,from,local,to,&scatter2));
  CHKERRQ(ISDestroy(&to));
  CHKERRQ(ISDestroy(&from));

  /*
     fill the global vector with the natural global numbering for each local entry
     this is only done for testing purposes since it is easy to see if the scatter worked correctly
  */
  CHKERRQ(DMDACreateNaturalVector(da,&natural));
  CHKERRQ(VecGetOwnershipRange(natural,&start,&end));
  CHKERRQ(VecGetArray(natural,&xnatural));
  for (i=start; i<end; i++) {
    xnatural[i-start] = i;
  }
  CHKERRQ(VecRestoreArray(natural,&xnatural));
  CHKERRQ(DMDANaturalToGlobalBegin(da,natural,INSERT_VALUES,global));
  CHKERRQ(DMDANaturalToGlobalEnd(da,natural,INSERT_VALUES,global));
  CHKERRQ(VecDestroy(&natural));

  /* scatter from global to local */
  CHKERRQ(VecScatterBegin(scatter1,global,local,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scatter1,global,local,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterBegin(scatter2,global,local,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scatter2,global,local,INSERT_VALUES,SCATTER_FORWARD));
  /*
     normally here you would also call
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));
    to update all the interior ghost cells between neighboring processes.
    We don't do it here since this is only a test of "special" ghost points.
  */

  /* view each local ghosted vector */
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&subviewer));
  CHKERRQ(VecView(local,subviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&subviewer));

  CHKERRQ(VecScatterDestroy(&scatter1));
  CHKERRQ(VecScatterDestroy(&scatter2));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

   test:
      suffix: 4
      nsize: 4

   test:
      suffix: 9
      nsize: 9

TEST*/
