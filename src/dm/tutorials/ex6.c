
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
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_STAR,M,M,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  /* construct global to local scatter for the left side of the domain to the ghost on the bottom */
  ierr = DMDAGetCorners(da,&x,&y,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  if (!y) { /* only processes on the bottom of the domain fill up the ghost locations */
    ierr = ISCreateStride(PETSC_COMM_SELF,xm,1,1,&to);CHKERRQ(ierr);
  } else {
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,0,&to);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(xm,&ifrom);CHKERRQ(ierr);
  for (i=x;i<x+xm;i++) {
    ifrom[i-x] = M*i;
  }
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,xm,ifrom);CHKERRQ(ierr);
  if (!y) {
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,xm,ifrom,PETSC_OWN_POINTER,&from);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(ifrom);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,0,NULL,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  }
  ierr = VecScatterCreate(global,from,local,to,&scatter1);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

  /* construct global to local scatter for the bottom side of the domain to the ghost on the right */
  if (!x) { /* only processes on the left side of the domain fill up the ghost locations */
    ierr = ISCreateStride(PETSC_COMM_SELF,ym,xm+2,xm+2,&to);CHKERRQ(ierr);
  } else {
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,0,&to);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(ym,&ifrom);CHKERRQ(ierr);
  for (i=y;i<y+ym;i++) {
    ifrom[i-y] = i;
  }
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,ym,ifrom);CHKERRQ(ierr);
  if (!x) {
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,ym,ifrom,PETSC_OWN_POINTER,&from);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(ifrom);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,0,NULL,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  }
  ierr = VecScatterCreate(global,from,local,to,&scatter2);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

  /*
     fill the global vector with the natural global numbering for each local entry
     this is only done for testing purposes since it is easy to see if the scatter worked correctly
  */
  ierr = DMDACreateNaturalVector(da,&natural);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(natural,&start,&end);CHKERRQ(ierr);
  ierr = VecGetArray(natural,&xnatural);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    xnatural[i-start] = i;
  }
  ierr = VecRestoreArray(natural,&xnatural);CHKERRQ(ierr);
  ierr = DMDANaturalToGlobalBegin(da,natural,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMDANaturalToGlobalEnd(da,natural,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = VecDestroy(&natural);CHKERRQ(ierr);

  /* scatter from global to local */
  ierr = VecScatterBegin(scatter1,global,local,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter1,global,local,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter2,global,local,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter2,global,local,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /*
     normally here you would also call
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
    to update all the interior ghost cells between neighboring processes.
    We don't do it here since this is only a test of "special" ghost points.
  */

  /* view each local ghosted vector */
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);
  ierr = VecView(local,subviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&subviewer);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&scatter1);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter2);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
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
