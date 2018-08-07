static const char help[] = "Test DMDAGetOwnershipRanges()\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  DM             da;
  PetscViewer    vw;
  PetscInt       dim = 2,m,n,p;
  const PetscInt *lx,*ly,*lz;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,0,"-dim",&dim,0);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, 3,5,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, 3,5,7,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,NULL,&da);CHKERRQ(ierr);
    break;
  default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for %D dimensions",dim);
  }
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, 0,0,0, &m,&n,&p, 0,0, 0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&vw);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(vw,"[%d] lx ly%s\n",rank,dim>2 ? " lz" : "");CHKERRQ(ierr);
  ierr = PetscIntView(m,lx,vw);CHKERRQ(ierr);
  ierr = PetscIntView(n,ly,vw);CHKERRQ(ierr);
  if (dim > 2) {ierr = PetscIntView(n,lz,vw);CHKERRQ(ierr);}
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&vw);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 12
      args: -dm_view -dim 3 -da_grid_x 11 -da_grid_y 5 -da_grid_z 7

TEST*/
