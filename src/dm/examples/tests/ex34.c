static const char help[] = "Test DMDAGetOwnershipRanges()\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  DM da;
  PetscInt dim = 2,m,n,p,i;
  const PetscInt *lx,*ly,*lz;
  PetscMPIInt rank,size;

  PetscInitialize(&argc,&argv,0,help);
  ierr = PetscOptionsGetInt(0,"-dim",&dim,0);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR, -3,-5,PETSC_DECIDE,PETSC_DECIDE,2,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR, -3,-5,-7,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
    break;
  default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for %D dimensions",dim);
  }
  ierr = DMDAGetInfo(da, 0, 0,0,0, &m,&n,&p, 0,0, 0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    if (i == rank) {
      ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"[%d] lx ly%s\n",rank,dim>2?" lz":"");CHKERRQ(ierr);
      ierr = PetscIntView(m,lx,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      ierr = PetscIntView(n,ly,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      if (dim > 2) {ierr = PetscIntView(n,lz,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}
    }
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
