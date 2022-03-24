static const char help[] = "Test DMDAGetOwnershipRanges()\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char *argv[])
{
  DM             da;
  PetscViewer    vw;
  PetscInt       dim = 2,m,n,p;
  const PetscInt *lx,*ly,*lz;
  PetscMPIInt    rank;

  CHKERRQ(PetscInitialize(&argc,&argv,0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-dim",&dim,0));
  switch (dim) {
  case 2:
    CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, 3,5,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
    break;
  case 3:
    CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, 3,5,7,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,NULL,&da));
    break;
  default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for %D dimensions",dim);
  }
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDAGetInfo(da, 0, 0,0,0, &m,&n,&p, 0,0, 0,0,0,0));
  CHKERRQ(DMDAGetOwnershipRanges(da,&lx,&ly,&lz));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&vw));
  CHKERRQ(PetscViewerASCIIPrintf(vw,"[%d] lx ly%s\n",rank,dim>2 ? " lz" : ""));
  CHKERRQ(PetscIntView(m,lx,vw));
  CHKERRQ(PetscIntView(n,ly,vw));
  if (dim > 2) CHKERRQ(PetscIntView(n,lz,vw));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&vw));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 12
      args: -dm_view -dim 3 -da_grid_x 11 -da_grid_y 5 -da_grid_z 7

TEST*/
