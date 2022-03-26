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

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(PetscOptionsGetInt(NULL,0,"-dim",&dim,0));
  switch (dim) {
  case 2:
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, 3,5,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
    break;
  case 3:
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, 3,5,7,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,NULL,&da));
    break;
  default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for %D dimensions",dim);
  }
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetInfo(da, 0, 0,0,0, &m,&n,&p, 0,0, 0,0,0,0));
  PetscCall(DMDAGetOwnershipRanges(da,&lx,&ly,&lz));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&vw));
  PetscCall(PetscViewerASCIIPrintf(vw,"[%d] lx ly%s\n",rank,dim>2 ? " lz" : ""));
  PetscCall(PetscIntView(m,lx,vw));
  PetscCall(PetscIntView(n,ly,vw));
  if (dim > 2) PetscCall(PetscIntView(n,lz,vw));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&vw));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 12
      args: -dm_view -dim 3 -da_grid_x 11 -da_grid_y 5 -da_grid_z 7

TEST*/
