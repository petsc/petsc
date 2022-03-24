
static char help[] = "Tests DMGlobalToLocal() for 3d DA with stencil width of 2.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt         N             = 3,M=2,P=4,dof=1,rstart,rend,i;
  PetscInt         stencil_width = 2;
  PetscMPIInt      rank;
  DMBoundaryType   bx           = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz = DM_BOUNDARY_NONE;
  DMDAStencilType  stencil_type = DMDA_STENCIL_STAR;
  DM               da;
  Vec              global,local;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-stencil_type",(PetscInt*)&stencil_type,NULL));

  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(VecGetOwnershipRange(global,&rstart,&rend));
  for (i=rstart; i<rend; i++) CHKERRQ(VecSetValue(global,i,(PetscReal)(i + 100*rank),INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(global));
  CHKERRQ(VecAssemblyEnd(global));
  CHKERRQ(DMCreateLocalVector(da,&local));
  CHKERRQ(VecSet(local,-1));
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));
  if (rank == 0) CHKERRQ(VecView(local,0));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(PetscFinalize());
  return 0;
}
