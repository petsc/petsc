
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-stencil_type",(PetscInt*)&stencil_type,NULL));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(VecGetOwnershipRange(global,&rstart,&rend));
  for (i=rstart; i<rend; i++) PetscCall(VecSetValue(global,i,(PetscReal)(i + 100*rank),INSERT_VALUES));
  PetscCall(VecAssemblyBegin(global));
  PetscCall(VecAssemblyEnd(global));
  PetscCall(DMCreateLocalVector(da,&local));
  PetscCall(VecSet(local,-1));
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));
  if (rank == 0) PetscCall(VecView(local,0));
  PetscCall(DMDestroy(&da));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(PetscFinalize());
  return 0;
}
