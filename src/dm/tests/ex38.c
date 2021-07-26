
static char help[] = "Tests DMGlobalToLocal() for 3d DA with stencil width of 2.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt         N             = 3,M=2,P=4,dof=1,rstart,rend,i;
  PetscInt         stencil_width = 2;
  PetscErrorCode   ierr;
  PetscMPIInt      rank;
  DMBoundaryType   bx           = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz = DM_BOUNDARY_NONE;
  DMDAStencilType  stencil_type = DMDA_STENCIL_STAR;
  DM               da;
  Vec              global,local;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-stencil_width",&stencil_width,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-stencil_type",(PetscInt*)&stencil_type,NULL);CHKERRQ(ierr);

  ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stencil_type,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,0,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(global,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {ierr = VecSetValue(global,i,(PetscReal)(i + 100*rank),INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecAssemblyBegin(global);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = VecSet(local,-1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  if (rank == 0) {ierr = VecView(local,0);CHKERRQ(ierr);}
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
