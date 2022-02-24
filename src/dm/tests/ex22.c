
static char help[] = "Tests MatSetValuesBlockedStencil() in 3d.\n\n";

#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt        M = 3,N = 4,P = 2,s = 1,w = 2,i, m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  PetscErrorCode  ierr;
  DM              da;
  Mat             mat;
  DMDAStencilType stencil_type = DMDA_STENCIL_BOX;
  PetscBool       flg          = PETSC_FALSE;
  MatStencil      idx[2],idy[2];
  PetscScalar     *values;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-w",&w,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-star",&flg,NULL));
  if (flg) stencil_type =  DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,stencil_type,M,N,P,m,n,p,w,s,0,0,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMSetMatType(da,MATMPIBAIJ));
  CHKERRQ(DMCreateMatrix(da,&mat));

  idx[0].i = 1;   idx[0].j = 1; idx[0].k = 0;
  idx[1].i = 2;   idx[1].j = 1; idx[1].k = 0;
  idy[0].i = 1;   idy[0].j = 2; idy[0].k = 0;
  idy[1].i = 2;   idy[1].j = 2; idy[1].k = 0;
  CHKERRQ(PetscMalloc1(2*2*w*w,&values));
  for (i=0; i<2*2*w*w; i++) values[i] = i;
  CHKERRQ(MatSetValuesBlockedStencil(mat,2,idx,2,idy,values,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* Free memory */
  CHKERRQ(PetscFree(values));
  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}
