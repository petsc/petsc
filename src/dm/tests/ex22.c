
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-P",&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-w",&w,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-star",&flg,NULL);CHKERRQ(ierr);
  if (flg) stencil_type =  DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,stencil_type,M,N,P,m,n,p,w,s,0,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&mat);CHKERRQ(ierr);

  idx[0].i = 1;   idx[0].j = 1; idx[0].k = 0;
  idx[1].i = 2;   idx[1].j = 1; idx[1].k = 0;
  idy[0].i = 1;   idy[0].j = 2; idy[0].k = 0;
  idy[1].i = 2;   idy[1].j = 2; idy[1].k = 0;
  ierr     = PetscMalloc1(2*2*w*w,&values);CHKERRQ(ierr);
  for (i=0; i<2*2*w*w; i++) values[i] = i;
  ierr = MatSetValuesBlockedStencil(mat,2,idx,2,idy,values,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free memory */
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

