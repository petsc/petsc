      
static char help[] = "Tests MatSetValuesBlockedStencil() in 3d.\n\n";

#include <petscmat.h>
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M = 3,N = 4,P = 2,s = 1,w = 2,i, m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE;
  PetscErrorCode ierr;
  DM             da;
  Mat            mat;
  DMDAStencilType  stencil_type = DMDA_STENCIL_BOX;
  PetscBool      flg = PETSC_FALSE;
  MatStencil     idx[2],idy[2];
  PetscScalar    *values;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Read options */  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-star",&flg,PETSC_NULL);CHKERRQ(ierr); 
  if (flg) stencil_type =  DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,stencil_type,M,N,P,m,n,p,w,s,
                    0,0,0,&da);CHKERRQ(ierr);

  ierr = DMCreateMatrix(da,MATMPIBAIJ,&mat);CHKERRQ(ierr);

  idx[0].i = 1;   idx[0].j = 1; idx[0].k = 0;
  idx[1].i = 2;   idx[1].j = 1; idx[1].k = 0;
  idy[0].i = 1;   idy[0].j = 2; idy[0].k = 0;
  idy[1].i = 2;   idy[1].j = 2; idy[1].k = 0;
  ierr = PetscMalloc(2*2*w*w*sizeof(PetscScalar),&values);CHKERRQ(ierr);
  for ( i=0; i<2*2*w*w; i++ ) values[i] = i;
  ierr = MatSetValuesBlockedStencil(mat,2,idx,2,idy,values,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free memory */
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr); 
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
  




















