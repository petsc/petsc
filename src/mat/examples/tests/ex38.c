
static char help[] = "Tests MatSetValues().\n\n"; 

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C; 
  PetscInt       i,n = 5,midx[3],nidx[2];
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscScalar    v[6];

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-row_oriented",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatSetOption(C,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);}
  for (i=0; i<6; i++) v[i] = (PetscReal)i;
  midx[0] = 0; midx[1] = 2; midx[2] = 3;
  nidx[0] = 1; nidx[1] = 3;
  ierr = MatSetValues(C,3,midx,2,nidx,v,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

 
