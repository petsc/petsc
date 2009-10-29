
static char help[] = "Tests relaxation for dense matrices.\n\n"; 

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C; 
  Vec            u,x,b,e;
  PetscInt       i,n = 10,midx[3];
  PetscErrorCode ierr;
  PetscScalar    v[3];
  PetscReal      omega = 1.0,norm;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-omega",&omega,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,n,n,n,n);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQDENSE);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&e);CHKERRQ(ierr);
  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for (i=1; i<n-1; i++){
    midx[0] = i-1; midx[1] = i; midx[2] = i+1;
    ierr = MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  i = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRQ(ierr);
  i = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatMult(C,u,b);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    ierr = MatSOR(C,b,omega,SOR_FORWARD_SWEEP,0.0,1,1,x);CHKERRQ(ierr);
    ierr = VecWAXPY(e,-1.0,x,u);CHKERRQ(ierr);
    ierr = VecNorm(e,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"2-norm of error %G\n",norm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(e);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

 
