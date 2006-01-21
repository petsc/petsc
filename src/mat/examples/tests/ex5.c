 
static char help[] = "Tests MatMult(), MatMultAdd(), MatMultTranspose().\n\
Also MatMultTransposeAdd(), MatScale(), MatGetDiagonal(), and MatDiagonalScale().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C; 
  Vec            s,u,w,x,y,z;
  PetscErrorCode ierr;
  PetscInt       i,j,m = 8,n,rstart,rend,vstart,vend;
  PetscScalar    one = 1.0,negone = -1.0,v,alpha=0.1;
  PetscReal      norm;
  PetscTruth     flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  n = m;
  ierr = PetscOptionsHasName(PETSC_NULL,"-rectA",&flg);CHKERRQ(ierr);
  if (flg) n += 2;
  ierr = PetscOptionsHasName(PETSC_NULL,"-rectB",&flg);CHKERRQ(ierr);
  if (flg) n -= 2;

  /* ---------- Assemble matrix and vectors ----------- */

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&z);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&s);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y,&vstart,&vend);CHKERRQ(ierr);

  /* Assembly */
  for (i=rstart; i<rend; i++) { 
    v = 100*(i+1);
    ierr = VecSetValues(z,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    for (j=0; j<n; j++) { 
      v=10*(i+1)+j+1; 
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Flush off proc Vec values and do more assembly */
  ierr = VecAssemblyBegin(z);CHKERRQ(ierr);
  for (i=vstart; i<vend; i++) {
    v = one*((PetscReal)i);
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    v = 100.0*i;
    ierr = VecSetValues(u,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Flush off proc Mat values and do more assembly */
  ierr = MatAssemblyBegin(C,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) { 
    for (j=0; j<n; j++) { 
      v=10*(i+1)+j+1; 
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  /* Try overlap Coomunication with the next stage XXXSetValues */
  ierr = VecAssemblyEnd(z);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  CHKMEMQ;
  /* The Assembly for the second Stage */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = MatScale(C,alpha);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);

  /* ------------ Test MatMult(), MatMultAdd()  ---------- */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatMult()\n");CHKERRQ(ierr);
  CHKMEMQ;
  ierr = MatMult(C,y,x);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatMultAdd()\n");CHKERRQ(ierr);
  ierr = MatMultAdd(C,y,z,w);CHKERRQ(ierr);
  ierr = VecAXPY(x,one,z);CHKERRQ(ierr);
  ierr = VecAXPY(x,negone,w);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > 1.e-8){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error difference = %G\n",norm);CHKERRQ(ierr);
  }

  /* ------- Test MatMultTranspose(), MatMultTransposeAdd() ------- */

  for (i=rstart; i<rend; i++) {
    v = one*((PetscReal)i);
    ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatMultTranspose()\n");CHKERRQ(ierr);
  ierr = MatMultTranspose(C,x,y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatMultTransposeAdd()\n");CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(C,x,u,s);CHKERRQ(ierr);
  ierr = VecAXPY(y,one,u);CHKERRQ(ierr);
  ierr = VecAXPY(y,negone,s);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > 1.e-8){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error difference = %G\n",norm);CHKERRQ(ierr);
  }

  /* -------------------- Test MatGetDiagonal() ------------------ */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatGetDiagonal(), MatDiagonalScale()\n");CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = MatGetDiagonal(C,x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  for (i=vstart; i<vend; i++) {
    v = one*((PetscReal)(i+1));
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* -------------------- Test () MatDiagonalScale ------------------ */
  ierr = PetscOptionsHasName(PETSC_NULL,"-test_diagonalscale",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatDiagonalScale(C,x,y);CHKERRQ(ierr);
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  }
  /* Free data structures */
  ierr = VecDestroy(u);CHKERRQ(ierr); ierr = VecDestroy(s);CHKERRQ(ierr); 
  ierr = VecDestroy(w);CHKERRQ(ierr); ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr); ierr = VecDestroy(z);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
