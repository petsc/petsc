/*$Id: ex5.c,v 1.18 2000/05/05 22:16:17 balay Exp bsmith $*/
 
static char help[] = "Tests MatMult(), MatMultAdd(), MatMultTranspose(),\n\
MatMultTransposeAdd(), MatScale(), MatGetDiagonal(), and MatDiagonalScale().\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat        C; 
  Vec        s,u,w,x,y,z;
  int        ierr,i,j,m = 8,n,rstart,rend,vstart,vend;
  Scalar     one = 1.0,negone = -1.0,v,alpha=0.1;
  double     norm;
  PetscTruth flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_COMMON,0);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  n = m;
  ierr = OptionsHasName(PETSC_NULL,"-rectA",&flg);CHKERRA(ierr);
  if (flg) n += 2;
  ierr = OptionsHasName(PETSC_NULL,"-rectB",&flg);CHKERRA(ierr);
  if (flg) n -= 2;

  /* ---------- Assemble matrix and vectors ----------- */

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&C);CHKERRA(ierr);
  ierr = MatSetFromOptions(C);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRA(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&z);CHKERRA(ierr);
  ierr = VecDuplicate(x,&w);CHKERRA(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&y);CHKERRA(ierr);
  ierr = VecSetFromOptions(y);CHKERRA(ierr);
  ierr = VecDuplicate(y,&u);CHKERRA(ierr);
  ierr = VecDuplicate(y,&s);CHKERRA(ierr);
  ierr = VecGetOwnershipRange(y,&vstart,&vend);CHKERRA(ierr);

  /* Assembly */
  for (i=rstart; i<rend; i++) { 
    v = 100*(i+1);
    ierr = VecSetValues(z,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
    for (j=0; j<n; j++) { 
      v=10*(i+1)+j+1; 
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }

  /* Flush off proc Vec values and do more assembly */
  ierr = VecAssemblyBegin(z);CHKERRA(ierr);
  for (i=vstart; i<vend; i++) {
    v = one*((double)i);
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
    v = 100.0*i;
    ierr = VecSetValues(u,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
  }

  /* Flush off proc Mat values and do more assembly */
  ierr = MatAssemblyBegin(C,MAT_FLUSH_ASSEMBLY);CHKERRA(ierr);
  for (i=rstart; i<rend; i++) { 
    for (j=0; j<n; j++) { 
      v=10*(i+1)+j+1; 
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  /* Try overlap Coomunication with the next stage XXXSetValues */
  ierr = VecAssemblyEnd(z);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FLUSH_ASSEMBLY);CHKERRA(ierr);

  /* The Assembly for the second Stage */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = VecAssemblyBegin(y);CHKERRA(ierr);
  ierr = VecAssemblyEnd(y);CHKERRA(ierr);
  ierr = MatScale(&alpha,C);CHKERRA(ierr);
  ierr = VecAssemblyBegin(u);CHKERRA(ierr);
  ierr = VecAssemblyEnd(u);CHKERRA(ierr);

  /* ------------ Test MatMult(), MatMultAdd()  ---------- */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatMult()\n");CHKERRA(ierr);
  ierr = MatMult(C,y,x);CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatMultAdd()\n");CHKERRA(ierr);
  ierr = MatMultAdd(C,y,z,w);CHKERRA(ierr);
  ierr = VecAXPY(&one,z,x);CHKERRA(ierr);
  ierr = VecAXPY(&negone,w,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  if (norm > 1.e-8){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error difference = %g\n",norm);CHKERRA(ierr);
  }

  /* ------- Test MatMultTranspose(), MatMultTransposeAdd() ------- */

  for (i=rstart; i<rend; i++) {
    v = one*((double)i);
    ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatMultTranspose()\n");CHKERRA(ierr);
  ierr = MatMultTranspose(C,x,y);CHKERRA(ierr);
  ierr = VecView(y,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatMultTransposeAdd()\n");CHKERRA(ierr);
  ierr = MatMultTransposeAdd(C,x,u,s);CHKERRA(ierr);
  ierr = VecAXPY(&one,u,y);CHKERRA(ierr);
  ierr = VecAXPY(&negone,s,y);CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRA(ierr);
  if (norm > 1.e-8){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error difference = %g\n",norm);CHKERRA(ierr);
  }

  /* -------------------- Test MatGetDiagonal() ------------------ */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"testing MatGetDiagonal(), MatDiagonalScale()\n");CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = VecSet(&one,x);CHKERRA(ierr);
  ierr = MatGetDiagonal(C,x);CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  for (i=vstart; i<vend; i++) {
    v = one*((double)(i+1));
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
  }

  /* -------------------- Test () MatDiagonalScale ------------------ */
  ierr = OptionsHasName(PETSC_NULL,"-test_diagonalscale",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = MatDiagonalScale(C,x,y);CHKERRA(ierr);
    ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr); 
  }
  /* Free data structures */
  ierr = VecDestroy(u);CHKERRA(ierr); ierr = VecDestroy(s);CHKERRA(ierr); 
  ierr = VecDestroy(w);CHKERRA(ierr); ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr); ierr = VecDestroy(z);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
