/*$Id: ex3.c,v 1.11 2000/01/11 21:01:03 bsmith Exp balay $*/

static char help[] = "Tests relaxation for dense matrices.\n\n"; 

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  Vec         u,x,b,e;
  int         i,n = 10,midx[3],ierr;
  Scalar      v[3],one = 1.0,zero = 0.0,mone = -1.0;
  double      omega = 1.0,norm;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetDouble(PETSC_NULL,"-omega",&omega,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,PETSC_NULL,&C);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&e);CHKERRA(ierr);
  ierr = VecSet(&one,u);CHKERRA(ierr);
  ierr = VecSet(&zero,x);CHKERRA(ierr);

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for (i=1; i<n-1; i++){
    midx[0] = i-1; midx[1] = i; midx[2] = i+1;
    ierr = MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES);CHKERRA(ierr);
  }
  i = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRA(ierr);
  i = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRA(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = MatMult(C,u,b);CHKERRA(ierr);

  for (i=0; i<n; i++) {
    ierr = MatRelax(C,b,omega,SOR_FORWARD_SWEEP,0.0,1,x);CHKERRA(ierr);
    ierr = VecWAXPY(&mone,x,u,e);CHKERRA(ierr);
    ierr = VecNorm(e,NORM_2,&norm);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"2-norm of error %g\n",norm);CHKERRA(ierr);
  }
  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(e);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

 
