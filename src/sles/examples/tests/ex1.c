/*$Id: ex1.c,v 1.9 1999/05/04 20:35:14 balay Exp bsmith $*/

static char help[] = "Tests solving linear system on 0 by 0 matrix.\n\n";

#include "sles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         ierr, N = 0, its;
  Vec         u, b, x;
  SLES        sles;
  Scalar      zero = 0.0, mone = -1.0;
  double      norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* create stiffness matrix */
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,N,N,&C);CHKERRA(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* create right hand side and solution */

  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&u);CHKERRA(ierr); 
  ierr = VecDuplicate(u,&b);CHKERRA(ierr);
  ierr = VecDuplicate(u,&x);CHKERRA(ierr);
  ierr = VecSet(&zero,u);CHKERRA(ierr);
  ierr = VecSet(&zero,b);CHKERRA(ierr);

  ierr = VecAssemblyBegin(b);CHKERRA(ierr);
  ierr = VecAssemblyEnd(b);CHKERRA(ierr);


  /* solve linear system */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
  ierr = SLESSolve(sles,b,u,&its);CHKERRA(ierr);

  ierr = MatMult(C,u,x);CHKERRA(ierr);
  ierr = VecAXPY(&mone,b,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  printf("Norm of residual %g\n",norm);

  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
