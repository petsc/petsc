/*$Id: ex1.c,v 1.18 2001/08/07 21:30:50 bsmith Exp $*/

static char help[] = "Tests solving linear system on 0 by 0 matrix.\n\n";

#include "petscsles.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         ierr,N = 0;
  Vec         u,b,x;
  SLES        sles;
  PetscScalar zero = 0.0,mone = -1.0;
  PetscReal   norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* create stiffness matrix */
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,N,N,&C);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* create right hand side and solution */

  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&u);CHKERRQ(ierr); 
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
  ierr = VecSet(&zero,u);CHKERRQ(ierr);
  ierr = VecSet(&zero,b);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);


  /* solve linear system */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);
  ierr = SLESSolve(sles,b,u);CHKERRQ(ierr);

  ierr = MatMult(C,u,x);CHKERRQ(ierr);
  ierr = VecAXPY(&mone,b,x);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  printf("Norm of residual %g\n",norm);

  ierr = SLESDestroy(sles);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
