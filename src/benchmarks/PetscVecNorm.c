/*$Id: PetscVecNorm.c,v 1.16 2001/08/29 20:59:36 balay Exp $*/

#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec            x;
  double         norm;
  PetscLogDouble t1,t2;
  int            ierr,n = 10000;

  PetscInitialize(&argc,&argv,0,0);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  PreLoadBegin(PETSC_TRUE,"VecNorm");
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  PreLoadEnd();
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);


  fprintf(stdout,"%s : \n","PetscMemcpy");
  fprintf(stdout," Time %g\n",t2-t1);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
