/*$Id: PetscVecNorm.c,v 1.13 2001/01/17 22:28:38 bsmith Exp balay $*/

#include "petscvec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Vec        x;
  double     norm;
  PetscLogDouble t1,t2;
  int        ierr,n = 10000;

  PetscInitialize(&argc,&argv,0,0);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);

  /* To take care of paging effects */
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);

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
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);

  fprintf(stderr,"%s : \n","PetscMemcpy");
  fprintf(stderr," Time %g\n",t2-t1);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
