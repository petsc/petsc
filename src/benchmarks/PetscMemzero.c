/*$Id: PetscMemzero.c,v 1.19 2001/03/23 23:25:45 balay Exp bsmith $*/

#include "petsc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y,z;
  Scalar     A[10000];
  int        ierr;

  PetscInitialize(&argc,&argv,0,0);
  /* To take care of paging effects */
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscGetTime(&x);CHKERRQ(ierr);

  ierr = PetscGetTime(&x);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*10000);CHKERRQ(ierr);,
  ierr = PetscGetTime(&y);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscMemzero(A,sizeof(PetscScalar)*0);CHKERRQ(ierr);
  ierr = PetscGetTime(&z);CHKERRQ(ierr);

  fprintf(stdout,"%s : \n","PetscMemzero");
  fprintf(stdout,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stdout,"    %-11s : %e sec\n","Per Scalar",(2*y-x-z)/100000.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
