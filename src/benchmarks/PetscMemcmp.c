/*$Id: PetscMemcmp.c,v 1.15 2001/01/15 21:49:39 bsmith Exp bsmith $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y,z;
  Scalar     A[10000],B[10000];
  int        i,ierr;

  PetscInitialize(&argc,&argv,0,0);

  for (i=0; i<10000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  ierr = PetscGetTime(&x);CHKERRQ(ierr);

  ierr = PetscGetTime(&x);CHKERRQ(ierr);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  PetscMemcmp(A,B,sizeof(Scalar)*10000);
  ierr = PetscGetTime(&y);CHKERRQ(ierr);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  ierr = PetscGetTime(&z);CHKERRQ(ierr);

  fprintf(stdout,"%s : \n","PetscMemcmp");
  fprintf(stdout,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stdout,"    %-11s : %e sec\n","Per Scalar",(2*y-x-z)/100000);

  PetscFinalize();
  PetscFunctionReturn(0);
}
