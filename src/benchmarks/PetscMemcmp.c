/*$Id: PetscMemcmp.c,v 1.11 1999/05/04 20:38:02 balay Exp bsmith $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv)
{
  PLogDouble x, y, z;
  Scalar     A[10000], B[10000];
  int        i,ierr;

  PetscInitialize(&argc, &argv,0,0);

  for (i=0; i<10000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  PetscMemcmp(A,B,sizeof(Scalar)*0);
  ierr = PetscGetTime(&x);CHKERRA(ierr);

  ierr = PetscGetTime(&x);CHKERRA(ierr);
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
  ierr = PetscGetTime(&y);CHKERRA(ierr);
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
  ierr = PetscGetTime(&z);CHKERRA(ierr);

  fprintf(stderr,"%s : \n","PetscMemcmp");
  fprintf(stderr,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stderr,"    %-11s : %e sec\n","Per Scalar",(2*y-x-z)/100000);

  PetscFinalize();
  PetscFunctionReturn(0);
}
