/*$Id: PetscMemcpy.c,v 1.20 2001/01/23 20:58:37 balay Exp balay $*/

#include "petsc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y,z;
  int        i,ierr;
  Scalar     *A,*B;

  PetscInitialize(&argc,&argv,0,0);

  ierr = PetscMalloc(8000000*sizeof(Scalar),&A);CHKERRQ(ierr);
  ierr = PetscMalloc(8000000*sizeof(Scalar),&B);CHKERRQ(ierr);

  for (i=0; i<8000000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscGetTime(&x);CHKERRQ(ierr);

  ierr = PetscGetTime(&x);CHKERRQ(ierr);
  /*
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000);
  PetscMemcpy(A,B,sizeof(Scalar)*8000000); */
  { int j;
  for (j = 0; j<10; j++) {
    for (i=0; i<8000000; i++) {
      B[i] = A[i];
    }
  }}

  ierr = PetscGetTime(&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRQ(ierr);
  ierr = PetscGetTime(&z);CHKERRQ(ierr);

  fprintf(stdout,"%s : \n","PetscMemcpy");
  fprintf(stdout,"    %-11s : %e MB/s\n","Bandwidth",10.0*8*8/(y-x));
  fprintf(stdout,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stdout,"    %-11s : %e sec\n","Per Scalar",(2*y-x-z)/8000000.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
