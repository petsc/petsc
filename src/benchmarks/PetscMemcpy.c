/*$Id: PetscMemcpy.c,v 1.16 2000/11/28 17:32:38 bsmith Exp bsmith $*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  PetscLogDouble x,y,z;
  int        i,ierr;
  Scalar     *A,*B;

  PetscInitialize(&argc,&argv,0,0);

ierr = PetscMalloc(8000000*sizeof(Scalar),&(  A ));
ierr = PetscMalloc(8000000*sizeof(Scalar),&(  B ));

  for (i=0; i<8000000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscGetTime(&x);CHKERRA(ierr);

  ierr = PetscGetTime(&x);CHKERRA(ierr);
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

  ierr = PetscGetTime(&y);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscMemcpy(A,B,sizeof(Scalar)*0);CHKERRA(ierr);
  ierr = PetscGetTime(&z);CHKERRA(ierr);

  fprintf(stdout,"%s : \n","PetscMemcpy");
  fprintf(stdout,"    %-11s : %e MB/s\n","Bandwidth",10.0*8*8/(y-x));
  fprintf(stdout,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stdout,"    %-11s : %e sec\n","Per Scalar",(2*y-x-z)/8000000.0);

  PetscFinalize();
  PetscFunctionReturn(0);
}
