#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscMemcpy.c,v 1.9 1997/10/19 03:30:47 bsmith Exp balay $";
#endif

#include "petsc.h"

int main( int argc, char **argv)
{
  PLogDouble x, y, z;
  int        i,ierr;
  Scalar     A[10000], B[10000];

  PetscInitialize(&argc, &argv,0,0);
  for (i=0; i<10000; i++) {
    A[i] = i%61897;
    B[i] = i%61897;
  }
  /* To take care of paging effects */
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  ierr = PetscGetTime(&x); CHKERRA(ierr);

  ierr = PetscGetTime(&x); CHKERRA(ierr);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  PetscMemcpy(A,B,sizeof(Scalar)*10000);
  ierr = PetscGetTime(&y); CHKERRA(ierr);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  PetscMemcpy(A,B,sizeof(Scalar)*0);
  ierr = PetscGetTime(&z); CHKERRA(ierr);

  fprintf(stderr,"%s : \n","PetscMemcpy");
  fprintf(stderr,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stderr,"    %-11s : %e sec\n","Per Scalar",(2*y-x-z)/100000.0);

  PetscFinalize();
  PetscFunctionReturn(0);
}
