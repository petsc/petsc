#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: PetscMemzero.c,v 1.10 1998/03/31 23:33:59 balay Exp bsmith $";
#endif

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv)
{
  PLogDouble x, y, z;
  Scalar     A[10000];
  int        ierr;

  PetscInitialize(&argc, &argv,0,0);
  /* To take care of paging effects */
  PetscMemzero(A,sizeof(Scalar)*0);
  ierr = PetscGetTime(&x); CHKERRA(ierr);

  ierr = PetscGetTime(&x); CHKERRA(ierr);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  PetscMemzero(A,sizeof(Scalar)*10000);
  ierr = PetscGetTime(&y); CHKERRA(ierr);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  PetscMemzero(A,sizeof(Scalar)*0);
  ierr = PetscGetTime(&z); CHKERRA(ierr);

  fprintf(stderr,"%s : \n","PetscMemzero");
  fprintf(stderr,"    %-11s : %e sec\n","Latency",(z-y)/10.0);
  fprintf(stderr,"    %-11s : %e sec\n","Per Scalar",(2*y-x-z)/100000.0);

  PetscFinalize();
  PetscFunctionReturn(0);
}
