#ifndef lint
static char vcid[] = "$Id: ex18.c,v 1.9 1995/11/30 22:31:54 bsmith Exp bsmith $";
#endif

static char help[] = "Compares BLAS dots on different machines. Input\n\
arguments are\n\
  -n <length> : local vector length\n\n";

#include "petsc.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int          n = 15, ierr, size,rank,i;
  Scalar       v;
  Vec          x,y;
  int          idx;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n); if (n < 5) n = 5;


  /* create two vectors */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&y); CHKERRA(ierr);

  for ( i=0; i<n; i++ ) {
    v = ((double) i) + 1.0/(((double) i) + .35);
    ierr = VecSetValues(x,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
    v += 1.375547826473644376;
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(y); CHKERRA(ierr);
  ierr = VecAssemblyEnd(y); CHKERRA(ierr);

  ierr = VecDot(x,y,&v);
  fprintf(stdout,"Vector inner product %16.12e\n",v);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
