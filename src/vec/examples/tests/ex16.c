#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex15.c,v 1.1 1997/12/06 18:21:52 bsmith Exp bsmith $";
#endif

static char help[] = "Tests VecSetValuesBlocked() on Seq vectors\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include <math.h>

int main(int argc,char **argv)
{
  int          n = 9, ierr, size,bs = 3,indices[2];
  Scalar       values[6];
  Vec          x;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  if (size != 1) SETERRA(1,0,"Must be run with oneprocessor");

  /* create vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x); CHKERRA(ierr);

  for ( i=0; i<6; i++ ) values = 4.0*i;
  indices[0] = 0; indices[1] = 2;
  ierr = VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES);CHKERRA(ierr);

  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  /* 
      Resulting vector should be 0 1 2 0 0 0 3 4 5
  */
  ierr = VecView(y,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
