#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2func.c,v 1.1 1997/04/08 04:03:11 bsmith Exp bsmith $";
#endif


/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines
     sys.h    - system routines
*/

#include "vec.h"

 
int Function(Vec x,Vec y)
{
  int      ierr;
  Scalar   three = 3.0,two = 2.0;

  ierr = VecCopy(x,y);
  ierr = VecScale(y,three); CHKERRQ(ierr);
  ierr = VecShift(y,two); CHKERRQ(ierr);
  return 0;
}






