#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.1 1997/03/20 23:12:06 bsmith Exp bsmith $";
#endif


/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines
     sys.h    - system routines
*/

#include "vec.h"
#include <math.h>

 
int Function(Vec x,Vec y)
{
  int      ierr;
  Scalar   three = 3.0,two = 2.0;

  ierr = VecCopy(x,y);
  ierr = VecScale(&three,y); CHKERRQ(ierr);
  ierr = VecShift(&two,y); CHKERRQ(ierr);
  return 0;
}






