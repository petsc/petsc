#ifndef lint
static char vcid[] = "$Id: sorder.c,v 1.27 1996/09/21 23:00:10 bsmith Exp $";
#endif
 
#include "petsc.h"
#include "mat.h"
#include "src/mat/impls/color/color.h"

extern int MatColoring_Natural(Mat,MatColoring,int*,IS**);
extern int MatFDColoringSL_Minpack(Mat,MatColoring,int *,IS **);

/*@C
  MatColoringRegisterAll - Registers all of the matrix coloring routines in PETSc.

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to MatColoringRegister() for 
  the new method, after the current list.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register a zero,
  instead of the function name, for those methods you do not wish to
  register. Make sure you keep the list of methods in the same order.
  Make sure that the replacement routine is linked before libpetscmat.a.

.keywords: matrix, coloring, register, all

.seealso: MatColoringRegister(), MatColoringRegisterDestroy()
@*/
int MatColoringRegisterAll()
{
  int         ierr;
  MatColoring name;
  static int  called = 0;
  if (called) return 0; else called = 1;

  /*
       Do not change the order of these, just add new ones to the end
  */
  ierr = MatColoringRegister(&name,"natural",MatColoring_Natural);CHKERRQ(ierr);
  ierr = MatColoringRegister(&name,"sl",MatFDColoringSL_Minpack);CHKERRQ(ierr);

  return 0;
}



