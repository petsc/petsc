#ifndef lint
static char vcid[] = "$Id: sregis.c,v 1.13 1996/09/14 03:08:25 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"     /*I       "mat.h"   I*/

extern int MatOrder_Natural(Mat,MatReordering,IS*,IS*);
extern int MatOrder_ND(Mat,MatReordering,IS*,IS*);
extern int MatOrder_1WD(Mat,MatReordering,IS*,IS*);
extern int MatOrder_QMD(Mat,MatReordering,IS*,IS*);
extern int MatOrder_RCM(Mat,MatReordering,IS*,IS*);
extern int MatOrder_RowLength(Mat,MatReordering,IS*,IS*);
extern int MatOrder_Flow(Mat,MatReordering,IS*,IS*);

/*@C
  MatReorderingRegisterAll - Registers all of the matrix 
  reordering routines in PETSc.

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to MatReorderRegister() for 
  the new method, after the current list.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register a zero,
  instead of the function name, for those methods you do not wish to
  register. Make sure you keep the list of methods in the same order.
  Make sure that the replacement routine is linked before libpetscmat.a.

.keywords: matrix, reordering, register, all

.seealso: MatReorderingRegister(), MatReorderingRegisterDestroy()
@*/
int MatReorderingRegisterAll()
{
  int           ierr;
  MatReordering name;
  static int  called = 0;
  if (called) return 0; else called = 1;

  /*
       Do not change the order of these, just add ones to the end 
  */
  ierr = MatReorderingRegister(&name,"natural",MatOrder_Natural);CHKERRQ(ierr);
  ierr = MatReorderingRegister(&name,"nd"     ,MatOrder_ND);CHKERRQ(ierr);
  ierr = MatReorderingRegister(&name,"1wd"    ,MatOrder_1WD);CHKERRQ(ierr);
  ierr = MatReorderingRegister(&name,"rcm"    ,MatOrder_RCM);CHKERRQ(ierr);
  ierr = MatReorderingRegister(&name,"qmd"    ,MatOrder_QMD);CHKERRQ(ierr);
  ierr = MatReorderingRegister(&name,"rl"     ,MatOrder_RowLength);CHKERRQ(ierr);
  ierr = MatReorderingRegister(&name,"flow"   ,MatOrder_Flow);CHKERRQ(ierr);
  return 0;
}

