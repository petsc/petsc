#ifndef lint
static char vcid[] = "$Id: sregis.c,v 1.17 1997/02/03 05:58:45 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"     /*I       "mat.h"   I*/

extern int MatOrder_Natural(Mat,MatReordering,IS*,IS*);
extern int MatOrder_ND(Mat,MatReordering,IS*,IS*);
extern int MatOrder_1WD(Mat,MatReordering,IS*,IS*);
extern int MatOrder_QMD(Mat,MatReordering,IS*,IS*);
extern int MatOrder_RCM(Mat,MatReordering,IS*,IS*);
extern int MatOrder_RowLength(Mat,MatReordering,IS*,IS*);
extern int MatOrder_Flow(Mat,MatReordering,IS*,IS*);

#undef __FUNC__  
#define __FUNC__ "MatReorderingRegisterAll" /* ADIC Ignore */
/*@C
  MatReorderingRegisterAll - Registers all of the matrix 
  reordering routines in PETSc.

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to MatReorderRegister() for 
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and comment out
  those orderigs you do not wish to include.  Make sure that the
  replacement routine is linked before libpetscmat.a.

.keywords: matrix, reordering, register, all

.seealso: MatReorderingRegister(), MatReorderingRegisterDestroy()
@*/
int MatReorderingRegisterAll()
{
  int           ierr;
  MatReorderingRegisterAllCalled = 1;

  ierr = MatReorderingRegister(ORDER_NATURAL,  0,"natural",MatOrder_Natural);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_ND,       0,"nd"     ,MatOrder_ND);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_1WD,      0,"1wd"    ,MatOrder_1WD);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_RCM,      0,"rcm"    ,MatOrder_RCM);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_QMD,      0,"qmd"    ,MatOrder_QMD);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_ROWLENGTH,0,"rl"     ,MatOrder_RowLength);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_FLOW,     0,"flow"   ,MatOrder_Flow);CHKERRQ(ierr);
  return 0;
}

