#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sregis.c,v 1.21 1997/10/19 03:25:56 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"     /*I       "mat.h"   I*/

extern int MatOrder_Natural(Mat,MatReorderingType,IS*,IS*);
extern int MatOrder_ND(Mat,MatReorderingType,IS*,IS*);
extern int MatOrder_1WD(Mat,MatReorderingType,IS*,IS*);
extern int MatOrder_QMD(Mat,MatReorderingType,IS*,IS*);
extern int MatOrder_RCM(Mat,MatReorderingType,IS*,IS*);
extern int MatOrder_RowLength(Mat,MatReorderingType,IS*,IS*);
extern int MatOrder_Flow(Mat,MatReorderingType,IS*,IS*);

#undef __FUNC__  
#define __FUNC__ "MatReorderingRegisterAll"
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

  PetscFunctionBegin;
  MatReorderingRegisterAllCalled = 1;

  ierr = MatReorderingRegister(ORDER_NATURAL,  0,"natural",MatOrder_Natural);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_ND,       0,"nd"     ,MatOrder_ND);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_1WD,      0,"1wd"    ,MatOrder_1WD);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_RCM,      0,"rcm"    ,MatOrder_RCM);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_QMD,      0,"qmd"    ,MatOrder_QMD);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_ROWLENGTH,0,"rl"     ,MatOrder_RowLength);CHKERRQ(ierr);
  ierr = MatReorderingRegister(ORDER_FLOW,     0,"flow"   ,MatOrder_Flow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

