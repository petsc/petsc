#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sregis.c,v 1.24 1998/04/13 17:38:58 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"     /*I       "mat.h"   I*/

extern int MatOrdering_Natural(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_ND(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_1WD(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_QMD(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_RCM(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_RowLength(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_Flow(Mat,MatOrderingType,IS*,IS*);

#undef __FUNC__  
#define __FUNC__ "MatOrderingRegisterAll"
/*@C
  MatOrderingRegisterAll - Registers all of the matrix 
  reordering routines in PETSc.

  Not Collective

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to MatReorderRegister() for 
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and comment out
  those orderigs you do not wish to include.  Make sure that the
  replacement routine is linked before libpetscmat.a.

.keywords: matrix, reordering, register, all

.seealso: MatOrderingRegister(), MatOrderingRegisterDestroy()
@*/
int MatOrderingRegisterAll(void)
{
  int           ierr;

  PetscFunctionBegin;
  MatOrderingRegisterAllCalled = 1;

  ierr = MatOrderingRegister(ORDER_NATURAL,  0,"natural",MatOrdering_Natural);CHKERRQ(ierr);
  ierr = MatOrderingRegister(ORDER_ND,       0,"nd"     ,MatOrdering_ND);CHKERRQ(ierr);
  ierr = MatOrderingRegister(ORDER_1WD,      0,"1wd"    ,MatOrdering_1WD);CHKERRQ(ierr);
  ierr = MatOrderingRegister(ORDER_RCM,      0,"rcm"    ,MatOrdering_RCM);CHKERRQ(ierr);
  ierr = MatOrderingRegister(ORDER_QMD,      0,"qmd"    ,MatOrdering_QMD);CHKERRQ(ierr);
  ierr = MatOrderingRegister(ORDER_ROWLENGTH,0,"rl"     ,MatOrdering_RowLength);CHKERRQ(ierr);
  ierr = MatOrderingRegister(ORDER_FLOW,     0,"flow"   ,MatOrdering_Flow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

