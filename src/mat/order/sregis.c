#ifndef lint
static char vcid[] = "$Id: sregis.c,v 1.12 1996/08/08 14:43:21 bsmith Exp bsmith $";
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
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to MatReorderRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/mat.h"
$       by appending the method's identifier as an
$       enumerator of the MatReordering enumeration.
$       As long as the enumerator is appended to
$       the existing list, only the MatReorderRegisterAll()
$       routine requires recompilation.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscmat.a.

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
       Do not change the order of these unless similarly changing 
    them in include/mat.h
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

