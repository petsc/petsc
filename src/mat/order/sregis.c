#ifndef lint
static char vcid[] = "$Id: sregis.c,v 1.10 1996/03/08 05:47:37 bsmith Exp bsmith $";
#endif

#include "../../matimpl.h"     /*I       "mat.h"   I*/

extern int MatOrder_Natural(int*,int*,int*,int*,int*);
extern int MatOrder_ND(int*,int*,int*,int*,int*);
extern int MatOrder_1WD(int*,int*,int*,int*,int*);
extern int MatOrder_QMD(int*,int*,int*,int*,int*);
extern int MatOrder_RCM(int*,int*,int*,int*,int*);
extern int MatOrder_RowLength(int*,int*,int*,int*,int*);

/*@C
  MatReorderingRegisterAll - Registers all of the sequential matrix 
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
  MatReordering name;
  static int  called = 0;
  if (called) return 0; else called = 1;

  /*
       Do not change the order of these unless similarly changing 
    them in include/mat.h
  */
  MatReorderingRegister(&name,"natural",PETSC_TRUE,1,MatOrder_Natural);
  MatReorderingRegister(&name,"nd"     ,PETSC_TRUE,1,MatOrder_ND);
  MatReorderingRegister(&name,"1wd"    ,PETSC_TRUE,1,MatOrder_1WD);
  MatReorderingRegister(&name,"rcm"    ,PETSC_TRUE,1,MatOrder_RCM);
  MatReorderingRegister(&name,"qmd"    ,PETSC_TRUE,1,MatOrder_QMD);
  MatReorderingRegister(&name,"rl"     ,PETSC_FALSE,0,MatOrder_RowLength);
  MatReorderingRegister(&name,"flow"   ,PETSC_FALSE,0,0);
  return 0;
}

