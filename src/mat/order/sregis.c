#ifndef lint
static char vcid[] = "$Id: sregis.c,v 1.7 1995/10/12 04:16:18 bsmith Exp bsmith $";
#endif

#include "../../matimpl.h"     /*I       "mat.h"   I*/

extern int MatOrder_Natural(int*,int*,int*,int*,int*);
extern int MatOrder_ND(int*,int*,int*,int*,int*);
extern int MatOrder_1WD(int*,int*,int*,int*,int*);
extern int MatOrder_QMD(int*,int*,int*,int*,int*);
extern int MatOrder_RCM(int*,int*,int*,int*,int*);

/*@C
  MatReorderingRegisterAll - Registers all of the sequential matrix 
  reordering routines in PETSc.

  Adding new methods:
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to MatReorderRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/mat.h"
$       by appending the method's identifier as an
$       enumerator of the MatOrdering enumeration.
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
  MatReorderingRegister(ORDER_NATURAL   , "natural"  ,MatOrder_Natural);
  MatReorderingRegister(ORDER_ND        , "nd"       ,MatOrder_ND);
  MatReorderingRegister(ORDER_1WD       , "1wd"      ,MatOrder_1WD);
  MatReorderingRegister(ORDER_RCM       , "rcm"      ,MatOrder_RCM);
  MatReorderingRegister(ORDER_QMD       , "qmd"      ,MatOrder_QMD);
  return 0;
}

