
#include "../../../../matimpl.h"     /*I       "mat.h"   I*/

extern int MatOrderNatural(int*,int*,int*,int*,int*);
extern int MatOrderND(int*,int*,int*,int*,int*);
extern int MatOrder1WD(int*,int*,int*,int*,int*);
extern int MatOrderQMD(int*,int*,int*,int*,int*);
extern int MatOrderRCM(int*,int*,int*,int*,int*);

/*@
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

  Notes:
  To prevent all the methods from being registered and thus save
  memory, copy this routine and register only those methods desired.

.keywords: reordering, register, all

.seealso: MatReorderingRegister(), MatReorderingRegisterDestroy()
@*/
int MatReorderingRegisterAll()
{
  MatReorderingRegister(ORDER_NATURAL   , "natural"  ,MatOrderNatural);
  MatReorderingRegister(ORDER_ND        , "nd"       ,MatOrderND);
  MatReorderingRegister(ORDER_1WD       , "1wd"      ,MatOrder1WD);
  MatReorderingRegister(ORDER_RCM       , "rcm"      ,MatOrderRCM);
  MatReorderingRegister(ORDER_QMD       , "qmd"      ,MatOrderQMD);
  return 0;
}

