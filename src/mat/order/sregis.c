#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sregis.c,v 1.26 1999/03/17 23:23:07 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"     /*I       "mat.h"   I*/

EXTERN_C_BEGIN
extern int MatOrdering_Natural(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_ND(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_1WD(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_QMD(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_RCM(Mat,MatOrderingType,IS*,IS*);
extern int MatOrdering_RowLength(Mat,MatOrderingType,IS*,IS*);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatOrderingRegisterAll"
/*@C
  MatOrderingRegisterAll - Registers all of the matrix 
  reordering routines in PETSc.

  Not Collective

  Level: developer

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
int MatOrderingRegisterAll(char *path)
{
  int           ierr;

  PetscFunctionBegin;
  MatOrderingRegisterAllCalled = 1;

  ierr = MatOrderingRegister(MATORDERING_NATURAL,  path,"MatOrdering_Natural"  ,MatOrdering_Natural);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERING_ND,       path,"MatOrdering_ND"       ,MatOrdering_ND);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERING_1WD,      path,"MatOrdering_1WD"      ,MatOrdering_1WD);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERING_RCM,      path,"MatOrdering_RCM"      ,MatOrdering_RCM);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERING_QMD,      path,"MatOrdering_QMD"      ,MatOrdering_QMD);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERING_ROWLENGTH,path,"MatOrdering_RowLength",MatOrdering_RowLength);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

