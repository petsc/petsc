#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: scolor.c,v 1.14 1999/03/17 23:23:24 bsmith Exp bsmith $";
#endif
 
#include "mat.h"
#include "src/mat/impls/color/color.h"

EXTERN_C_BEGIN
extern int MatColoring_Natural(Mat,MatColoringType,ISColoring*);
extern int MatFDColoringSL_Minpack(Mat,MatColoringType,ISColoring*);
extern int MatFDColoringLF_Minpack(Mat,MatColoringType,ISColoring*);
extern int MatFDColoringID_Minpack(Mat,MatColoringType,ISColoring*);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatColoringRegisterAll" 
/*@C
  MatColoringRegisterAll - Registers all of the matrix coloring routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to MatColoringRegister() for 
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and modify it to
  register a zero, instead of the function name, for those methods you
  do not wish to register.  Make sure that the replacement routine is
  linked before libpetscmat.a.

.keywords: matrix, coloring, register, all

.seealso: MatColoringRegister(), MatColoringRegisterDestroy()
@*/
int MatColoringRegisterAll(char *path)
{
  int         ierr;

  PetscFunctionBegin;
  MatColoringRegisterAllCalled = 1;  
  ierr = MatColoringRegister(MATCOLORING_NATURAL,path,"MatColoring_Natural",    MatColoring_Natural);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORING_SL,     path,"MatFDColoringSL_Minpack",MatFDColoringSL_Minpack);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORING_LF,     path,"MatFDColoringLF_Minpack",MatFDColoringLF_Minpack);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORING_ID,     path,"MatFDColoringID_Minpack",MatFDColoringID_Minpack);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



