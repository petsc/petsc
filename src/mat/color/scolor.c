#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: scolor.c,v 1.13 1998/04/13 17:40:46 bsmith Exp bsmith $";
#endif
 
#include "petsc.h"
#include "mat.h"
#include "src/mat/impls/color/color.h"

extern int MatColoring_Natural(Mat,MatColoringType,ISColoring*);
extern int MatFDColoringSL_Minpack(Mat,MatColoringType,ISColoring*);
extern int MatFDColoringLF_Minpack(Mat,MatColoringType,ISColoring*);
extern int MatFDColoringID_Minpack(Mat,MatColoringType,ISColoring*);

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
int MatColoringRegisterAll(void)
{
  int         ierr;

  PetscFunctionBegin;
  MatColoringRegisterAllCalled = 1;  
  ierr = MatColoringRegister(COLORING_NATURAL,0,"natural",MatColoring_Natural);CHKERRQ(ierr);
  ierr = MatColoringRegister(COLORING_SL,     0,"sl",MatFDColoringSL_Minpack);CHKERRQ(ierr);
  ierr = MatColoringRegister(COLORING_LF,     0,"lf",MatFDColoringLF_Minpack);CHKERRQ(ierr);
  ierr = MatColoringRegister(COLORING_ID,     0,"id",MatFDColoringID_Minpack);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



