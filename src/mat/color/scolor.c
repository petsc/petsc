/*$Id: scolor.c,v 1.24 2001/03/23 23:22:57 balay Exp $*/
 
#include "petscmat.h"
#include "src/mat/color/color.h"

EXTERN_C_BEGIN
EXTERN int MatColoring_Natural(Mat,const MatColoringType,ISColoring*);
EXTERN int MatFDColoringSL_Minpack(Mat,const MatColoringType,ISColoring*);
EXTERN int MatFDColoringLF_Minpack(Mat,const MatColoringType,ISColoring*);
EXTERN int MatFDColoringID_Minpack(Mat,const MatColoringType,ISColoring*);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatColoringRegisterAll" 
/*@C
  MatColoringRegisterAll - Registers all of the matrix coloring routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to MatColoringRegisterDynamic() for 
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and modify it to
  register a zero, instead of the function name, for those methods you
  do not wish to register.  Make sure that the replacement routine is
  linked before libpetscmat.a.

.keywords: matrix, coloring, register, all

.seealso: MatColoringRegisterDynamic(), MatColoringRegisterDestroy()
@*/
int MatColoringRegisterAll(const char path[])
{
  int         ierr;

  PetscFunctionBegin;
  MatColoringRegisterAllCalled = PETSC_TRUE;  
  ierr = MatColoringRegisterDynamic(MATCOLORING_NATURAL,path,"MatColoring_Natural",    MatColoring_Natural);CHKERRQ(ierr);
  ierr = MatColoringRegisterDynamic(MATCOLORING_SL,     path,"MatFDColoringSL_Minpack",MatFDColoringSL_Minpack);CHKERRQ(ierr);
  ierr = MatColoringRegisterDynamic(MATCOLORING_LF,     path,"MatFDColoringLF_Minpack",MatFDColoringLF_Minpack);CHKERRQ(ierr);
  ierr = MatColoringRegisterDynamic(MATCOLORING_ID,     path,"MatFDColoringID_Minpack",MatFDColoringID_Minpack);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



