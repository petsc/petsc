#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: registerdestroy.c,v 1.5 2000/01/10 03:54:25 knepley Exp $";
#endif

#include "src/bilinear/bilinearimpl.h"  /*I "bilinear.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "BilinearRegisterDestroy"
/*@C
  BilinearRegisterDestroy - Frees the list of creation routines for bilinear operators that were registered by BilinearRegister().

  Not collective

  Level: advanced

.keywords: Bilinear, register, destroy
.seealso: BilinearRegister(), BilinearRegisterAll(), BilinearSerializeRegisterDestroy()
@*/
int BilinearRegisterDestroy(void) {
  int ierr;

  PetscFunctionBegin;
  if (BilinearList != PETSC_NULL) {
    ierr = PetscFListDestroy(&BilinearList);                                                              CHKERRQ(ierr);
    BilinearList = PETSC_NULL;
  }
  BilinearRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
