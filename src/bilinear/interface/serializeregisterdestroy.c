#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: serializeregisterdestroy.c,v 1.5 2000/01/10 03:54:25 knepley Exp $";
#endif

#include "src/bilinear/bilinearimpl.h"  /*I "bilinear.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "BilinearSerializeRegisterDestroy"
/*@C
  BilinearSerializeRegisterDestroy - Frees the list of serialization routines for
  bilinear operators that were registered by BilinearSerializeRegister().

  Not collective

  Level: advanced

.keywords: Bilinear, serialization, register, destroy
.seealso: BilinearSerializeRegisterAll(), BilinearRegisterDestroy()
@*/
int BilinearSerializeRegisterDestroy(void) {
  int ierr;

  PetscFunctionBegin;
  if (BilinearSerializeList != PETSC_NULL) {
    ierr = PetscFListDestroy(&BilinearSerializeList);                                                     CHKERRQ(ierr);
    BilinearSerializeList = PETSC_NULL;
  }
  BilinearSerializeRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
