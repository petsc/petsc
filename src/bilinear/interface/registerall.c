#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: registerall.c,v 1.3 1999/09/18 16:36:44 knepley Exp $";
#endif

#include "src/bilinear/bilinearimpl.h"     /*I  "bilinear.h"  I*/

EXTERN_C_BEGIN
extern int BilinearCreate_Dense_Seq(Bilinear);
EXTERN_C_END

PetscTruth BilinearRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "BilinearRegisterAll"
/*@C
  BilinearRegisterAll - Registers all of the creation routines in the Bilinear package. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: Bilinear, register, all
.seealso: BilinearCreate(), BilinearRegister(), BilinearRegisterDestroy()
@*/
int BilinearRegisterAll(const char path[]) {
  int ierr;

  PetscFunctionBegin;
  BilinearRegisterAllCalled = PETSC_TRUE;

  ierr = BilinearRegisterDynamic(BILINEAR_DENSE_SEQ, path, "BilinearCreate_Dense_Seq", BilinearCreate_Dense_Seq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
