#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: serializeregisterall.c,v 1.3 1999/09/18 16:36:44 knepley Exp $";
#endif

#include "src/bilinear/bilinearimpl.h"     /*I  "bilinear.h"  I*/

EXTERN_C_BEGIN
extern int BilinearSerialize_Dense_Seq(MPI_Comm, Bilinear *, PetscViewer, PetscTruth);
EXTERN_C_END

PetscTruth BilinearSerializeRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "BilinearSerializeRegisterAll"
/*@C
  BilinearSerializeRegisterAll - Registers all of the serialization routines in the Bilinear package. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: Bilinear, register, all, serialize
.seealso: BilinearSerialize(), BilinearSerializeRegister(), BilinearSerializeRegisterDestroy()
@*/
int BilinearSerializeRegisterAll(const char path[]) {
  int ierr;

  PetscFunctionBegin;
  BilinearSerializeRegisterAllCalled = PETSC_TRUE;

  ierr = BilinearSerializeRegisterDynamic(BILINEAR_SER_DENSE_SEQ, path, "BilinearSerialize_Dense_Seq", BilinearSerialize_Dense_Seq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
