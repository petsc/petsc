#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bilinearregall.c,v 1.3 1999/06/01 16:44:06 knepley Exp $";
#endif

#include "src/bilinear/bilinearimpl.h"     /*I  "bilinear.h"  I*/
EXTERN_C_BEGIN
extern int BilinearSerialize_SeqDense(MPI_Comm, Bilinear *, PetscViewer, PetscTruth);
extern int BilinearSerialize_MPIDense(MPI_Comm, Bilinear *, PetscViewer, PetscTruth);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "BilinearSerializeRegisterAll"
/*@C
  BilinearSerializeRegisterAll - Registers all of the serialization routines in the Bilinear package. 

  Not collective

  Input parameters:
. path - Dynamic library path

  Level: advanced

.keywords: Bilinear, register, all, serialize

.seealso: BilinearSerializeRegister(), BilinearSerializeRegisterDestroy()
@*/
int BilinearSerializeRegisterAll(const char *path)
{
  int ierr;

  PetscFunctionBegin;
  BilinearSerializeRegisterAllCalled = 1;

  ierr = BilinearSerializeRegister(BILINEAR_SER_SEQDENSE_BINARY, path, "BilinearSerialize_SeqDense",
                                   BilinearSerialize_SeqDense);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
