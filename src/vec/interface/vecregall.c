#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vecregall.c,v 1.5 1999/06/07 17:17:56 knepley Exp $";
#endif

#include "src/vec/vecimpl.h"     /*I  "vec.h"  I*/
EXTERN_C_BEGIN
extern int VecCreate_Seq(Vec);
extern int VecCreate_MPI(Vec);
extern int VecCreate_Shared(Vec);
extern int VecCreate_FETI(Vec);
extern int VecCreate_ESI(Vec);
extern int VecCreate_PetscESI(Vec);

extern int VecSerialize_Seq(MPI_Comm, Vec *, PetscViewer, PetscTruth);
extern int VecSerialize_MPI(MPI_Comm, Vec *, PetscViewer, PetscTruth);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "VecRegisterAll"
/*@C
  VecRegisterAll - Registers all of the vector components in the Vec package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: Vec, register, all
.seealso:  VecRegister(), VecRegisterDestroy()
@*/
int VecRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  VecRegisterAllCalled = PETSC_TRUE;

  ierr = VecRegisterDynamic(VECSEQ,      path, "VecCreate_Seq",      VecCreate_Seq);                     CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECMPI,      path, "VecCreate_MPI",      VecCreate_MPI);                     CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECSHARED,   path, "VecCreate_Shared",   VecCreate_Shared);                  CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECFETI,     path, "VecCreate_FETI",     VecCreate_FETI);                    CHKERRQ(ierr);
#if defined(__cplusplus)
  ierr = VecRegisterDynamic(VECESI,      path, "VecCreate_ESI",      VecCreate_ESI);                     CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECPETSCESI, path, "VecCreate_PetscESI", VecCreate_PetscESI);                CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSerializeRegisterAll"
/*@C
  VecSerializeRegisterAll - Registers all of the serialization routines in the Vec package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: Vec, register, all, serialize
.seealso: VecSerializeRegister(), VecSerializeRegisterDestroy()
@*/
int VecSerializeRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  VecSerializeRegisterAllCalled = PETSC_TRUE;

  ierr = VecSerializeRegister(VEC_SER_SEQ_BINARY, path, "VecSerialize_Seq", VecSerialize_Seq);            CHKERRQ(ierr);
  ierr = VecSerializeRegister(VEC_SER_MPI_BINARY, path, "VecSerialize_MPI", VecSerialize_MPI);            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
