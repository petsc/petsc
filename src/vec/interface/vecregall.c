#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vecregall.c,v 1.5 1999/06/07 17:17:56 knepley Exp $";
#endif

#include "src/vec/vecimpl.h"     /*I  "vec.h"  I*/
EXTERN_C_BEGIN
extern int VecCreate_Seq(Vec, ParameterDict);
extern int VecCreate_MPI(Vec, ParameterDict);
extern int VecCreate_Shared(Vec, ParameterDict);
extern int VecCreate_FETI(Vec, ParameterDict);
extern int VecCreate_ESI(Vec, ParameterDict);
extern int VecCreate_PetscESI(Vec, ParameterDict);

extern int VecSerialize_Seq(MPI_Comm, Vec *, PetscViewer, PetscTruth);
extern int VecSerialize_MPI(MPI_Comm, Vec *, PetscViewer, PetscTruth);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "VecRegisterAll"
/*@C
  VecRegisterAll - Registers all of the Vec components in the PETSc package.

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
#if defined(PETSC_HAVE_ESI) && defined(__cplusplus)
  ierr = VecRegisterDynamic(VECESI,      path, "VecCreate_ESI",      VecCreate_ESI);                     CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECPETSCESI, path, "VecCreate_PetscESI", VecCreate_PetscESI);                CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSerializeRegisterAll"
/*@C
  VecSerializeRegisterAll - Registers all of the serialization routines in the Vec package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: Vec, vector, register, all, serialize
.seealso: VecSerializeRegister(), VecSerializeRegisterDestroy()
@*/
int VecSerializeRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  VecSerializeRegisterAllCalled = 1;

  ierr = VecSerializeRegister(VEC_SER_SEQ_BINARY, path, "VecSerialize_Seq", VecSerialize_Seq);            CHKERRQ(ierr);
  ierr = VecSerializeRegister(VEC_SER_MPI_BINARY, path, "VecSerialize_MPI", VecSerialize_MPI);            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
