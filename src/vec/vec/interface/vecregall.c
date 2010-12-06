#define PETSCVEC_DLL

#include "private/vecimpl.h"     /*I  "vec.h"  I*/
EXTERN_C_BEGIN
extern PetscErrorCode  VecCreate_Seq(Vec);
extern PetscErrorCode  VecCreate_MPI(Vec);
extern PetscErrorCode  VecCreate_Standard(Vec);
extern PetscErrorCode  VecCreate_Shared(Vec);
#if defined(PETSC_HAVE_CUDA)
extern PetscErrorCode  VecCreate_SeqCUDA(Vec);
extern PetscErrorCode  VecCreate_MPICUDA(Vec);
extern PetscErrorCode  VecCreate_CUDA(Vec);
#endif
#if 0
#if defined(PETSC_HAVE_SIEVE)
extern PetscErrorCode  VecCreate_Sieve(Vec);
#endif
#endif
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
.seealso:  VecRegister(), VecRegisterDestroy(), VecRegisterDynamic()
@*/
PetscErrorCode  VecRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecRegisterAllCalled = PETSC_TRUE;

  ierr = VecRegisterDynamic(VECSEQ,      path, "VecCreate_Seq",      VecCreate_Seq);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECMPI,      path, "VecCreate_MPI",      VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECSTANDARD, path, "VecCreate_Standard", VecCreate_Standard);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECSHARED,   path, "VecCreate_Shared",   VecCreate_Shared);CHKERRQ(ierr);
#if defined PETSC_HAVE_CUDA
  ierr = VecRegisterDynamic(VECSEQCUDA,  path, "VecCreate_SeqCUDA",  VecCreate_SeqCUDA);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECMPICUDA,  path, "VecCreate_MPICUDA",  VecCreate_MPICUDA);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECCUDA,     path, "VecCreate_CUDA",     VecCreate_CUDA);CHKERRQ(ierr);
#endif
#if 0
#if defined(PETSC_HAVE_SIEVE)
  ierr = VecRegisterDynamic(VECSIEVE,    path, "VecCreate_Sieve",    VecCreate_Sieve);CHKERRQ(ierr);
#endif
#endif
  PetscFunctionReturn(0);
}

