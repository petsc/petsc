
#include <petscvec.h>
#include <petsc/private/vecimpl.h>
PETSC_EXTERN PetscErrorCode VecCreate_Seq(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPI(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_Standard(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_Shared(Vec);
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
PETSC_EXTERN PetscErrorCode VecCreate_Node(Vec);
#endif
#if defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode VecCreate_SeqViennaCL(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPIViennaCL(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_ViennaCL(Vec);
#endif
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode VecCreate_SeqCUDA(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPICUDA(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_CUDA(Vec);
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_EXTERN PetscErrorCode VecCreate_SeqKokkos(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPIKokkos(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_Kokkos(Vec);
#endif
#if defined(PETSC_HAVE_HIP)
PETSC_EXTERN PetscErrorCode VecCreate_SeqHIP(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPIHIP(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_HIP(Vec);
#endif

/*@C
  VecRegisterAll - Registers all of the vector components in the Vec package.

  Not Collective

  Level: advanced

.seealso:  VecRegister(), VecRegisterDestroy(), VecRegister()
@*/
PetscErrorCode VecRegisterAll(void)
{
  PetscFunctionBegin;
  if (VecRegisterAllCalled) PetscFunctionReturn(0);
  VecRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(VecRegister(VECSEQ,        VecCreate_Seq));
  CHKERRQ(VecRegister(VECMPI,        VecCreate_MPI));
  CHKERRQ(VecRegister(VECSTANDARD,   VecCreate_Standard));
  CHKERRQ(VecRegister(VECSHARED,     VecCreate_Shared));
#if defined PETSC_HAVE_VIENNACL
  CHKERRQ(VecRegister(VECSEQVIENNACL,    VecCreate_SeqViennaCL));
  CHKERRQ(VecRegister(VECMPIVIENNACL,    VecCreate_MPIViennaCL));
  CHKERRQ(VecRegister(VECVIENNACL,       VecCreate_ViennaCL));
#endif
#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(VecRegister(VECSEQCUDA,    VecCreate_SeqCUDA));
  CHKERRQ(VecRegister(VECMPICUDA,    VecCreate_MPICUDA));
  CHKERRQ(VecRegister(VECCUDA,       VecCreate_CUDA));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  CHKERRQ(VecRegister(VECSEQKOKKOS,  VecCreate_SeqKokkos));
  CHKERRQ(VecRegister(VECMPIKOKKOS,  VecCreate_MPIKokkos));
  CHKERRQ(VecRegister(VECKOKKOS,     VecCreate_Kokkos));
#endif
#if defined(PETSC_HAVE_HIP)
  CHKERRQ(VecRegister(VECSEQHIP,    VecCreate_SeqHIP));
  CHKERRQ(VecRegister(VECMPIHIP,    VecCreate_MPIHIP));
  CHKERRQ(VecRegister(VECHIP,       VecCreate_HIP));
#endif
  PetscFunctionReturn(0);
}
