
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
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_EXTERN PetscErrorCode VecCreate_SeqKokkos(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPIKokkos(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_Kokkos(Vec);
#endif

/*@C
  VecRegisterAll - Registers all of the vector types in the `Vec` package.

  Not Collective

  Level: advanced

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecRegister()`, `VecRegisterDestroy()`, `VecRegister()`
@*/
PetscErrorCode VecRegisterAll(void)
{
  PetscFunctionBegin;
  if (VecRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  VecRegisterAllCalled = PETSC_TRUE;

  PetscCall(VecRegister(VECSEQ, VecCreate_Seq));
  PetscCall(VecRegister(VECMPI, VecCreate_MPI));
  PetscCall(VecRegister(VECSTANDARD, VecCreate_Standard));
  PetscCall(VecRegister(VECSHARED, VecCreate_Shared));
#if defined PETSC_HAVE_VIENNACL
  PetscCall(VecRegister(VECSEQVIENNACL, VecCreate_SeqViennaCL));
  PetscCall(VecRegister(VECMPIVIENNACL, VecCreate_MPIViennaCL));
  PetscCall(VecRegister(VECVIENNACL, VecCreate_ViennaCL));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(VecRegister(VECSEQCUDA, VecCreate_SeqCUDA));
  PetscCall(VecRegister(VECMPICUDA, VecCreate_MPICUDA));
  PetscCall(VecRegister(VECCUDA, VecCreate_CUDA));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(VecRegister(VECSEQKOKKOS, VecCreate_SeqKokkos));
  PetscCall(VecRegister(VECMPIKOKKOS, VecCreate_MPIKokkos));
  PetscCall(VecRegister(VECKOKKOS, VecCreate_Kokkos));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(VecRegister(VECSEQHIP, VecCreate_SeqHIP));
  PetscCall(VecRegister(VECMPIHIP, VecCreate_MPIHIP));
  PetscCall(VecRegister(VECHIP, VecCreate_HIP));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
