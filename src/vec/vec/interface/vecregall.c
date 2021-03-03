
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (VecRegisterAllCalled) PetscFunctionReturn(0);
  VecRegisterAllCalled = PETSC_TRUE;

  ierr = VecRegister(VECSEQ,        VecCreate_Seq);CHKERRQ(ierr);
  ierr = VecRegister(VECMPI,        VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegister(VECSTANDARD,   VecCreate_Standard);CHKERRQ(ierr);
  ierr = VecRegister(VECSHARED,     VecCreate_Shared);CHKERRQ(ierr);
#if defined PETSC_HAVE_VIENNACL
  ierr = VecRegister(VECSEQVIENNACL,    VecCreate_SeqViennaCL);CHKERRQ(ierr);
  ierr = VecRegister(VECMPIVIENNACL,    VecCreate_MPIViennaCL);CHKERRQ(ierr);
  ierr = VecRegister(VECVIENNACL,       VecCreate_ViennaCL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUDA)
  ierr = VecRegister(VECSEQCUDA,    VecCreate_SeqCUDA);CHKERRQ(ierr);
  ierr = VecRegister(VECMPICUDA,    VecCreate_MPICUDA);CHKERRQ(ierr);
  ierr = VecRegister(VECCUDA,       VecCreate_CUDA);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  ierr = VecRegister(VECSEQKOKKOS,  VecCreate_SeqKokkos);CHKERRQ(ierr);
  ierr = VecRegister(VECMPIKOKKOS,  VecCreate_MPIKokkos);CHKERRQ(ierr);
  ierr = VecRegister(VECKOKKOS,     VecCreate_Kokkos);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HIP)
  ierr = VecRegister(VECSEQHIP,    VecCreate_SeqHIP);CHKERRQ(ierr);
  ierr = VecRegister(VECMPIHIP,    VecCreate_MPIHIP);CHKERRQ(ierr);
  ierr = VecRegister(VECHIP,       VecCreate_HIP);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
