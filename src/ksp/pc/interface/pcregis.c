
#include <petsc/private/pcimpl.h> /*I   "petscpc.h"   I*/

PETSC_EXTERN PetscErrorCode PCCreate_Jacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_BJacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_PBJacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_VPBJacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_ILU(PC);
PETSC_EXTERN PetscErrorCode PCCreate_None(PC);
PETSC_EXTERN PetscErrorCode PCCreate_LU(PC);
PETSC_EXTERN PetscErrorCode PCCreate_QR(PC);
PETSC_EXTERN PetscErrorCode PCCreate_SOR(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Shell(PC);
PETSC_EXTERN PetscErrorCode PCCreate_MG(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Eisenstat(PC);
PETSC_EXTERN PetscErrorCode PCCreate_ICC(PC);
PETSC_EXTERN PetscErrorCode PCCreate_ASM(PC);
PETSC_EXTERN PetscErrorCode PCCreate_GASM(PC);
PETSC_EXTERN PetscErrorCode PCCreate_KSP(PC);
PETSC_EXTERN PetscErrorCode PCCreate_BJKOKKOS(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Composite(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Redundant(PC);
PETSC_EXTERN PetscErrorCode PCCreate_NN(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Cholesky(PC);
PETSC_EXTERN PetscErrorCode PCCreate_FieldSplit(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Galerkin(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Exotic(PC);
PETSC_EXTERN PetscErrorCode PCCreate_CP(PC);
PETSC_EXTERN PetscErrorCode PCCreate_LSC(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Redistribute(PC);
PETSC_EXTERN PetscErrorCode PCCreate_SVD(PC);
PETSC_EXTERN PetscErrorCode PCCreate_GAMG(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Kaczmarz(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Telescope(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Patch(PC);
PETSC_EXTERN PetscErrorCode PCCreate_LMVM(PC);
PETSC_EXTERN PetscErrorCode PCCreate_HMG(PC);
#if defined(PETSC_HAVE_AMGX)
PETSC_EXTERN PetscErrorCode PCCreate_AMGX(PC);
#endif
#if defined(PETSC_HAVE_ML)
PETSC_EXTERN PetscErrorCode PCCreate_ML(PC);
#endif
#if defined(PETSC_HAVE_SPAI)
PETSC_EXTERN PetscErrorCode PCCreate_SPAI(PC);
#endif
PETSC_EXTERN PetscErrorCode PCCreate_Mat(PC);
#if defined(PETSC_HAVE_HYPRE)
PETSC_EXTERN PetscErrorCode PCCreate_HYPRE(PC);
PETSC_EXTERN PetscErrorCode PCCreate_PFMG(PC);
PETSC_EXTERN PetscErrorCode PCCreate_SysPFMG(PC);
PETSC_EXTERN PetscErrorCode PCCreate_SMG(PC);
#endif
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN PetscErrorCode PCCreate_TFS(PC);
#endif
#if defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode PCCreate_CHOWILUVIENNACL(PC);
PETSC_EXTERN PetscErrorCode PCCreate_ROWSCALINGVIENNACL(PC);
PETSC_EXTERN PetscErrorCode PCCreate_SAVIENNACL(PC);
#endif
#if defined(PETSC_HAVE_PARMS)
PETSC_EXTERN PetscErrorCode PCCreate_PARMS(PC);
#endif
PETSC_EXTERN PetscErrorCode PCCreate_BDDC(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Deflation(PC);
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
PETSC_EXTERN PetscErrorCode PCCreate_HPDDM(PC);
#endif
#if defined(PETSC_HAVE_H2OPUS)
PETSC_EXTERN PetscErrorCode PCCreate_H2OPUS(PC);
#endif
PETSC_EXTERN PetscErrorCode PCCreate_MPI(PC);

/*@C
   PCRegisterAll - Registers all of the preconditioners in the PC package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.seealso: `PCRegister()`
@*/
PetscErrorCode PCRegisterAll(void)
{
  PetscFunctionBegin;
  if (PCRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PCRegisterAllCalled = PETSC_TRUE;

  PetscCall(PCRegister(PCNONE, PCCreate_None));
  PetscCall(PCRegister(PCJACOBI, PCCreate_Jacobi));
  PetscCall(PCRegister(PCPBJACOBI, PCCreate_PBJacobi));
  PetscCall(PCRegister(PCVPBJACOBI, PCCreate_VPBJacobi));
  PetscCall(PCRegister(PCBJACOBI, PCCreate_BJacobi));
  PetscCall(PCRegister(PCSOR, PCCreate_SOR));
  PetscCall(PCRegister(PCLU, PCCreate_LU));
  PetscCall(PCRegister(PCQR, PCCreate_QR));
  PetscCall(PCRegister(PCSHELL, PCCreate_Shell));
  PetscCall(PCRegister(PCMG, PCCreate_MG));
  PetscCall(PCRegister(PCEISENSTAT, PCCreate_Eisenstat));
  PetscCall(PCRegister(PCILU, PCCreate_ILU));
  PetscCall(PCRegister(PCICC, PCCreate_ICC));
  PetscCall(PCRegister(PCCHOLESKY, PCCreate_Cholesky));
  PetscCall(PCRegister(PCASM, PCCreate_ASM));
  PetscCall(PCRegister(PCGASM, PCCreate_GASM));
  PetscCall(PCRegister(PCKSP, PCCreate_KSP));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PCRegister(PCBJKOKKOS, PCCreate_BJKOKKOS));
#endif
  PetscCall(PCRegister(PCCOMPOSITE, PCCreate_Composite));
  PetscCall(PCRegister(PCREDUNDANT, PCCreate_Redundant));
  PetscCall(PCRegister(PCNN, PCCreate_NN));
  PetscCall(PCRegister(PCMAT, PCCreate_Mat));
  PetscCall(PCRegister(PCFIELDSPLIT, PCCreate_FieldSplit));
  PetscCall(PCRegister(PCGALERKIN, PCCreate_Galerkin));
  PetscCall(PCRegister(PCEXOTIC, PCCreate_Exotic));
  PetscCall(PCRegister(PCCP, PCCreate_CP));
  PetscCall(PCRegister(PCLSC, PCCreate_LSC));
  PetscCall(PCRegister(PCREDISTRIBUTE, PCCreate_Redistribute));
  PetscCall(PCRegister(PCSVD, PCCreate_SVD));
  PetscCall(PCRegister(PCGAMG, PCCreate_GAMG));
  PetscCall(PCRegister(PCKACZMARZ, PCCreate_Kaczmarz));
  PetscCall(PCRegister(PCTELESCOPE, PCCreate_Telescope));
  PetscCall(PCRegister(PCPATCH, PCCreate_Patch));
  PetscCall(PCRegister(PCHMG, PCCreate_HMG));
#if defined(PETSC_HAVE_AMGX)
  PetscCall(PCRegister(PCAMGX, PCCreate_AMGX));
#endif
#if defined(PETSC_HAVE_ML)
  PetscCall(PCRegister(PCML, PCCreate_ML));
#endif
#if defined(PETSC_HAVE_SPAI)
  PetscCall(PCRegister(PCSPAI, PCCreate_SPAI));
#endif
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PCRegister(PCHYPRE, PCCreate_HYPRE));
  PetscCall(PCRegister(PCPFMG, PCCreate_PFMG));
  PetscCall(PCRegister(PCSYSPFMG, PCCreate_SysPFMG));
  PetscCall(PCRegister(PCSMG, PCCreate_SMG));
#endif
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PCRegister(PCTFS, PCCreate_TFS));
#endif
#if defined(PETSC_HAVE_VIENNACL)
  PetscCall(PCRegister(PCCHOWILUVIENNACL, PCCreate_CHOWILUVIENNACL));
  PetscCall(PCRegister(PCROWSCALINGVIENNACL, PCCreate_ROWSCALINGVIENNACL));
  PetscCall(PCRegister(PCSAVIENNACL, PCCreate_SAVIENNACL));
#endif
#if defined(PETSC_HAVE_PARMS)
  PetscCall(PCRegister(PCPARMS, PCCreate_PARMS));
#endif
  PetscCall(PCRegister(PCBDDC, PCCreate_BDDC));
  PetscCall(PCRegister(PCLMVM, PCCreate_LMVM));
  PetscCall(PCRegister(PCDEFLATION, PCCreate_Deflation));
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
  PetscCall(PCRegister(PCHPDDM, PCCreate_HPDDM));
#endif
#if defined(PETSC_HAVE_H2OPUS)
  PetscCall(PCRegister(PCH2OPUS, PCCreate_H2OPUS));
#endif
  PetscCall(PCRegister(PCMPI, PCCreate_MPI));
  PetscFunctionReturn(PETSC_SUCCESS);
}
