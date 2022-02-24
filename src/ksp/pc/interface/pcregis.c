
#include <petsc/private/pcimpl.h>          /*I   "petscpc.h"   I*/

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

/*@C
   PCRegisterAll - Registers all of the preconditioners in the PC package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.seealso: PCRegister()
@*/
PetscErrorCode  PCRegisterAll(void)
{
  PetscFunctionBegin;
  if (PCRegisterAllCalled) PetscFunctionReturn(0);
  PCRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PCRegister(PCNONE         ,PCCreate_None));
  CHKERRQ(PCRegister(PCJACOBI       ,PCCreate_Jacobi));
  CHKERRQ(PCRegister(PCPBJACOBI     ,PCCreate_PBJacobi));
  CHKERRQ(PCRegister(PCVPBJACOBI    ,PCCreate_VPBJacobi));
  CHKERRQ(PCRegister(PCBJACOBI      ,PCCreate_BJacobi));
  CHKERRQ(PCRegister(PCSOR          ,PCCreate_SOR));
  CHKERRQ(PCRegister(PCLU           ,PCCreate_LU));
  CHKERRQ(PCRegister(PCQR           ,PCCreate_QR));
  CHKERRQ(PCRegister(PCSHELL        ,PCCreate_Shell));
  CHKERRQ(PCRegister(PCMG           ,PCCreate_MG));
  CHKERRQ(PCRegister(PCEISENSTAT    ,PCCreate_Eisenstat));
  CHKERRQ(PCRegister(PCILU          ,PCCreate_ILU));
  CHKERRQ(PCRegister(PCICC          ,PCCreate_ICC));
  CHKERRQ(PCRegister(PCCHOLESKY     ,PCCreate_Cholesky));
  CHKERRQ(PCRegister(PCASM          ,PCCreate_ASM));
  CHKERRQ(PCRegister(PCGASM         ,PCCreate_GASM));
  CHKERRQ(PCRegister(PCKSP          ,PCCreate_KSP));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  CHKERRQ(PCRegister(PCBJKOKKOS     ,PCCreate_BJKOKKOS));
#endif
  CHKERRQ(PCRegister(PCCOMPOSITE    ,PCCreate_Composite));
  CHKERRQ(PCRegister(PCREDUNDANT    ,PCCreate_Redundant));
  CHKERRQ(PCRegister(PCNN           ,PCCreate_NN));
  CHKERRQ(PCRegister(PCMAT          ,PCCreate_Mat));
  CHKERRQ(PCRegister(PCFIELDSPLIT   ,PCCreate_FieldSplit));
  CHKERRQ(PCRegister(PCGALERKIN     ,PCCreate_Galerkin));
  CHKERRQ(PCRegister(PCEXOTIC       ,PCCreate_Exotic));
  CHKERRQ(PCRegister(PCCP           ,PCCreate_CP));
  CHKERRQ(PCRegister(PCLSC          ,PCCreate_LSC));
  CHKERRQ(PCRegister(PCREDISTRIBUTE ,PCCreate_Redistribute));
  CHKERRQ(PCRegister(PCSVD          ,PCCreate_SVD));
  CHKERRQ(PCRegister(PCGAMG         ,PCCreate_GAMG));
  CHKERRQ(PCRegister(PCKACZMARZ     ,PCCreate_Kaczmarz));
  CHKERRQ(PCRegister(PCTELESCOPE    ,PCCreate_Telescope));
  CHKERRQ(PCRegister(PCPATCH        ,PCCreate_Patch));
  CHKERRQ(PCRegister(PCHMG          ,PCCreate_HMG));
#if defined(PETSC_HAVE_ML)
  CHKERRQ(PCRegister(PCML           ,PCCreate_ML));
#endif
#if defined(PETSC_HAVE_SPAI)
  CHKERRQ(PCRegister(PCSPAI         ,PCCreate_SPAI));
#endif
#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(PCRegister(PCHYPRE        ,PCCreate_HYPRE));
  CHKERRQ(PCRegister(PCPFMG         ,PCCreate_PFMG));
  CHKERRQ(PCRegister(PCSYSPFMG      ,PCCreate_SysPFMG));
#endif
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(PCRegister(PCTFS          ,PCCreate_TFS));
#endif
#if defined(PETSC_HAVE_VIENNACL)
  CHKERRQ(PCRegister(PCCHOWILUVIENNACL,PCCreate_CHOWILUVIENNACL));
  CHKERRQ(PCRegister(PCROWSCALINGVIENNACL,PCCreate_ROWSCALINGVIENNACL));
  CHKERRQ(PCRegister(PCSAVIENNACL   ,PCCreate_SAVIENNACL));
#endif
#if defined(PETSC_HAVE_PARMS)
  CHKERRQ(PCRegister(PCPARMS        ,PCCreate_PARMS));
#endif
  CHKERRQ(PCRegister(PCBDDC         ,PCCreate_BDDC));
  CHKERRQ(PCRegister(PCLMVM         ,PCCreate_LMVM));
  CHKERRQ(PCRegister(PCDEFLATION    ,PCCreate_Deflation));
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
  CHKERRQ(PCRegister(PCHPDDM        ,PCCreate_HPDDM));
#endif
#if defined(PETSC_HAVE_H2OPUS)
  CHKERRQ(PCRegister(PCH2OPUS       ,PCCreate_H2OPUS));
#endif
  PetscFunctionReturn(0);
}
