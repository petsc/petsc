
#include <petsc/private/pcimpl.h>          /*I   "petscpc.h"   I*/

PETSC_EXTERN PetscErrorCode PCCreate_Jacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_BJacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_PBJacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_VPBJacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_ILU(PC);
PETSC_EXTERN PetscErrorCode PCCreate_None(PC);
PETSC_EXTERN PetscErrorCode PCCreate_LU(PC);
PETSC_EXTERN PetscErrorCode PCCreate_SOR(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Shell(PC);
PETSC_EXTERN PetscErrorCode PCCreate_MG(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Eisenstat(PC);
PETSC_EXTERN PetscErrorCode PCCreate_ICC(PC);
PETSC_EXTERN PetscErrorCode PCCreate_ASM(PC);
PETSC_EXTERN PetscErrorCode PCCreate_GASM(PC);
PETSC_EXTERN PetscErrorCode PCCreate_KSP(PC);
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
#if defined(PETSC_HAVE_HARA)
PETSC_EXTERN PetscErrorCode PCCreate_HARA(PC);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PCRegisterAllCalled) PetscFunctionReturn(0);
  PCRegisterAllCalled = PETSC_TRUE;

  ierr = PCRegister(PCNONE         ,PCCreate_None);CHKERRQ(ierr);
  ierr = PCRegister(PCJACOBI       ,PCCreate_Jacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCPBJACOBI     ,PCCreate_PBJacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCVPBJACOBI    ,PCCreate_VPBJacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCBJACOBI      ,PCCreate_BJacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCSOR          ,PCCreate_SOR);CHKERRQ(ierr);
  ierr = PCRegister(PCLU           ,PCCreate_LU);CHKERRQ(ierr);
  ierr = PCRegister(PCSHELL        ,PCCreate_Shell);CHKERRQ(ierr);
  ierr = PCRegister(PCMG           ,PCCreate_MG);CHKERRQ(ierr);
  ierr = PCRegister(PCEISENSTAT    ,PCCreate_Eisenstat);CHKERRQ(ierr);
  ierr = PCRegister(PCILU          ,PCCreate_ILU);CHKERRQ(ierr);
  ierr = PCRegister(PCICC          ,PCCreate_ICC);CHKERRQ(ierr);
  ierr = PCRegister(PCCHOLESKY     ,PCCreate_Cholesky);CHKERRQ(ierr);
  ierr = PCRegister(PCASM          ,PCCreate_ASM);CHKERRQ(ierr);
  ierr = PCRegister(PCGASM         ,PCCreate_GASM);CHKERRQ(ierr);
  ierr = PCRegister(PCKSP          ,PCCreate_KSP);CHKERRQ(ierr);
  ierr = PCRegister(PCCOMPOSITE    ,PCCreate_Composite);CHKERRQ(ierr);
  ierr = PCRegister(PCREDUNDANT    ,PCCreate_Redundant);CHKERRQ(ierr);
  ierr = PCRegister(PCNN           ,PCCreate_NN);CHKERRQ(ierr);
  ierr = PCRegister(PCMAT          ,PCCreate_Mat);CHKERRQ(ierr);
  ierr = PCRegister(PCFIELDSPLIT   ,PCCreate_FieldSplit);CHKERRQ(ierr);
  ierr = PCRegister(PCGALERKIN     ,PCCreate_Galerkin);CHKERRQ(ierr);
  ierr = PCRegister(PCEXOTIC       ,PCCreate_Exotic);CHKERRQ(ierr);
  ierr = PCRegister(PCCP           ,PCCreate_CP);CHKERRQ(ierr);
  ierr = PCRegister(PCLSC          ,PCCreate_LSC);CHKERRQ(ierr);
  ierr = PCRegister(PCREDISTRIBUTE ,PCCreate_Redistribute);CHKERRQ(ierr);
  ierr = PCRegister(PCSVD          ,PCCreate_SVD);CHKERRQ(ierr);
  ierr = PCRegister(PCGAMG         ,PCCreate_GAMG);CHKERRQ(ierr);
  ierr = PCRegister(PCKACZMARZ     ,PCCreate_Kaczmarz);CHKERRQ(ierr);
  ierr = PCRegister(PCTELESCOPE    ,PCCreate_Telescope);CHKERRQ(ierr);
  ierr = PCRegister(PCPATCH        ,PCCreate_Patch);CHKERRQ(ierr);
  ierr = PCRegister(PCHMG          ,PCCreate_HMG);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ML)
  ierr = PCRegister(PCML           ,PCCreate_ML);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPAI)
  ierr = PCRegister(PCSPAI         ,PCCreate_SPAI);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HYPRE)
  ierr = PCRegister(PCHYPRE        ,PCCreate_HYPRE);CHKERRQ(ierr);
  ierr = PCRegister(PCPFMG         ,PCCreate_PFMG);CHKERRQ(ierr);
  ierr = PCRegister(PCSYSPFMG      ,PCCreate_SysPFMG);CHKERRQ(ierr);
#endif
#if !defined(PETSC_USE_COMPLEX)
  ierr = PCRegister(PCTFS          ,PCCreate_TFS);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_VIENNACL)
  ierr = PCRegister(PCCHOWILUVIENNACL,PCCreate_CHOWILUVIENNACL);CHKERRQ(ierr);
  ierr = PCRegister(PCROWSCALINGVIENNACL,PCCreate_ROWSCALINGVIENNACL);CHKERRQ(ierr);
  ierr = PCRegister(PCSAVIENNACL   ,PCCreate_SAVIENNACL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PARMS)
  ierr = PCRegister(PCPARMS        ,PCCreate_PARMS);CHKERRQ(ierr);
#endif
  ierr = PCRegister(PCBDDC         ,PCCreate_BDDC);CHKERRQ(ierr);
  ierr = PCRegister(PCLMVM         ,PCCreate_LMVM);CHKERRQ(ierr);
  ierr = PCRegister(PCDEFLATION    ,PCCreate_Deflation);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
  ierr = PCRegister(PCHPDDM        ,PCCreate_HPDDM);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HARA)
  ierr = PCRegister(PCHARA         ,PCCreate_HARA);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
