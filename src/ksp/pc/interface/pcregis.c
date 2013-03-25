
#include <petsc-private/pcimpl.h>          /*I   "petscpc.h"   I*/

PETSC_EXTERN PetscErrorCode PCCreate_Jacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_BJacobi(PC);
PETSC_EXTERN PetscErrorCode PCCreate_PBJacobi(PC);
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
PETSC_EXTERN PetscErrorCode PCCreate_HMPI(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Exotic(PC);
PETSC_EXTERN PetscErrorCode PCCreate_ASA(PC);
PETSC_EXTERN PetscErrorCode PCCreate_CP(PC);
PETSC_EXTERN PetscErrorCode PCCreate_LSC(PC);
PETSC_EXTERN PetscErrorCode PCCreate_Redistribute(PC);
PETSC_EXTERN PetscErrorCode PCCreate_SVD(PC);
PETSC_EXTERN PetscErrorCode PCCreate_GAMG(PC);

#if defined(PETSC_HAVE_BOOST) && defined(PETSC_CLANGUAGE_CXX)
PETSC_EXTERN PetscErrorCode PCCreate_SupportGraph(PC);
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
#endif
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN PetscErrorCode PCCreate_TFS(PC);
#endif
#if defined(PETSC_HAVE_CUSP_SMOOTHED_AGGREGATION) && defined(PETSC_HAVE_CUSP)
PETSC_EXTERN PetscErrorCode PCCreate_SACUSP(PC);
PETSC_EXTERN PetscErrorCode PCCreate_SACUSPPoly(PC);
PETSC_EXTERN PetscErrorCode PCCreate_BiCGStabCUSP(PC);
PETSC_EXTERN PetscErrorCode PCCreate_AINVCUSP(PC);
#endif
#if defined(PETSC_HAVE_PARMS)
PETSC_EXTERN PetscErrorCode PCCreate_PARMS(PC);
#endif
#if defined(PETSC_HAVE_PCBDDC)
PETSC_EXTERN PetscErrorCode PCCreate_BDDC(PC);
#endif

#undef __FUNCT__
#define __FUNCT__ "PCRegisterAll"
/*@C
   PCRegisterAll - Registers all of the preconditioners in the PC package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.keywords: PC, register, all

.seealso: PCRegister(), PCRegisterDestroy()
@*/
PetscErrorCode  PCRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PCRegisterAllCalled = PETSC_TRUE;

  ierr = PCRegister(PCNONE         ,path,"PCCreate_None",PCCreate_None);CHKERRQ(ierr);
  ierr = PCRegister(PCJACOBI       ,path,"PCCreate_Jacobi",PCCreate_Jacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCPBJACOBI     ,path,"PCCreate_PBJacobi",PCCreate_PBJacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCBJACOBI      ,path,"PCCreate_BJacobi",PCCreate_BJacobi);CHKERRQ(ierr);
  ierr = PCRegister(PCSOR          ,path,"PCCreate_SOR",PCCreate_SOR);CHKERRQ(ierr);
  ierr = PCRegister(PCLU           ,path,"PCCreate_LU",PCCreate_LU);CHKERRQ(ierr);
  ierr = PCRegister(PCSHELL        ,path,"PCCreate_Shell",PCCreate_Shell);CHKERRQ(ierr);
  ierr = PCRegister(PCMG           ,path,"PCCreate_MG",PCCreate_MG);CHKERRQ(ierr);
  ierr = PCRegister(PCEISENSTAT    ,path,"PCCreate_Eisenstat",PCCreate_Eisenstat);CHKERRQ(ierr);
  ierr = PCRegister(PCILU          ,path,"PCCreate_ILU",PCCreate_ILU);CHKERRQ(ierr);
  ierr = PCRegister(PCICC          ,path,"PCCreate_ICC",PCCreate_ICC);CHKERRQ(ierr);
  ierr = PCRegister(PCCHOLESKY     ,path,"PCCreate_Cholesky",PCCreate_Cholesky);CHKERRQ(ierr);
  ierr = PCRegister(PCASM          ,path,"PCCreate_ASM",PCCreate_ASM);CHKERRQ(ierr);
  ierr = PCRegister(PCGASM         ,path,"PCCreate_GASM",PCCreate_GASM);CHKERRQ(ierr);
  ierr = PCRegister(PCKSP          ,path,"PCCreate_KSP",PCCreate_KSP);CHKERRQ(ierr);
  ierr = PCRegister(PCCOMPOSITE    ,path,"PCCreate_Composite",PCCreate_Composite);CHKERRQ(ierr);
  ierr = PCRegister(PCREDUNDANT    ,path,"PCCreate_Redundant",PCCreate_Redundant);CHKERRQ(ierr);
  ierr = PCRegister(PCNN           ,path,"PCCreate_NN",PCCreate_NN);CHKERRQ(ierr);
  ierr = PCRegister(PCMAT          ,path,"PCCreate_Mat",PCCreate_Mat);CHKERRQ(ierr);
  ierr = PCRegister(PCFIELDSPLIT   ,path,"PCCreate_FieldSplit",PCCreate_FieldSplit);CHKERRQ(ierr);
  ierr = PCRegister(PCGALERKIN     ,path,"PCCreate_Galerkin",PCCreate_Galerkin);CHKERRQ(ierr);
  ierr = PCRegister(PCEXOTIC       ,path,"PCCreate_Exotic",PCCreate_Exotic);CHKERRQ(ierr);
  ierr = PCRegister(PCHMPI         ,path,"PCCreate_HMPI",PCCreate_HMPI);CHKERRQ(ierr);
  ierr = PCRegister(PCASA          ,path,"PCCreate_ASA",PCCreate_ASA);CHKERRQ(ierr);
  ierr = PCRegister(PCCP           ,path,"PCCreate_CP",PCCreate_CP);CHKERRQ(ierr);
  ierr = PCRegister(PCLSC          ,path,"PCCreate_LSC",PCCreate_LSC);CHKERRQ(ierr);
  ierr = PCRegister(PCREDISTRIBUTE ,path,"PCCreate_Redistribute",PCCreate_Redistribute);CHKERRQ(ierr);
  ierr = PCRegister(PCSVD          ,path,"PCCreate_SVD",PCCreate_SVD);CHKERRQ(ierr);
  ierr = PCRegister(PCGAMG         ,path,"PCCreate_GAMG",PCCreate_GAMG);CHKERRQ(ierr);
#if defined(PETSC_HAVE_BOOST) && defined(PETSC_CLANGUAGE_CXX)
  ierr = PCRegister(PCSUPPORTGRAPH ,path,"PCCreate_SupportGraph",PCCreate_SupportGraph);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ML)
  ierr = PCRegister(PCML           ,path,"PCCreate_ML",PCCreate_ML);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPAI)
  ierr = PCRegister(PCSPAI         ,path,"PCCreate_SPAI",PCCreate_SPAI);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HYPRE)
  ierr = PCRegister(PCHYPRE        ,path,"PCCreate_HYPRE",PCCreate_HYPRE);CHKERRQ(ierr);
  ierr = PCRegister(PCPFMG         ,path,"PCCreate_PFMG",PCCreate_PFMG);CHKERRQ(ierr);
  ierr = PCRegister(PCSYSPFMG      ,path,"PCCreate_SysPFMG",PCCreate_SysPFMG);CHKERRQ(ierr);
#endif
#if !defined(PETSC_USE_COMPLEX)
  ierr = PCRegister(PCTFS          ,path,"PCCreate_TFS",PCCreate_TFS);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUSP_SMOOTHED_AGGREGATION) && defined(PETSC_HAVE_CUSP)
  ierr = PCRegister(PCSACUSP       ,path,"PCCreate_SACUSP",PCCreate_SACUSP);CHKERRQ(ierr);
  ierr = PCRegister(PCAINVCUSP     ,path,"PCCreate_AINVCUSP",PCCreate_AINVCUSP);CHKERRQ(ierr);
  ierr = PCRegister(PCBICGSTABCUSP ,path,"PCCreate_BiCGStabCUSP",PCCreate_BiCGStabCUSP);CHKERRQ(ierr);
  ierr = PCRegister(PCSACUSPPOLY   ,path,"PCCreate_SACUSPPoly",PCCreate_SACUSPPoly);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PARMS)
  ierr = PCRegister(PCPARMS        ,path,"PCCreate_PARMS",PCCreate_PARMS);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PCBDDC)
  ierr = PCRegister(PCBDDC         ,path,"PCCreate_BDDC",PCCreate_BDDC);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
