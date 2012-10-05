
#include <petsc-private/pcimpl.h>          /*I   "petscpc.h"   I*/

EXTERN_C_BEGIN
extern PetscErrorCode  PCCreate_Jacobi(PC);
extern PetscErrorCode  PCCreate_BJacobi(PC);
extern PetscErrorCode  PCCreate_PBJacobi(PC);
extern PetscErrorCode  PCCreate_ILU(PC);
extern PetscErrorCode  PCCreate_None(PC);
extern PetscErrorCode  PCCreate_LU(PC);
extern PetscErrorCode  PCCreate_SOR(PC);
extern PetscErrorCode  PCCreate_Shell(PC);
extern PetscErrorCode  PCCreate_MG(PC);
extern PetscErrorCode  PCCreate_Eisenstat(PC);
extern PetscErrorCode  PCCreate_ICC(PC);
extern PetscErrorCode  PCCreate_ASM(PC);
extern PetscErrorCode  PCCreate_GASM(PC);
extern PetscErrorCode  PCCreate_KSP(PC);
extern PetscErrorCode  PCCreate_Composite(PC);
extern PetscErrorCode  PCCreate_Redundant(PC);
extern PetscErrorCode  PCCreate_NN(PC);
extern PetscErrorCode  PCCreate_Cholesky(PC);
extern PetscErrorCode  PCCreate_FieldSplit(PC);
extern PetscErrorCode  PCCreate_Galerkin(PC);
extern PetscErrorCode  PCCreate_HMPI(PC);
extern PetscErrorCode  PCCreate_Exotic(PC);
extern PetscErrorCode  PCCreate_ASA(PC);
extern PetscErrorCode  PCCreate_CP(PC);
extern PetscErrorCode  PCCreate_LSC(PC);
extern PetscErrorCode  PCCreate_Redistribute(PC);
extern PetscErrorCode  PCCreate_SVD(PC);
extern PetscErrorCode  PCCreate_GAMG(PC);

#if defined(PETSC_HAVE_BOOST) && defined(PETSC_CLANGUAGE_CXX)
extern PetscErrorCode  PCCreate_SupportGraph(PC);
#endif
#if defined(PETSC_HAVE_ML)
extern PetscErrorCode  PCCreate_ML(PC);
#endif
#if defined(PETSC_HAVE_SPAI)
extern PetscErrorCode  PCCreate_SPAI(PC);
#endif
extern PetscErrorCode  PCCreate_Mat(PC);
#if defined(PETSC_HAVE_HYPRE)
extern PetscErrorCode  PCCreate_HYPRE(PC);
extern PetscErrorCode  PCCreate_PFMG(PC);
extern PetscErrorCode  PCCreate_SysPFMG(PC);
#endif
#if !defined(PETSC_USE_COMPLEX)
extern PetscErrorCode  PCCreate_TFS(PC);
#endif
#if defined(PETSC_HAVE_CUSP_SMOOTHED_AGGREGATION) && defined(PETSC_HAVE_CUSP)
extern PetscErrorCode  PCCreate_SACUSP(PC);
extern PetscErrorCode  PCCreate_SACUSPPoly(PC);
extern PetscErrorCode  PCCreate_BiCGStabCUSP(PC);
extern PetscErrorCode  PCCreate_AINVCUSP(PC);
#endif
#if defined(PETSC_HAVE_PARMS)
extern PetscErrorCode  PCCreate_PARMS(PC);
#endif
#if defined(PETSC_HAVE_PCBDDC)
extern PetscErrorCode  PCCreate_BDDC(PC);
#endif

EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PCRegisterAll"
/*@C
   PCRegisterAll - Registers all of the preconditioners in the PC package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.keywords: PC, register, all

.seealso: PCRegisterDynamic(), PCRegisterDestroy()
@*/
PetscErrorCode  PCRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PCRegisterAllCalled = PETSC_TRUE;

  ierr = PCRegisterDynamic(PCNONE         ,path,"PCCreate_None",PCCreate_None);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCJACOBI       ,path,"PCCreate_Jacobi",PCCreate_Jacobi);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCPBJACOBI     ,path,"PCCreate_PBJacobi",PCCreate_PBJacobi);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCBJACOBI      ,path,"PCCreate_BJacobi",PCCreate_BJacobi);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCSOR          ,path,"PCCreate_SOR",PCCreate_SOR);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCLU           ,path,"PCCreate_LU",PCCreate_LU);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCSHELL        ,path,"PCCreate_Shell",PCCreate_Shell);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCMG           ,path,"PCCreate_MG",PCCreate_MG);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCEISENSTAT    ,path,"PCCreate_Eisenstat",PCCreate_Eisenstat);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCILU          ,path,"PCCreate_ILU",PCCreate_ILU);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCICC          ,path,"PCCreate_ICC",PCCreate_ICC);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCCHOLESKY     ,path,"PCCreate_Cholesky",PCCreate_Cholesky);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCASM          ,path,"PCCreate_ASM",PCCreate_ASM);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCGASM         ,path,"PCCreate_GASM",PCCreate_GASM);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCKSP          ,path,"PCCreate_KSP",PCCreate_KSP);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCCOMPOSITE    ,path,"PCCreate_Composite",PCCreate_Composite);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCREDUNDANT    ,path,"PCCreate_Redundant",PCCreate_Redundant);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCNN           ,path,"PCCreate_NN",PCCreate_NN);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCMAT          ,path,"PCCreate_Mat",PCCreate_Mat);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCFIELDSPLIT   ,path,"PCCreate_FieldSplit",PCCreate_FieldSplit);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCGALERKIN     ,path,"PCCreate_Galerkin",PCCreate_Galerkin);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCEXOTIC       ,path,"PCCreate_Exotic",PCCreate_Exotic);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCHMPI         ,path,"PCCreate_HMPI",PCCreate_HMPI);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCASA          ,path,"PCCreate_ASA",PCCreate_ASA);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCCP           ,path,"PCCreate_CP",PCCreate_CP);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCLSC          ,path,"PCCreate_LSC",PCCreate_LSC);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCREDISTRIBUTE ,path,"PCCreate_Redistribute",PCCreate_Redistribute);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCSVD          ,path,"PCCreate_SVD",PCCreate_SVD);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCGAMG         ,path,"PCCreate_GAMG",PCCreate_GAMG);CHKERRQ(ierr);
#if defined(PETSC_HAVE_BOOST) && defined(PETSC_CLANGUAGE_CXX)
  ierr = PCRegisterDynamic(PCSUPPORTGRAPH ,path,"PCCreate_SupportGraph",PCCreate_SupportGraph);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ML)
  ierr = PCRegisterDynamic(PCML           ,path,"PCCreate_ML",PCCreate_ML);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPAI)
  ierr = PCRegisterDynamic(PCSPAI         ,path,"PCCreate_SPAI",PCCreate_SPAI);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HYPRE)
  ierr = PCRegisterDynamic(PCHYPRE        ,path,"PCCreate_HYPRE",PCCreate_HYPRE);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCPFMG         ,path,"PCCreate_PFMG",PCCreate_PFMG);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCSYSPFMG      ,path,"PCCreate_SysPFMG",PCCreate_SysPFMG);CHKERRQ(ierr);
#endif
#if !defined(PETSC_USE_COMPLEX)
  ierr = PCRegisterDynamic(PCTFS          ,path,"PCCreate_TFS",PCCreate_TFS);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUSP_SMOOTHED_AGGREGATION) && defined(PETSC_HAVE_CUSP)
  ierr = PCRegisterDynamic(PCSACUSP       ,path,"PCCreate_SACUSP",PCCreate_SACUSP);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCAINVCUSP     ,path,"PCCreate_AINVCUSP",PCCreate_AINVCUSP);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCBICGSTABCUSP ,path,"PCCreate_BiCGStabCUSP",PCCreate_BiCGStabCUSP);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCSACUSPPOLY    ,path,"PCCreate_SACUSPPoly",PCCreate_SACUSPPoly);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PARMS)
  ierr = PCRegisterDynamic(PCPARMS   ,path,"PCCreate_PARMS",PCCreate_PARMS);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PCBDDC)
  ierr = PCRegisterDynamic(PCBDDC         ,path,"PCCreate_BDDC",PCCreate_BDDC);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
