
#include "src/ksp/pc/pcimpl.h"          /*I   "petscpc.h"   I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode PCCreate_Jacobi(PC);
EXTERN PetscErrorCode PCCreate_BJacobi(PC);
EXTERN PetscErrorCode PCCreate_PBJacobi(PC);
EXTERN PetscErrorCode PCCreate_ILU(PC);
EXTERN PetscErrorCode PCCreate_None(PC);
EXTERN PetscErrorCode PCCreate_LU(PC);
EXTERN PetscErrorCode PCCreate_SOR(PC);
EXTERN PetscErrorCode PCCreate_Shell(PC);
EXTERN PetscErrorCode PCCreate_MG(PC);
EXTERN PetscErrorCode PCCreate_Eisenstat(PC);
EXTERN PetscErrorCode PCCreate_ICC(PC);
EXTERN PetscErrorCode PCCreate_ASM(PC);
EXTERN PetscErrorCode PCCreate_KSP(PC);
EXTERN PetscErrorCode PCCreate_Composite(PC);
EXTERN PetscErrorCode PCCreate_Redundant(PC);
EXTERN PetscErrorCode PCCreate_NN(PC);
EXTERN PetscErrorCode PCCreate_Cholesky(PC);
#if defined(PETSC_HAVE_SPAI) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN PetscErrorCode PCCreate_SPAI(PC);
#endif
#if defined(PETSC_HAVE_RAMG)  && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN PetscErrorCode PCCreate_RAMG(PC);
#endif
#if defined(PETSC_HAVE_SAMG)  && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN PetscErrorCode PCCreate_SAMG(PC);
#endif
EXTERN PetscErrorCode PCCreate_Mat(PC);
#if defined(PETSC_HAVE_HYPRE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN PetscErrorCode PCCreate_HYPRE(PC);
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
PetscErrorCode PCRegisterAll(const char path[])
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
  ierr = PCRegisterDynamic(PCKSP          ,path,"PCCreate_KSP",PCCreate_KSP);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCCOMPOSITE    ,path,"PCCreate_Composite",PCCreate_Composite);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCREDUNDANT    ,path,"PCCreate_Redundant",PCCreate_Redundant);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCNN           ,path,"PCCreate_NN",PCCreate_NN);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCMAT          ,path,"PCCreate_Mat",PCCreate_Mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SPAI) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PCRegisterDynamic(PCSPAI         ,path,"PCCreate_SPAI",PCCreate_SPAI);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_RAMG) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PCRegisterDynamic(PCRAMG         ,path,"PCCreate_RAMG",PCCreate_RAMG);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SAMG) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PCRegisterDynamic(PCSAMG         ,path,"PCCreate_SAMG",PCCreate_SAMG);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HYPRE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PCRegisterDynamic(PCHYPRE        ,path,"PCCreate_HYPRE",PCCreate_HYPRE);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}




