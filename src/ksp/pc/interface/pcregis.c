/*$Id: pcregis.c,v 1.66 2001/08/10 03:32:21 bsmith Exp $*/

#include "src/ksp/pc/pcimpl.h"          /*I   "petscpc.h"   I*/

EXTERN_C_BEGIN
EXTERN int PCCreate_Jacobi(PC);
EXTERN int PCCreate_BJacobi(PC);
EXTERN int PCCreate_PBJacobi(PC);
EXTERN int PCCreate_PBSOR(PC);
EXTERN int PCCreate_ILU(PC);
EXTERN int PCCreate_None(PC);
EXTERN int PCCreate_LU(PC);
EXTERN int PCCreate_SOR(PC);
EXTERN int PCCreate_Shell(PC);
EXTERN int PCCreate_MG(PC);
EXTERN int PCCreate_Eisenstat(PC);
EXTERN int PCCreate_ICC(PC);
EXTERN int PCCreate_ASM(PC);
EXTERN int PCCreate_KSP(PC);
EXTERN int PCCreate_Composite(PC);
EXTERN int PCCreate_Redundant(PC);
EXTERN int PCCreate_NN(PC);
EXTERN int PCCreate_Cholesky(PC);
#if defined(PETSC_HAVE_SPAI) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN int PCCreate_SPAI(PC);
#endif
#if defined(PETSC_HAVE_RAMG)  && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN int PCCreate_RAMG(PC);
#endif
#if defined(PETSC_HAVE_SAMG)  && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN int PCCreate_SAMG(PC);
#endif
EXTERN int PCCreate_mILU(PC);
EXTERN int PCCreate_PetscESI(PC);
EXTERN int PCCreate_ESI(PC);
EXTERN int PCCreate_Mat(PC);
#if defined(PETSC_HAVE_HYPRE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN int PCCreate_HYPRE(PC);
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
int PCRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  PCRegisterAllCalled = PETSC_TRUE;

  ierr = PCRegisterDynamic(PCNONE         ,path,"PCCreate_None",PCCreate_None);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCJACOBI       ,path,"PCCreate_Jacobi",PCCreate_Jacobi);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCPBJACOBI     ,path,"PCCreate_PBJacobi",PCCreate_PBJacobi);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCBJACOBI      ,path,"PCCreate_BJacobi",PCCreate_BJacobi);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCSOR          ,path,"PCCreate_SOR",PCCreate_SOR);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCPBSOR        ,path,"PCCreate_PBSOR",PCCreate_PBSOR);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCLU           ,path,"PCCreate_LU",PCCreate_LU);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCSHELL        ,path,"PCCreate_Shell",PCCreate_Shell);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCMG           ,path,"PCCreate_MG",PCCreate_MG);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCEISENSTAT    ,path,"PCCreate_Eisenstat",PCCreate_Eisenstat);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCILU          ,path,"PCCreate_ILU",PCCreate_ILU);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCICC          ,path,"PCCreate_ICC",PCCreate_ICC);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCCHOLESKY     ,path,"PCCreate_Cholesky",PCCreate_Cholesky);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCASM          ,path,"PCCreate_ASM",PCCreate_ASM);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCKSP         ,path,"PCCreate_KSP",PCCreate_KSP);CHKERRQ(ierr);
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
  ierr = PCRegisterDynamic(PCMILU         ,path,"PCCreate_mILU",PCCreate_mILU);CHKERRQ(ierr);
#if defined(__cplusplus) && !defined(PETSC_USE_SINGLE) && !defined (PETSC_USE_COMPLEX) && defined(PETSC_HAVE_CXX_NAMESPACE)
  ierr = PCRegisterDynamic(PCESI          ,path,"PCCreate_ESI",PCCreate_ESI);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCPETSCESI     ,path,"PCCreate_PetscESI",PCCreate_PetscESI);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HYPRE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PCRegisterDynamic(PCHYPRE        ,path,"PCCreate_HYPRE",PCCreate_HYPRE);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}




