#ifndef _COMPAT_PETSC_PC_H
#define _COMPAT_PETSC_PC_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#include "private/pcimpl.h"
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define PCSACUSP          "sacusp"
#define PCSACUSPPOLY      "sacusppoly"
#define PCBICGSTABCUSP    "bicgstabcusp"
#define PCSVD             "svd"
#define PCAINVCUSP        "ainvcusp"
#define PCHMPI            "hmpi"
#define PCGAMG            "amg"
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "PCSetDM"
static PetscErrorCode PCSetDM(PC pc,DM dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(dm,DM_COOKIE,2);
  ierr = PetscObjectCompose((PetscObject)pc, "__DM__",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PCGetDM"
static PetscErrorCode PCGetDM(PC pc,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(dm,2);
  ierr = PetscObjectQuery((PetscObject)pc, "__DM__",(PetscObject*)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__  
#define __FUNCT__ "PCReset"
static PetscErrorCode PCReset_Compat(PC pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(0);
}
#define PCReset PCReset_Compat
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
/* PCFieldSplitSetFields */
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetFields"
static PetscErrorCode PCFieldSplitSetFields_Compat(PC pc,const char splitname[],
                                                   PetscInt n,const PetscInt *fields)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCFieldSplitSetFields(pc,n,(PetscInt*)fields);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PCFieldSplitSetFields PCFieldSplitSetFields_Compat
/* PCFieldSplitSetIS */
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetIS"
static PetscErrorCode PCFieldSplitSetIS_Compat(PC pc,const char splitname[],IS is)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCFieldSplitSetIS(pc,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PCFieldSplitSetIS PCFieldSplitSetIS_Compat
#endif

#if PETSC_VERSION_(3,0,0)
/* PCASMSetLocalSubdomains */
#undef __FUNCT__
#define __FUNCT__ "PCASMSetLocalSubdomains"
static PetscErrorCode PCASMSetLocalSubdomains_Compat(PC pc,PetscInt n,
                                              IS is[],IS is_local[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (is_local) SETERRQ(PETSC_ERR_SUP,"local subdomains not supported");
  ierr = PCASMSetLocalSubdomains(pc,n,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PCASMSetLocalSubdomains PCASMSetLocalSubdomains_Compat
/* PCASMSetTotalSubdomains */
#undef __FUNCT__
#define __FUNCT__ "PCASMSetTotalSubdomains"
static PetscErrorCode PCASMSetTotalSubdomains_Compat(PC pc,PetscInt N,
                                                     IS is[],IS is_local[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (is_local) SETERRQ(PETSC_ERR_SUP,"local subdomains not supported");
  ierr = PCASMSetTotalSubdomains(pc,N,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PCASMSetTotalSubdomains PCASMSetTotalSubdomains_Compat
#endif

#if (PETSC_VERSION_(3,0,0))
#define PCLSC          "lsc"
#define PCPFMG         "pfmg"
#define PCSYSPFMG      "syspfmg"
#define PCREDISTRIBUTE "redistribute"
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "PCFactorSetShiftType"
PetscErrorCode PCFactorSetShiftType(PC pc,MatFactorShiftType shifttype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (shifttype == MAT_SHIFT_NONE) {
    ierr = PCFactorSetShiftNonzero(pc,(PetscReal)0);CHKERRQ(ierr);
    ierr = PCFactorSetShiftPd(pc,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PCFactorSetShiftInBlocks(pc,(PetscReal)0);CHKERRQ(ierr);
  } else if (shifttype == MAT_SHIFT_NONZERO) {
    ierr = PCFactorSetShiftNonzero(pc,(PetscReal)PETSC_DECIDE);CHKERRQ(ierr);
  } else if (shifttype == MAT_SHIFT_POSITIVE_DEFINITE) {
    ierr = PCFactorSetShiftPd(pc,PETSC_TRUE);CHKERRQ(ierr);
  } else if (shifttype == MAT_SHIFT_INBLOCKS) {
    ierr = PCFactorSetShiftInBlocks(pc,(PetscReal)PETSC_DECIDE);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"unknown value for shift type");
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PCFactorSetShiftType"
PetscErrorCode PCFactorSetShiftAmount(PC pc,PetscReal shiftamount)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PCFactorSetShiftNonzero(pc,shiftamount);CHKERRQ(ierr);
  ierr = PCFactorSetShiftInBlocks(pc,shiftamount);CHKERRQ(ierr);
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#endif

#endif /* _COMPAT_PETSC_PC_H */
