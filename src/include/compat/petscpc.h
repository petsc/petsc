#ifndef _COMPAT_PETSC_PC_H
#define _COMPAT_PETSC_PC_H

#include "private/pcimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define PCSACUSP "sacusp"
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

#endif /* _COMPAT_PETSC_PC_H */
