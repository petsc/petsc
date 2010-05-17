#ifndef _COMPAT_PETSC_PC_H
#define _COMPAT_PETSC_PC_H

#include "private/pcimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
/* PCFieldSplitSetFields */
#undef __FUNCT__
#define __FUNCT__ "PCFieldSplitSetFields"
static PetscErrorCode
PCFieldSplitSetFields_Compat(PC pc,const char splitname[],
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
static PetscErrorCode
PCFieldSplitSetIS_Compat(PC pc,const char splitname[],IS is)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCFieldSplitSetIS(pc,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PCFieldSplitSetIS PCFieldSplitSetIS_Compat
#endif

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define PCLSC          "lsc"
#define PCPFMG         "pfmg"
#define PCSYSPFMG      "syspfmg"
#define PCREDISTRIBUTE "redistribute"
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define PCEXOTIC       "exotic"
#define PCSUPPORTGRAPH "supportgraph"
#define PCASA	       "asa"
#define PCCP	       "cp"
#define PCBFBT         "bfbt"
#endif

#if (PETSC_VERSION_(2,3,2))
#define PCOPENMP       "openmp"
#endif

#endif /* _COMPAT_PETSC_PC_H */
