#ifndef _COMPAT_PETSC_DM_H
#define _COMPAT_PETSC_DM_H

#if PETSC_VERSION_(3,1,0) || \
    PETSC_VERSION_(3,0,0)

#include <petscda.h>

#undef __FUNCT__
#define __FUNCT__ "DMCreate"
static PetscErrorCode DMCreate(MPI_Comm comm,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm,2);
#if !PETSC_VERSION_(3,0,0)
  ierr = DACreate(comm,(DA*)dm);CHKERRQ(ierr);
#else
  ierr = PETSC_ERR_SUP;
  SETERRQ(ierr,__FUNCT__"() not supported in this PETSc version");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetType"
static PetscErrorCode DMSetType(DM dm, const char *method)
{
  PetscTruth     flag;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(method,2);
  ierr = PetscStrcmp(method,"da",&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,
                      "Unknown DM type: %s", method);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetType"
static PetscErrorCode DMGetType(DM dm, const char **method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(method,2);
  *method = "da";
  PetscFunctionReturn(0);
}

#if !PETSC_VERSION_(3,0,0)
#define DMSetOptionsPrefix(dm,p) DASetOptionsPrefix((DA)dm,p)
#define DMSetFromOptions(dm)     DASetFromOptions((DA)dm)
#else
#define DMSetOptionsPrefix(dm,p) PetscObjectSetOptionsPrefix((PetscObject)dm,p)
#define DMSetFromOptions(dm)     PetscObjectSetFromOptions((PetscObject)dm)
#endif

#undef __FUNCT__
#define __FUNCT__ "DMSetUp"
static PetscErrorCode DMSetUp(DM dm) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
#if !PETSC_VERSION_(3,0,0)
  ierr = DASetFromOptions((DA)dm);CHKERRQ(ierr);
#else
  ierr = 0;CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#define DMGetLocalToGlobalMapping(dm,l2g) \
        DAGetLocalToGlobalMapping((DA)dm,l2g)
#define DMGetLocalToGlobalMappingBlock(dm,l2g) \
        DAGetLocalToGlobalMappingBlock((DA)dm,l2g)

#undef __FUNCT__  
#define __FUNCT__ "DMLocalToGlobalBegin"
static PetscErrorCode DMLocalToGlobalBegin(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  VecScatter     ltog, gtol;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidHeaderSpecific(l,VEC_COOKIE,2);
  PetscValidHeaderSpecific(g,VEC_COOKIE,3);
  ierr = DAGetScatter((DA)dm,&ltog,&gtol,0);CHKERRQ(ierr);
  if (mode == ADD_VALUES) {
    ierr = VecScatterBegin(gtol,l,g,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (mode == INSERT_VALUES) {
    ierr = VecScatterBegin(ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"Not yet implemented");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMLocalToGlobalEnd"
static PetscErrorCode DMLocalToGlobalEnd(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  VecScatter     ltog, gtol;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidHeaderSpecific(l,VEC_COOKIE,2);
  PetscValidHeaderSpecific(g,VEC_COOKIE,3);
  ierr = DAGetScatter((DA)dm,&ltog,&gtol,0);CHKERRQ(ierr);
  if (mode == ADD_VALUES) {
    ierr = VecScatterEnd(gtol,l,g,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (mode == INSERT_VALUES) {
    ierr = VecScatterEnd(ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"Not yet implemented");
  }
  PetscFunctionReturn(0);
}

#endif

#endif /* _COMPAT_PETSC_DM_H */
