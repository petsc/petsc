#ifndef _COMPAT_PETSC_DM_H
#define _COMPAT_PETSC_DM_H

#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)

#include <petscda.h>

#define DMType char*
#define DMDA        "da"
#define DMADDA      "adda"
#define DMCOMPOSITE "composite"
#define DMSLICED    "sliced"
#define DMMESH      "mesh"
#define DMCARTESIAN "cartesian"
#define DMIGA       "iga"

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
static PetscErrorCode DMSetType(DM dm, const DMType method)
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
static PetscErrorCode DMGetType(DM dm, const DMType *method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(method,2);
  *method = "da";
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetOptionsPrefix"
static PetscErrorCode DMSetOptionsPrefix(DM dm,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)dm,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions"
static PetscErrorCode DMSetFromOptions(DM dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
#if PETSC_VERSION_(3,1,0)
  ierr = DASetFromOptions((DA)dm);CHKERRQ(ierr);
#endif
  ierr = 0;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp"
static PetscErrorCode DMSetUp(DM dm) 
{
#if PETSC_VERSION_(3,1,0)
  PetscInt       dim;
  const DAType   datype = NULL;
  PetscErrorCode ierr;
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
#if PETSC_VERSION_(3,1,0)
  /*ierr = DAGetInfo((DA)dm,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (dim >= 1 && dim <= 3) {
    switch (dim) {
    case 1: datype = DA1D; break;
    case 2: datype = DA2D; break;
    case 3: datype = DA3D; break;}
    ierr = DASetType((DA)dm,datype);CHKERRQ(ierr);
    }*/
  ierr = DASetFromOptions((DA)dm);CHKERRQ(ierr);
  ierr = DAGetInfo((DA)dm,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (dim >= 1 && dim <= 3) {
    switch (dim) {
    case 1: datype = DA1D; break;
    case 2: datype = DA2D; break;
    case 3: datype = DA3D; break;}
    ierr = DASetType((DA)dm,datype);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetBlockSize"
static PetscErrorCode DMGetBlockSize(DM dm, PetscInt *bs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidIntPointer(bs,2);
  ierr = DAGetInfo((DA)dm,0,
		   0,0,0,0,0,0,
		   bs,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetVecType"
static PetscErrorCode DMSetVecType(DM dm, const VecType vec_type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidCharPointer(vec_type,2);
  ierr = PETSC_ERR_SUP;
  SETERRQ(ierr,__FUNCT__"() not supported in this PETSc version");
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


#if PETSC_VERSION_(3,0,0)
#undef __FUNCT__
#define __FUNCT__ "DMRefineHierarchy_Compat"
static PetscErrorCode
DMRefineHierarchy_Compat(DM dm,PetscInt nlevels,DM dmf[])
{
  DM             *dmftmp;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(dmf,2);
  ierr = DMRefineHierarchy(dm,nlevels,&dmftmp);CHKERRQ(ierr);
  ierr = PetscMemcpy(dmf,dmftmp,nlevels*sizeof(DM));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DMRefineHierarchy DMRefineHierarchy_Compat
#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHierarchy_Compat"
static PetscErrorCode
DMCoarsenHierarchy_Compat(DM dm,PetscInt nlevels,DM dmc[])
{
  DM             *dmctmp;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_COOKIE,1);
  PetscValidPointer(dmc,2);
  ierr = DMCoarsenHierarchy(dm,nlevels,&dmctmp);CHKERRQ(ierr);
  ierr = PetscMemcpy(dmc,dmctmp,nlevels*sizeof(DM));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DMCoarsenHierarchy DMCoarsenHierarchy_Compat
#endif


#endif /* _COMPAT_PETSC_DM_H */
