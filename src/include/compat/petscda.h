#ifndef _COMPAT_PETSC_DA_H
#define _COMPAT_PETSC_DA_H

#if PETSC_VERSION_(3,0,0)
#undef __FUNCT__
#define __FUNCT__ "DASetCoordinates"
static PETSC_UNUSED
PetscErrorCode DASetCoordinates_Compat(DA da,Vec c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidHeaderSpecific(c,VEC_COOKIE,2);
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  ierr = DASetCoordinates(da,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DASetCoordinates DASetCoordinates_Compat
#endif

#if (PETSC_VERSION_(3,1,0) ||                    \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DAGetCoordinates"
static PETSC_UNUSED
PetscErrorCode DAGetCoordinates_Compat(DA da,Vec *c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetCoordinates(da,c);CHKERRQ(ierr);
  if (*c) {ierr = PetscObjectDereference((PetscObject)*c);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DAGetCoordinates DAGetCoordinates_Compat
#undef __FUNCT__
#define __FUNCT__ "DAGetCoordinateDA"
static PETSC_UNUSED
PetscErrorCode DAGetCoordinateDA_Compat(DA da,DA *cda)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetCoordinateDA(da,cda);CHKERRQ(ierr);
  if (*cda) {ierr = PetscObjectDereference((PetscObject)*cda);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DAGetCoordinateDA DAGetCoordinateDA_Compat
#undef __FUNCT__
#define __FUNCT__ "DAGetGhostedCoordinates"
static PETSC_UNUSED
PetscErrorCode DAGetGhostedCoordinates_Compat(DA da,Vec *c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetGhostedCoordinates(da,c);CHKERRQ(ierr);
  if (*c) {ierr = PetscObjectDereference((PetscObject)*c);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DAGetGhostedCoordinates DAGetGhostedCoordinates_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define DM_COOKIE DA_COOKIE
#define DA_XYZGHOSTED ((DAPeriodicType)-1)
#endif

#if PETSC_VERSION_(2,3,3)
#undef __FUNCT__
#define __FUNCT__ "DASetCoordinates"
static PETSC_UNUSED
PetscErrorCode DASetCoordinates_Compat(DA da,Vec c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE,1);
  PetscValidHeaderSpecific(c,VEC_COOKIE,2);
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  ierr = DASetCoordinates(da,c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define DASetCoordinates DASetCoordinates_Compat
#endif

#if PETSC_VERSION_(2,3,2)
#undef __FUNCT__
#define __FUNCT__ "DASetCoordinates"
static PETSC_UNUSED
PetscErrorCode DASetCoordinates_Compat(DA da,Vec c)
{
  Vec            cold;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE,1);
  PetscValidHeaderSpecific(c,VEC_COOKIE,2);
  ierr = DAGetCoordinates(da,&cold);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  ierr = DASetCoordinates(da,c);CHKERRQ(ierr);
  if (cold) {ierr=VecDestroy(cold);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DASetCoordinates DASetCoordinates_Compat
#undef __FUNCT__
#define __FUNCT__ "DASetUniformCoordinates"
static PETSC_UNUSED
PetscErrorCode DASetUniformCoordinates_Compat(DA da,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,PetscReal zmin,PetscReal zmax)
{
  Vec            cold;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE,1);
  ierr = DAGetCoordinates(da,&cold);CHKERRQ(ierr);
  ierr = DASetUniformCoordinates(da,xmin,xmax,ymin,ymax,zmin,zmax);CHKERRQ(ierr);
  if (cold) {ierr=VecDestroy(cold);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DASetUniformCoordinates DASetUniformCoordinates_Compat
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "DACreate"
static PETSC_UNUSED
PetscErrorCode DACreate_Compat(MPI_Comm comm,PetscInt dim,DAPeriodicType wrap,DAStencilType stencil_type,
			       PetscInt M, PetscInt N,PetscInt P,PetscInt m,PetscInt n,PetscInt p,
			       PetscInt dof,PetscInt sw,
			       const PetscInt lx[],const PetscInt ly[],const PetscInt lz[],DA *da)
{
  return DACreate(comm,dim,wrap,stencil_type,M,N,P,m,n,p,dof,sw,
		  (PetscInt*)lx,(PetscInt*)ly,(PetscInt*)lz,da);
}
#define DACreate DACreate_Compat
#endif

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "DASetOptionsPrefix"
static PETSC_UNUSED
PetscErrorCode DASetOptionsPrefix(DA da,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)da,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "DASetFromOptions"
static PETSC_UNUSED
PetscErrorCode DASetFromOptions(DA da) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DA Options","DA");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#undef __FUNCT__
#define __FUNCT__ "DACoarsen"
static PETSC_UNUSED
PetscErrorCode DACoarsen(DA da, MPI_Comm comm,DA *daref)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(daref,3);
  SETERRQ(PETSC_ERR_SUP,__FUNCT__" not available in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#endif

#endif /* _COMPAT_PETSC_DA_H */
