#ifndef _COMPAT_PETSC_DA_H
#define _COMPAT_PETSC_DA_H

#if PETSC_VERSION_(3,0,0)
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

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define DM_COOKIE DA_COOKIE
#define DA_XYZGHOSTED ((DAPeriodicType)-1)
#endif

#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
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

#endif /* _COMPAT_PETSC_DA_H */
