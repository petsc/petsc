#ifndef _COMPAT_PETSC_DA_H
#define _COMPAT_PETSC_DA_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define DASetOwnershipRanges DASetVertexDivision
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DASetCoordinates"
static PetscErrorCode DASetCoordinates_Compat(DA da,Vec c)
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

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DAGetCoordinates"
static PetscErrorCode DAGetCoordinates_Compat(DA da,Vec *c)
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
static PetscErrorCode DAGetCoordinateDA_Compat(DA da,DA *cda)
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
static PetscErrorCode DAGetGhostedCoordinates_Compat(DA da,Vec *c)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DAGetGhostedCoordinates(da,c);CHKERRQ(ierr);
  if (*c) {ierr = PetscObjectDereference((PetscObject)*c);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
#define DAGetGhostedCoordinates DAGetGhostedCoordinates_Compat
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DASetOptionsPrefix"
static PetscErrorCode DASetOptionsPrefix(DA da,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)da,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if (PETSC_VERSION_(3,0,0))
#undef __FUNCT__
#define __FUNCT__ "DASetFromOptions"
static PetscErrorCode DASetFromOptions(DA da) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscOptionsBegin(((PetscObject)da)->comm,((PetscObject)da)->prefix,"DA Options","DA");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#endif /* _COMPAT_PETSC_DA_H */
