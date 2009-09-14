#ifndef _PETSC_COMPAT_DA_H
#define _PETSC_COMPAT_DA_H

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

#endif /* _PETSC_COMPAT_DA_H */
