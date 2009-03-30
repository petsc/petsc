#ifndef _PETSC_COMPAT_H
#define _PETSC_COMPAT_H

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsHasName_300"
static PETSC_UNUSED
PetscErrorCode PetscOptionsHasName_300(const char pre[],const char name[],PetscTruth *flg)
{
  char dummy[2] = { 0, 0 };
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsGetString(pre,name,dummy,1,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscOptionsHasName PetscOptionsHasName_300

#endif /* _PETSC_COMPAT_H */
