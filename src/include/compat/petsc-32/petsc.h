#undef __FUNCT__  
#define __FUNCT__ "PetscTokenDestroy_Compat"
PetscErrorCode PetscTokenDestroy_Compat(PetscToken *a)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(a,1);
  if (!*a) PetscFunctionReturn(0);
  ierr = PetscTokenDestroy(*a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define PetscTokenDestroy PetscTokenDestroy_Compat
