/* ------------------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "PetscInitializePackageAll"
static PetscErrorCode PetscInitializePackageAll(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
  ierr = PetscViewerInitializePackage();CHKERRQ(ierr);
  ierr = PetscRandomInitializePackage();CHKERRQ(ierr);
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = VecInitializePackage();CHKERRQ(ierr);
  ierr = PFInitializePackage();CHKERRQ(ierr);
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PCInitializePackage();CHKERRQ(ierr);
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = SNESInitializePackage();CHKERRQ(ierr);
  ierr = TSInitializePackage();CHKERRQ(ierr);
  ierr = TaoInitializePackage();CHKERRQ(ierr);
  ierr = AOInitializePackage();CHKERRQ(ierr);
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PetscSFInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  __FUNCT__
#define __FUNCT__ "<petsc4py.PETSc>"

/* ------------------------------------------------------------------------- */

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
