/* ------------------------------------------------------------------------- */

static PetscErrorCode PetscInitializePackageAll(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
  ierr = PetscDrawInitializePackage();CHKERRQ(ierr);
  ierr = PetscViewerInitializePackage();CHKERRQ(ierr);
  ierr = PetscRandomInitializePackage();CHKERRQ(ierr);
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = AOInitializePackage();CHKERRQ(ierr);
  ierr = PFInitializePackage();CHKERRQ(ierr);
  ierr = PetscSFInitializePackage();CHKERRQ(ierr);
  ierr = VecInitializePackage();CHKERRQ(ierr);
  ierr = VecScatterInitializePackage();CHKERRQ(ierr);
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PCInitializePackage();CHKERRQ(ierr);
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = SNESInitializePackage();CHKERRQ(ierr);
  ierr = TaoInitializePackage();CHKERRQ(ierr);
  ierr = TSInitializePackage();CHKERRQ(ierr);
  ierr = PetscPartitionerInitializePackage();CHKERRQ(ierr);
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PetscDSInitializePackage();CHKERRQ(ierr);
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------- */

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
