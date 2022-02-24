/* ------------------------------------------------------------------------- */

static PetscErrorCode PetscInitializePackageAll(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSysInitializePackage());
  CHKERRQ(PetscDrawInitializePackage());
  CHKERRQ(PetscViewerInitializePackage());
  CHKERRQ(PetscRandomInitializePackage());
  CHKERRQ(ISInitializePackage());
  CHKERRQ(AOInitializePackage());
  CHKERRQ(PFInitializePackage());
  CHKERRQ(PetscSFInitializePackage());
  CHKERRQ(VecInitializePackage());
  CHKERRQ(MatInitializePackage());
  CHKERRQ(PCInitializePackage());
  CHKERRQ(KSPInitializePackage());
  CHKERRQ(SNESInitializePackage());
  CHKERRQ(TaoInitializePackage());
  CHKERRQ(TSInitializePackage());
  CHKERRQ(PetscPartitionerInitializePackage());
  CHKERRQ(DMInitializePackage());
  CHKERRQ(PetscDSInitializePackage());
  CHKERRQ(PetscFEInitializePackage());
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------- */

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
