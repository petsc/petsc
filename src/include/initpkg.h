/* ------------------------------------------------------------------------- */

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define PetscSysInitializePackage PetscInitializePackage
#endif

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#undef  __FUNCT__
#define __FUNCT__ "PetscFwkInitializePackage"
static PetscErrorCode PetscFwkInitializePackage(const char path[])
{
  static PetscTruth initialized = PETSC_FALSE;
  PetscErrorCode ierr;
  if (initialized) return 0;
  initialized = PETSC_TRUE;
  PetscFunctionBegin;
  ierr = PetscCookieRegister("PetscFwk",&PETSC_FWK_COOKIE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/* ------------------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "PetscInitializeAllPackages"
static PetscErrorCode PetscInitializeAllPackages(const char path[])
{
  PetscErrorCode ierr;
  static PetscTruth initialized = PETSC_FALSE;
  if (initialized) return ierr=0;
  initialized = PETSC_TRUE;
  PetscFunctionBegin;
  ierr = PetscSysInitializePackage(path);CHKERRQ(ierr);
  ierr = PetscFwkInitializePackage(path);CHKERRQ(ierr);
  ierr = PetscDrawInitializePackage(path);CHKERRQ(ierr);
  ierr = PetscViewerInitializePackage(path);CHKERRQ(ierr);
  ierr = PetscRandomInitializePackage(path);CHKERRQ(ierr);
  ierr = ISInitializePackage(path);CHKERRQ(ierr);
  ierr = VecInitializePackage(path);CHKERRQ(ierr);
  ierr = PFInitializePackage(path);CHKERRQ(ierr);
  ierr = MatInitializePackage(path);CHKERRQ(ierr);
  ierr = PCInitializePackage(path);CHKERRQ(ierr);
  ierr = KSPInitializePackage(path);CHKERRQ(ierr);
  ierr = SNESInitializePackage(path);CHKERRQ(ierr);
  ierr = TSInitializePackage(path);CHKERRQ(ierr);
  ierr = DMInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------- */

/*
  Local variables:
  c-basic-offset: 2
  indent-tabs-mode: nil
  End:
*/
