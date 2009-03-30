/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(2,3,2)
#undef  __FUNCT__
#define __FUNCT__ "ISInitializePackage"
static PetscErrorCode ISInitializePackage(char path[])
{
  static PetscTruth initialized = PETSC_FALSE;
  PetscErrorCode ierr;
  if (initialized) return 0;
  initialized = PETSC_TRUE;
  PetscFunctionBegin;
  if (IS_LTOGM_COOKIE == -1) {
    ierr = PetscLogClassRegister(&IS_LTOGM_COOKIE,"IS L to G Mapping");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#define PetscInitializePackage(path)       PetscInitializePackage((char*)path)
#define PetscDrawInitializePackage(path)   0
#define PetscViewerInitializePackage(path) 0
#define PetscRandomInitializePackage(path) PetscRandomInitializePackage((char*)path)
#define ISInitializePackage(path)          ISInitializePackage((char*)path)
#define VecInitializePackage(path)         VecInitializePackage((char*)path)
#define PFInitializePackage(path)          0
#define MatInitializePackage(path)         MatInitializePackage((char*)path)
#endif /* PETSC_VERSION_(2,3,2) */

/* ---------------------------------------------------------------- */

#if PETSC_VERSION_(2,3,3)
#undef  __FUNCT__
#define __FUNCT__ "ISInitializePackage"
static PetscErrorCode ISInitializePackage(const char path[])
{
  static PetscTruth initialized = PETSC_FALSE;
  PetscErrorCode ierr;
  if (initialized) return 0;
  initialized = PETSC_TRUE;
  PetscFunctionBegin;
  if (IS_LTOGM_COOKIE == -1) {
    ierr = PetscLogClassRegister(&IS_LTOGM_COOKIE,"IS L to G Mapping");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif /* PETSC_VERSION_(2,3,3) */

/* ---------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "PetscInitializeAllPackages"
static PetscErrorCode PetscInitializeAllPackages(const char path[])
{
  PetscErrorCode ierr;
  static PetscTruth initialized = PETSC_FALSE;
  if (initialized) return ierr=0;
  initialized = PETSC_TRUE;
  PetscFunctionBegin;
  ierr = PetscInitializePackage(path);CHKERRQ(ierr);
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

/* ---------------------------------------------------------------- */
