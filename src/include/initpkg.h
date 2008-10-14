/* ---------------------------------------------------------------- */

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
#define PetscInitializePackage(path)       PetscInitializePackage((char*)path)
#define PetscViewerInitializePackage(path) 0
#define PetscRandomInitializePackage(path) PetscRandomInitializePackage((char*)path)
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
#define VecInitializePackage(path)         VecInitializePackage((char*)path)
#define MatInitializePackage(path)         MatInitializePackage((char*)path)
#endif /* PETSC_232 */

/* ---------------------------------------------------------------- */

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 3 && \
     PETSC_VERSION_RELEASE  == 1)
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
#endif /* PETSC_233 */

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
  ierr = PetscViewerInitializePackage(path);CHKERRQ(ierr);
  ierr = PetscRandomInitializePackage(path);CHKERRQ(ierr);
  ierr = ISInitializePackage(path);CHKERRQ(ierr);
  ierr = VecInitializePackage(path);CHKERRQ(ierr);
  ierr = MatInitializePackage(path);CHKERRQ(ierr);
  ierr = PCInitializePackage(path);CHKERRQ(ierr);
  ierr = KSPInitializePackage(path);CHKERRQ(ierr);
  ierr = SNESInitializePackage(path);CHKERRQ(ierr);
  ierr = TSInitializePackage(path);CHKERRQ(ierr);
  ierr = DMInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
