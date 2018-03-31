
#include <petscvec.h>
#include <petsc/private/vecimpl.h>

static PetscBool  VecScatterPackageInitialized = PETSC_FALSE;

/*@C
  VecScatterFinalizePackage - This function destroys everything in the VecScatter package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode VecScatterFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&VecScatterList);CHKERRQ(ierr);
  VecScatterPackageInitialized = PETSC_FALSE;
  VecScatterRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
      VecScatterInitializePackage - This function initializes everything in the VecScatter package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to VecScatterCreateXXXX()
  when using static libraries.

  Level: developer

.keywords: Vec, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode VecScatterInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (VecScatterPackageInitialized) PetscFunctionReturn(0);
  VecScatterPackageInitialized = PETSC_TRUE;
  /* Register Constructors */
  ierr = VecScatterRegisterAll();CHKERRQ(ierr);

  /* Register Classes */
  ierr = PetscClassIdRegister("Vec Scatter",&VEC_SCATTER_CLASSID);CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "vec scatter", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(VEC_SCATTER_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL, "-log_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "vec scatter", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(VEC_SCATTER_CLASSID);CHKERRQ(ierr);
    }
  }

  ierr = PetscRegisterFinalize(VecScatterFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------- */
/*@C
  VecScatterRegisterAll - Registers all of the vector components in the Vec package.

  Not Collective

  Level: advanced

.keywords: Vec, register, all
.seealso:  VecScatterRegister(), VecScatterRegisterDestroy(), VecScatterRegister()
@*/
PetscErrorCode VecScatterRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (VecScatterRegisterAllCalled) PetscFunctionReturn(0);
  VecScatterRegisterAllCalled = PETSC_TRUE;

  ierr = VecScatterRegister(VECSCATTERSEQ,        VecScatterCreate_Seq);CHKERRQ(ierr);
  ierr = VecScatterRegister(VECSCATTERMPI1,       VecScatterCreate_MPI1);CHKERRQ(ierr);
  ierr = VecScatterRegister(VECSCATTERMPI3,       VecScatterCreate_MPI3);CHKERRQ(ierr);
  ierr = VecScatterRegister(VECSCATTERMPI3NODE,   VecScatterCreate_MPI3Node);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
