
#include <petsc-private/dmdaimpl.h>
#include <petsc-private/dmpleximpl.h>

static PetscBool DMPackageInitialized = PETSC_FALSE;
#undef __FUNCT__
#define __FUNCT__ "DMFinalizePackage"
/*@C
  DMFinalizePackage - This function finalizes everything in the DM package. It is called
  from PetscFinalize().

  Level: developer

.keywords: AO, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  DMFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&DMList);CHKERRQ(ierr);
  DMPackageInitialized = PETSC_FALSE;
  DMRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HYPRE)
PETSC_EXTERN PetscErrorCode MatCreate_HYPREStruct(Mat);
#endif

#undef __FUNCT__
#define __FUNCT__ "DMInitializePackage"
/*@C
  DMInitializePackage - This function initializes everything in the DM package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to AOCreate()
  or DMDACreate() when using static libraries.

  Level: developer

.keywords: AO, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  DMInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DMPackageInitialized) PetscFunctionReturn(0);
  DMPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  ierr = PetscClassIdRegister("Distributed Mesh",&DM_CLASSID);CHKERRQ(ierr);

#if defined(PETSC_HAVE_HYPRE)
  ierr = MatRegister(MATHYPRESTRUCT, MatCreate_HYPREStruct);CHKERRQ(ierr);
#endif

  /* Register Constructors */
  ierr = DMRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("DMConvert",              DM_CLASSID,&DM_Convert);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMGlobalToLocal",        DM_CLASSID,&DM_GlobalToLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMLocalToGlobal",        DM_CLASSID,&DM_LocalToGlobal);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("DMDALocalADFunc",        DM_CLASSID,&DMDA_LocalADFunction);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("DMPlexDistribute",    DM_CLASSID,&DMPLEX_Distribute);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexStratify",      DM_CLASSID,&DMPLEX_Stratify);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "da", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(DM_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "da", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(DM_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(DMFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_petscdm"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the mesh generators and partitioners that are in
  the basic DM library.

*/
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscdm(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AOInitializePackage();CHKERRQ(ierr);
  ierr = DMInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
