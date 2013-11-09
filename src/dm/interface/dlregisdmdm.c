
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

  ierr = PetscLogEventRegister("DMPlexInterpolate",      DM_CLASSID,&DMPLEX_Interpolate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexPartition",        DM_CLASSID,&DMPLEX_Partition);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistribute",       DM_CLASSID,&DMPLEX_Distribute);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistributeLabels", DM_CLASSID,&DMPLEX_DistributeLabels);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexDistributeSF",     DM_CLASSID,&DMPLEX_DistributeSF);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexStratify",         DM_CLASSID,&DMPLEX_Stratify);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexPreallocate",      DM_CLASSID,&DMPLEX_Preallocate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexResidualFEM",      DM_CLASSID,&DMPLEX_ResidualFEM);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexJacobianFEM",      DM_CLASSID,&DMPLEX_JacobianFEM);CHKERRQ(ierr);
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
#include <petscfe.h>

static PetscBool PetscFEPackageInitialized = PETSC_FALSE;
#undef __FUNCT__
#define __FUNCT__ "PetscFEFinalizePackage"
/*@C
  PetscFEFinalizePackage - This function finalizes everything in the PetscFE package. It is called
  from PetscFinalize().

  Level: developer

.keywords: PetscFE, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PetscFEFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscSpaceList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscDualSpaceList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscFEList);CHKERRQ(ierr);
  PetscFEPackageInitialized       = PETSC_FALSE;
  PetscSpaceRegisterAllCalled     = PETSC_FALSE;
  PetscDualSpaceRegisterAllCalled = PETSC_FALSE;
  PetscFERegisterAllCalled        = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEInitializePackage"
/*@C
  PetscFEInitializePackage - This function initializes everything in the FE package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PetscSpaceCreate()
  when using static libraries.

  Level: developer

.keywords: PetscFE, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PetscFEInitializePackage(void)
{
  char           logList[256];
  char          *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscFEPackageInitialized) PetscFunctionReturn(0);
  PetscFEPackageInitialized = PETSC_TRUE;

  /* Register Classes */
  ierr = PetscClassIdRegister("Linear Space", &PETSCSPACE_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Dual Space",   &PETSCDUALSPACE_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("FE Space",     &PETSCFE_CLASSID);CHKERRQ(ierr);

  /* Register Constructors */
  ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);
  ierr = PetscDualSpaceRegisterAll();CHKERRQ(ierr);
  ierr = PetscFERegisterAll();CHKERRQ(ierr);
  /* Register Events */
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "fe", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(PETSCFE_CLASSID);CHKERRQ(ierr);
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
  ierr = PetscRegisterFinalize(PetscFEFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
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
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
