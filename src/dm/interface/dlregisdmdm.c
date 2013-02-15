
#include <petsc-private/daimpl.h>
#include <petsc-private/pleximpl.h>
#if defined(PETSC_HAVE_SIEVE)
#include <petsc-private/meshimpl.h>
#endif

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
#if defined(PETSC_HAVE_SIEVE)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  DMPackageInitialized = PETSC_FALSE;
  DMList               = NULL;
  DMRegisterAllCalled  = PETSC_FALSE;
#if defined(PETSC_HAVE_SIEVE)
  ierr = DMMeshFinalize();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HYPRE)
EXTERN_C_BEGIN
extern PetscErrorCode  MatCreate_HYPREStruct(Mat);
EXTERN_C_END
#endif

#undef __FUNCT__
#define __FUNCT__ "DMInitializePackage"
/*@C
  DMInitializePackage - This function initializes everything in the DM package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to AOCreate()
  or DMDACreate() when using static libraries.

  Input Parameter:
  path - The dynamic library path, or NULL

  Level: developer

.keywords: AO, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  DMInitializePackage(const char path[])
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
#if defined(PETSC_HAVE_SIEVE)
  ierr = PetscClassIdRegister("SectionReal",&SECTIONREAL_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("SectionInt",&SECTIONINT_CLASSID);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_HYPRE)
  ierr = MatRegisterDynamic(MATHYPRESTRUCT,    path,"MatCreate_HYPREStruct", MatCreate_HYPREStruct);CHKERRQ(ierr);
#endif

  /* Register Constructors */
  ierr = DMRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("DMConvert",              DM_CLASSID,&DM_Convert);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMGlobalToLocal",        DM_CLASSID,&DM_GlobalToLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMLocalToGlobal",        DM_CLASSID,&DM_LocalToGlobal);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("DMDALocalADFunc",        DM_CLASSID,&DMDA_LocalADFunction);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("DMPlexDistribute",    DM_CLASSID,&DMPLEX_Distribute);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMPlexStratify",      DM_CLASSID,&DMPLEX_Stratify);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SIEVE)
  ierr = PetscLogEventRegister("DMMeshView",             DM_CLASSID,&DMMesh_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMMeshGetGlobalScatter", DM_CLASSID,&DMMesh_GetGlobalScatter);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMMeshRestrictVector",   DM_CLASSID,&DMMesh_restrictVector);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMMeshAssembleVector",   DM_CLASSID,&DMMesh_assembleVector);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMMeshAssemVecComplete", DM_CLASSID,&DMMesh_assembleVectorComplete);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMMeshAssembleMatrix",   DM_CLASSID,&DMMesh_assembleMatrix);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMMeshUpdateOperator",   DM_CLASSID,&DMMesh_updateOperator);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SectionRealView",        SECTIONREAL_CLASSID,&SectionReal_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SectionIntView",         SECTIONINT_CLASSID,&SectionInt_View);CHKERRQ(ierr);
#endif
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "da", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(DM_CLASSID);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_SIEVE)
    ierr = PetscStrstr(logList, "sectionreal", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SECTIONREAL_CLASSID);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "sectionint", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SECTIONINT_CLASSID);CHKERRQ(ierr);
    }
#endif
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "da", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(DM_CLASSID);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_SIEVE)
    ierr = PetscStrstr(logList, "sectionreal", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SECTIONREAL_CLASSID);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "sectionint", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SECTIONINT_CLASSID);CHKERRQ(ierr);
    }
#endif
  }
  ierr = PetscRegisterFinalize(DMFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_petscdm"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the mesh generators and partitioners that are in
  the basic DM library.

  Input Parameter:
  path - library path
*/
PetscErrorCode  PetscDLLibraryRegister_petscdm(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AOInitializePackage(path);CHKERRQ(ierr);
  ierr = DMInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
