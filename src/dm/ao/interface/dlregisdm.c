#define PETSCDM_DLL

#include "../src/dm/ao/aoimpl.h"
#include "private/daimpl.h"
#ifdef PETSC_HAVE_SIEVE
#include "private/meshimpl.h"
#endif

static PetscBool  AOPackageInitialized = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "AOFinalizePackage"
/*@C
  AOFinalizePackage - This function finalizes everything in the AO package. It is called
  from PetscFinalize().

  Level: developer

.keywords: AO, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AOFinalizePackage(void)
{
  PetscFunctionBegin;
  AOPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOInitializePackage"
/*@C
  AOInitializePackage - This function initializes everything in the AO package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to AOCreate().

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: AO, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AOInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (AOPackageInitialized) PetscFunctionReturn(0);
  AOPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Application Order",&AO_CLASSID);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("AOPetscToApplication", AO_CLASSID,&AO_PetscToApplication);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AOApplicationToPetsc", AO_CLASSID,&AO_ApplicationToPetsc);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ao", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(AO_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ao", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(AO_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(AOFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscBool  DMPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "DMFinalizePackage"
/*@C
  DMFinalizePackage - This function finalizes everything in the DM package. It is called
  from PetscFinalize().

  Level: developer

.keywords: AO, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMFinalizePackage(void)
{
#ifdef PETSC_HAVE_SIEVE
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  DMPackageInitialized = PETSC_FALSE;
  DMList               = PETSC_NULL;
  DMRegisterAllCalled  = PETSC_FALSE;
#ifdef PETSC_HAVE_SIEVE
  ierr = MeshFinalize();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HYPRE)
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MatCreate_HYPREStruct(Mat);
EXTERN_C_END
#endif

#undef __FUNCT__  
#define __FUNCT__ "DMInitializePackage"
/*@C
  DMInitializePackage - This function initializes everything in the DM package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to AOCreate()
  or DMDACreate() when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: AO, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (DMPackageInitialized) PetscFunctionReturn(0);
  DMPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Distributed array",&DM_CLASSID);CHKERRQ(ierr);
#ifdef PETSC_HAVE_SIEVE
  ierr = PetscClassIdRegister("Mesh",&MESH_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("SectionReal",&SECTIONREAL_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("SectionInt",&SECTIONINT_CLASSID);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_HYPRE)
  ierr = MatRegisterDynamic(MATHYPRESTRUCT,    path,"MatCreate_HYPREStruct", MatCreate_HYPREStruct);CHKERRQ(ierr);
#endif

  /* Register Constructors */
  ierr = DMRegisterAll(path);CHKERRQ(ierr);
#ifdef PETSC_HAVE_SIEVE
  ierr = MeshRegisterAll(path);CHKERRQ(ierr);
#endif
  /* Register Events */
  ierr = PetscLogEventRegister("DMGlobalToLocal",      DM_CLASSID,&DMDA_GlobalToLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMLocalToGlobal",      DM_CLASSID,&DMDA_LocalToGlobal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DMDALocalADFunc",        DM_CLASSID,&DMDA_LocalADFunction);CHKERRQ(ierr);
#ifdef PETSC_HAVE_SIEVE
  ierr = PetscLogEventRegister("MeshView",             MESH_CLASSID,&Mesh_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshGetGlobalScatter", MESH_CLASSID,&Mesh_GetGlobalScatter);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshRestrictVector",   MESH_CLASSID,&Mesh_restrictVector);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshAssembleVector",   MESH_CLASSID,&Mesh_assembleVector);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshAssemVecComplete", MESH_CLASSID,&Mesh_assembleVectorComplete);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshAssembleMatrix",   MESH_CLASSID,&Mesh_assembleMatrix);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshUpdateOperator",   MESH_CLASSID,&Mesh_updateOperator);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SectionRealView",      SECTIONREAL_CLASSID,&SectionReal_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SectionIntView",       SECTIONINT_CLASSID,&SectionInt_View);CHKERRQ(ierr);
#endif
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "da", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(DM_CLASSID);CHKERRQ(ierr);
    }
#ifdef PETSC_HAVE_SIEVE
    ierr = PetscStrstr(logList, "mesh", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(MESH_CLASSID);CHKERRQ(ierr);
    }
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
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "da", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(DM_CLASSID);CHKERRQ(ierr);
    }
#ifdef PETSC_HAVE_SIEVE
    ierr = PetscStrstr(logList, "mesh", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MESH_CLASSID);CHKERRQ(ierr);
    }
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



#ifdef PETSC_USE_DYNAMIC_LIBRARIES
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
PetscErrorCode PETSCDM_DLLEXPORT PetscDLLibraryRegister_petscdm(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AOInitializePackage(path);CHKERRQ(ierr);
  ierr = DMInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
