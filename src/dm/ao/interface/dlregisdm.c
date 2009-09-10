#define PETSCDM_DLL

#include "../src/dm/ao/aoimpl.h"
#include "private/daimpl.h"
#ifdef PETSC_HAVE_SIEVE
#include "private/meshimpl.h"
#endif

static PetscTruth DMPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "DMFinalizePackage"
/*@C
  DMFinalizePackage - This function finalizes everything in the DM package. It is called
  from PetscFinalize().

  Level: developer

.keywords: AO, DA, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMFinalizePackage(void) {
#ifdef PETSC_HAVE_SIEVE
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  DMPackageInitialized = PETSC_FALSE;
#ifdef PETSC_HAVE_SIEVE
  ierr = MeshFinalize();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HYPRE)
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_HYPREStruct(Mat);
EXTERN_C_END
#endif

#undef __FUNCT__  
#define __FUNCT__ "DMInitializePackage"
/*@C
  DMInitializePackage - This function initializes everything in the DM package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to AOCreate()
  or DACreate() when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: AO, DA, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (DMPackageInitialized) PetscFunctionReturn(0);
  DMPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Application Order",&AO_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Distributed array",&DM_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Arbitrary Dimension Distributed array",&ADDA_COOKIE);CHKERRQ(ierr);
#ifdef PETSC_HAVE_SIEVE
  ierr = PetscCookieRegister("Mesh",&MESH_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("SectionReal",&SECTIONREAL_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("SectionInt",&SECTIONINT_COOKIE);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_HYPRE)
  ierr = MatRegisterDynamic(MATHYPRESTRUCT,    path,"MatCreate_HYPREStruct", MatCreate_HYPREStruct);CHKERRQ(ierr);
#endif

  /* Register Constructors */
  ierr = DARegisterAll(path);CHKERRQ(ierr);
#ifdef PETSC_HAVE_SIEVE
  ierr = MeshRegisterAll(path);CHKERRQ(ierr);
#endif
  /* Register Events */
  ierr = PetscLogEventRegister("AOPetscToApplication", AO_COOKIE,&AO_PetscToApplication);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AOApplicationToPetsc", AO_COOKIE,&AO_ApplicationToPetsc);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DAGlobalToLocal",      DM_COOKIE,&DA_GlobalToLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DALocalToGlobal",      DM_COOKIE,&DA_LocalToGlobal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("DALocalADFunc",        DM_COOKIE,&DA_LocalADFunction);CHKERRQ(ierr);
#ifdef PETSC_HAVE_SIEVE
  ierr = PetscLogEventRegister("MeshView",             MESH_COOKIE,&Mesh_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshGetGlobalScatter", MESH_COOKIE,&Mesh_GetGlobalScatter);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshRestrictVector",   MESH_COOKIE,&Mesh_restrictVector);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshAssembleVector",   MESH_COOKIE,&Mesh_assembleVector);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshAssemVecComplete", MESH_COOKIE,&Mesh_assembleVectorComplete);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshAssembleMatrix",   MESH_COOKIE,&Mesh_assembleMatrix);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MeshUpdateOperator",   MESH_COOKIE,&Mesh_updateOperator);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SectionRealView",      SECTIONREAL_COOKIE,&SectionReal_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SectionIntView",       SECTIONINT_COOKIE,&SectionInt_View);CHKERRQ(ierr);
#endif
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ao", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(AO_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "da", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(DM_COOKIE);CHKERRQ(ierr);
    }
#ifdef PETSC_HAVE_SIEVE
    ierr = PetscStrstr(logList, "mesh", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(MESH_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "sectionreal", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SECTIONREAL_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "sectionint", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SECTIONINT_COOKIE);CHKERRQ(ierr);
    }
#endif
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ao", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(AO_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "da", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(DM_COOKIE);CHKERRQ(ierr);
    }
#ifdef PETSC_HAVE_SIEVE
    ierr = PetscStrstr(logList, "mesh", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MESH_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "sectionreal", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SECTIONREAL_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "sectionint", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SECTIONINT_COOKIE);CHKERRQ(ierr);
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

  ierr = PetscInitializeNoArguments();
  if (ierr) return(1);

  /*
      If we got here then PETSc was properly loaded
  */
  ierr = DMInitializePackage(path);CHKERRQ(ierr);
  return(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
