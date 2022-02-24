
#include <petscdraw.h>
#include <petscviewer.h>
#include <petsc/private/viewerimpl.h>

static PetscBool PetscSysPackageInitialized = PETSC_FALSE;

/*@C
  PetscSysFinalizePackage - This function destroys everything in the PETSc created internally in the system library portion of PETSc.
  It is called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscSysFinalizePackage(void)
{
  PetscFunctionBegin;
  if (Petsc_Seq_keyval != MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Seq_keyval));
  }
  PetscSysPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscSysInitializePackage - This function initializes everything in the main Petsc package. It is called
  from PetscDLLibraryRegister_petsc() when using dynamic libraries, and on the call to PetscInitialize()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  PetscSysInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscSysPackageInitialized) PetscFunctionReturn(0);
  PetscSysPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Object",&PETSC_OBJECT_CLASSID));
  CHKERRQ(PetscClassIdRegister("Container",&PETSC_CONTAINER_CLASSID));

  /* Register Events */
  CHKERRQ(PetscLogEventRegister("PetscBarrier", PETSC_SMALLEST_CLASSID,&PETSC_Barrier));
  CHKERRQ(PetscLogEventRegister("BuildTwoSided",PETSC_SMALLEST_CLASSID,&PETSC_BuildTwoSided));
  CHKERRQ(PetscLogEventRegister("BuildTwoSidedF",PETSC_SMALLEST_CLASSID,&PETSC_BuildTwoSidedF));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSC_SMALLEST_CLASSID;
    CHKERRQ(PetscInfoProcessClass("sys", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("null",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSC_SMALLEST_CLASSID));
  }
  CHKERRQ(PetscRegisterFinalize(PetscSysFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscvec(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscmat(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscdm(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscksp(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsnes(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscts(void);
#endif

/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the system level objects.

 */
#if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsc(void)
#else
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsys(void)
#endif
{
  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  CHKERRQ(PetscSysInitializePackage());
  CHKERRQ(PetscDrawInitializePackage());
  CHKERRQ(PetscViewerInitializePackage());
  CHKERRQ(PetscRandomInitializePackage());

#if defined(PETSC_USE_SINGLE_LIBRARY)
  CHKERRQ(PetscDLLibraryRegister_petscvec());
  CHKERRQ(PetscDLLibraryRegister_petscmat());
  CHKERRQ(PetscDLLibraryRegister_petscdm());
  CHKERRQ(PetscDLLibraryRegister_petscksp());
  CHKERRQ(PetscDLLibraryRegister_petscsnes());
  CHKERRQ(PetscDLLibraryRegister_petscts());
#endif
  PetscFunctionReturn(0);
}
#endif  /* PETSC_HAVE_DYNAMIC_LIBRARIES */
