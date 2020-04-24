
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Petsc_Seq_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Seq_keyval);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscSysPackageInitialized) PetscFunctionReturn(0);
  PetscSysPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Object",&PETSC_OBJECT_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Container",&PETSC_CONTAINER_CLASSID);CHKERRQ(ierr);

  /* Register Events */
  ierr = PetscLogEventRegister("PetscBarrier", PETSC_SMALLEST_CLASSID,&PETSC_Barrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BuildTwoSided",PETSC_SMALLEST_CLASSID,&PETSC_BuildTwoSided);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BuildTwoSidedF",PETSC_SMALLEST_CLASSID,&PETSC_BuildTwoSidedF);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSC_SMALLEST_CLASSID;
    ierr = PetscInfoProcessClass("sys", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("null",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSC_SMALLEST_CLASSID);CHKERRQ(ierr);}
  }
  ierr = PetscRegisterFinalize(PetscSysFinalizePackage);CHKERRQ(ierr);
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

#if defined(PETSC_USE_SINGLE_LIBRARY)
#else
#endif
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the draw and PetscViewer objects.

 */
#if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsc(void)
#else
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsys(void)
#endif
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
  ierr = PetscDrawInitializePackage();CHKERRQ(ierr);
  ierr = PetscViewerInitializePackage();CHKERRQ(ierr);
  ierr = PetscRandomInitializePackage();CHKERRQ(ierr);

#if defined(PETSC_USE_SINGLE_LIBRARY)
  ierr = PetscDLLibraryRegister_petscvec();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscmat();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscdm();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscksp();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscsnes();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscts();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
#endif  /* PETSC_HAVE_DYNAMIC_LIBRARIES */
