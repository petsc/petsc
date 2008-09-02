/* ---------------------------------------------------------------- */

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_ISX(Mat A);
EXTERN_C_END
#endif

/* ---------------------------------------------------------------- */

#define MATPYTHON "python"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Python(Mat);
EXTERN_C_END
PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetContext(Mat,void*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPythonGetContext(Mat,void**);
PETSC_EXTERN_CXX_END

/* ---------------------------------------------------------------- */

#define KSPPYTHON "python"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_Python(KSP);
EXTERN_C_END
PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetContext(KSP,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonGetContext(KSP,void**);
PETSC_EXTERN_CXX_END


/* ---------------------------------------------------------------- */

#define PCPYTHON "python"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Python(PC);
EXTERN_C_END
PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPythonSetContext(PC,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPythonGetContext(PC,void**);
PETSC_EXTERN_CXX_END


#define PCSCHUR "schur"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Schur(PC);
EXTERN_C_END

/* ---------------------------------------------------------------- */

#define SNESPYTHON "python"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_Python(SNES);
EXTERN_C_END
PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESPythonSetContext(SNES,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESPythonGetContext(SNES,void**);
PETSC_EXTERN_CXX_END

/* ---------------------------------------------------------------- */

#define TS_PYTHON "python"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Python(TS);
EXTERN_C_END
PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSPythonSetContext(TS,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSPythonGetContext(TS,void**);
PETSC_EXTERN_CXX_END



#define TS_USER "user"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_User(TS);
EXTERN_C_END


/* ---------------------------------------------------------------- */

#undef  __FUNCT__  
#define __FUNCT__ "PyPetscRegisterAll"
static PetscErrorCode PyPetscRegisterAll(const char path[])
{
  static PetscTruth initialized = PETSC_FALSE;
  PetscErrorCode ierr;

  if (initialized) return 0;
  initialized = PETSC_TRUE;

  PetscFunctionBegin;

#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(path);CHKERRQ(ierr);
  ierr = MatInitializePackage(path);CHKERRQ(ierr);
  ierr = KSPInitializePackage(path);CHKERRQ(ierr);
  ierr = PCInitializePackage(path);CHKERRQ(ierr);
  ierr = SNESInitializePackage(path);CHKERRQ(ierr);
  ierr = TSInitializePackage(path);CHKERRQ(ierr);
#endif

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
  ierr = MatRegisterDynamic(MATIS, path, "MatCreate_ISX", MatCreate_ISX);CHKERRQ(ierr);
#endif

  /* Mat */
  ierr = MatRegisterDynamic(MATPYTHON, path, "MatCreate_Python",  MatCreate_Python);CHKERRQ(ierr);
  /* KSP */
  ierr = KSPRegisterDynamic(KSPPYTHON, path, "KSPCreate_Python", KSPCreate_Python);CHKERRQ(ierr);
  /* PC */
  ierr = PCRegisterDynamic(PCSCHUR,  path, "PCCreate_Schur",  PCCreate_Schur);CHKERRQ(ierr);
  ierr = PCRegisterDynamic(PCPYTHON, path, "PCCreate_Python", PCCreate_Python);CHKERRQ(ierr);
  /* SNES */
  ierr = SNESRegisterDynamic(SNESPYTHON, path, "SNESCreate_Python", SNESCreate_Python);CHKERRQ(ierr);
  /* TS */
  ierr = TSRegisterDynamic(TS_USER,   path, "TSCreate_User",   TSCreate_User);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_PYTHON, path, "TSCreate_Python", TSCreate_Python);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
