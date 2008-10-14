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



/* ---------------------------------------------------------------- */

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_ISX(Mat A);
EXTERN_C_END
#endif /* PETSC_232 */

#define PCSCHUR "schur"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Schur(PC);
EXTERN_C_END

#define TS_USER "user"

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_User(TS);
EXTERN_C_END


/* ---------------------------------------------------------------- */

#undef  __FUNCT__  
#define __FUNCT__ "PetscPythonRegisterAll"
static PetscErrorCode PetscPythonRegisterAll(const char path[])
{
  static PetscTruth registered = PETSC_FALSE;
  PetscErrorCode ierr;
  if (registered) return 0;
  registered = PETSC_TRUE;

  PetscFunctionBegin;
  
#if 0  

  /* XXX This should be the right way ... !!! */

  ierr = MatRegisterDynamic  ( MATPYTHON,  path, "MatCreate_Python",  MatCreate_Python  ); CHKERRQ(ierr);
  ierr = PCRegisterDynamic   ( PCPYTHON,   path, "PCCreate_Python",   PCCreate_Python   ); CHKERRQ(ierr);
  ierr = KSPRegisterDynamic  ( KSPPYTHON,  path, "KSPCreate_Python",  KSPCreate_Python  ); CHKERRQ(ierr);
  ierr = SNESRegisterDynamic ( SNESPYTHON, path, "SNESCreate_Python", SNESCreate_Python ); CHKERRQ(ierr);
  ierr = TSRegisterDynamic   ( TS_PYTHON,  path, "TSCreate_Python",   TSCreate_Python   ); CHKERRQ(ierr);
#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
  ierr = MatRegisterDynamic ( MATIS,   path, "MatCreate_ISX",  MatCreate_ISX  ); CHKERRQ(ierr);
#endif
  ierr = PCRegisterDynamic  ( PCSCHUR, path, "PCCreate_Schur", PCCreate_Schur ); CHKERRQ(ierr);
  ierr = TSRegisterDynamic  ( TS_USER, path, "TSCreate_User",  TSCreate_User  ); CHKERRQ(ierr);

#else

  /* XXX But up to now, this is the way it works !!! */

  ierr = MatRegister  ( MATPYTHON,  PETSC_NULL, "MatCreate_Python",  MatCreate_Python  ); CHKERRQ(ierr);
  ierr = PCRegister   ( PCPYTHON,   PETSC_NULL, "PCCreate_Python",   PCCreate_Python   ); CHKERRQ(ierr);
  ierr = KSPRegister  ( KSPPYTHON,  PETSC_NULL, "KSPCreate_Python",  KSPCreate_Python  ); CHKERRQ(ierr);
  ierr = SNESRegister ( SNESPYTHON, PETSC_NULL, "SNESCreate_Python", SNESCreate_Python ); CHKERRQ(ierr);
  ierr = TSRegister   ( TS_PYTHON,  PETSC_NULL, "TSCreate_Python",   TSCreate_Python   ); CHKERRQ(ierr);
#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
  ierr = MatRegister ( MATIS,   PETSC_NULL, "MatCreate_ISX",  MatCreate_ISX  ); CHKERRQ(ierr);
#endif
  ierr = PCRegister  ( PCSCHUR, PETSC_NULL, "PCCreate_Schur", PCCreate_Schur ); CHKERRQ(ierr);
  ierr = TSRegister  ( TS_USER, PETSC_NULL, "TSCreate_User",  TSCreate_User  ); CHKERRQ(ierr);

#endif

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
