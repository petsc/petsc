
#if !defined(__PETSCVERSION_H)
#define __PETSCVERSION_H
PETSC_EXTERN_CXX_BEGIN

/* ========================================================================== */
/* 
   Current PETSc version number and release date, also listed in
    Web page
    src/docs/tex/manual/intro.tex,
    src/docs/tex/manual/manual.tex.
    src/docs/website/index.html.
*/
#define PETSC_VERSION_RELEASE    1
#define PETSC_VERSION_MAJOR      2
#define PETSC_VERSION_MINOR      3
#define PETSC_VERSION_SUBMINOR   2
#define PETSC_VERSION_PATCH      3
#define PETSC_VERSION_DATE       "Septenber, 1, 2006"
#define PETSC_VERSION_PATCH_DATE "unknown"
#define PETSC_VERSION_HG         "unknown"
#define PETSC_AUTHOR_INFO        "\
       The PETSc Team\n\
    petsc-maint@mcs.anl.gov\n\
 http://www.mcs.anl.gov/petsc/\n"

#if (PETSC_VERSION_RELEASE == 1)
#define PetscGetVersion(version,len) (PetscSNPrintf(*(version),len,"Petsc Release Version %d.%d.%d, Patch %d, ", \
                                         PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, \
                                         PETSC_VERSION_PATCH),PetscStrcat(*(version),PETSC_VERSION_PATCH_DATE), \
                                         PetscStrcat(*(version)," HG revision: "),PetscStrcat(*(version),PETSC_VERSION_HG),0)
#else
#define PetscGetVersion(version,len) (PetscSNPrintf(*(version),len,"Petsc Development Version %d.%d.%d, Patch %d, ", \
                                         PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, \
                                         PETSC_VERSION_PATCH),PetscStrcat(*(version),PETSC_VERSION_PATCH_DATE), \
                                         PetscStrcat(*(version)," HG revision: "),PetscStrcat(*(version),PETSC_VERSION_HG),0)
#endif

/*MC
    PetscGetVersion - Gets the Petsc Version information in a string.

    Output Parameter:
.   version - version string

    Input Parameter:
.   len - length of the string

    Level: developer

    Usage:
    char version[256];
    PetscGetVersion(&version,256);

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscGetProgramName()

M*/

PETSC_EXTERN_CXX_END
#endif
