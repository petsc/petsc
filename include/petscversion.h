
#if !defined(__PETSCVERSION_H)
#define __PETSCVERSION_H
PETSC_EXTERN_CXX_BEGIN

/* ========================================================================== */
/* 
   Current PETSc version number and release date, also listed in
    Web page
    docs/tex/manual/intro.tex,
    docs/tex/manual/manual.tex.
    docs/website/index.html.
*/
#define PETSC_VERSION_RELEASE    1
#define PETSC_VERSION_MAJOR      2
#define PETSC_VERSION_MINOR      3
#define PETSC_VERSION_SUBMINOR   1
#define PETSC_VERSION_PATCH      10
#define PETSC_VERSION_DATE       "February, 3, 2006"
#define PETSC_VERSION_PATCH_DATE "unknown"
#define PETSC_VERSION_BK         "unknown"
#define PETSC_AUTHOR_INFO        "\
       The PETSc Team\n\
    petsc-maint@mcs.anl.gov\n\
 http://www.mcs.anl.gov/petsc/\n"

#if (PETSC_VERSION_RELEASE == 1)
#define PetscGetVersion(version) (sprintf(*(version),"Petsc Release Version %d.%d.%d, Patch %d, ", \
                                         PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, \
                                         PETSC_VERSION_PATCH),PetscStrcat(*(version),PETSC_VERSION_PATCH_DATE), \
                                         PetscStrcat(*(version),"\nBK revision: "),PetscStrcat(*(version),PETSC_VERSION_BK),0)
#else
#define PetscGetVersion(version) (sprintf(*(version),"Petsc Development Version %d.%d.%d, Patch %d, ", \
                                         PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, \
                                         PETSC_VERSION_PATCH),PetscStrcat(*(version),PETSC_VERSION_PATCH_DATE), \
                                         PetscStrcat(*(version),"\nBK revision: "),PetscStrcat(*(version),PETSC_VERSION_BK),0)
#endif

/*MC
    PetscGetVersion - Gets the Petsc Version information in a string.

    Output Parameter:
.   version - version string

    Level: developer

    Usage:
    char version[256];
    PetscGetVersion(&version);

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscGetProgramName()

M*/

PETSC_EXTERN_CXX_END
#endif
