
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
#define PETSC_VERSION_MAJOR      2
#define PETSC_VERSION_MINOR      2
#define PETSC_VERSION_SUBMINOR   1
#define PETSC_VERSION_PATCH      26
#define PETSC_VERSION_DATE       "August, 18, 2004"
#define PETSC_VERSION_PATCH_DATE "November, 18, 2004"
#define PETSC_AUTHOR_INFO        "\
       The PETSc Team\n\
    petsc-maint@mcs.anl.gov\n\
 http://www.mcs.anl.gov/petsc/\n"

#define PetscGetVersion(version) (sprintf(*(version),"Petsc Version %d.%d.%d, Patch %d, Released ", \
                                         PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, \
                                         PETSC_VERSION_PATCH),PetscStrcat(*(version),PETSC_VERSION_DATE),0)

/*M
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
