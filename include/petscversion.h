
/* $Id: petscversion.h,v 1.26 2001/06/21 21:20:02 bsmith Exp $ */
#if !defined(__PETSCVERSION_H)
#define __PETSCVERSION_H

/* ========================================================================== */
/* 
   Current PETSc version number and release date, also listed in
    Web page
    docs/tex/manual/intro.tex,
    docs/tex/manual/manual.tex.
*/
#define PETSC_VERSION_MAJOR    2
#define PETSC_VERSION_MINOR    1
#define PETSC_VERSION_SUBMINOR 2
#define PETSC_VERSION_PATCH    7
#define PETSC_VERSION_DATE     "May 10, 2002"
#define PETSC_AUTHOR_INFO      "\
       The PETSc Team\n\
    petsc-maint@mcs.anl.gov\n\
 http://www.mcs.anl.gov/petsc/\n"

#define PetscGetVersion(version) (sprintf(*(version),"Petsc Version %d.%d.%d, Patch %d, Released ", \
                                         PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR, \
                                         PETSC_VERSION_PATCH),PetscStrcat(*(version),PETSC_VERSION_DATE),0)
#endif

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
