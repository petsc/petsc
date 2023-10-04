
#include <petsc/private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmdasetfieldname_     DMDASETFIELDNAME
  #define dmdagetfieldname_     DMDAGETFIELDNAME
  #define dmdagetcorners_       DMDAGETCORNERS
  #define dmdagetcorners000000_ DMDAGETCORNERS000000
  #define dmdagetcorners001001_ DMDAGETCORNERS001001
  #define dmdagetcorners011011_ DMDAGETCORNERS011011
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmdasetfieldname_     dmdasetfieldname
  #define dmdagetfieldname_     dmdagetfieldname
  #define dmdagetcorners_       dmdagetcorners
  #define dmdagetcorners000000_ dmdagetcorners000000
  #define dmdagetcorners001001_ dmdagetcorners001001
  #define dmdagetcorners011011_ dmdagetcorners011011
#endif

PETSC_EXTERN void dmdasetfieldname_(DM *da, PetscInt *nf, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(name, len, t);
  *ierr = DMDASetFieldName(*da, *nf, t);
  if (*ierr) return;
  FREECHAR(name, t);
}

PETSC_EXTERN void dmdagetfieldname_(DM *da, PetscInt *nf, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = DMDAGetFieldName(*da, *nf, &tname);
  if (*ierr) return;
  *ierr = PetscStrncpy(name, tname, len);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}

PETSC_EXTERN void dmdagetcorners_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  CHKFORTRANNULLINTEGER(y);
  CHKFORTRANNULLINTEGER(z);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);

  *ierr = DMDAGetCorners(*da, x, y, z, m, n, p);
}

PETSC_EXTERN void dmdagetcorners000000_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetcorners_(da, x, y, z, m, n, p, ierr);
}

PETSC_EXTERN void dmdagetcorners001001_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetcorners_(da, x, y, z, m, n, p, ierr);
}

PETSC_EXTERN void dmdagetcorners011011_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetcorners_(da, x, y, z, m, n, p, ierr);
}
