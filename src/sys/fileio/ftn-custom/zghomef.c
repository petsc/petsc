#include <petsc/private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscfixfilename_ PETSCFIXFILENAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscfixfilename_ petscfixfilename
#endif

/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif

PETSC_EXTERN void petscfixfilename_(char *filein, char *fileout, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  PetscInt i, n;
  char    *in, *out;

  in  = filein;
  out = fileout;
  n   = len1;

  for (i = 0; i < n; i++) {
    if (in[i] == PETSC_REPLACE_DIR_SEPARATOR) out[i] = PETSC_DIR_SEPARATOR;
    else out[i] = in[i];
  }
  out[i] = 0;
}

#if defined(__cplusplus)
}
#endif
