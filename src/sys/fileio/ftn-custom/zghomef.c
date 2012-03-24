#include <petsc-private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscfixfilename_          PETSCFIXFILENAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscfixfilename_          petscfixfilename
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscfixfilename_(CHAR filein PETSC_MIXED_LEN(len1),CHAR fileout PETSC_MIXED_LEN(len2),
                                     PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  PetscInt  i,n;
  char *in,*out;

  in  = filein;
  out = fileout;
  n   = len1;

  for (i=0; i<n; i++) {
    if (in[i] == PETSC_REPLACE_DIR_SEPARATOR) out[i] = PETSC_DIR_SEPARATOR;
    else out[i] = in[i];
  }
  out[i] = 0;
}

EXTERN_C_END
