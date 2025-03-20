#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscinfo_ PETSCINFO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscinfo_ petscinfo
#endif

static PetscErrorCode PetscFixSlashN(const char *in, char **out)
{
  PetscInt i;
  size_t   len;

  PetscFunctionBegin;
  PetscCall(PetscStrallocpy(in, out));
  PetscCall(PetscStrlen(*out, &len));
  for (i = 0; i < (int)len - 1; i++) {
    if ((*out)[i] == '\\' && (*out)[i + 1] == 'n') {
      (*out)[i]     = ' ';
      (*out)[i + 1] = '\n';
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN void petscinfo_(char *text, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1)
{
  char *c1, *tmp;

  FIXCHAR(text, len1, c1);
  *ierr = PetscFixSlashN(c1, &tmp);
  if (*ierr) return;
  FREECHAR(text, c1);
  *ierr = PetscInfo(NULL, "%s", tmp);
  if (*ierr) return;
  *ierr = PetscFree(tmp);
}
