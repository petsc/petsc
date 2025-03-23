#include <petsc/private/ftnimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerasciiprintf_             PETSCVIEWERASCIIPRINTF
  #define petscviewerasciisynchronizedprintf_ PETSCVIEWERASCIISYNCHRONIZEDPRINTF
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerasciiprintf_             petscviewerasciiprintf
  #define petscviewerasciisynchronizedprintf_ petscviewerasciisynchronizedprintf
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

PETSC_EXTERN void petscviewerasciiprintf_(PetscViewer *viewer, char *str, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1)
{
  char       *c1, *tmp;
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer, v);
  FIXCHAR(str, len1, c1);
  *ierr = PetscFixSlashN(c1, &tmp);
  if (*ierr) return;
  FREECHAR(str, c1);
  *ierr = PetscViewerASCIIPrintf(v, "%s", tmp);
  if (*ierr) return;
  *ierr = PetscFree(tmp);
}

PETSC_EXTERN void petscviewerasciisynchronizedprintf_(PetscViewer *viewer, char *str, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1)
{
  char       *c1, *tmp;
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer, v);
  FIXCHAR(str, len1, c1);
  *ierr = PetscFixSlashN(c1, &tmp);
  if (*ierr) return;
  FREECHAR(str, c1);
  *ierr = PetscViewerASCIISynchronizedPrintf(v, "%s", tmp);
  if (*ierr) return;
  *ierr = PetscFree(tmp);
}
