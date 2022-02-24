#include <petsc/private/fortranimpl.h>
#include <petscsys.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscmallocdump_               PETSCMALLOCDUMP
#define petscmallocview_               PETSCMALLOCVIEW
#define petscmallocvalidate_           PETSCMALLOCVALIDATE
#define petscmemoryview_               PETSCMEMORYVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscmallocdump_               petscmallocdump
#define petscmallocview_               petscmallocview
#define petscmallocvalidate_           petscmallocvalidate
#define petscmemoryview_               petscmemoryview
#endif

static PetscErrorCode PetscFixSlashN(const char *in, char **out)
{
  PetscInt       i;
  size_t         len;

  PetscFunctionBegin;
  CHKERRQ(PetscStrallocpy(in,out));
  CHKERRQ(PetscStrlen(*out,&len));
  for (i=0; i<(int)len-1; i++) {
    if ((*out)[i] == '\\' && (*out)[i+1] == 'n') {(*out)[i] = ' '; (*out)[i+1] = '\n';}
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN void petscmallocdump_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDump(stdout);
}
PETSC_EXTERN void petscmallocview_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocView(stdout);
}

PETSC_EXTERN void petscmallocvalidate_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocValidate(0,"Unknown Fortran",0);
}

PETSC_EXTERN void petscmemoryview_(PetscViewer *vin, char* message, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  PetscViewer v;
  char        *msg, *tmp;

  FIXCHAR(message,len,msg);
  *ierr = PetscFixSlashN(msg,&tmp);if (*ierr) return;
  FREECHAR(message,msg);
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscMemoryView(v,tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);
}
