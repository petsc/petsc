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
  PetscErrorCode ierr;
  PetscInt       i;
  size_t         len;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(in,out);CHKERRQ(ierr);
  ierr = PetscStrlen(*out,&len);CHKERRQ(ierr);
  for (i=0; i<(int)len-1; i++) {
    if ((*out)[i] == '\\' && (*out)[i+1] == 'n') {(*out)[i] = ' '; (*out)[i+1] = '\n';}
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN void PETSC_STDCALL petscmallocdump_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDump(stdout);
}
PETSC_EXTERN void PETSC_STDCALL petscmallocview_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocView(stdout);
}

PETSC_EXTERN void PETSC_STDCALL petscmallocvalidate_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocValidate(0,"Unknown Fortran",0);
}

PETSC_EXTERN void PETSC_STDCALL petscmemoryview_(PetscViewer *vin, char* message PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
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

