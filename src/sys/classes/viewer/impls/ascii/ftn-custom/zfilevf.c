#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerfilesetname_                PETSCVIEWERFILESETNAME
#define petscviewerfilegetname_                PETSCVIEWERFILEGETNAME
#define petscviewerasciiprintf_                PETSCVIEWERASCIIPRINTF
#define petscviewerasciipushtab_               PETSCVIEWERASCIIPUSHTAB
#define petscviewerasciipoptab_                PETSCVIEWERASCIIPOPTAB
#define petscviewerasciisynchronizedprintf_    PETSCVIEWERASCIISYNCHRONIZEDPRINTF
#define petscviewerasciipushsynchronized_      PETSCVIEWERASCIIPUSHSYNCHRONIZED
#define petscviewerasciipopsynchronized_       PETSCVIEWERASCIIPOPSYNCHRONIZED
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerfilesetname_                petscviewerfilesetname
#define petscviewerfilegetname_                petscviewerfilegetname
#define petscviewerasciiprintf_                petscviewerasciiprintf
#define petscviewerasciipushtab_               petscviewerasciipushtab
#define petscviewerasciipoptab_                petscviewerasciipoptab
#define petscviewerasciisynchronizedprintf_    petscviewerasciisynchronizedprintf
#define petscviewerasciipushsynchronized_      petscviewerasciipushsynchronized
#define petscviewerasciipopsynchronized_       petscviewerasciipopsynchronized
#endif

PETSC_EXTERN void petscviewerfilesetname_(PetscViewer *viewer,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char        *c1;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerFileSetName(v,c1);if (*ierr) return;
  FREECHAR(name,c1);
}

PETSC_EXTERN void petscviewerfilegetname_(PetscViewer *viewer, char* name, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
   const char *c1;

   *ierr = PetscViewerGetType(*viewer, &c1);if (*ierr) return;
   *ierr = PetscStrncpy(name, c1, len);if (*ierr) return;
   FIXRETURNCHAR(PETSC_TRUE, name, len);
}

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

PETSC_EXTERN void petscviewerasciiprintf_(PetscViewer *viewer,char* str,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len1)
{
  char        *c1, *tmp;
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(str,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  FREECHAR(str,c1);
  *ierr = PetscViewerASCIIPrintf(v,"%s",tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);
}

PETSC_EXTERN void petscviewerasciipushtab_(PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerASCIIPushTab(v);
}

PETSC_EXTERN void petscviewerasciipoptab_(PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerASCIIPopTab(v);
}

PETSC_EXTERN void petscviewerasciisynchronizedprintf_(PetscViewer *viewer,char* str,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len1)
{
  char        *c1, *tmp;
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(str,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  FREECHAR(str,c1);
  *ierr = PetscViewerASCIISynchronizedPrintf(v,"%s",tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);
}

PETSC_EXTERN void petscviewerasciipushsynchronized_(PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerASCIIPushSynchronized(v);
}

PETSC_EXTERN void petscviewerasciipopsynchronized_(PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerASCIIPopSynchronized(v);
}
