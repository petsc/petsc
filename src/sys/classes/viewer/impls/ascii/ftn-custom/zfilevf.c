#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerfilesetname_                PETSCVIEWERFILESETNAME
#define petscviewerasciiprintf_                PETSCVIEWERASCIIPRINTF
#define petscviewerasciisynchronizedprintf_    PETSCVIEWERASCIISYNCHRONIZEDPRINTF
#define petscviewerasciipushsynchronized_      PETSCVIEWERASCIIPUSHSYNCHRONIZE
#define petscviewerasciipopsynchronized_       PETSCVIEWERASCIIPOPSYNCHRONIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerfilesetname_                petscviewerfilesetname
#define petscviewerasciiprintf_                petscviewerasciiprintf
#define petscviewerasciisynchronizedprintf_    petscviewerasciisynchronizedprintf
#define petscviewerasciipushsynchronized_      petscviewerasciipushsynchronized
#define petscviewerasciipopsynchronized_       petscviewerasciipopsynchronized
#endif

PETSC_EXTERN void PETSC_STDCALL petscviewerfilesetname_(PetscViewer *viewer,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char        *c1;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerFileSetName(v,c1);
  FREECHAR(name,c1);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFixSlashN"
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

PETSC_EXTERN void PETSC_STDCALL petscviewerasciiprintf_(PetscViewer *viewer,CHAR str PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char        *c1, *tmp;
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(str,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  FREECHAR(str,c1);
  *ierr = PetscViewerASCIIPrintf(v,tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerasciisynchronizedprintf_(PetscViewer *viewer,CHAR str PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char        *c1, *tmp;
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(str,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  FREECHAR(str,c1);
  *ierr = PetscViewerASCIISynchronizedPrintf(v,tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerasciipushsynchronized_(PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerASCIIPushSynchronized(v);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerasciipopsynchronized_(PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerASCIIPopSynchronized(v);
}
