#include <petsc/private/fortranimpl.h>
#include <petscpc.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcview_                    PCVIEW
#define pcgetoperators_            PCGETOPERATORS
#define pcsetoptionsprefix_        PCSETOPTIONSPREFIX
#define pcappendoptionsprefix_     PCAPPENDOPTIONSPREFIX
#define pcgetoptionsprefix_        PCGETOPTIONSPREFIX
#define pcviewfromoptions_         PCVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcview_                    pcview
#define pcgetoperators_            pcgetoperators
#define pcsetoptionsprefix_        pcsetoptionsprefix
#define pcappendoptionsprefix_     pcappendoptionsprefix
#define pcgetoptionsprefix_        pcgetoptionsprefix
#define pcviewfromoptions_         pcviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL pcview_(PC *pc,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PCView(*pc,v);
}

PETSC_EXTERN void PETSC_STDCALL pcsetoptionsprefix_(PC *pc,char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PCSetOptionsPrefix(*pc,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL pcappendoptionsprefix_(PC *pc,char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PCAppendOptionsPrefix(*pc,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL pcgetoptionsprefix_(PC *pc,char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PCGetOptionsPrefix(*pc,&tname);
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void PETSC_STDCALL pcviewfromoptions_(PC *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
