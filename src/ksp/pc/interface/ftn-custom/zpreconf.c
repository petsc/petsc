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
#define pcdestroy_                 PCDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcview_                    pcview
#define pcgetoperators_            pcgetoperators
#define pcsetoptionsprefix_        pcsetoptionsprefix
#define pcappendoptionsprefix_     pcappendoptionsprefix
#define pcgetoptionsprefix_        pcgetoptionsprefix
#define pcviewfromoptions_         pcviewfromoptions
#define pcdestroy_                 pcdestroy
#endif

PETSC_EXTERN void pcview_(PC *pc,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PCView(*pc,v);
}

PETSC_EXTERN void pcsetoptionsprefix_(PC *pc,char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PCSetOptionsPrefix(*pc,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void pcappendoptionsprefix_(PC *pc,char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PCAppendOptionsPrefix(*pc,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void pcgetoptionsprefix_(PC *pc,char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PCGetOptionsPrefix(*pc,&tname);
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void pcviewfromoptions_(PC *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PCViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void pcdestroy_(PC *x,int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = PCDestroy(x); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}
