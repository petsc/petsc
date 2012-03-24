#include <petsc-private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcview_                    PCVIEW
#define pcgetoperators_            PCGETOPERATORS
#define pcsetoptionsprefix_        PCSETOPTIONSPREFIX
#define pcappendoptionsprefix_     PCAPPENDOPTIONSPREFIX
#define pcgetoptionsprefix_        PCGETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcview_                    pcview
#define pcgetoperators_            pcgetoperators
#define pcsetoptionsprefix_        pcsetoptionsprefix
#define pcappendoptionsprefix_     pcappendoptionsprefix
#define pcgetoptionsprefix_        pcgetoptionsprefix
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pcview_(PC *pc,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PCView(*pc,v);
}

void PETSC_STDCALL pcgetoperators_(PC *pc,Mat *mat,Mat *pmat,MatStructure *flag,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mat);
  CHKFORTRANNULLOBJECT(pmat);
  CHKFORTRANNULLINTEGER(flag);
  *ierr = PCGetOperators(*pc,mat,pmat,flag);
}

void PETSC_STDCALL pcsetoptionsprefix_(PC *pc,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PCSetOptionsPrefix(*pc,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL pcappendoptionsprefix_(PC *pc,CHAR prefix PETSC_MIXED_LEN(len),
                                          PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PCAppendOptionsPrefix(*pc,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL pcgetoptionsprefix_(PC *pc,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PCGetOptionsPrefix(*pc,&tname);
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
}

EXTERN_C_END
