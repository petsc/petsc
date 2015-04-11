#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspgetoptionsprefix_       KSPGETOPTIONSPREFIX
#define kspappendoptionsprefix_    KSPAPPENDOPTIONSPREFIX
#define kspsetoptionsprefix_       KSPSETOPTIONSPREFIX
#define kspsetfischerguess_        KSPSETFISCHERGUESS
#define kspsetusefischerguess_     KSPSETUSEFISCHERGUESS
#define kspgetfischerguess_        KSPGETFISCHERGUESS
#define kspfischerguesscreate_     KSPFISCHERGUESSCREATE
#define kspfischerguessdestroy_    KSPFISCHERGUESSDESTROY
#define kspbuildsolution_          KSPBUILDSOLUTION
#define kspbuildresidual_          KSPBUILDRESIDUAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgetoptionsprefix_       kspgetoptionsprefix
#define kspappendoptionsprefix_    kspappendoptionsprefix
#define kspsetoptionsprefix_       kspsetoptionsprefix
#define kspsetfischerguess_        kspsetfischerguess
#define kspsetusefischerguess_     kspsetusefischerguess
#define kspgetfischerguess_        kspgetfischerguess
#define kspfischerguesscreate_     kspfischerguesscreate
#define kspfischerguessdestroy_    kspfischerguessdestroy
#define kspbuildsolution_          kspbuildsolution
#define kspbuildresidual_          kspbuildresidual

#endif

PETSC_EXTERN void PETSC_STDCALL kspbuildsolution_(KSP *ksp,Vec *v,Vec *V, int *ierr)
{
  Vec vp = 0;
  CHKFORTRANNULLOBJECT(v);
  CHKFORTRANNULLOBJECT(V);
  if (v) vp = *v;
  *ierr = KSPBuildSolution(*ksp,vp,V);
}

PETSC_EXTERN void PETSC_STDCALL kspbuildresidual_(KSP *ksp,Vec *t,Vec *v,Vec *V, int *ierr)
{
  Vec tp = 0,vp = 0;
  CHKFORTRANNULLOBJECT(t);
  CHKFORTRANNULLOBJECT(v);
  CHKFORTRANNULLOBJECT(V);
  if (t) tp = *t;
  if (v) vp = *v;
  *ierr = KSPBuildResidual(*ksp,tp,vp,V);
}

PETSC_EXTERN void PETSC_STDCALL kspgetoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = KSPGetOptionsPrefix(*ksp,&tname);
  *ierr = PetscStrncpy(prefix,tname,len); if (*ierr) return;
}
PETSC_EXTERN void PETSC_STDCALL kspappendoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPAppendOptionsPrefix(*ksp,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL kspsetoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPSetOptionsPrefix(*ksp,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL kspsetfischerguess_(KSP *ksp,KSPFischerGuess *guess, PetscErrorCode *ierr)
{
  *ierr = KSPSetFischerGuess(*ksp,*guess);
}

PETSC_EXTERN void PETSC_STDCALL kspgetfischerguess_(KSP *ksp,KSPFischerGuess *guess, PetscErrorCode *ierr)
{
  *ierr = KSPGetFischerGuess(*ksp,guess);
}

PETSC_EXTERN void PETSC_STDCALL kspsetusefischerguess_(KSP *ksp,PetscInt *model,PetscInt *size, PetscErrorCode *ierr)
{
  *ierr = KSPSetUseFischerGuess(*ksp,*model,*size);
}

PETSC_EXTERN void PETSC_STDCALL kspfischerguesscreate_(KSP *ksp,PetscInt *model,PetscInt *size, KSPFischerGuess *guess,PetscErrorCode *ierr)
{
  *ierr = KSPFischerGuessCreate(*ksp,*model,*size,guess);
}

PETSC_EXTERN void PETSC_STDCALL kspfischerguessdestroy_(KSPFischerGuess *guess,PetscErrorCode *ierr)
{
  *ierr = KSPFischerGuessDestroy(guess);
}

