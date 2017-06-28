#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspgetoptionsprefix_       KSPGETOPTIONSPREFIX
#define kspappendoptionsprefix_    KSPAPPENDOPTIONSPREFIX
#define kspsetoptionsprefix_       KSPSETOPTIONSPREFIX
#define kspbuildsolution_          KSPBUILDSOLUTION
#define kspbuildresidual_          KSPBUILDRESIDUAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgetoptionsprefix_       kspgetoptionsprefix
#define kspappendoptionsprefix_    kspappendoptionsprefix
#define kspsetoptionsprefix_       kspsetoptionsprefix
#define kspbuildsolution_          kspbuildsolution
#define kspbuildresidual_          kspbuildresidual
#endif

PETSC_EXTERN void PETSC_STDCALL kspbuildsolution_(KSP *ksp,Vec *v,Vec *V, int *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(v);
  CHKFORTRANNULLOBJECT(V);
  *ierr = KSPBuildSolution(*ksp,*v,V);
}

PETSC_EXTERN void PETSC_STDCALL kspbuildresidual_(KSP *ksp,Vec *t,Vec *v,Vec *V, int *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(t);
  CHKFORTRANNULLOBJECTDEREFERENCE(v);
  CHKFORTRANNULLOBJECT(V);
  *ierr = KSPBuildResidual(*ksp,*t,*v,V);
}

PETSC_EXTERN void PETSC_STDCALL kspgetoptionsprefix_(KSP *ksp,char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = KSPGetOptionsPrefix(*ksp,&tname);
  *ierr = PetscStrncpy(prefix,tname,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void PETSC_STDCALL kspappendoptionsprefix_(KSP *ksp,char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPAppendOptionsPrefix(*ksp,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL kspsetoptionsprefix_(KSP *ksp,char* prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPSetOptionsPrefix(*ksp,t);
  FREECHAR(prefix,t);
}
