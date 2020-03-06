#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspgetoptionsprefix_       KSPGETOPTIONSPREFIX
#define kspappendoptionsprefix_    KSPAPPENDOPTIONSPREFIX
#define kspsetoptionsprefix_       KSPSETOPTIONSPREFIX
#define kspbuildsolution_          KSPBUILDSOLUTION
#define kspbuildresidual_          KSPBUILDRESIDUAL
#define matcreateschurcomplement_  MATCREATESCHURCOMPLEMENT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgetoptionsprefix_       kspgetoptionsprefix
#define kspappendoptionsprefix_    kspappendoptionsprefix
#define kspsetoptionsprefix_       kspsetoptionsprefix
#define kspbuildsolution_          kspbuildsolution
#define kspbuildresidual_          kspbuildresidual
#define matcreateschurcomplement_  matcreateschurcomplement
#endif

PETSC_EXTERN void kspbuildsolution_(KSP *ksp,Vec *v,Vec *V, int *ierr)
{
  CHKFORTRANNULLOBJECT(V);
  *ierr = KSPBuildSolution(*ksp,*v,V);
}

PETSC_EXTERN void kspbuildresidual_(KSP *ksp,Vec *t,Vec *v,Vec *V, int *ierr)
{
  CHKFORTRANNULLOBJECT(V);
  *ierr = KSPBuildResidual(*ksp,*t,*v,V);
}

PETSC_EXTERN void kspgetoptionsprefix_(KSP *ksp,char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = KSPGetOptionsPrefix(*ksp,&tname);
  *ierr = PetscStrncpy(prefix,tname,len); if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void kspappendoptionsprefix_(KSP *ksp,char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPAppendOptionsPrefix(*ksp,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

PETSC_EXTERN void kspsetoptionsprefix_(KSP *ksp,char* prefix, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPSetOptionsPrefix(*ksp,t);if (*ierr) return;
  FREECHAR(prefix,t);
}
