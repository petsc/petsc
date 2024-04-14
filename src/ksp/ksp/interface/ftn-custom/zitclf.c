#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspgetoptionsprefix_      KSPGETOPTIONSPREFIX
  #define kspbuildsolution_         KSPBUILDSOLUTION
  #define kspbuildresidual_         KSPBUILDRESIDUAL
  #define matcreateschurcomplement_ MATCREATESCHURCOMPLEMENT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspgetoptionsprefix_      kspgetoptionsprefix
  #define kspbuildsolution_         kspbuildsolution
  #define kspbuildresidual_         kspbuildresidual
  #define matcreateschurcomplement_ matcreateschurcomplement
#endif

PETSC_EXTERN void kspbuildsolution_(KSP *ksp, Vec *v, Vec *V, int *ierr)
{
  CHKFORTRANNULLOBJECT(V);
  *ierr = KSPBuildSolution(*ksp, *v, V);
}

PETSC_EXTERN void kspbuildresidual_(KSP *ksp, Vec *t, Vec *v, Vec *V, int *ierr)
{
  CHKFORTRANNULLOBJECT(V);
  *ierr = KSPBuildResidual(*ksp, *t, *v, V);
}
