#include <petsc/private/fortranimpl.h>
#include <petscao.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define aocreatebasicis_          AOCREATEBASICIS
  #define aocreatememoryscalableis_ AOCREATEMEMORYSCALABLEIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define aocreatebasicis_          aocreatebasicis
  #define aocreatememoryscalableis_ aocreatememoryscalableis
#endif

PETSC_EXTERN void aocreatebasicis_(IS *isapp, IS *ispetsc, AO *aoout, PetscErrorCode *ierr)
{
  IS cispetsc = NULL;
  CHKFORTRANNULLOBJECT(ispetsc);
  if (ispetsc) cispetsc = *ispetsc;
  *ierr = AOCreateBasicIS(*isapp, cispetsc, aoout);
}

PETSC_EXTERN void aocreatememoryscalableis_(IS *isapp, IS *ispetsc, AO *aoout, PetscErrorCode *ierr)
{
  IS cispetsc = NULL;
  CHKFORTRANNULLOBJECT(ispetsc);
  if (ispetsc) cispetsc = *ispetsc;
  *ierr = AOCreateMemoryScalableIS(*isapp, cispetsc, aoout);
}
