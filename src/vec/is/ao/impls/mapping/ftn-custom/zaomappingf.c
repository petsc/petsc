#include <petsc/private/fortranimpl.h>
#include <petscao.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define aocreatemappingis_ AOCREATEMAPPINGIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define aocreatemappingis_ aocreatemappingis
#endif

PETSC_EXTERN void aocreatemappingis_(IS *myapp, IS *mypetsc, AO *aoout, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mypetsc);
  *ierr = AOCreateMappingIS(*myapp, *mypetsc, aoout);
}
