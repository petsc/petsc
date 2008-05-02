
#include "private/fortranimpl.h"
#include "petscao.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define aocreatemapping_   AOCREATEMAPPING
#define aocreatemappingis_ AOCREATEMAPPINGIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aocreatemapping_   aocreatemapping
#define aocreatemappingis_ aocreatemappingis
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL aocreatemapping_(MPI_Comm *comm,PetscInt *napp,PetscInt *myapp,PetscInt *mypetsc,AO *aoout,PetscErrorCode *ierr)
{
  if (*napp) {
    CHKFORTRANNULLINTEGER(myapp);
    CHKFORTRANNULLINTEGER(mypetsc);
  }
  *ierr = AOCreateMapping(MPI_Comm_f2c(*(MPI_Fint*)comm),*napp,myapp,mypetsc,aoout);
}

void PETSC_STDCALL aocreatemappingis_(IS *isapp,IS *ispetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ispetsc);
  *ierr = AOCreateMappingIS(*isapp,*ispetsc,aoout);
}

EXTERN_C_END
