
#include <petsc-private/fortranimpl.h>
#include <petscao.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define aocreatebasic_   AOCREATEBASIC
#define aocreatebasicis_ AOCREATEBASICIS
#define aocreatememoryscalable_ AOCREATEMEMORYSCALABLE     
#define aocreatememoryscalableis_ AOCREATEMEMORYSCALABLEIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aocreatebasic_   aocreatebasic
#define aocreatebasicis_ aocreatebasicis
#define aocreatememoryscalable_   aocreatememoryscalable
#define aocreatememoryscalableis_ aocreatememoryscalableis
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL aocreatebasic_(MPI_Comm *comm,PetscInt *napp,PetscInt *myapp,PetscInt *mypetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(myapp);
  CHKFORTRANNULLINTEGER(mypetsc);
  *ierr = AOCreateBasic(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*napp,myapp,mypetsc,aoout);
}

void PETSC_STDCALL aocreatebasicis_(IS *isapp,IS *ispetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ispetsc);
  *ierr = AOCreateBasicIS(*isapp,*ispetsc,aoout);
}

void PETSC_STDCALL aocreatememoryscalable_(MPI_Comm *comm,PetscInt *napp,PetscInt *myapp,PetscInt *mypetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(myapp);
  CHKFORTRANNULLINTEGER(mypetsc);
  *ierr = AOCreateMemoryScalable(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*napp,myapp,mypetsc,aoout);
}

void PETSC_STDCALL aocreatememoryscalableis_(IS *isapp,IS *ispetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ispetsc);
  *ierr = AOCreateMemoryScalableIS(*isapp,*ispetsc,aoout);
}

EXTERN_C_END
