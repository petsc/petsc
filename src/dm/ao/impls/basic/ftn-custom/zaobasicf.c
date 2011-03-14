
#include <private/fortranimpl.h>
#include <petscao.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define aocreatebasic_   AOCREATEBASIC
#define aocreatebasicis_ AOCREATEBASICIS
#define aocreatebasic_   AOCREATEBASICMemoryScalable
#define aocreatebasicis_ AOCREATEBASICMemoryScalableIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aocreatebasic_   aocreatebasic
#define aocreatebasicis_ aocreatebasicis
#define aocreatebasicmemoryscalable_   aocreatebasicmemoryscalable
#define aocreatebasicmemoryscalableis_ aocreatebasicmemoryscalableis
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL aocreatebasic_(MPI_Comm *comm,PetscInt *napp,PetscInt *myapp,PetscInt *mypetsc,AO *aoout,PetscErrorCode *ierr)
{
  if (*napp) {
    CHKFORTRANNULLINTEGER(myapp);
    CHKFORTRANNULLINTEGER(mypetsc);
  }
  *ierr = AOCreateBasic(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*napp,myapp,mypetsc,aoout);
}

void PETSC_STDCALL aocreatebasicis_(IS *isapp,IS *ispetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ispetsc);
  *ierr = AOCreateBasicIS(*isapp,*ispetsc,aoout);
}

void PETSC_STDCALL aocreatebasicmemoryscalable_(MPI_Comm *comm,PetscInt *napp,PetscInt *myapp,PetscInt *mypetsc,AO *aoout,PetscErrorCode *ierr)
{
  if (*napp) {
    CHKFORTRANNULLINTEGER(myapp);
    CHKFORTRANNULLINTEGER(mypetsc);
  }
  *ierr = AOCreateBasicMemoryScalable(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*napp,myapp,mypetsc,aoout);
}

void PETSC_STDCALL aocreatebasicmemoryscalableis_(IS *isapp,IS *ispetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ispetsc);
  *ierr = AOCreateBasicMemoryScalableIS(*isapp,*ispetsc,aoout);
}

EXTERN_C_END
