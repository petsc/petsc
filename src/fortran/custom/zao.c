
#include "src/fortran/custom/zpetsc.h"
#include "petscao.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define aocreatebasic_   AOCREATEBASIC
#define aocreatebasicis_ AOCREATEBASICIS
#define aoview_          AOVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aocreatebasic_   aocreatebasic
#define aocreatebasicis_ aocreatebasicis
#define aoview_          aoview
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL aoview_(AO *ao,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = AOView(*ao,v);
}

void PETSC_STDCALL aocreatebasic_(MPI_Comm *comm,PetscInt *napp,PetscInt *myapp,PetscInt *mypetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(myapp);
  CHKFORTRANNULLINTEGER(mypetsc);
  *ierr = AOCreateBasic((MPI_Comm)PetscToPointerComm(*comm),*napp,myapp,mypetsc,aoout);
}

void PETSC_STDCALL aocreatebasicis_(IS *isapp,IS *ispetsc,AO *aoout,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(aoout);
  *ierr = AOCreateBasicIS(*isapp,*ispetsc,aoout);
}

EXTERN_C_END
