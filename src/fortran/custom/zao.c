#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zao.c,v 1.11 1999/05/12 03:34:35 bsmith Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "ao.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define aocreatebasic_ AOCREATEBASIC
#define aocreatebasicis_ AOCREATEBASICIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aocreatebasic_ aocreatebasic
#define aocreatebasicis_ aocreatebasicis
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL aocreatebasic_(MPI_Comm *comm,int *napp,int *myapp,int *mypetsc,AO *aoout, int *__ierr )
{
  *__ierr = AOCreateBasic((MPI_Comm)PetscToPointerComm( *comm ),*napp,myapp,mypetsc,aoout);
}

void PETSC_STDCALL aocreatebasicis_(IS *isapp,IS *ispetsc,AO *aoout, int *__ierr )
{
  *__ierr = AOCreateBasicIS(*isapp,*ispetsc,aoout);
}

EXTERN_C_END
