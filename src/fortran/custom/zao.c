#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zao.c,v 1.9 1998/10/05 18:36:15 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "ao.h"

#ifdef HAVE_FORTRAN_CAPS
#define aocreatebasic_ AOCREATEBASIC
#define aocreatebasicis_ AOCREATEBASICIS
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define aocreatebasic_ aocreatebasic
#define aocreatebasicis_ aocreatebasicis
#endif

EXTERN_C_BEGIN

void aocreatebasic_(MPI_Comm *comm,int *napp,int *myapp,int *mypetsc,AO *aoout, int *__ierr )
{
  *__ierr = AOCreateBasic((MPI_Comm)PetscToPointerComm( *comm ),*napp,myapp,mypetsc,aoout);
}

void aocreatebasicis_(IS *isapp,IS *ispetsc,AO *aoout, int *__ierr )
{
  *__ierr = AOCreateBasicIS(*isapp,*ispetsc,aoout);
}

EXTERN_C_END
