#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zao.c,v 1.6 1997/09/26 02:16:37 bsmith Exp bsmith $";
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

/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
void aocreatebasic_(MPI_Comm *comm,int *napp,int *myapp,int *mypetsc,AO *aoout, int *__ierr ){
*__ierr = AOCreateBasic(
	(MPI_Comm)PetscToPointerComm( *comm ),*napp,myapp,mypetsc,aoout);
}
void aocreatebasicis_(IS isapp,IS ispetsc,AO *aoout, int *__ierr ){
*__ierr = AOCreateBasicIS(
	(IS)PetscToPointer( *(int*)(isapp) ),
	(IS)PetscToPointer( *(int*)(ispetsc) ),aoout);
}
#if defined(__cplusplus)
}
#endif
