#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zao.c,v 1.5 1997/07/09 20:55:52 balay Exp bsmith $";
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
void aocreatebasicis_(MPI_Comm *comm,IS isapp,IS ispetsc,AO *aoout, int *__ierr ){
*__ierr = AOCreateBasicIS(
	(MPI_Comm)PetscToPointerComm( *comm ),
	(IS)PetscToPointer( *(int*)(isapp) ),
	(IS)PetscToPointer( *(int*)(ispetsc) ),aoout);
}
#if defined(__cplusplus)
}
#endif
