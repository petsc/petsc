#include "petscsys.h"
#include "petscfix.h"
#include "petsc-private/fortranimpl.h"
/* matcoloring.c */
/* Fortran interface file */

#ifdef PETSC_USE_POINTER_CONVERSION
#if defined(__cplusplus)
extern "C" {
#endif
  extern void *PetscToPointer(void*);
  extern int PetscFromPointer(void *);
  extern void PetscRmPointer(void*);
#if defined(__cplusplus)
}
#endif

#else

#define PetscToPointer(a) (*(long *)(a))
#define PetscFromPointer(a) (long)(a)
#define PetscRmPointer(a)
#endif

#include "petscmat.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matcoloringcreate_  MATCOLORINGCREATE
#define matcoloringsettype_ MATCOLORINGSETTYPE
#define matcoloringsettype_ MATCOLORINGDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define matcoloringcreate_  matcoloringcreate
#define matcoloringsettype_ matcoloringsettype
#define matcoloringdestroy_ matcoloringdestroy
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
void PETSC_STDCALL  matcoloringcreate_(Mat m,MatColoring *mcptr, int *__ierr ){
  *__ierr = MatColoringCreate((Mat)PetscToPointer((m)),mcptr);
}
PETSC_EXTERN void PETSC_STDCALL matcoloringsettype_(MatColoring *mc,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = MatColoringSetType(*mc,t);
  FREECHAR(type,t);
}
PETSC_EXTERN void PETSC_STDCALL  matcoloringdestroy_(MatColoring *mc, int *__ierr ){
*__ierr = MatColoringDestroy(mc);
}
#if defined(__cplusplus)
}
#endif
