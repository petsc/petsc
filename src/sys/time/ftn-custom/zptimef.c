#include "petscsys.h"
#include "petscfix.h"
#include <petsctime.h>
#include "petsc-private/fortranimpl.h"
/* ptime.c */
/* Fortran interface file */

/*
* This file was generated automatically by bfort from the C source
* file.  
 */

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

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petsctime_ PETSCTIME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petsctime_ petsctime
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
void PETSC_STDCALL  petsctime_(PetscLogDouble *t, int *__ierr ){
*__ierr = PetscTime(t);
}
#if defined(__cplusplus)
}
#endif
