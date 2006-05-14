#include "petsc.h"
#include "petscfix.h"
/* mesh.c */
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

#include "petscmat.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define restrictvector_ RESTRICTVECTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define restrictvector_ restrictvector
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define assemblevectorcomplete_ ASSEMBLEVECTORCOMPLETE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define assemblevectorcomplete_ assemblevectorcomplete
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define assemblevector_ ASSEMBLEVECTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define assemblevector_ assemblevector
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define assemblematrix_ ASSEMBLEMATRIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define assemblematrix_ assemblematrix
#endif

PETSC_EXTERN_CXX_BEGIN
PetscErrorCode restrictVector(Vec, Vec, InsertMode);
PetscErrorCode assembleVectorComplete(Vec, Vec, InsertMode);
PetscErrorCode assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
PetscErrorCode assembleMatrix(Mat, PetscInt, PetscScalar [], InsertMode);
PETSC_EXTERN_CXX_END

/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
void PETSC_STDCALL  restrictvector_(Vec g,Vec l,InsertMode *mode, int *__ierr ){
*__ierr = restrictVector(
	(Vec)PetscToPointer((g) ),
	(Vec)PetscToPointer((l) ),*mode);
}
void PETSC_STDCALL  assemblevectorcomplete_(Vec g,Vec l,InsertMode *mode, int *__ierr ){
*__ierr = assembleVectorComplete(
	(Vec)PetscToPointer((g) ),
	(Vec)PetscToPointer((l) ),*mode);
}
void PETSC_STDCALL  assemblevector_(Vec b,PetscInt *e,PetscScalar v[],InsertMode *mode, int *__ierr ){
*__ierr = assembleVector(
	(Vec)PetscToPointer((b) ),*e,v,*mode);
}
void PETSC_STDCALL  assemblematrix_(Mat A,PetscInt *e,PetscScalar v[],InsertMode *mode, int *__ierr ){
*__ierr = assembleMatrix(
	(Mat)PetscToPointer((A) ),*e,v,*mode);
}
#if defined(__cplusplus)
}
#endif
