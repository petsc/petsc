
/* This file contains info for the use of PETSc Fortran interface stubs */

#include "petsc.h"

       int     PetscScalarAddressToFortran(Scalar*,Scalar*);
       Scalar* PetscScalarAddressFromFortran(Scalar*,int);
       int     PetscIntAddressToFortran(int*,int*);
       int    *PetscIntAddressFromFortran(int*,int); 
extern void   *PETSC_NULL_Fortran;
extern char   *PETSC_NULL_CHARACTER_Fortran;

/*
     On 32 bit machines we store the PETSc object C pointers directly 
   as a Fortran integer. On 64 bit machines we convert it with the routines
       C pointer       = PetscToPointer(Fortran integer)
       Fortran integer = PetscFromPointer(C pointer)

     For 32 bit machines and MPI implementations that use integers as MPI_Comms
   (i.e. when USES_INT_MPI_COMM is defined) the C and Fortran representations 
   are the same. For 64 bit machines using MPICH we convert it with the routines
       C pointer       = PetscToPointerComm(Fortran integer)
       Fortran integer = PetscFromPointerComm(C pointer)
*/
#if defined(HAVE_64BITS) && !defined(PETSC_USING_MPIUNI)
#if defined(__cplusplus)
extern "C" {
#endif
extern void *PetscToPointer(int);
extern int  PetscFromPointer(void*);
extern void PetscRmPointer(int);
#if defined(__cplusplus)
}
#endif

#if defined(USES_INT_MPI_COMM)
#define PetscToPointerComm(a)        (a)
#define PetscFromPointerComm(a) (int)(a)
#else
/*
    Here we assume that only MPICH uses pointers for 
  MPI_Comms on 64 bit machines.
*/
#if defined(__cplusplus)
extern "C" {
#endif
extern void *MPIR_ToPointer(int);
extern int   MPIR_FromPointer(void*);
#if defined(__cplusplus)
}
#endif
#define PetscToPointerComm(a)    MPIR_ToPointer(a)
#define PetscFromPointerComm(a)  MPIR_FromPointer(a)
#endif

#else
#define PetscToPointer(a)            (a)
#define PetscFromPointer(a)     (int)(a)
#define PetscRmPointer(a)
#define PetscToPointerComm(a)        (a)
#define PetscFromPointerComm(a) (int)(a)
#endif

/*
    This defines the mappings from Fortran charactor strings 
  to C charactor strings on the Cray T3D.
*/
#if defined(USES_CPTOFCD)
#include <fortran.h>

#define CHAR _fcd
#define FIXCHAR(a,n,b) \
{ \
  b = _fcdtocp(a); \
  if (b == PETSC_NULL_CHARACTER_Fortran) {b = 0;} \
}
#define FREECHAR(a,b) 

#else

#define CHAR char*
#define FIXCHAR(a,n,b) \
{\
  if (a == ((char*) PETSC_NULL_Fortran)) {  \
    fprintf(stderr,"PETSC ERROR: Must use PETSC_NULL_CHARACTER!"); \
    *__ierr = 1; return; \
  }  \
  if (a == PETSC_NULL_CHARACTER_Fortran) { \
    b = a = 0; \
  } else if (a[n] != 0) { \
    b = (char *) PetscMalloc( (n+1)*sizeof(char)); \
    PetscStrncpy(b,a,n); \
    b[n] = 0; \
  } else b = a;\
}
#define FREECHAR(a,b) if (a != b) PetscFree(b);

#endif

#define FORTRANNULL(a) (((void *) a) == PETSC_NULL_Fortran)

/*
    These are used to support the default viewers that are 
  created at run time, in C using the , trick.

    The numbers here must match the numbers in include/FINCLUDE/petsc.h
*/
#define VIEWER_DRAWX_WORLD_0_FORTRAN  -4
#define VIEWER_DRAWX_WORLD_1_FORTRAN  -5
#define VIEWER_DRAWX_WORLD_2_FORTRAN  -6
#define VIEWER_DRAWX_SELF_FORTRAN     -7
#define VIEWER_MATLAB_WORLD_FORTRAN   -8 

#define PetscPatchDefaultViewers_Fortran(v) \
{ \
    if ( (*(int*)v) == VIEWER_DRAWX_WORLD_0_FORTRAN) { \
      v = VIEWER_DRAWX_WORLD_0; \
    } else if ( (*(int*)v) == VIEWER_DRAWX_WORLD_1_FORTRAN) { \
      v = VIEWER_DRAWX_WORLD_1; \
    } else if ( (*(int*)v) == VIEWER_DRAWX_WORLD_2_FORTRAN) { \
      v = VIEWER_DRAWX_WORLD_2; \
    } else if ( (*(int*)v) == VIEWER_DRAWX_SELF_FORTRAN) { \
      v = VIEWER_DRAWX_SELF; \
    } else if ( (*(int*)v) == VIEWER_MATLAB_WORLD_FORTRAN) { \
      v = VIEWER_MATLAB_WORLD; \
    } else {\
      v = (Viewer)PetscToPointer(*(int*)(v)); \
    } \
}
