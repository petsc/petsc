
/* This file contains info for the use of PETSc Fortran interface stubs */

#include "petsc.h"
#include "pinclude/petscfix.h"

extern int     PetscScalarAddressToFortran(PetscObject,Scalar*,Scalar*,int,long*);
extern int     PetscScalarAddressFromFortran(PetscObject,Scalar*,long,int,Scalar **);
extern long    PetscIntAddressToFortran(int*,int*);
extern int    *PetscIntAddressFromFortran(int*,long); 
extern char   *PETSC_NULL_CHARACTER_Fortran;
extern void   *PETSC_NULL_INTEGER_Fortran;
extern void   *PETSC_NULL_SCALAR_Fortran;
extern void   *PETSC_NULL_DOUBLE_Fortran;
extern void   *PETSC_NULL_FUNCTION_Fortran;
/*  ----------------------------------------------------------------------*/
/*
   We store each PETSc object C pointer directly as a
   Fortran integer*4 or *8 depending on the size of pointers.
*/
#define PetscFInt long

#define PetscToPointer(a)     (*(long *)(a))
#define PetscFromPointer(a)        (long)(a)
#define PetscRmPointer(a)

/*  ----------------------------------------------------------------------*/
/*

   Some MPI implementations use the same representation of MPI_Comm in C and 
Fortran. 

   MPICH
     -For 32 bit machines there is no conversion between C and Fortran
     -For 64 bit machines
         = Before version 1.1 conversion with MPIR_xxx()
         = Version 1.1 and later no conversion

   Cray T3E/T3D 
     No conversion

   SGI
     No conversion

   HP-Convex
     - Before release 1.3 MPI_*_F2C() and MPI_*_C2F()
     - Release 1.3 and later MPI_*_f2c() and MPI_*_c2f()

   MPI-2 standard
     - MPI_*_f2c() and MPI_*_c2f()

   LAM 6.1
     - Upgrate to LAM 6.2

  LAM 6.2
     - MPI_*_f2c() and MPI_*_c2f()

   We define the macros
     PetscToPointerComm - from Fortran to C
     PetscFromPointerComm - From C to Fortran

*/
#if defined(USES_INT_MPI_COMM)
#define PetscToPointerComm(a)        (a)
#define PetscFromPointerComm(a) (int)(a)

#elif defined (PETSC_HAVE_MPI_COMM_F2C)
#define PetscToPointerComm(a)        MPI_Comm_f2c(*(MPI_Fint *)(&a))
#define PetscFromPointerComm(a)      MPI_Comm_c2f(a)

#elif (PETSC_SIZEOF_VOIDP == 8)
/*
    Here we assume that only MPICH uses pointers for 
  MPI_Comms on 64 bit machines.
*/
EXTERN_C_BEGIN
extern void *MPIR_ToPointer(int);
extern int   MPIR_FromPointer(void*);
EXTERN_C_END
#define PetscToPointerComm(a)    MPIR_ToPointer(*(int *)(&a))
#define PetscFromPointerComm(a)  MPIR_FromPointer(a)

#else
#define PetscToPointerComm(a)        (a)
#define PetscFromPointerComm(a) (int)(a)
#endif

/* --------------------------------------------------------------------*/
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
  n = _fcdlen (a); \
  if (b == PETSC_NULL_CHARACTER_Fortran) { \
      b = 0; \
  } else {  \
    while((n > 0) && (b[n-1] == ' ')) n--; \
    b = (char *) PetscMalloc( (n+1)*sizeof(char)); \
    PetscStrncpy(b,_fcdtocp(a),n); \
    b[n] = 0; \
  } \
}
#define FREECHAR(a,b) if (b) PetscFree(b);

#else
/*
  if (a == ((char*) PETSC_NULL_Fortran)) {  \
    (*PetscErrorPrintf)("PETSC ERROR: Must use PETSC_NULL_CHARACTER!"); \
    *__ierr = 1; return; \
  } 
*/
#define CHAR char*
#define FIXCHAR(a,n,b) \
{\
  if (a == PETSC_NULL_CHARACTER_Fortran) { \
    b = a = 0; \
  } else { \
    while((n > 0) && (a[n-1] == ' ')) n--; \
    if (a[n] != 0) { \
      b = (char *) PetscMalloc( (n+1)*sizeof(char)); \
      PetscStrncpy(b,a,n); \
      b[n] = 0; \
    } else b = a;\
  } \
}

#define FREECHAR(a,b) if (a != b) PetscFree(b);

#endif

#define FORTRANNULLINTEGER(a)  (((void *) a) == PETSC_NULL_INTEGER_Fortran)
#define FORTRANNULLSCALAR(a)   (((void *) a) == PETSC_NULL_SCALAR_Fortran)
#define FORTRANNULLDOUBLE(a)   (((void *) a) == PETSC_NULL_DOUBLE_Fortran)
#define FORTRANNULLFUNCTION(a) (((void *) a) == PETSC_NULL_FUNCTION_Fortran)
/*
    These are used to support the default viewers that are 
  created at run time, in C using the , trick.

    The numbers here must match the numbers in include/finclude/petsc.h
*/
#define VIEWER_DRAW_WORLD_0_FORTRAN  -4
#define VIEWER_DRAW_WORLD_1_FORTRAN  -5
#define VIEWER_DRAW_WORLD_2_FORTRAN  -6
#define VIEWER_DRAW_SELF_FORTRAN     -7
#define VIEWER_SOCKET_WORLD_FORTRAN   -8 

#define PetscPatchDefaultViewers_Fortran(vin,v) \
{ \
    if ( (*(PetscFortranAddr*)vin) == VIEWER_DRAW_WORLD_0_FORTRAN) { \
      v = VIEWER_DRAW_WORLD_0; \
    } else if ( (*(PetscFortranAddr*)vin) == VIEWER_DRAW_WORLD_1_FORTRAN) { \
      v = VIEWER_DRAW_WORLD_1; \
    } else if ( (*(PetscFortranAddr*)vin) == VIEWER_DRAW_WORLD_2_FORTRAN) { \
      v = VIEWER_DRAW_WORLD_2; \
    } else if ( (*(PetscFortranAddr*)vin) == VIEWER_DRAW_SELF_FORTRAN) { \
      v = VIEWER_DRAW_SELF; \
    } else if ( (*(PetscFortranAddr*)vin) == VIEWER_SOCKET_WORLD_FORTRAN) { \
      v = VIEWER_SOCKET_WORLD; \
    } else { \
      v = *vin; \
    } \
}
