/*$Id: zpetsc.h,v 1.67 2001/09/10 03:41:06 bsmith Exp $*/

/* This file contains info for the use of PETSc Fortran interface stubs */

#include "petsc.h"
#include "petscfix.h"

EXTERN int     PetscScalarAddressToFortran(PetscObject,PetscScalar*,PetscScalar*,int,long*);
EXTERN int     PetscScalarAddressFromFortran(PetscObject,PetscScalar*,long,int,PetscScalar **);
EXTERN long    PetscIntAddressToFortran(int*,int*);
EXTERN int    *PetscIntAddressFromFortran(int*,long); 
extern char   *PETSC_NULL_CHARACTER_Fortran;
extern void   *PETSC_NULL_INTEGER_Fortran;
extern void   *PETSC_NULL_SCALAR_Fortran;
extern void   *PETSC_NULL_DOUBLE_Fortran;
extern void   *PETSC_NULL_REAL_Fortran;
EXTERN_C_BEGIN
extern void   (*PETSC_NULL_FUNCTION_Fortran)(void);
EXTERN_C_END
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
#if defined (HAVE_MPI_COMM_F2C)
#define PetscToPointerComm(a)        MPI_Comm_f2c(*(MPI_Fint *)(&a))
#define PetscFromPointerComm(a)      MPI_Comm_c2f(a)

#elif (SIZEOF_INT == SIZEOF_MPI_COMM)
#define PetscToPointerComm(a)        (a)
#define PetscFromPointerComm(a) (int)(a)

#else
#error "In the variable MPI_INCLUDE in bmake/PETSC_ARCH/packages file you must specify either: \
-DHAVE_MPI_COMM_F2C or -DSIZEOF_MPI_COMM"

#endif


/* --------------------------------------------------------------------*/
/*
    This lets us map the str-len argument either, immediately following
    the char argument (DVF on Win32) or at the end of the argument list
    (general unix compilers)
*/
#if defined(PETSC_USE_FORTRAN_MIXED_STR_ARG)
#define PETSC_MIXED_LEN(len) ,int len
#define PETSC_END_LEN(len)
#else
#define PETSC_MIXED_LEN(len)
#define PETSC_END_LEN(len)   ,int len
#endif

/* --------------------------------------------------------------------*/
/*
    This defines the mappings from Fortran character strings 
  to C character strings on the Cray T3D.
*/
#if defined(PETSC_USES_CPTOFCD)
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
    *ierr = PetscMalloc((n+1)*sizeof(char),&b); \
    if(*ierr) return; \
    *ierr = PetscStrncpy(b,_fcdtocp(a),n); \
    if(*ierr) return; \
    b[n] = 0; \
  } \
}
#define FREECHAR(a,b) if (b) PetscFree(b);

#else

#define CHAR char*
#define FIXCHAR(a,n,b) \
{\
  if (a == PETSC_NULL_CHARACTER_Fortran) { \
    b = a = 0; \
  } else { \
    while((n > 0) && (a[n-1] == ' ')) n--; \
    if (a[n] != 0) { \
      *ierr = PetscMalloc((n+1)*sizeof(char),&b); \
      if(*ierr) return; \
      *ierr = PetscStrncpy(b,a,n); \
      if(*ierr) return; \
      b[n] = 0; \
    } else b = a;\
  } \
}

#define FREECHAR(a,b) if (a != b) PetscFree(b);

#endif



#define FORTRANNULLINTEGER(a)  (((void*)a) == PETSC_NULL_INTEGER_Fortran)
#define FORTRANNULLOBJECT(a)   (((void*)a) == PETSC_NULL_INTEGER_Fortran)
#define FORTRANNULLSCALAR(a)   (((void*)a) == PETSC_NULL_SCALAR_Fortran)
#define FORTRANNULLDOUBLE(a)   (((void*)a) == PETSC_NULL_DOUBLE_Fortran)
#define FORTRANNULLREAL(a)     (((void*)a) == PETSC_NULL_REAL_Fortran)
#define FORTRANNULLFUNCTION(a) (((void(*)(void))a) == PETSC_NULL_FUNCTION_Fortran)
/*
    These are used to support the default viewers that are 
  created at run time, in C using the , trick.

    The numbers here must match the numbers in include/finclude/petsc.h
*/
#define PETSC_VIEWER_DRAW_WORLD_FORTRAN     -4
#define PETSC_VIEWER_DRAW_SELF_FORTRAN      -5
#define PETSC_VIEWER_SOCKET_WORLD_FORTRAN   -6 
#define PETSC_VIEWER_SOCKET_SELF_FORTRAN    -7
#define PETSC_VIEWER_STDOUT_WORLD_FORTRAN   -8 
#define PETSC_VIEWER_STDOUT_SELF_FORTRAN    -9
#define PETSC_VIEWER_STDERR_WORLD_FORTRAN   -10 
#define PETSC_VIEWER_STDERR_SELF_FORTRAN    -11
#define PETSC_VIEWER_BINARY_WORLD_FORTRAN   -12
#define PETSC_VIEWER_BINARY_SELF_FORTRAN    -13

#define PetscPatchDefaultViewers_Fortran(vin,v) \
{ \
    if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_DRAW_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_DRAW_WORLD; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_DRAW_SELF_FORTRAN) { \
      v = PETSC_VIEWER_DRAW_SELF; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_SOCKET_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_SOCKET_WORLD; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_SOCKET_SELF_FORTRAN) { \
      v = PETSC_VIEWER_SOCKET_SELF; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_STDOUT_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_STDOUT_WORLD; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_STDOUT_SELF_FORTRAN) { \
      v = PETSC_VIEWER_STDOUT_SELF; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_STDERR_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_STDERR_WORLD; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_STDERR_SELF_FORTRAN) { \
      v = PETSC_VIEWER_STDERR_SELF; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_BINARY_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_BINARY_WORLD; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_BINARY_SELF_FORTRAN) { \
      v = PETSC_VIEWER_BINARY_SELF; \
    } else { \
      v = *vin; \
    } \
}
