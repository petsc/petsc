
/* This file contains info for the use of PETSc Fortran interface stubs */

#include "petscsys.h"

EXTERN PetscErrorCode PetscScalarAddressToFortran(PetscObject,PetscInt,PetscScalar*,PetscScalar*,PetscInt,size_t*);
EXTERN PetscErrorCode PetscScalarAddressFromFortran(PetscObject,PetscScalar*,size_t,PetscInt,PetscScalar **);
EXTERN size_t         PetscIntAddressToFortran(PetscInt*,PetscInt*);
EXTERN PetscInt       *PetscIntAddressFromFortran(PetscInt*,size_t); 
extern void   *PETSC_NULL_Fortran;
extern char   *PETSC_NULL_CHARACTER_Fortran;
extern void   *PETSC_NULL_INTEGER_Fortran;
extern void   *PETSC_NULL_SCALAR_Fortran;
extern void   *PETSC_NULL_DOUBLE_Fortran;
extern void   *PETSC_NULL_REAL_Fortran;
extern void   *PETSC_NULL_OBJECT_Fortran;
extern void   *PETSC_NULL_TRUTH_Fortran;
EXTERN_C_BEGIN
extern void   (*PETSC_NULL_FUNCTION_Fortran)(void);
EXTERN_C_END
/*  ----------------------------------------------------------------------*/
/*
   PETSc object C pointers are stored directly as
   Fortran integer*4 or *8 depending on the size of pointers.
*/


/* --------------------------------------------------------------------*/
/*
    This lets us map the str-len argument either, immediately following
    the char argument (DVF on Win32) or at the end of the argument list
    (general unix compilers)
*/
#if defined(PETSC_HAVE_FORTRAN_MIXED_STR_ARG)
#define PETSC_MIXED_LEN(len) ,int len
#define PETSC_END_LEN(len)
#else
#define PETSC_MIXED_LEN(len)
#define PETSC_END_LEN(len)   ,int len
#endif

/* --------------------------------------------------------------------*/
#define CHAR char*
#define FIXCHAR(a,n,b) \
{\
  if (a == PETSC_NULL_CHARACTER_Fortran) { \
    b = a = 0; \
  } else { \
    while((n > 0) && (a[n-1] == ' ')) n--; \
    *ierr = PetscMalloc((n+1)*sizeof(char),&b); \
    if(*ierr) return; \
    *ierr = PetscStrncpy(b,a,n); \
    if(*ierr) return; \
    b[n] = 0; \
  } \
}

#define FREECHAR(a,b) if (a != b) PetscFreeVoid(b);

#define FIXRETURNCHAR(flg,a,n)			\
if (flg) {					\
  int __i; \
  for (__i=0; __i<n && a[__i] != 0; __i++) ; \
  for (; __i<n; __i++) a[__i] = ' ' ; \
}

#define FORTRANNULL(a)         (((void*)a) == PETSC_NULL_Fortran)
#define FORTRANNULLINTEGER(a)  (((void*)a) == PETSC_NULL_INTEGER_Fortran)
#define FORTRANNULLSCALAR(a)   (((void*)a) == PETSC_NULL_SCALAR_Fortran)
#define FORTRANNULLDOUBLE(a)   (((void*)a) == PETSC_NULL_DOUBLE_Fortran)
#define FORTRANNULLREAL(a)     (((void*)a) == PETSC_NULL_REAL_Fortran)
#define FORTRANNULLOBJECT(a)   (((void*)a) == PETSC_NULL_OBJECT_Fortran)
#define FORTRANNULLTRUTH(a)    (((void*)a) == PETSC_NULL_TRUTH_Fortran)
#define FORTRANNULLFUNCTION(a) (((void(*)(void))a) == PETSC_NULL_FUNCTION_Fortran)


#define CHKFORTRANNULLINTEGER(a)  \
  if (FORTRANNULL(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLOBJECT(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(__LINE__,"fortran_interface_unknown_file",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,1, \
    "Use PETSC_NULL_INTEGER"); *ierr = 1; return; } \
  else if (FORTRANNULLINTEGER(a)) { a = PETSC_NULL; }

#define CHKFORTRANNULLSCALAR(a)  \
  if (FORTRANNULL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLOBJECT(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(__LINE__,"fortran_interface_unknown_file",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,1, \
    "Use PETSC_NULL_SCALAR"); *ierr = 1; return; } \
  else if (FORTRANNULLSCALAR(a)) { a = PETSC_NULL; }

#define CHKFORTRANNULLDOUBLE(a)  \
  if (FORTRANNULL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLOBJECT(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(__LINE__,"fortran_interface_unknown_file",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,1, \
    "Use PETSC_NULL_DOUBLE"); *ierr = 1; return; } \
  else if (FORTRANNULLDOUBLE(a)) { a = PETSC_NULL; }

#define CHKFORTRANNULLREAL(a)  \
  if (FORTRANNULL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLOBJECT(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(__LINE__,"fortran_interface_unknown_file",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,1, \
    "Use PETSC_NULL_REAL"); *ierr = 1; return; } \
  else if (FORTRANNULLREAL(a)) { a = PETSC_NULL; }

#define CHKFORTRANNULLOBJECT(a)  \
  if (FORTRANNULL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(__LINE__,"fortran_interface_unknown_file",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,1, \
    "Use PETSC_NULL_OBJECT"); *ierr = 1; return; } \
  else if (FORTRANNULLOBJECT(a)) { a = PETSC_NULL; }

extern void *PETSCNULLPOINTERADDRESS;

#define CHKFORTRANNULLOBJECTDEREFERENCE(a)  \
  if (FORTRANNULL(a) || FORTRANNULLSCALAR(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(__LINE__,"fortran_interface_unknown_file",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,1, \
    "Use PETSC_NULL_OBJECT"); *ierr = 1; return; } \
  else if (FORTRANNULLOBJECT(a)) { *((void***)&a) = &PETSCNULLPOINTERADDRESS; }

#define CHKFORTRANNULLFUNCTION(a)  \
  if (FORTRANNULL(a) || FORTRANNULLSCALAR(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLOBJECT(a)) { \
    PetscError(__LINE__,"fortran_interface_unknown_file",__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,1, \
    "Use PETSC_NULL_FUNCTION"); *ierr = 1; return; } \
  else if (FORTRANNULLFUNCTION(a)) { a = PETSC_NULL; }
  
/*
    These are used to support the default viewers that are 
  created at run time, in C using the , trick.

    The numbers here must match the numbers in include/finclude/petscsys.h
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
#define PETSC_VIEWER_MATLAB_WORLD_FORTRAN   -14
#define PETSC_VIEWER_MATLAB_SELF_FORTRAN    -15

#if defined (PETSC_USE_SOCKET_VIEWER)
#define PetscPatchDefaultViewers_Fortran_Socket(vin,v) \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_SOCKET_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_SOCKET_WORLD; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_SOCKET_SELF_FORTRAN) { \
      v = PETSC_VIEWER_SOCKET_SELF
#else
#define PetscPatchDefaultViewers_Fortran_Socket(vin,v)
#endif

#define PetscPatchDefaultViewers_Fortran(vin,v) \
{ \
    if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_DRAW_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_DRAW_WORLD; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_DRAW_SELF_FORTRAN) { \
      v = PETSC_VIEWER_DRAW_SELF; \
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
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_MATLAB_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_BINARY_WORLD; \
    } else if ((*(PetscFortranAddr*)vin) == PETSC_VIEWER_MATLAB_SELF_FORTRAN) { \
      v = PETSC_VIEWER_BINARY_SELF; \
    PetscPatchDefaultViewers_Fortran_Socket(vin,v); \
    } else { \
      v = *vin; \
    } \
}

/*
      Allocates enough space to store Fortran function pointers in PETSc object
   that are needed by the Fortran interface.
*/
#define PetscObjectAllocateFortranPointers(obj,N)	\
  if (!((PetscObject)(obj))->fortran_func_pointers) { \
    *ierr = PetscMalloc(N*sizeof(void*),&((PetscObject)(obj))->fortran_func_pointers);if (*ierr) return; \
  }
