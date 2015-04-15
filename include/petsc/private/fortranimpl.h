
/* This file contains info for the use of PETSc Fortran interface stubs */
#if !defined(_FORTRANIMPL_H)
#define _FORTRANIMPL_H

#include <petsc/private/petscimpl.h>

/* PETSC_STDCALL is defined on some Microsoft Windows systems and is used for functions compiled by the Fortran compiler */
#if !defined(PETSC_STDCALL)
#define PETSC_STDCALL
#endif

PETSC_EXTERN PetscErrorCode PetscScalarAddressToFortran(PetscObject,PetscInt,PetscScalar*,PetscScalar*,PetscInt,size_t*);
PETSC_EXTERN PetscErrorCode PetscScalarAddressFromFortran(PetscObject,PetscScalar*,size_t,PetscInt,PetscScalar **);
PETSC_EXTERN size_t         PetscIntAddressToFortran(const PetscInt*,const PetscInt*);
PETSC_EXTERN PetscInt        *PetscIntAddressFromFortran(const PetscInt*,size_t);
PETSC_EXTERN char   *PETSC_NULL_CHARACTER_Fortran;
PETSC_EXTERN void    *PETSC_NULL_INTEGER_Fortran;
PETSC_EXTERN void    *PETSC_NULL_SCALAR_Fortran;
PETSC_EXTERN void    *PETSC_NULL_DOUBLE_Fortran;
PETSC_EXTERN void    *PETSC_NULL_REAL_Fortran;
PETSC_EXTERN void    *PETSC_NULL_OBJECT_Fortran;
PETSC_EXTERN void    *PETSC_NULL_BOOL_Fortran;
PETSC_EXTERN void (*PETSC_NULL_FUNCTION_Fortran)(void);
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
#define PETSC_MIXED_LEN_CALL(len) ,len
#define PETSC_END_LEN_CALL(len)
#define PETSC_MIXED_LEN_PROTO ,int
#define PETSC_END_LEN_PROTO
#else
#define PETSC_MIXED_LEN(len)
#define PETSC_END_LEN(len)   ,int len
#define PETSC_MIXED_LEN_CALL(len)
#define PETSC_END_LEN_CALL(len)   ,len
#define PETSC_MIXED_LEN_PROTO
#define PETSC_END_LEN_PROTO   ,int
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
    if (*ierr) return; \
    *ierr = PetscStrncpy(b,a,n+1); \
    if (*ierr) return; \
  } \
}

#define FREECHAR(a,b) if (a != b) PetscFreeVoid(b);

#define FIXRETURNCHAR(flg,a,n)               \
if (flg) {                                   \
  int __i;                                   \
  for (__i=0; __i<n && a[__i] != 0; __i++) ; \
  for (; __i<n; __i++) a[__i] = ' ' ; \
}

/*
    The cast through PETSC_UINTPTR_T is so that compilers that warn about casting to/from void * to void(*)(void)
    will not complain about these comparisons. It is not know if this works for all compilers
*/
#define FORTRANNULLINTEGER(a)  (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_INTEGER_Fortran)
#define FORTRANNULLSCALAR(a)   (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_SCALAR_Fortran)
#define FORTRANNULLDOUBLE(a)   (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_DOUBLE_Fortran)
#define FORTRANNULLREAL(a)     (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_REAL_Fortran)
#define FORTRANNULLOBJECT(a)   (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_OBJECT_Fortran)
#define FORTRANNULLBOOL(a)    (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_BOOL_Fortran)
#define FORTRANNULLFUNCTION(a) (((void(*)(void))(PETSC_UINTPTR_T)a) == PETSC_NULL_FUNCTION_Fortran)


#define CHKFORTRANNULLINTEGER(a)  \
  if (FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLOBJECT(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_INTEGER"); *ierr = 1; return; } \
  else if (FORTRANNULLINTEGER(a)) { a = NULL; }

#define CHKFORTRANNULLSCALAR(a)   \
  if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLOBJECT(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_SCALAR"); *ierr = 1; return; } \
  else if (FORTRANNULLSCALAR(a)) { a = NULL; }

#define CHKFORTRANNULLDOUBLE(a)  \
  if (FORTRANNULLINTEGER(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLOBJECT(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_DOUBLE"); *ierr = 1; return; } \
  else if (FORTRANNULLDOUBLE(a)) { a = NULL; }

#define CHKFORTRANNULLREAL(a)  \
  if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLOBJECT(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_REAL"); *ierr = 1; return; } \
  else if (FORTRANNULLREAL(a)) { a = NULL; }

#define CHKFORTRANNULLOBJECT(a)  \
  if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_OBJECT"); *ierr = 1; return; } \
  else if (FORTRANNULLOBJECT(a)) { a = NULL; }

PETSC_EXTERN void  *PETSCNULLPOINTERADDRESS;

#define CHKFORTRANNULLOBJECTDEREFERENCE(a)  \
  if (FORTRANNULLSCALAR(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLFUNCTION(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_OBJECT"); *ierr = 1; return; } \
  else if (FORTRANNULLOBJECT(a)) { *((void***)&a) = &PETSCNULLPOINTERADDRESS; }

#define CHKFORTRANNULLFUNCTION(a)  \
  if (FORTRANNULLSCALAR(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLOBJECT(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_FUNCTION"); *ierr = 1; return; } \
  else if (FORTRANNULLFUNCTION(a)) { a = NULL; }



/*
    Variable type where we stash PETSc object pointers in Fortran.
*/
typedef PETSC_UINTPTR_T PetscFortranAddr;

/*
    These are used to support the default viewers that are
  created at run time, in C using the , trick.

    The numbers here must match the numbers in include/petsc/finclude/petscsys.h
*/
#define PETSC_VIEWER_DRAW_WORLD_FORTRAN     4
#define PETSC_VIEWER_DRAW_SELF_FORTRAN      5
#define PETSC_VIEWER_SOCKET_WORLD_FORTRAN   6
#define PETSC_VIEWER_SOCKET_SELF_FORTRAN    7
#define PETSC_VIEWER_STDOUT_WORLD_FORTRAN   8
#define PETSC_VIEWER_STDOUT_SELF_FORTRAN    9
#define PETSC_VIEWER_STDERR_WORLD_FORTRAN   10
#define PETSC_VIEWER_STDERR_SELF_FORTRAN    11
#define PETSC_VIEWER_BINARY_WORLD_FORTRAN   12
#define PETSC_VIEWER_BINARY_SELF_FORTRAN    13
#define PETSC_VIEWER_MATLAB_WORLD_FORTRAN   14
#define PETSC_VIEWER_MATLAB_SELF_FORTRAN    15

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
#define PetscObjectAllocateFortranPointers(obj,N) do {                  \
    if (!((PetscObject)(obj))->fortran_func_pointers) {                 \
      *ierr = PetscMalloc((N)*sizeof(void(*)(void)),&((PetscObject)(obj))->fortran_func_pointers);if (*ierr) return; \
      *ierr = PetscMemzero(((PetscObject)(obj))->fortran_func_pointers,(N)*sizeof(void(*)(void)));if (*ierr) return; \
      ((PetscObject)obj)->num_fortran_func_pointers = (N);              \
    }                                                                   \
  } while (0)

/* Entire function body, _ctx is a "special" variable that can be passed along */
#define PetscObjectUseFortranCallback_Private(obj,cid,types,args,cbclass) { \
    PetscErrorCode ierr;                                                \
    void (PETSC_STDCALL *func) types,*_ctx;                             \
    PetscFunctionBegin;                                                 \
    ierr = PetscObjectGetFortranCallback((PetscObject)(obj),(cbclass),(cid),(PetscVoidFunction*)&func,&_ctx);CHKERRQ(ierr); \
    (*func)args;CHKERRQ(ierr);                                          \
    PetscFunctionReturn(0);                                             \
  }
#define PetscObjectUseFortranCallback(obj,cid,types,args) PetscObjectUseFortranCallback_Private(obj,cid,types,args,PETSC_FORTRAN_CALLBACK_CLASS)
#define PetscObjectUseFortranCallbackSubType(obj,cid,types,args) PetscObjectUseFortranCallback_Private(obj,cid,types,args,PETSC_FORTRAN_CALLBACK_SUBTYPE)


#endif
