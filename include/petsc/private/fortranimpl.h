
/* This file contains info for the use of PETSc Fortran interface stubs */
#if !defined(PETSCFORTRANIMPL_H)
#define PETSCFORTRANIMPL_H

#include <petsc/private/petscimpl.h>
PETSC_INTERN PetscErrorCode PETScParseFortranArgs_Private(int*,char***);
PETSC_EXTERN PetscErrorCode PetscMPIFortranDatatypeToC(MPI_Fint,MPI_Datatype*);

PETSC_EXTERN PetscErrorCode PetscScalarAddressToFortran(PetscObject,PetscInt,PetscScalar*,PetscScalar*,PetscInt,size_t*);
PETSC_EXTERN PetscErrorCode PetscScalarAddressFromFortran(PetscObject,PetscScalar*,size_t,PetscInt,PetscScalar **);
PETSC_EXTERN size_t         PetscIntAddressToFortran(const PetscInt*,const PetscInt*);
PETSC_EXTERN PetscInt      *PetscIntAddressFromFortran(const PetscInt*,size_t);
PETSC_EXTERN char    *PETSC_NULL_CHARACTER_Fortran;
PETSC_EXTERN void    *PETSC_NULL_INTEGER_Fortran;
PETSC_EXTERN void    *PETSC_NULL_SCALAR_Fortran;
PETSC_EXTERN void    *PETSC_NULL_DOUBLE_Fortran;
PETSC_EXTERN void    *PETSC_NULL_REAL_Fortran;
PETSC_EXTERN void    *PETSC_NULL_BOOL_Fortran;
PETSC_EXTERN void   (*PETSC_NULL_FUNCTION_Fortran)(void);
PETSC_EXTERN void    *PETSC_NULL_MPI_COMM_Fortran;

PETSC_INTERN PetscErrorCode PetscInitFortran_Private(PetscBool,const char*,PetscInt);

/*  ----------------------------------------------------------------------*/
/*
   PETSc object C pointers are stored directly as
   Fortran integer*4 or *8 depending on the size of pointers.
*/

/* --------------------------------------------------------------------*/
/*
    Since Fortran does not null terminate strings we need to insure the string is null terminated before passing it
    to C. This may require a memory allocation which is then freed with FREECHAR().
*/
#define FIXCHAR(a,n,b) \
{\
  if (a == PETSC_NULL_CHARACTER_Fortran) { \
    b = a = NULL; \
  } else { \
    while ((n > 0) && (a[n-1] == ' ')) n--; \
    *ierr = PetscMalloc1(n+1,&b); \
    if (*ierr) return; \
    *ierr = PetscStrncpy(b,a,n+1); \
    if (*ierr) return; \
  } \
}
#define FREECHAR(a,b) if (a != b) *ierr = PetscFree(b);

/*
    Fortran expects any unneeded characters at the end of its strings to be filled with the blank character.
*/
#define FIXRETURNCHAR(flg,a,n)               \
if (flg) {                                   \
  PETSC_FORTRAN_CHARLEN_T __i;               \
  for (__i=0; __i<n && a[__i] != 0; __i++) {};  \
  for (; __i<n; __i++) a[__i] = ' ' ; \
}

/*
    The cast through PETSC_UINTPTR_T is so that compilers that warn about casting to/from void * to void(*)(void)
    will not complain about these comparisons. It is not know if this works for all compilers
*/
#define FORTRANNULLINTEGER(a)   (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_INTEGER_Fortran)
#define FORTRANNULLSCALAR(a)    (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_SCALAR_Fortran)
#define FORTRANNULLDOUBLE(a)    (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_DOUBLE_Fortran)
#define FORTRANNULLREAL(a)      (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_REAL_Fortran)
#define FORTRANNULLBOOL(a)      (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_BOOL_Fortran)
#define FORTRANNULLCHARACTER(a) (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_CHARACTER_Fortran)
#define FORTRANNULLFUNCTION(a)  (((void(*)(void))(PETSC_UINTPTR_T)a) == PETSC_NULL_FUNCTION_Fortran)
#define FORTRANNULLOBJECT(a)    (*(void**)(PETSC_UINTPTR_T)a == (void*)0)
#define FORTRANNULLMPICOMM(a)   (((void*)(PETSC_UINTPTR_T)a) == PETSC_NULL_MPI_COMM_Fortran)

#define CHKFORTRANNULLINTEGER(a)  \
  if (FORTRANNULLINTEGER(a)) { a = NULL; } \
  else if (FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a)  || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_INTEGER"); *ierr = 1; return; }

#define CHKFORTRANNULLSCALAR(a)   \
  if (FORTRANNULLSCALAR(a)) { a = NULL; } \
  else if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a)  || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_SCALAR"); *ierr = 1; return; }

#define CHKFORTRANNULLDOUBLE(a)  \
  if (FORTRANNULLDOUBLE(a)) { a = NULL; } \
  else if (FORTRANNULLINTEGER(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a)  || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_DOUBLE"); *ierr = 1; return; }

#define CHKFORTRANNULLREAL(a)  \
  if (FORTRANNULLREAL(a)) { a = NULL; } \
  else if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a)  || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_REAL"); *ierr = 1; return; }

#define CHKFORTRANNULLOBJECT(a)  \
  if (*(void**)a == (void*)0) { a = NULL; } \
  else if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_XXX where XXX is the name of a particular object class"); *ierr = 1; return; }

#define CHKFORTRANNULLBOOL(a)  \
  if (FORTRANNULLBOOL(a)) { a = NULL; } \
  else if (FORTRANNULLSCALAR(a) || FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a)  || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_BOOL"); *ierr = 1; return; }

#define CHKFORTRANNULLFUNCTION(a)  \
  if (FORTRANNULLFUNCTION(a)) { a = NULL; } \
  else if (FORTRANNULLOBJECT(a) || FORTRANNULLSCALAR(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLBOOL(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_FUNCTION"); *ierr = 1; return; }

#define CHKFORTRANNULLMPICOMM(a)  \
  if (FORTRANNULLMPICOMM(a)) { a = NULL; } \
  else if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a)  || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a)) { \
    PetscError(PETSC_COMM_SELF,__LINE__,"fortran_interface_unknown_file",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL, \
    "Use PETSC_NULL_MPI_COMM"); *ierr = 1; return; }

/* The two macros are used at the beginning and end of PETSc object Fortran destroy routines XxxDestroy(). -2 is in consistent with
   the one used in checkFortranTypeInitialize() at compilersFortran.py.
 */

/* In the beginning of Fortran XxxDestroy(a), if the input object was destroyed, change it to a petsc C NULL object so that it won't crash C XxxDestory() */
#define PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(a) do {if (*((void**)(a)) == (void*)-2) *(a) = NULL;} while (0)

/* After C XxxDestroy(a) is called, change a's state from NULL to destroyed, so that it can be used/destroyed again by Fortran.
   E.g., in VecScatterCreateToAll(x,vscat,seq,ierr), if seq = PETSC_NULL_VEC, petsc won't create seq. But if seq is a
   destroyed object (e.g., as a result of a previous Fortran VecDestroy), petsc will create seq.
*/
#define PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(a) do {*((void**)(a)) = (void*)-2;} while (0)

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
      *ierr = PetscCalloc((N)*sizeof(void(*)(void)),&((PetscObject)(obj))->fortran_func_pointers);if (*ierr) return; \
      ((PetscObject)obj)->num_fortran_func_pointers = (N);              \
    }                                                                   \
  } while (0)

#define PetscCallFortranVoidFunction(...) do {          \
    PetscErrorCode ierr = 0;                            \
    /* the function may or may not access ierr */       \
    __VA_ARGS__;                                        \
    PetscCall(ierr);                                    \
  } while (0)

/* Entire function body, _ctx is a "special" variable that can be passed along */
#define PetscObjectUseFortranCallback_Private(obj,cid,types,args,cbclass) {                    \
    void (*func) types,*_ctx;                                                                  \
    PetscFunctionBegin;                                                                        \
    PetscCall(PetscObjectGetFortranCallback((PetscObject)(obj),(cbclass),(cid),(PetscVoidFunction*)&func,&_ctx)); \
    if (func) PetscCallFortranVoidFunction((*func)args);                                       \
    PetscFunctionReturn(0);                                                                    \
  }
#define PetscObjectUseFortranCallback(obj,cid,types,args) PetscObjectUseFortranCallback_Private(obj,cid,types,args,PETSC_FORTRAN_CALLBACK_CLASS)
#define PetscObjectUseFortranCallbackSubType(obj,cid,types,args) PetscObjectUseFortranCallback_Private(obj,cid,types,args,PETSC_FORTRAN_CALLBACK_SUBTYPE)

/* Disable deprecation warnings while building Fortran wrappers */
#undef  PETSC_DEPRECATED_FUNCTION
#define PETSC_DEPRECATED_FUNCTION(arg)

#endif
