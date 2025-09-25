#pragma once

#include <petsc/private/petscimpl.h>
PETSC_INTERN PetscErrorCode PETScParseFortranArgs_Private(int *, char ***);
PETSC_EXTERN PetscErrorCode PetscMPIFortranDatatypeToC(MPI_Fint, MPI_Datatype *);

PETSC_EXTERN PetscErrorCode          PetscScalarAddressToFortran(PetscObject, PetscInt, PetscScalar *, PetscScalar *, PetscInt, size_t *);
PETSC_EXTERN PetscErrorCode          PetscScalarAddressFromFortran(PetscObject, PetscScalar *, size_t, PetscInt, PetscScalar **);
PETSC_EXTERN size_t                  PetscIntAddressToFortran(const PetscInt *, const PetscInt *);
PETSC_EXTERN PetscInt               *PetscIntAddressFromFortran(const PetscInt *, size_t);
PETSC_EXTERN char                   *PETSC_NULL_CHARACTER_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_INTEGER_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_SCALAR_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_DOUBLE_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_REAL_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_BOOL_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_ENUM_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_INTEGER_ARRAY_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_SCALAR_ARRAY_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_REAL_ARRAY_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_MPI_COMM_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_INTEGER_POINTER_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_SCALAR_POINTER_Fortran;
PETSC_EXTERN void                   *PETSC_NULL_REAL_POINTER_Fortran;
PETSC_EXTERN PetscFortranCallbackFn *PETSC_NULL_FUNCTION_Fortran;

PETSC_INTERN PetscErrorCode PetscInitFortran_Private(const char *, PetscInt);

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
#define FIXCHAR(a, n, b) \
  do { \
    if ((a) == PETSC_NULL_CHARACTER_Fortran) { \
      (b) = PETSC_NULLPTR; \
      (a) = PETSC_NULLPTR; \
    } else { \
      while (((n) > 0) && ((a)[(n) - 1] == ' ')) (n)--; \
      *ierr = PetscMalloc1((n) + 1, &(b)); \
      if (*ierr) return; \
      *ierr  = PetscMemcpy((b), (a), (n)); \
      (b)[n] = '\0'; \
      if (*ierr) return; \
    } \
  } while (0)
#define FREECHAR(a, b) \
  do { \
    if (a != b) *ierr = PetscFree(b); \
  } while (0)

/*
    Fortran expects any unneeded characters at the end of its strings to be filled with the blank character.
*/
#define FIXRETURNCHAR(flg, a, n) \
  do { \
    if (flg) { \
      PETSC_FORTRAN_CHARLEN_T __i; \
      for (__i = 0; __i < n && a[__i] != 0; __i++) { }; \
      for (; __i < n; __i++) a[__i] = ' '; \
    } \
  } while (0)

/*
    The cast through PETSC_UINTPTR_T is so that compilers that warn about casting to/from void * to void(*)(void)
    will not complain about these comparisons. It is not know if this works for all compilers
*/
#define FORTRANNULLINTEGERPOINTER(a) (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_INTEGER_POINTER_Fortran)
#define FORTRANNULLSCALARPOINTER(a)  (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_SCALAR_POINTER_Fortran)
#define FORTRANNULLREALPOINTER(a)    (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_REAL_POINTER_Fortran)
#define FORTRANNULLINTEGER(a)        (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_INTEGER_Fortran || ((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_INTEGER_ARRAY_Fortran)
#define FORTRANNULLSCALAR(a)         (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_SCALAR_Fortran || ((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_SCALAR_ARRAY_Fortran)
#define FORTRANNULLREAL(a)           (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_REAL_Fortran || ((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_REAL_ARRAY_Fortran)
#define FORTRANNULLDOUBLE(a)         (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_DOUBLE_Fortran)
#define FORTRANNULLBOOL(a)           (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_BOOL_Fortran)
#define FORTRANNULLENUM(a)           ((((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_ENUM_Fortran) || (((void *)(PETSC_UINTPTR_T)a) == (void *)-50))
#define FORTRANNULLCHARACTER(a)      (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_CHARACTER_Fortran)
#define FORTRANNULLFUNCTION(a)       (((PetscFortranCallbackFn *)(PETSC_UINTPTR_T)a) == PETSC_NULL_FUNCTION_Fortran)
#define FORTRANNULLOBJECT(a)         (*(void **)(PETSC_UINTPTR_T)a == (void *)0)
#define FORTRANNULLMPICOMM(a)        (((void *)(PETSC_UINTPTR_T)a) == PETSC_NULL_MPI_COMM_Fortran)

/*
    A Fortran object with a value of (void*) 0 is indicated in Fortran by PETSC_NULL_XXXX, it is passed to routines to indicate the argument value is not requested or provided
    similar to how NULL is used with PETSc objects in C

    A Fortran object with a value of (void*) PETSC_FORTRAN_TYPE_INITIALIZE is an object that was never created or was destroyed (see checkFortranTypeInitialize()).

    A Fortran object with a value of (void*) PETSC_FORTRAN_TYPE_NULL_RETURN happens when a PETSc routine returns in one of its arguments a NULL object
    (it cannot return a value of (void*) PETSC_FORTRAN_TYPE_NULL because if later the returned variable is passed to a creation routine, it would think one has passed in a PETSC_NULL_XXX and error).

    These three values are used because Fortran always uses pass by reference so one cannot pass a NULL address, only an address with special
    values at the location.

    PETSC_FORTRAN_TYPE_INITIALIZE  is also defined in include/petsc/finclude/petscsysbase.h
*/
#define PETSC_FORTRAN_TYPE_INITIALIZE  (void *)-2
#define PETSC_FORTRAN_TYPE_NULL_RETURN (void *)-3

#define CHKFORTRANNULL(a) \
  do { \
    if (FORTRANNULLINTEGER(a) || FORTRANNULLENUM(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) a = PETSC_NULLPTR; \
  } while (0)

#define CHKFORTRANNULLENUM(a) \
  do { \
    if (FORTRANNULLENUM(a)) a = PETSC_NULLPTR; \
  } while (0)

#define CHKFORTRANNULLINTEGER(a) \
  do { \
    if (FORTRANNULLINTEGER(a) || FORTRANNULLENUM(a)) a = PETSC_NULLPTR; \
    else if (FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_INTEGER"); \
      *ierr = PETSC_ERR_ARG_BADPTR; \
      return; \
    } \
  } while (0)

#define CHKFORTRANNULLSCALAR(a) \
  do { \
    if (FORTRANNULLSCALAR(a)) { \
      a = PETSC_NULLPTR; \
    } else if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_SCALAR"); \
      *ierr = PETSC_ERR_ARG_BADPTR; \
      return; \
    } \
  } while (0)

#define CHKFORTRANNULLDOUBLE(a) \
  do { \
    if (FORTRANNULLDOUBLE(a)) { \
      a = PETSC_NULLPTR; \
    } else if (FORTRANNULLINTEGER(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_DOUBLE"); \
      *ierr = PETSC_ERR_ARG_BADPTR; \
      return; \
    } \
  } while (0)

#define CHKFORTRANNULLREAL(a) \
  do { \
    if (FORTRANNULLREAL(a)) { \
      a = PETSC_NULLPTR; \
    } else if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_REAL"); \
      *ierr = PETSC_ERR_ARG_BADPTR; \
      return; \
    } \
  } while (0)

#define CHKFORTRANNULLOBJECT(a) \
  do { \
    if (!(*(void **)a)) { \
      a = PETSC_NULLPTR; \
    } else if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_XXX where XXX is the name of a particular object class"); \
      *ierr = PETSC_ERR_ARG_BADPTR; \
      return; \
    } \
  } while (0)

#define CHKFORTRANNULLBOOL(a) \
  do { \
    if (FORTRANNULLBOOL(a)) { \
      a = PETSC_NULLPTR; \
    } else if (FORTRANNULLSCALAR(a) || FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_BOOL"); \
      *ierr = PETSC_ERR_ARG_BADPTR; \
      return; \
    } \
  } while (0)

#define CHKFORTRANNULLFUNCTION(a) \
  do { \
    if (FORTRANNULLFUNCTION(a)) { \
      a = PETSC_NULLPTR; \
    } else if (FORTRANNULLOBJECT(a) || FORTRANNULLSCALAR(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLREAL(a) || FORTRANNULLINTEGER(a) || FORTRANNULLBOOL(a) || FORTRANNULLCHARACTER(a) || FORTRANNULLMPICOMM(a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_FUNCTION"); \
      *ierr = PETSC_ERR_ARG_BADPTR; \
      return; \
    } \
  } while (0)

#define CHKFORTRANNULLMPICOMM(a) \
  do { \
    if (FORTRANNULLMPICOMM(a)) { \
      a = PETSC_NULLPTR; \
    } else if (FORTRANNULLINTEGER(a) || FORTRANNULLDOUBLE(a) || FORTRANNULLSCALAR(a) || FORTRANNULLREAL(a) || FORTRANNULLBOOL(a) || FORTRANNULLFUNCTION(a) || FORTRANNULLCHARACTER(a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Use PETSC_NULL_MPI_COMM"); \
      *ierr = PETSC_ERR_ARG_BADPTR; \
      return; \
    } \
  } while (0)

/* In the beginning of Fortran XxxCreate() ensure object is not NULL or already created */
#define PETSC_FORTRAN_OBJECT_CREATE(a) \
  do { \
    if (!(*(void **)a)) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Cannot create PETSC_NULL_XXX object"); \
      *ierr = PETSC_ERR_ARG_WRONG; \
      return; \
    } else if (*((void **)(a)) != PETSC_FORTRAN_TYPE_INITIALIZE && *((void **)(a)) != PETSC_FORTRAN_TYPE_NULL_RETURN) { \
      *ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_WRONG, PETSC_ERROR_INITIAL, "Cannot create already existing object"); \
      *ierr = PETSC_ERR_ARG_WRONG; \
      return; \
    } \
  } while (0)

/*
  In the beginning of Fortran XxxDestroy(a), if the input object was destroyed, change it to a PETSc C NULL object so that it won't crash C XxxDestory()
  If it is PETSC_NULL_XXX just return since these objects cannot be destroyed
*/
#define PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(a) \
  do { \
    if (!*(void **)a || *((void **)(a)) == PETSC_FORTRAN_TYPE_INITIALIZE || *((void **)(a)) == PETSC_FORTRAN_TYPE_NULL_RETURN) { \
      *ierr = PETSC_SUCCESS; \
      return; \
    } \
  } while (0)

/* After C XxxDestroy(a) is called, change a's state from NULL to destroyed, so that it can be used/destroyed again by Fortran.
   E.g., in VecScatterCreateToAll(x,vscat,seq,ierr), if seq = PETSC_NULL_VEC, PETSc won't create seq. But if seq is a
   destroyed object (e.g., as a result of a previous Fortran VecDestroy), PETSc will create seq.
*/
#define PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(a) \
  do { \
    *((void **)(a)) = PETSC_FORTRAN_TYPE_INITIALIZE; \
  } while (0)

/*
    Variable type where we stash PETSc object pointers in Fortran.
*/
typedef PETSC_UINTPTR_T PetscFortranAddr;

/*
    These are used to support the default viewers that are
  created at run time, in C using the , trick.

    The numbers here must match the numbers in include/petsc/finclude/petscsys.h
*/
#define PETSC_VIEWER_DRAW_WORLD_FORTRAN   4
#define PETSC_VIEWER_DRAW_SELF_FORTRAN    5
#define PETSC_VIEWER_SOCKET_WORLD_FORTRAN 6
#define PETSC_VIEWER_SOCKET_SELF_FORTRAN  7
#define PETSC_VIEWER_STDOUT_WORLD_FORTRAN 8
#define PETSC_VIEWER_STDOUT_SELF_FORTRAN  9
#define PETSC_VIEWER_STDERR_WORLD_FORTRAN 10
#define PETSC_VIEWER_STDERR_SELF_FORTRAN  11
#define PETSC_VIEWER_BINARY_WORLD_FORTRAN 12
#define PETSC_VIEWER_BINARY_SELF_FORTRAN  13
#define PETSC_VIEWER_MATLAB_WORLD_FORTRAN 14
#define PETSC_VIEWER_MATLAB_SELF_FORTRAN  15

#include <petscviewer.h>

static inline PetscViewer PetscPatchDefaultViewers(PetscViewer *v)
{
  if (!v) return PETSC_NULLPTR;
  if (!(*(void **)v)) return PETSC_NULLPTR;
  switch (*(PetscFortranAddr *)v) {
  case PETSC_VIEWER_DRAW_WORLD_FORTRAN:
    return PETSC_VIEWER_DRAW_WORLD;
  case PETSC_VIEWER_DRAW_SELF_FORTRAN:
    return PETSC_VIEWER_DRAW_SELF;

  case PETSC_VIEWER_STDOUT_WORLD_FORTRAN:
    return PETSC_VIEWER_STDOUT_WORLD;
  case PETSC_VIEWER_STDOUT_SELF_FORTRAN:
    return PETSC_VIEWER_STDOUT_SELF;

  case PETSC_VIEWER_STDERR_WORLD_FORTRAN:
    return PETSC_VIEWER_STDERR_WORLD;
  case PETSC_VIEWER_STDERR_SELF_FORTRAN:
    return PETSC_VIEWER_STDERR_SELF;

  case PETSC_VIEWER_BINARY_WORLD_FORTRAN:
    return PETSC_VIEWER_BINARY_WORLD;
  case PETSC_VIEWER_BINARY_SELF_FORTRAN:
    return PETSC_VIEWER_BINARY_SELF;

#if defined(PETSC_HAVE_MATLAB)
  case PETSC_VIEWER_MATLAB_SELF_FORTRAN:
    return PETSC_VIEWER_MATLAB_SELF;
  case PETSC_VIEWER_MATLAB_WORLD_FORTRAN:
    return PETSC_VIEWER_MATLAB_WORLD;
#endif

#if defined(PETSC_USE_SOCKET_VIEWER)
  case PETSC_VIEWER_SOCKET_WORLD_FORTRAN:
    return PETSC_VIEWER_SOCKET_WORLD;
  case PETSC_VIEWER_SOCKET_SELF_FORTRAN:
    return PETSC_VIEWER_SOCKET_SELF;
#endif

  default:
    return *v;
  }
}

#if defined(PETSC_USE_SOCKET_VIEWER)
  #define PetscPatchDefaultViewers_Fortran_Socket(vin, v) \
    } \
    else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_SOCKET_WORLD_FORTRAN) \
    { \
      v = PETSC_VIEWER_SOCKET_WORLD; \
    } \
    else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_SOCKET_SELF_FORTRAN) \
    { \
      v = PETSC_VIEWER_SOCKET_SELF
#else
  #define PetscPatchDefaultViewers_Fortran_Socket(vin, v)
#endif

#define PetscPatchDefaultViewers_Fortran(vin, v) \
  do { \
    if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_DRAW_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_DRAW_WORLD; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_DRAW_SELF_FORTRAN) { \
      v = PETSC_VIEWER_DRAW_SELF; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_STDOUT_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_STDOUT_WORLD; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_STDOUT_SELF_FORTRAN) { \
      v = PETSC_VIEWER_STDOUT_SELF; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_STDERR_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_STDERR_WORLD; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_STDERR_SELF_FORTRAN) { \
      v = PETSC_VIEWER_STDERR_SELF; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_BINARY_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_BINARY_WORLD; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_BINARY_SELF_FORTRAN) { \
      v = PETSC_VIEWER_BINARY_SELF; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_MATLAB_WORLD_FORTRAN) { \
      v = PETSC_VIEWER_BINARY_WORLD; \
    } else if ((*(PetscFortranAddr *)vin) == PETSC_VIEWER_MATLAB_SELF_FORTRAN) { \
      v = PETSC_VIEWER_BINARY_SELF; \
      PetscPatchDefaultViewers_Fortran_Socket(vin, v); \
    } else { \
      v = *vin; \
    } \
  } while (0)

/*
      Allocates enough space to store Fortran function pointers in PETSc object
   that are needed by the Fortran interface.
*/
#define PetscObjectAllocateFortranPointers(obj, N) \
  do { \
    if (!((PetscObject)(obj))->fortran_func_pointers) { \
      *ierr = PetscCalloc((N) * sizeof(PetscFortranCallbackFn *), &((PetscObject)(obj))->fortran_func_pointers); \
      if (*ierr) return; \
      ((PetscObject)obj)->num_fortran_func_pointers = (N); \
    } \
  } while (0)

#define PetscCallFortranVoidFunction(...) \
  do { \
    PetscErrorCode ierr = PETSC_SUCCESS; \
    /* the function may or may not access ierr */ \
    __VA_ARGS__; \
    PetscCall(ierr); \
  } while (0)

/* Entire function body, _ctx is a "special" variable that can be passed along */
#define PetscObjectUseFortranCallback_Private(obj, cid, types, args, cbclass) \
  do { \
    void(*func) types, *_ctx; \
    PetscFunctionBegin; \
    PetscCall(PetscObjectGetFortranCallback((PetscObject)(obj), (cbclass), (cid), (PetscFortranCallbackFn **)&func, &_ctx)); \
    if (func) PetscCallFortranVoidFunction((*func)args); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } while (0)
#define PetscObjectUseFortranCallback(obj, cid, types, args)        PetscObjectUseFortranCallback_Private(obj, cid, types, args, PETSC_FORTRAN_CALLBACK_CLASS)
#define PetscObjectUseFortranCallbackSubType(obj, cid, types, args) PetscObjectUseFortranCallback_Private(obj, cid, types, args, PETSC_FORTRAN_CALLBACK_SUBTYPE)

/* Disable deprecation warnings while building Fortran wrappers */
#undef PETSC_DEPRECATED_OBJECT
#define PETSC_DEPRECATED_OBJECT(...)
#undef PETSC_DEPRECATED_FUNCTION
#define PETSC_DEPRECATED_FUNCTION(...)
#undef PETSC_DEPRECATED_ENUM
#define PETSC_DEPRECATED_ENUM(...)
#undef PETSC_DEPRECATED_TYPEDEF
#define PETSC_DEPRECATED_TYPEDEF(...)
#undef PETSC_DEPRECATED_MACRO
#define PETSC_DEPRECATED_MACRO(...)

/* PGI compilers pass in f90 pointers as 2 arguments */
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  #define PETSC_F90_2PTR_PROTO_NOVAR , void *
  #define PETSC_F90_2PTR_PROTO(ptr)  , void *ptr
  #define PETSC_F90_2PTR_PARAM(ptr)  , ptr
#else
  #define PETSC_F90_2PTR_PROTO_NOVAR
  #define PETSC_F90_2PTR_PROTO(ptr)
  #define PETSC_F90_2PTR_PARAM(ptr)
#endif

typedef struct {
  char dummy;
} F90Array1d;
typedef struct {
  char dummy;
} F90Array2d;
typedef struct {
  char dummy;
} F90Array3d;
typedef struct {
  char dummy;
} F90Array4d;

PETSC_EXTERN PetscErrorCode F90Array1dCreate(void *, MPI_Datatype, PetscInt, PetscInt, F90Array1d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array1dAccess(F90Array1d *, MPI_Datatype, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array1dDestroy(F90Array1d *, MPI_Datatype PETSC_F90_2PTR_PROTO_NOVAR);

PETSC_EXTERN PetscErrorCode F90Array2dCreate(void *, MPI_Datatype, PetscInt, PetscInt, PetscInt, PetscInt, F90Array2d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array2dAccess(F90Array2d *, MPI_Datatype, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array2dDestroy(F90Array2d *, MPI_Datatype PETSC_F90_2PTR_PROTO_NOVAR);

PETSC_EXTERN PetscErrorCode F90Array3dCreate(void *, MPI_Datatype, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, F90Array3d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array3dAccess(F90Array3d *, MPI_Datatype, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array3dDestroy(F90Array3d *, MPI_Datatype PETSC_F90_2PTR_PROTO_NOVAR);

PETSC_EXTERN PetscErrorCode F90Array4dCreate(void *, MPI_Datatype, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, F90Array4d *PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array4dAccess(F90Array4d *, MPI_Datatype, void **PETSC_F90_2PTR_PROTO_NOVAR);
PETSC_EXTERN PetscErrorCode F90Array4dDestroy(F90Array4d *, MPI_Datatype PETSC_F90_2PTR_PROTO_NOVAR);

/*
  F90Array1dCreate - Given a C pointer to a one dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
+   array - regular C pointer (address)
.   type  - DataType of the array
.   start - starting index of the array
-   len   - length of array (in items)

  Output Parameter:
.   ptr - Fortran 90 pointer
*/
