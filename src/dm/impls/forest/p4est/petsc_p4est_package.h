#if !defined(__PETSC_P4EST_PACKAGE_H)
#define      __PETSC_P4EST_PACKAGE_H
#include <petscsys.h>

#if defined(PETSC_HAVE_SETJMP_H) && defined(PETSC_USE_ERRORCHECKING)
#include <setjmp.h>
PETSC_INTERN jmp_buf PetscScJumpBuf;

#define PetscStackCallP4est(func,args) do {                                                                                                           \
  if (setjmp(PetscScJumpBuf)) {                                                                                                                       \
    return PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,PETSC_ERR_LIB,PETSC_ERROR_REPEAT,"Error in p4est/libsc call %s()",#func); \
  }                                                                                                                                                   \
  else {                                                                                                                                              \
    PetscStackPush(#func);                                                                                                                            \
    func args;                                                                                                                                        \
    PetscStackPop;                                                                                                                                    \
  }                                                                                                                                                   \
} while (0)
#define PetscStackCallP4estReturn(ret,func,args) do {                                                                                                 \
  if (setjmp(PetscScJumpBuf)) {                                                                                                                       \
    return PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,PETSC_ERR_LIB,PETSC_ERROR_REPEAT,"Error in p4est/libsc call %s()",#func); \
  }                                                                                                                                                   \
  else {                                                                                                                                              \
    PetscStackPush(#func);                                                                                                                            \
    ret = func args;                                                                                                                                  \
    PetscStackPop;                                                                                                                                    \
  }                                                                                                                                                   \
} while (0)
#else
#define PetscStackCallP4est(func,args) do {                                         \
  if (setjmp(PetscScJumpBuf)) {                                                     \
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in p4est/libsc call %s()",#func); \
  }                                                                                 \
  else {                                                                            \
    PetscStackPush(#func);                                                          \
    func args;                                                                      \
    PetscStackPop;                                                                  \
  }                                                                                 \
} while (0)
#define PetscStackCallP4estReturn(ret,func,args) do {                               \
  if (setjmp(PetscScJumpBuf)) {                                                     \
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in p4est/libsc call %s()",#func); \
  }                                                                                 \
  else {                                                                            \
    PetscStackPush(#func);                                                          \
    ret = func args;                                                                \
    PetscStackPop;                                                                  \
  }                                                                                 \
} while (0)
#endif

PETSC_EXTERN PetscErrorCode PetscP4estInitialize();

#endif
