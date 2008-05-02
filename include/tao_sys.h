#if !defined(__TAO_SYS_H)
#define __TAO_SYS_H


#if defined(_MSC_VER)

//Special Microsoft builtins
#define TaoNaN(a)        _isnan(a)
#define TaoInf(a)        (!_finite(a))
#define TaoInfOrNaN(a)   TaoInf(a)

#else

// These tests are from ISO C99
#define TaoNaN(a)        isnan(a)
#define TaoInf(a)        isinf(a)
#define TaoInfOrNaN(a)   (TaoInf(a) || TaoNaN(a))

#endif

#define TAO_DEFAULT        -13456834
#define TAO_INFINITY        1.0e20
#define TAO_NINFINITY       -1.0e20
#define TAO_NULL           0
#define TAO_EPSILON     DBL_EPSILON

/*
PetscErrorCode PETSC_DLLEXPORT TaoInitialize(int*,char ***,char[],const char[]);
PetscErrorCode PETSC_DLLEXPORT TaoInitializeNoArguments(void);
PetscErrorCode PETSC_DLLEXPORT TaoInitializeFortran(void);
PetscErrorCode PETSC_DLLEXPORT TaoFinalize(void);
*/



#endif


