#if !defined(PETSCCXXCOMPLEXFIX_H)
#define PETSCCXXCOMPLEXFIX_H
#if defined(__cplusplus) && defined(PETSC_HAVE_COMPLEX) && defined(PETSC_HAVE_CXX_COMPLEX)

/*
    The pragma below silence all compiler warnings comming from code in this header file.
    In particular, it silences `-Wfloat-equal` warnings in `operator==()` and `operator!=` below.
    Other compilers beyond GCC support this pragma.
*/
#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif

/*
     Defines additional operator overloading for the C++ complex class that are "missing" in the standard
     include files. For example, the code fragment

     std::complex<double> c = 22.0;
     c = 11 + c;

     will produce a compile time error such as

     error: no match for 'operator+' (operand types are 'int' and 'std::complex<double>')

     The code fragment

     std::complex<float> c = 22.0;
     c = 11.0 + c;

     will produce a compile time error such as

     error: no match for 'operator+' (operand types are 'double' and 'std::complex<float>')

     This deficiency means one may need to write cumbersome code while working with the C++ complex classes.

     This include file defines a few additional operator overload methods for the C++ complex classes to handle
     these cases naturally within PETSc code.

     This file is included automatically by PETSc include files. In the small number of cases where these additional methods
     may conflict with other code one may add

     #define PETSC_SKIP_CXX_COMPLEX_FIX

     before including any PETSc include files to prevent these methods from being provided.
*/

#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif

#define PETSC_CXX_COMPLEX_FIX(Type) \
static inline PetscComplex operator+(const PetscComplex& lhs, const Type& rhs) { return const_cast<PetscComplex&>(lhs) + PetscReal(rhs); } \
static inline PetscComplex operator+(const Type& lhs, const PetscComplex& rhs) { return PetscReal(lhs) + const_cast<PetscComplex&>(rhs); } \
static inline PetscComplex operator-(const PetscComplex& lhs, const Type& rhs) { return const_cast<PetscComplex&>(lhs) - PetscReal(rhs); } \
static inline PetscComplex operator-(const Type& lhs, const PetscComplex& rhs) { return PetscReal(lhs) - const_cast<PetscComplex&>(rhs); } \
static inline PetscComplex operator*(const PetscComplex& lhs, const Type& rhs) { return const_cast<PetscComplex&>(lhs) * PetscReal(rhs); } \
static inline PetscComplex operator*(const Type& lhs, const PetscComplex& rhs) { return PetscReal(lhs) * const_cast<PetscComplex&>(rhs); } \
static inline PetscComplex operator/(const PetscComplex& lhs, const Type& rhs) { return const_cast<PetscComplex&>(lhs) / PetscReal(rhs); } \
static inline PetscComplex operator/(const Type& lhs, const PetscComplex& rhs) { return PetscReal(lhs) / const_cast<PetscComplex&>(rhs); } \
static inline bool operator==(const PetscComplex& lhs, const Type& rhs) { return const_cast<PetscComplex&>(lhs).imag() == PetscReal(0) && const_cast<PetscComplex&>(lhs).real() == PetscReal(rhs); } \
static inline bool operator==(const Type& lhs, const PetscComplex& rhs) { return const_cast<PetscComplex&>(rhs).imag() == PetscReal(0) && const_cast<PetscComplex&>(rhs).real() == PetscReal(lhs); } \
static inline bool operator!=(const PetscComplex& lhs, const Type& rhs) { return const_cast<PetscComplex&>(lhs).imag() != PetscReal(0) || const_cast<PetscComplex&>(lhs).real() != PetscReal(rhs); } \
static inline bool operator!=(const Type& lhs, const PetscComplex& rhs) { return const_cast<PetscComplex&>(rhs).imag() != PetscReal(0) || const_cast<PetscComplex&>(rhs).real() != PetscReal(lhs); } \
/* PETSC_CXX_COMPLEX_FIX */

/*
    Due to the C++ automatic promotion rules for floating point and integer values only the two cases below
    need to be handled.
*/
#if defined(PETSC_USE_REAL_SINGLE)
PETSC_CXX_COMPLEX_FIX(double)
#elif defined(PETSC_USE_REAL_DOUBLE)
PETSC_CXX_COMPLEX_FIX(PetscInt)
#endif /* PETSC_USE_REAL_* */

#endif /* __cplusplus && PETSC_HAVE_COMPLEX && PETSC_HAVE_CXX_COMPLEX */
#endif
