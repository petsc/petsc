#if !defined(PETSCCXXCOMPLEXFIX_H)
#define PETSCCXXCOMPLEXFIX_H
#if defined(__cplusplus) && defined(PETSC_HAVE_COMPLEX) && defined(PETSC_HAVE_CXX_COMPLEX)

#define PETSC_CXX_COMPLEX_FIX(Type) \
static inline PetscComplex operator+(const PetscComplex& lhs, const Type& rhs) { return lhs + PetscReal(rhs); } \
static inline PetscComplex operator+(const Type& lhs, const PetscComplex& rhs) { return PetscReal(lhs) + rhs; } \
static inline PetscComplex operator-(const PetscComplex& lhs, const Type& rhs) { return lhs - PetscReal(rhs); } \
static inline PetscComplex operator-(const Type& lhs, const PetscComplex& rhs) { return PetscReal(lhs) - rhs; } \
static inline PetscComplex operator*(const PetscComplex& lhs, const Type& rhs) { return lhs * PetscReal(rhs); } \
static inline PetscComplex operator*(const Type& lhs, const PetscComplex& rhs) { return PetscReal(lhs) * rhs; } \
static inline PetscComplex operator/(const PetscComplex& lhs, const Type& rhs) { return lhs / PetscReal(rhs); } \
static inline PetscComplex operator/(const Type& lhs, const PetscComplex& rhs) { return PetscReal(lhs) / rhs; } \
static inline bool operator==(const PetscComplex& lhs, const Type& rhs) { return lhs.imag() == PetscReal(0) && lhs.real() == PetscReal(rhs); } \
static inline bool operator==(const Type& lhs, const PetscComplex& rhs) { return rhs.imag() == PetscReal(0) && rhs.real() == PetscReal(lhs); } \
static inline bool operator!=(const PetscComplex& lhs, const Type& rhs) { return lhs.imag() != PetscReal(0) || lhs.real() != PetscReal(rhs); } \
static inline bool operator!=(const Type& lhs, const PetscComplex& rhs) { return rhs.imag() != PetscReal(0) || rhs.real() != PetscReal(lhs); } \
/* PETSC_CXX_COMPLEX_FIX */

#if defined(PETSC_USE_REAL_SINGLE)
PETSC_CXX_COMPLEX_FIX(double)
#elif defined(PETSC_USE_REAL_DOUBLE)
PETSC_CXX_COMPLEX_FIX(PetscInt)
#endif /* PETSC_USE_REAL_* */

#endif /* __cplusplus && PETSC_HAVE_COMPLEX && PETSC_HAVE_CXX_COMPLEX */
#endif
