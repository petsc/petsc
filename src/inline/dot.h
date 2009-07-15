
#ifndef DOT
#include "petsc.h"

EXTERN_C_BEGIN

#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmdot4_      FORTRANMDOT4
#define fortranmdot3_      FORTRANMDOT3
#define fortranmdot2_      FORTRANMDOT2
#define fortranmdot1_      FORTRANMDOT1
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmdot4_      fortranmdot4
#define fortranmdot3_      fortranmdot3
#define fortranmdot2_      fortranmdot2
#define fortranmdot1_      fortranmdot1
#endif
EXTERN void fortranmdot4_(void*,void*,void*,void*,void*,PetscInt*,void*,void*,void*,void*);
EXTERN void fortranmdot3_(void*,void*,void*,void*,PetscInt*,void*,void*,void*);
EXTERN void fortranmdot2_(void*,void*,void*,PetscInt*,void*,void*);
EXTERN void fortranmdot1_(void*,void*,PetscInt*,void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_NORM)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortrannormsqr_    FORTRANNORMSQR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortrannormsqr_    fortrannormsqr
#endif
EXTERN void fortrannormsqr_(void*,PetscInt*,void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultcrl_    FORTRANMULTCRL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultcrl_    fortranmultcrl
#endif
EXTERN void fortranmultcrl_(PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCSRPERM)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultcsrperm_    FORTRANMULTCSRPERM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultcsrperm_    fortranmultcsrperm
#endif
EXTERN void fortranmultcsrperm_(PetscInt*,PetscScalar*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultaij_    FORTRANMULTAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultaij_    fortranmultaij
#endif
EXTERN void fortranmultaij_(PetscInt*,const PetscScalar*,PetscInt*,PetscInt*,const MatScalar*,PetscScalar*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmulttransposeaddaij_    FORTRANMULTTRANSPOSEADDAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmulttransposeaddaij_    fortranmulttransposeaddaij
#endif
EXTERN void fortranmulttransposeaddaij_(PetscInt*,void*,PetscInt*,PetscInt*,void*,void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultaddaij_ FORTRANMULTADDAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultaddaij_ fortranmultaddaij
#endif
EXTERN void fortranmultaddaij_(PetscInt*,void*,PetscInt*,PetscInt*,const MatScalar*,void*,void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortransolveaij_   FORTRANSOLVEAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortransolveaij_   fortransolveaij
#endif
EXTERN void fortransolveaij_(const PetscInt*,void*,const PetscInt*,const PetscInt*,const PetscInt*,const void*,const void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranrelaxaijforward_   FORTRANRELAXAIJFORWARD
#define fortranrelaxaijbackward_   FORTRANRELAXAIJBACKWARD
#define fortranrelaxaijforwardzero_   FORTRANRELAXAIJFORWARDZERO
#define fortranrelaxaijbackwardzero_   FORTRANRELAXAIJBACKWARDZERO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranrelaxaijforward_   fortranrelaxaijforward
#define fortranrelaxaijbackward_   fortranrelaxaijbackward
#define fortranrelaxaijforwardzero_   fortranrelaxaijforwardzero
#define fortranrelaxaijbackwardzero_   fortranrelaxaijbackwardzero
#endif
EXTERN void fortranrelaxaijforward_(PetscInt*,PetscReal*,void*,PetscInt*,PetscInt*,const PetscInt*,void*,void*);
EXTERN void fortranrelaxaijbackward_(PetscInt*,PetscReal*,void*,PetscInt*,PetscInt*,const PetscInt*,void*,void*);
EXTERN void fortranrelaxaijforwardzero_(PetscInt*,PetscReal*,void*,PetscInt*,PetscInt*,const PetscInt*,void*,void*,void*);
EXTERN void fortranrelaxaijbackwardzero_(PetscInt*,PetscReal*,void*,PetscInt*,PetscInt*,const PetscInt*,void*,void*,void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortransolvebaij4_         FORTRANSOLVEBAIJ4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortransolvebaij4_          fortransolvebaij4
#endif
EXTERN void fortransolvebaij4_(const PetscInt*,void*,const PetscInt*,const PetscInt*,const PetscInt*,const void*,const void*,const void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJUNROLL)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortransolvebaij4unroll_   FORTRANSOLVEBAIJ4UNROLL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortransolvebaij4unroll_    fortransolvebaij4unroll
#endif
EXTERN void fortransolvebaij4unroll_(const PetscInt*,void*,const PetscInt*,const PetscInt*,const PetscInt*,const void*,const void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJBLAS)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortransolvebaij4blas_     FORTRANSOLVEBAIJ4BLAS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortransolvebaij4blas_      fortransolvebaij4blas
#endif
EXTERN void fortransolvebaij4blas_(const PetscInt*,void*,const PetscInt*,const PetscInt*,const PetscInt*,const void*,const void*,const void*);
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define fortranxtimesy_ FORTRANXTIMESY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranxtimesy_ fortranxtimesy
#endif
EXTERN void fortranxtimesy_(void*,void*,void*,PetscInt*);
#endif

EXTERN_C_END


#endif
