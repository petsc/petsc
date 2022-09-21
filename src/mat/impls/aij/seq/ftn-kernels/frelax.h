
#ifndef __FRELAX_H
  #include <petscsys.h>
  #if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
    #if defined(PETSC_HAVE_FORTRAN_CAPS)
      #define fortranrelaxaijforward_      FORTRANRELAXAIJFORWARD
      #define fortranrelaxaijbackward_     FORTRANRELAXAIJBACKWARD
      #define fortranrelaxaijforwardzero_  FORTRANRELAXAIJFORWARDZERO
      #define fortranrelaxaijbackwardzero_ FORTRANRELAXAIJBACKWARDZERO
    #elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
      #define fortranrelaxaijforward_      fortranrelaxaijforward
      #define fortranrelaxaijbackward_     fortranrelaxaijbackward
      #define fortranrelaxaijforwardzero_  fortranrelaxaijforwardzero
      #define fortranrelaxaijbackwardzero_ fortranrelaxaijbackwardzero
    #endif

PETSC_EXTERN void fortranrelaxaijforward_(PetscInt *, PetscReal *, void *, PetscInt *, PetscInt *, const PetscInt *, void *, void *);
PETSC_EXTERN void fortranrelaxaijbackward_(PetscInt *, PetscReal *, void *, PetscInt *, PetscInt *, const PetscInt *, void *, void *);
PETSC_EXTERN void fortranrelaxaijforwardzero_(PetscInt *, PetscReal *, void *, PetscInt *, PetscInt *, const PetscInt *, const void *, void *, void *);
PETSC_EXTERN void fortranrelaxaijbackwardzero_(PetscInt *, PetscReal *, void *, PetscInt *, PetscInt *, const PetscInt *, const void *, void *, void *);

  #endif
#endif
