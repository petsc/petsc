!
!  Include file for Fortran use of the SNES package in PETSc
!
#if !defined (__PETSCSNESDEF_H)
#define __PETSCSNESDEF_H

#if defined(PETSC_USE_FORTRAN_MODULES)
#define SNES_HIDE type(SNES)
#else
#define SNES_HIDE SNES

#define SNES PetscFortranAddr
#endif
#define SNESType character*(80)
#define SNESConvergedReason PetscEnum
#define MatMFFD PetscFortranAddr
#define MatMFFDType PetscFortranAddr
!
!  SNESType
!
#define SNESLS 'ls'
#define SNESTR 'tr'
#define SNESTEST 'test'
!
! MatSNESMF
! 
#define MATMFFD_DEFAULT 'ds'
#define MATMFFD_WP 'wp'

#endif
