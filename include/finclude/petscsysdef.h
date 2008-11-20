!
!
!  Include file for Fortran use of the System package in PETSc
!
#if !defined (__PETSCSYSDEF_H)
#define __PETSCSYSDEF_H

#define PetscRandom PetscFortranAddr
#define PetscRandomType character*(80)
#define PetscBinarySeekType PetscEnum

#endif
