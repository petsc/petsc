!
!
!  Include file for Fortran use of the AO (application ordering) package in PETSc
!
#if !defined (__PETSCAO_H)
#define __PETSCAO_H

#define AO PetscFortranAddr
#define AOData PetscFortranAddr
#define AOType integer
#define AODataType integer
#define AOData2dGrid PetscFortranAddr

#endif


#if !defined (PETSC_AVOID_DECLARATIONS)

      integer AO_BASIC,AO_ADVANCED
      parameter (AO_BASIC = 0,AO_ADVANCED = 1)

      integer AODATA_BASIC,AODATA_ADVANCED
      parameter (AODATA_BASIC=0,AODATA_ADVANCED=1)
!
!  End of Fortran include file for the AO package in PETSc

#endif
