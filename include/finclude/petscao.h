!
!  $Id: ao.h,v 1.10 1999/09/14 15:58:08 balay Exp bsmith $;
!
!  Include file for Fortran use of the AO (application ordering) package in PETSc
!
#if !defined (__AO_H)
#define __AO_H

#define AO         PetscFortranAddr
#define AOData     PetscFortranAddr
#define AOType     integer
#define AODataType integer

#endif


#if !defined (PETSC_AVOID_DECLARATIONS)

      integer AO_BASIC,AO_ADVANCED
      parameter (AO_BASIC = 0,AO_ADVANCED = 1)

      integer AODATA_BASIC,AODATA_ADVANCED
      parameter (AODATA_BASIC=0,AODATA_ADVANCED=1)
!
!  End of Fortran include file for the AO package in PETSc

#endif
