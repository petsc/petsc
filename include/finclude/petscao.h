!
!  $Id: ao.h,v 1.8 1998/03/27 21:18:20 balay Exp balay $;
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
      integer AO_BASIC, AO_ADVANCED
      parameter (AO_BASIC = 0, AO_ADVANCED = 1)

      integer AODATA_BASIC, AODATA_ADVANCED
      parameter (AODATA_BASIC=0, AODATA_ADVANCED=1)
!
!  End of Fortran include file for the AO package in PETSc

