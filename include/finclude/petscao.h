!
!  $Id: ao.h,v 1.4 1997/11/13 18:44:59 balay Exp balay $;
!
!  Include file for Fortran use of the AO (application ordering) package in PETSc
!
#define AO         integer
#define AOType     integer
#define AOData     integer
#define AODataType integer

      integer AO_BASIC, AO_ADVANCED
      parameter (AO_BASIC = 0, AO_ADVANCED = 1)

      integer AODATA_BASIC, AODATA_ADVANCED
      parameter (AODATA_BASIC=0, AODATA_ADVANCED=1)
!
!  End of Fortran include file for the AO package in PETSc

