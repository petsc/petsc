C
C  $Id: ao.h,v 1.3 1997/09/26 02:22:28 bsmith Exp balay $;
C
C  Include file for Fortran use of the AO (application ordering) package in PETSc
C
#define AO         integer
#define AOType     integer
#define AOData     integer
#define AODataType integer

      integer AO_BASIC, AO_ADVANCED
      parameter (AO_BASIC = 0, AO_ADVANCED = 1)

      integer AODATA_BASIC, AODATA_ADVANCED
      parameter (AODATA_BASIC=0, AODATA_ADVANCED=1)
C
C  End of Fortran include file for the AO package in PETSc

