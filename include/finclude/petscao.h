C
C  $Id: ao.h,v 1.2 1996/08/06 04:04:35 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the AO (application ordering) package in PETSc
C
#define AO       integer
#define AOType   integer

      integer AO_BASIC, AO_ADVANCED

      parameter (AO_BASIC = 0, AO_ADVANCED = 1)

C
C  End of Fortran include file for the AO package in PETSc

