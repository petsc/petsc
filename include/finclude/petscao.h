C
C  $Id: ao.h,v 1.1 1996/08/05 22:22:55 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the AO (application ordering) package in PETSc
C
#define AO       integer
#define AOType   integer

      integer AO_DEBUG, AO_BASIC

      parameter (AO_DEBUG = 0, AO_BASIC = 1)

C
C  End of Fortran include file for the AO package in PETSc

