!
!  Include file for Fortran use of the MM (multi-model) package
!
#define MM           integer
#define MMType       integer
!
!  Various multimodels
!
      integer MMEULER, MMFP, MMHYBRID_EF1, MMHYBRID_E
      integer MMHYBRID_F, MMNEW 

      parameter (MMEULER = 0, MMFP = 1, MMHYBRID_EF1 = 2)
      parameter (MMHYBRID_E = 3, MMHYBRID_F = 4, MMNEW = 5) 

