!
!  Include file for Fortran use of the MM (multi-model) package
!
#define MM           integer
#define MMType       integer
!
!  Various multimodels
!
      integer MMEULER_INT, MMFP_INT, MMHYBRID_EF1_INT, MMHYBRID_E_INT
      integer MMHYBRID_F_INT, MMNEW_INT 

      parameter (MMEULER_INT = 0, MMFP_INT = 1, MMHYBRID_EF1_INT = 2)
      parameter (MMHYBRID_E_INT = 3, MMHYBRID_F_INT = 4, MMNEW_INT = 5) 

