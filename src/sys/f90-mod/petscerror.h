
!
!  Include file for Fortran error codes
!
      PetscEnum PETSC_ERROR_INITIAL
      PetscEnum PETSC_ERROR_REPEAT
      parameter (PETSC_ERROR_INITIAL = 0)
      parameter (PETSC_ERROR_REPEAT = 1)

      PetscErrorCode PETSC_ERR_MEM
      PetscErrorCode PETSC_ERR_SUP
      PetscErrorCode PETSC_ERR_SUP_SYS
      PetscErrorCode PETSC_ERR_ORDER
      PetscErrorCode PETSC_ERR_SIG
      PetscErrorCode PETSC_ERR_FP
      PetscErrorCode PETSC_ERR_COR
      PetscErrorCode PETSC_ERR_LIB
      PetscErrorCode PETSC_ERR_PLIB
      PetscErrorCode PETSC_ERR_MEMC
      PetscErrorCode PETSC_ERR_CONV_FAILED
      PetscErrorCode PETSC_ERR_USER
      PetscErrorCode PETSC_ERR_SYS
      PetscErrorCode PETSC_ERR_POINTER
      PetscErrorCode PETSC_ERR_MPI_LIB_INCOMP

      PetscErrorCode PETSC_ERR_ARG_SIZ
      PetscErrorCode PETSC_ERR_ARG_IDN
      PetscErrorCode PETSC_ERR_ARG_WRONG
      PetscErrorCode PETSC_ERR_ARG_CORRUPT
      PetscErrorCode PETSC_ERR_ARG_OUTOFRANGE
      PetscErrorCode PETSC_ERR_ARG_BADPTR
      PetscErrorCode PETSC_ERR_ARG_NOTSAMETYPE
      PetscErrorCode PETSC_ERR_ARG_NOTSAMECOMM
      PetscErrorCode PETSC_ERR_ARG_WRONGSTATE
      PetscErrorCode PETSC_ERR_ARG_TYPENOTSET
      PetscErrorCode PETSC_ERR_ARG_INCOMP
      PetscErrorCode PETSC_ERR_ARG_NULL
      PetscErrorCode PETSC_ERR_ARG_UNKNOWN_TYPE

      PetscErrorCode PETSC_ERR_FILE_OPEN
      PetscErrorCode PETSC_ERR_FILE_READ
      PetscErrorCode PETSC_ERR_FILE_WRITE
      PetscErrorCode PETSC_ERR_FILE_UNEXPECTED

      PetscErrorCode PETSC_ERR_MAT_LU_ZRPVT
      PetscErrorCode PETSC_ERR_MAT_CH_ZRPVT

      PetscErrorCode PETSC_ERR_INT_OVERFLOW

      PetscErrorCode PETSC_ERR_FLOP_COUNT
      PetscErrorCode PETSC_ERR_NOT_CONVERGED
      PetscErrorCode PETSC_ERR_MISSING_FACTOR
      PetscErrorCode PETSC_ERR_OPT_OVERWRITE
      PetscErrorCode PETSC_ERR_WRONG_MPI_SIZE
      PetscErrorCode PETSC_ERR_USER_INPUT

      parameter(PETSC_ERR_MEM =     55)
      parameter(PETSC_ERR_SUP =     56)
      parameter(PETSC_ERR_SUP_SYS = 57)
      parameter(PETSC_ERR_ORDER =   58)
      parameter(PETSC_ERR_SIG =     59)
      parameter(PETSC_ERR_FP =      72)
      parameter(PETSC_ERR_COR =     74)
      parameter(PETSC_ERR_LIB =     76)
      parameter(PETSC_ERR_PLIB =    77)
      parameter(PETSC_ERR_MEMC =    78)
      parameter(PETSC_ERR_CONV_FAILED  =    82)
      parameter(PETSC_ERR_USER =    83)
      parameter(PETSC_ERR_SYS =     88)
      parameter(PETSC_ERR_POINTER = 70)
      parameter(PETSC_ERR_MPI_LIB_INCOMP =  87)

      parameter(PETSC_ERR_ARG_SIZ = 60)
      parameter(PETSC_ERR_ARG_IDN = 61)
      parameter(PETSC_ERR_ARG_WRONG =       62)
      parameter(PETSC_ERR_ARG_CORRUPT =    64)
      parameter(PETSC_ERR_ARG_OUTOFRANGE =  63)
      parameter(PETSC_ERR_ARG_BADPTR =      68)
      parameter(PETSC_ERR_ARG_NOTSAMETYPE =  69)
      parameter(PETSC_ERR_ARG_NOTSAMECOMM = 80)
      parameter(PETSC_ERR_ARG_WRONGSTATE  = 73)
      parameter(PETSC_ERR_ARG_TYPENOTSET =  89)
      parameter(PETSC_ERR_ARG_INCOMP   =    75)
      parameter(PETSC_ERR_ARG_NULL      =   85)
      parameter(PETSC_ERR_ARG_UNKNOWN_TYPE = 86)

      parameter(PETSC_ERR_FILE_OPEN      =  65)
      parameter(PETSC_ERR_FILE_READ      =  66)
      parameter(PETSC_ERR_FILE_WRITE     =  67)
      parameter(PETSC_ERR_FILE_UNEXPECTED = 79)

      parameter(PETSC_ERR_MAT_LU_ZRPVT   =  71)
      parameter(PETSC_ERR_MAT_CH_ZRPVT   =  81)

      parameter(PETSC_ERR_INT_OVERFLOW   =  84)

      parameter(PETSC_ERR_FLOP_COUNT     =  90)
      parameter(PETSC_ERR_NOT_CONVERGED  =  91)
      parameter(PETSC_ERR_MISSING_FACTOR =  92)
      parameter(PETSC_ERR_OPT_OVERWRITE  =  93)
      parameter(PETSC_ERR_WRONG_MPI_SIZE =  94)
      parameter(PETSC_ERR_USER_INPUT     =  95)



