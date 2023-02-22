!
!  Used by petscsysmod.F90 to create Fortran module file
!
      PetscEnum, parameter :: PETSC_ERROR_INITIAL = 0
      PetscEnum, parameter :: PETSC_ERROR_REPEAT  = 1

      PetscErrorCode, parameter :: PETSC_ERR_MEM              = 55
      PetscErrorCode, parameter :: PETSC_ERR_SUP              = 56
      PetscErrorCode, parameter :: PETSC_ERR_SUP_SYS          = 57
      PetscErrorCode, parameter :: PETSC_ERR_ORDER            = 58
      PetscErrorCode, parameter :: PETSC_ERR_SIG              = 59
      PetscErrorCode, parameter :: PETSC_ERR_FP               = 72
      PetscErrorCode, parameter :: PETSC_ERR_COR              = 74
      PetscErrorCode, parameter :: PETSC_ERR_LIB              = 76
      PetscErrorCode, parameter :: PETSC_ERR_PLIB             = 77
      PetscErrorCode, parameter :: PETSC_ERR_MEMC             = 78
      PetscErrorCode, parameter :: PETSC_ERR_CONV_FAILED      = 82
      PetscErrorCode, parameter :: PETSC_ERR_USER             = 83
      PetscErrorCode, parameter :: PETSC_ERR_SYS              = 88
      PetscErrorCode, parameter :: PETSC_ERR_POINTER          = 70
      PetscErrorCode, parameter :: PETSC_ERR_MPI_LIB_INCOMP   = 87

      PetscErrorCode, parameter :: PETSC_ERR_ARG_SIZ          = 60
      PetscErrorCode, parameter :: PETSC_ERR_ARG_IDN          = 61
      PetscErrorCode, parameter :: PETSC_ERR_ARG_WRONG        = 62
      PetscErrorCode, parameter :: PETSC_ERR_ARG_CORRUPT      = 64
      PetscErrorCode, parameter :: PETSC_ERR_ARG_OUTOFRANGE   = 63
      PetscErrorCode, parameter :: PETSC_ERR_ARG_BADPTR       = 68
      PetscErrorCode, parameter :: PETSC_ERR_ARG_NOTSAMETYPE  = 69
      PetscErrorCode, parameter :: PETSC_ERR_ARG_NOTSAMECOMM  = 80
      PetscErrorCode, parameter :: PETSC_ERR_ARG_WRONGSTATE   = 73
      PetscErrorCode, parameter :: PETSC_ERR_ARG_TYPENOTSET   = 89
      PetscErrorCode, parameter :: PETSC_ERR_ARG_INCOMP       = 75
      PetscErrorCode, parameter :: PETSC_ERR_ARG_NULL         = 85
      PetscErrorCode, parameter :: PETSC_ERR_ARG_UNKNOWN_TYPE = 86

      PetscErrorCode, parameter :: PETSC_ERR_FILE_OPEN        = 65
      PetscErrorCode, parameter :: PETSC_ERR_FILE_READ        = 66
      PetscErrorCode, parameter :: PETSC_ERR_FILE_WRITE       = 67
      PetscErrorCode, parameter :: PETSC_ERR_FILE_UNEXPECTED  = 79

      PetscErrorCode, parameter :: PETSC_ERR_MAT_LU_ZRPVT     = 71
      PetscErrorCode, parameter :: PETSC_ERR_MAT_CH_ZRPVT     = 81

      PetscErrorCode, parameter :: PETSC_ERR_INT_OVERFLOW     = 84

      PetscErrorCode, parameter :: PETSC_ERR_FLOP_COUNT       = 90
      PetscErrorCode, parameter :: PETSC_ERR_NOT_CONVERGED    = 91
      PetscErrorCode, parameter :: PETSC_ERR_MISSING_FACTOR   = 92
      PetscErrorCode, parameter :: PETSC_ERR_OPT_OVERWRITE    = 93
      PetscErrorCode, parameter :: PETSC_ERR_WRONG_MPI_SIZE   = 94
      PetscErrorCode, parameter :: PETSC_ERR_USER_INPUT       = 95
      PetscErrorCode, parameter :: PETSC_ERR_GPU_RESOURCE     = 96
      PetscErrorCode, parameter :: PETSC_ERR_GPU              = 97
      PetscErrorCode, parameter :: PETSC_ERR_MPI              = 98
      PetscErrorCode, parameter :: PETSC_ERR_RETURN           = 99
