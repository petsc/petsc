!
!
!  Include file for Fortran use of the Mat package in PETSc
!
#if !defined (__PETSCMAT_H)
#define __PETSCMAT_H

#define Mat PetscFortranAddr
#define MatFDColoring PetscFortranAddr
#define MatPartitioning PetscFortranAddr
#define MatAIJIndices PetscFortranAddr
#define MatType character*(80)
#define MatOption PetscEnum 
#define MatAssemblyType PetscEnum
#define MatOrderingType character*(80)
#define MatSORType PetscEnum
#define MatInfoType PetscEnum
#define MatReuse PetscEnum
#define MatOperation PetscEnum
#define MatColoringType character*(80)
#define MatInfo double precision
#define MatFactorInfo double precision
#define MatDuplicateOption PetscEnum      
#define MatStructure PetscEnum
#define MatPartitioningType character*(80)
#define MatNullSpace PetscFortranAddr

#define MatStencil PetscInt
#define MatStencil_k 1
#define MatStencil_j 2
#define MatStencil_i 3
#define MatStencil_c 4

#define MATPARTITIONING_CURRENT 'current'
#define MATPARTITIONING_PARMETIS 'parmetis'

#define MATCOLORING_NATURAL 'natural'
#define MATCOLORING_SL 'sl'
#define MATCOLORING_LF 'lf'
#define MATCOLORING_ID 'id'

#define MATORDERING_NATURAL 'natural'
#define MATORDERING_ND 'nd'
#define MATORDERING_1WD '1wd'
#define MATORDERING_RCM 'rcm'
#define MATORDERING_QMD 'qmd'
#define MATORDERING_ROWLENGTH 'rowlength'
#define MATORDERING_DSC_ND 'dsc_nd'
#define MATORDERING_DSC_MMD 'dsc_mmd'
#define MATORDERING_DSC_MDF 'dsc_mdf'

#endif

!
!  Matrix types
!
#define MATSAME            'same'
#define MATSEQMAIJ         'seqmaij'
#define MATMPIMAIJ         'mpimaij'
#define MATMAIJ            'maij'
#define MATIS              'is'
#define MATMPIROWBS        'mpirowbs'
#define MATSEQAIJ          'seqaij'
#define MATMPIAIJ          'mpiaij'
#define MATAIJ             'aij'
#define MATSHELL           'shell'
#define MATSEQBDIAG        'seqbdiag'
#define MATMPIBDIAG        'mpibdiag'
#define MATBDIAG           'bdiag'
#define MATSEQDENSE        'seqdense'
#define MATMPIDENSE        'mpidense'
#define MATDENSE           'dense'
#define MATSEQBAIJ         'seqbaij'
#define MATMPIBAIJ         'mpibaij'
#define MATBAIJ            'baij'
#define MATMPIADJ          'mpiadj'
#define MATSEQSBAIJ        'seqsbaij'
#define MATMPISBAIJ        'mpisbaij'
#define MATSBAIJ           'sbaij'
#define MATDAAD            'daad'
#define MATMFFD            'mffd'
#define MATESI             'esi'
#define MATPETSCESI        'petscesi'
#define MATNORMAL          'normal'
#define MATSEQAIJSPOOLES   'seqaijspooles'
#define MATMPIAIJSPOOLES   'mpiaijspooles'
#define MATSEQSBAIJSPOOLES 'seqsbaijspooles'
#define MATMPISBAIJSPOOLES 'mpisbaijspooles'
#define MATSUPERLU         'superlu'
#define MATSUPERLU_DIST    'superlu_dist'
#define MATUMFPACK         'umfpack'
#define MATESSL            'essl'
#define MATLUSOL           'lusol'
#define MATAIJMUMPS        'aijmumps'
#define MATSBAIJMUMPS      'sbaijmumps'
#define MATDSCPACK         'dscpack'
#define MATMATLAB          'matlab'

#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  Flag for matrix assembly
!
      PetscEnum MAT_FLUSH_ASSEMBLY,MAT_FINAL_ASSEMBLY

      parameter(MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0)
!
!  Matrix options; must match those in include/petscmat.h
!
      PetscEnum MAT_ROW_ORIENTED,MAT_COLUMN_ORIENTED,MAT_ROWS_SORTED
      PetscEnum MAT_COLUMNS_SORTED,MAT_NO_NEW_NONZERO_LOCATIONS
      PetscEnum MAT_YES_NEW_NONZERO_LOCATIONS,MAT_SYMMETRIC
      PetscEnum MAT_STRUCTURALLY_SYMMETRIC,MAT_NO_NEW_DIAGONALS
      PetscEnum MAT_YES_NEW_DIAGONALS,MAT_INODE_LIMIT_1
      PetscEnum MAT_INODE_LIMIT_2,MAT_INODE_LIMIT_3,MAT_INODE_LIMIT_4
      PetscEnum MAT_INODE_LIMIT_5,MAT_IGNORE_OFF_PROC_ENTRIES
      PetscEnum MAT_ROWS_UNSORTED,MAT_COLUMNS_UNSORTED
      PetscEnum MAT_NEW_NONZERO_LOCATION_ERR
      PetscEnum MAT_NEW_NONZERO_ALLOCATION_ERR,MAT_USE_HASH_TABLE
      PetscEnum MAT_KEEP_ZEROED_ROWS,MAT_IGNORE_ZERO_ENTRIES
      PetscEnum MAT_USE_INODES,MAT_DO_NOT_USE_INODES
      PetscEnum MAT_NOT_SYMMETRIC,MAT_HERMITIAN
      PetscEnum MAT_NOT_STRUCTURALLY_SYMMETRIC,MAT_NOT_HERMITIAN
      PetscEnum MAT_SYMMETRY_ETERNAL,MAT_NOT_SYMMETRY_ETERNAL

      parameter (MAT_ROW_ORIENTED=1,MAT_COLUMN_ORIENTED=2)
      parameter (MAT_ROWS_SORTED=4,MAT_COLUMNS_SORTED=8)
      parameter (MAT_NO_NEW_NONZERO_LOCATIONS=16)
      parameter (MAT_YES_NEW_NONZERO_LOCATIONS=32)
      parameter (MAT_SYMMETRIC=64,MAT_STRUCTURALLY_SYMMETRIC=65)
      parameter (MAT_NO_NEW_DIAGONALS=66,MAT_YES_NEW_DIAGONALS=67)
      parameter (MAT_INODE_LIMIT_1=68,MAT_INODE_LIMIT_2=69)
      parameter (MAT_INODE_LIMIT_3=70,MAT_INODE_LIMIT_4=71)
      parameter (MAT_INODE_LIMIT_5=72,MAT_IGNORE_OFF_PROC_ENTRIES=73)
      parameter (MAT_ROWS_UNSORTED=74,MAT_COLUMNS_UNSORTED=75)
      parameter (MAT_NEW_NONZERO_LOCATION_ERR=76)
      parameter (MAT_NEW_NONZERO_ALLOCATION_ERR=77)
      parameter (MAT_USE_HASH_TABLE=78)
      parameter (MAT_KEEP_ZEROED_ROWS=79)
      parameter (MAT_IGNORE_ZERO_ENTRIES=80)
      parameter (MAT_USE_INODES=81,MAT_DO_NOT_USE_INODES=82)
      parameter (MAT_NOT_SYMMETRIC=83,MAT_HERMITIAN=84)
      parameter (MAT_NOT_STRUCTURALLY_SYMMETRIC=85)
      parameter (MAT_NOT_HERMITIAN=86)
      parameter (MAT_SYMMETRY_ETERNAL=87,MAT_NOT_SYMMETRY_ETERNAL=88)

!
!  MatDuplicateOption
!
      PetscEnum MAT_DO_NOT_COPY_VALUES,MAT_COPY_VALUES
      parameter (MAT_DO_NOT_COPY_VALUES=0,MAT_COPY_VALUES=1)
!
!  Flags for PCSetOperators()
!
      PetscEnum SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN
      PetscEnum SAME_PRECONDITIONER

      parameter (SAME_NONZERO_PATTERN = 0,DIFFERENT_NONZERO_PATTERN = 1)
      parameter (SAME_PRECONDITIONER = 2)

!
!  Note: MAT_INFO_SIZE must equal # elements in MatInfo structure
!  (See petsc/include/petscmat.h)
!
      PetscEnum   MAT_INFO_SIZE

      parameter (MAT_INFO_SIZE=14)

      PetscEnum MAT_INFO_ROWS_GLOBAL,MAT_INFO_COLUMNS_GLOBAL
      PetscEnum MAT_INFO_ROWS_LOCAL,MAT_INFO_COLUMNS_LOCAL
      PetscEnum MAT_INFO_BLOCK_SIZE,MAT_INFO_NZ_ALLOCATED
      PetscEnum MAT_INFO_NZ_USED,MAT_INFO_NZ_UNNEEDED
      PetscEnum MAT_INFO_MEMORY,MAT_INFO_ASSEMBLIES
      PetscEnum MAT_INFO_MALLOCS,MAT_INFO_FILL_RATIO_GIVEN
      PetscEnum MAT_INFO_FILL_RATIO_NEEDED,MAT_INFO_FACTOR_MALLOCS

      parameter (MAT_INFO_ROWS_GLOBAL=1,MAT_INFO_COLUMNS_GLOBAL=2)
      parameter (MAT_INFO_ROWS_LOCAL=3,MAT_INFO_COLUMNS_LOCAL=4)
      parameter (MAT_INFO_BLOCK_SIZE=5,MAT_INFO_NZ_ALLOCATED=6)
      parameter (MAT_INFO_NZ_USED=7,MAT_INFO_NZ_UNNEEDED=8)
      parameter (MAT_INFO_MEMORY=9,MAT_INFO_ASSEMBLIES=10)
      parameter (MAT_INFO_MALLOCS=11,MAT_INFO_FILL_RATIO_GIVEN=12)
      parameter (MAT_INFO_FILL_RATIO_NEEDED=13)
      parameter (MAT_INFO_FACTOR_MALLOCS=14)
!
!  MatReuse
!
      PetscEnum MAT_INITIAL_MATRIX,MAT_REUSE_MATRIX

      parameter (MAT_INITIAL_MATRIX=0,MAT_REUSE_MATRIX=1)

!
!  MatInfoType
!
      PetscEnum MAT_LOCAL,MAT_GLOBAL_MAX,MAT_GLOBAL_SUM

      parameter (MAT_LOCAL=1,MAT_GLOBAL_MAX=2,MAT_GLOBAL_SUM=3)

!
!  Note: MAT_FACTORINFO_SIZE must equal # elements in MatFactorInfo structure
!  (See petsc/include/petscmat.h)
!
      PetscEnum   MAT_FACTORINFO_SIZE

      parameter (MAT_FACTORINFO_SIZE=11)

      PetscEnum MAT_FACTORINFO_LEVELS
      PetscEnum MAT_FACTORINFO_FILL
      PetscEnum MAT_FACTORINFO_DIAGONAL_FILL
      PetscEnum MAT_FACTORINFO_DT
      PetscEnum MAT_FACTORINFO_DTCOL
      PetscEnum MAT_FACTORINFO_DTCOUNT
      PetscEnum MAT_FACTORINFO_DAMPING
      PetscEnum MAT_FACTORINFO_SHIFT
      PetscEnum MAT_FACTORINFO_SHIFT_FRACTION
      PetscEnum MAT_FACTORINFO_ZERO_PIVOT
      PetscEnum MAT_FACTORINFO_PIVOT_IN_BLOCKS

      parameter (MAT_FACTORINFO_DAMPING = 1)
      parameter (MAT_FACTORINFO_SHIFT = 2)
      parameter (MAT_FACTORINFO_SHIFT_FRACTION = 3)
      parameter (MAT_FACTORINFO_DIAGONAL_FILL = 4)
      parameter (MAT_FACTORINFO_DT = 5)
      parameter (MAT_FACTORINFO_DTCOL = 6)
      parameter (MAT_FACTORINFO_DTCOUNT = 7)
      parameter (MAT_FACTORINFO_LEVELS = 8)
      parameter (MAT_FACTORINFO_FILL = 9)
      parameter (MAT_FACTORINFO_PIVOT_IN_BLOCKS = 10)
      parameter (MAT_FACTORINFO_ZERO_PIVOT = 11)


!
!  Options for SOR and SSOR
!  MatSorType may be bitwise ORd together, so do not change the numbers
!
      PetscEnum SOR_FORWARD_SWEEP,SOR_BACKWARD_SWEEP,SOR_SYMMETRIC_SWEEP
      PetscEnum SOR_LOCAL_FORWARD_SWEEP,SOR_LOCAL_BACKWARD_SWEEP
      PetscEnum SOR_LOCAL_SYMMETRIC_SWEEP,SOR_ZERO_INITIAL_GUESS
      PetscEnum SOR_EISENSTAT,SOR_APPLY_UPPER,SOR_APPLY_LOWER

      parameter (SOR_FORWARD_SWEEP=1,SOR_BACKWARD_SWEEP=2)
      parameter (SOR_SYMMETRIC_SWEEP=3,SOR_LOCAL_FORWARD_SWEEP=4)
      parameter (SOR_LOCAL_BACKWARD_SWEEP=8)
      parameter (SOR_LOCAL_SYMMETRIC_SWEEP=12)
      parameter (SOR_ZERO_INITIAL_GUESS=16,SOR_EISENSTAT=32)
      parameter (SOR_APPLY_UPPER=64,SOR_APPLY_LOWER=128)
!
!  MatOperation
!
      PetscEnum MATOP_SET_VALUES
      PetscEnum MATOP_GET_ROW
      PetscEnum MATOP_RESTORE_ROW
      PetscEnum MATOP_MULT
      PetscEnum MATOP_MULT_ADD
      PetscEnum MATOP_MULT_TRANSPOSE
      PetscEnum MATOP_MULT_TRANSPOSE_ADD
      PetscEnum MATOP_SOLVE
      PetscEnum MATOP_SOLVE_ADD
      PetscEnum MATOP_SOLVE_TRANSPOSE
      PetscEnum MATOP_SOLVE_TRANSPOSE_ADD
      PetscEnum MATOP_LUFACTOR
      PetscEnum MATOP_CHOLESKYFACTOR
      PetscEnum MATOP_RELAX
      PetscEnum MATOP_TRANSPOSE
      PetscEnum MATOP_GETINFO
      PetscEnum MATOP_EQUAL
      PetscEnum MATOP_GET_DIAGONAL 
      PetscEnum MATOP_DIAGONAL_SCALE
      PetscEnum MATOP_NORM
      PetscEnum MATOP_ASSEMBLY_BEGIN
      PetscEnum MATOP_ASSEMBLY_END
      PetscEnum MATOP_COMPRESS
      PetscEnum MATOP_SET_OPTION
      PetscEnum MATOP_ZERO_ENTRIES
      PetscEnum MATOP_ZERO_ROWS
      PetscEnum MATOP_LUFACTOR_SYMBOLIC
      PetscEnum MATOP_LUFACTOR_NUMERIC
      PetscEnum MATOP_CHOLESKY_FACTOR_SYMBOLIC
      PetscEnum MATOP_CHOLESKY_FACTOR_NUMERIC
      PetscEnum MATOP_SETUP_PREALLOCATION
      PetscEnum MATOP_ILUFACTOR_SYMBOLIC
      PetscEnum MATOP_ICCFACTOR_SYMBOLIC
      PetscEnum MATOP_GET_ARRAY
      PetscEnum MATOP_RESTORE_ARRAY

      PetscEnum MATOP_DUPLICATE
      PetscEnum MATOP_FORWARD_SOLVE
      PetscEnum MATOP_BACKWARD_SOLVE
      PetscEnum MATOP_ILUFACTOR
      PetscEnum MATOP_ICCFACTOR
      PetscEnum MATOP_AXPY
      PetscEnum MATOP_GET_SUBMATRICES
      PetscEnum MATOP_INCREASE_OVERLAP
      PetscEnum MATOP_GET_VALUES
      PetscEnum MATOP_COPY
      PetscEnum MATOP_PRINT_HELP
      PetscEnum MATOP_SCALE
      PetscEnum MATOP_SHIFT
      PetscEnum MATOP_DIAGONAL_SHIFT
      PetscEnum MATOP_ILUDT_FACTOR
      PetscEnum MATOP_GET_BLOCK_SIZE

      PetscEnum MATOP_GET_ROW_IJ
      PetscEnum MATOP_RESTORE_ROW_IJ
      PetscEnum MATOP_GET_COLUMN_IJ
      PetscEnum MATOP_RESTORE_COLUMN_IJ
      PetscEnum MATOP_FDCOLORING_CREATE
      PetscEnum MATOP_COLORING_PATCH
      PetscEnum MATOP_SET_UNFACTORED
      PetscEnum MATOP_PERMUTE
      PetscEnum MATOP_SET_VALUES_BLOCKED
      PetscEnum MATOP_GET_SUBMATRIX
      PetscEnum MATOP_DESTROY
      PetscEnum MATOP_VIEW
      PetscEnum MATOP_GET_MAPS
      PetscEnum MATOP_USE_SCALED_FORM
      PetscEnum MATOP_SCALE_SYSTEM
      PetscEnum MATOP_UNSCALE_SYSTEM
      PetscEnum MATOP_SET_LOCAL_TO_GLOBAL_MAP
      PetscEnum MATOP_SET_VALUES_LOCAL
      PetscEnum MATOP_ZERO_ROWS_LOCAL
      PetscEnum MATOP_GET_ROW_MAX
      PetscEnum MATOP_CONVERT
      PetscEnum MATOP_SET_COLORING
      PetscEnum MATOP_SET_VALUES_ADIC
      PetscEnum MATOP_SET_VALUES_ADIFOR
      PetscEnum MATOP_FD_COLORING_APPLY
      PetscEnum MATOP_SET_FROM_OPTIONS
      PetscEnum MATOP_MULT_CON
      PetscEnum MATOP_MULT_TRANSPOSE_CON
      PetscEnum MATOP_ILU_FACTOR_SYMBOLIC_CON
      PetscEnum MATOP_PERMUTE_SPARSIFY
      PetscEnum MATOP_MULT_MULTIPLE
      PetscEnum MATOP_SOLVE_MULTIPLE
      PetscEnum MATOP_GET_INERTIA
      PetscEnum MATOP_LOAD
      PetscEnum MATOP_IS_SYMMETRIC
      PetscEnum MATOP_IS_HERMITIAN
      PetscEnum MATOP_IS_STRUCTURALLY_SYMMETRIC
      PetscEnum MATOP_PB_RELAX
      PetscEnum MATOP_GET_VECS
      PetscEnum MATOP_MAT_MULT
      PetscEnum MATOP_MAT_MULT_SYMBOLIC
      PetscEnum MATOP_MAT_MULT_NUMERIC
      PetscEnum MATOP_PTAP
      PetscEnum MATOP_PTAP_SYMBOLIC
      PetscEnum MATOP_PTAP_NUMERIC
      PetscEnum MATOP_MAT_MULT_TRANSPOSE
      PetscEnum MATOP_MAT_MULT_TRANSPOSE_SYM
      PetscEnum MATOP_MAT_MULT_TRANSPOSE_NUM
      PetscEnum MATOP_PTAP_SYMBOLIC_SEQAIJ
      PetscEnum MATOP_PTAP_NUMERIC_SEQAIJ
      PetscEnum MATOP_PTAP_SYMBOLIC_MPIAIJ
      PetscEnum MATOP_PTAP_NUMERIC_MPIAIJ
      PetscEnum MATOP_SET_VALUES_ROW
  

      parameter(MATOP_SET_VALUES=0)
      parameter(MATOP_GET_ROW=1)
      parameter(MATOP_RESTORE_ROW=2)
      parameter(MATOP_MULT=3)
      parameter(MATOP_MULT_ADD=4)
      parameter(MATOP_MULT_TRANSPOSE=5)
      parameter(MATOP_MULT_TRANSPOSE_ADD=6)
      parameter(MATOP_SOLVE=7)
      parameter(MATOP_SOLVE_ADD=8)
      parameter(MATOP_SOLVE_TRANSPOSE=9)
      parameter(MATOP_SOLVE_TRANSPOSE_ADD=10)
      parameter(MATOP_LUFACTOR=11)
      parameter(MATOP_CHOLESKYFACTOR=12)
      parameter(MATOP_RELAX=13)
      parameter(MATOP_TRANSPOSE=14)
      parameter(MATOP_GETINFO=15)
      parameter(MATOP_EQUAL=16)
      parameter(MATOP_GET_DIAGONAL=17) 
      parameter(MATOP_DIAGONAL_SCALE=18)
      parameter(MATOP_NORM=19)
      parameter(MATOP_ASSEMBLY_BEGIN=20)
      parameter(MATOP_ASSEMBLY_END=21)
      parameter(MATOP_COMPRESS=22)
      parameter(MATOP_SET_OPTION=23)
      parameter(MATOP_ZERO_ENTRIES=24)
      parameter(MATOP_ZERO_ROWS=25)
      parameter(MATOP_LUFACTOR_SYMBOLIC=26)
      parameter(MATOP_LUFACTOR_NUMERIC=27)
      parameter(MATOP_CHOLESKY_FACTOR_SYMBOLIC=28)
      parameter(MATOP_CHOLESKY_FACTOR_NUMERIC=29)
      parameter(MATOP_SETUP_PREALLOCATION=30)
      parameter(MATOP_ILUFACTOR_SYMBOLIC=31)
      parameter(MATOP_ICCFACTOR_SYMBOLIC=32)
      parameter(MATOP_GET_ARRAY=33)
      parameter(MATOP_RESTORE_ARRAY=34)
      parameter(MATOP_DUPLICATE=35)
      parameter(MATOP_FORWARD_SOLVE=36)
      parameter(MATOP_BACKWARD_SOLVE=37)
      parameter(MATOP_ILUFACTOR=38)
      parameter(MATOP_ICCFACTOR=39)
      parameter(MATOP_AXPY=40)
      parameter(MATOP_GET_SUBMATRICES=41)
      parameter(MATOP_INCREASE_OVERLAP=42)
      parameter(MATOP_GET_VALUES=43)
      parameter(MATOP_COPY=44)
      parameter(MATOP_PRINT_HELP=45)
      parameter(MATOP_SCALE=46)
      parameter(MATOP_SHIFT=47)
      parameter(MATOP_DIAGONAL_SHIFT=48)
      parameter(MATOP_ILUDT_FACTOR=49)
      parameter(MATOP_GET_BLOCK_SIZE=50)
      parameter(MATOP_GET_ROW_IJ=51)
      parameter(MATOP_RESTORE_ROW_IJ=52)
      parameter(MATOP_GET_COLUMN_IJ=53)
      parameter(MATOP_RESTORE_COLUMN_IJ=54)
      parameter(MATOP_FDCOLORING_CREATE=55)
      parameter(MATOP_COLORING_PATCH=56)
      parameter(MATOP_SET_UNFACTORED=57)
      parameter(MATOP_PERMUTE=58)
      parameter(MATOP_SET_VALUES_BLOCKED=59)
      parameter(MATOP_GET_SUBMATRIX=60)
      parameter(MATOP_DESTROY=61)
      parameter(MATOP_VIEW=62)
      parameter(MATOP_GET_MAPS=63)
      parameter(MATOP_USE_SCALED_FORM=64)
      parameter(MATOP_SCALE_SYSTEM=65)
      parameter(MATOP_UNSCALE_SYSTEM=66)
      parameter(MATOP_SET_LOCAL_TO_GLOBAL_MAP=67)
      parameter(MATOP_SET_VALUES_LOCAL=68)
      parameter(MATOP_ZERO_ROWS_LOCAL=69)
      parameter(MATOP_GET_ROW_MAX=70)
      parameter(MATOP_CONVERT=71)
      parameter(MATOP_SET_COLORING=72)
      parameter(MATOP_SET_VALUES_ADIC=73)
      parameter(MATOP_SET_VALUES_ADIFOR=74)
      parameter(MATOP_FD_COLORING_APPLY=75)
      parameter(MATOP_SET_FROM_OPTIONS=76)
      parameter(MATOP_MULT_CON=77)
      parameter(MATOP_MULT_TRANSPOSE_CON=78)
      parameter(MATOP_ILU_FACTOR_SYMBOLIC_CON=79)
      parameter(MATOP_PERMUTE_SPARSIFY=80)
      parameter(MATOP_MULT_MULTIPLE=81)
      parameter(MATOP_SOLVE_MULTIPLE=82)
      parameter(MATOP_GET_INERTIA=83)
      parameter(MATOP_LOAD=84)
      parameter(MATOP_IS_SYMMETRIC=85)
      parameter(MATOP_IS_HERMITIAN=86)
      parameter(MATOP_IS_STRUCTURALLY_SYMMETRIC=87)
      parameter(MATOP_PB_RELAX=88)
      parameter(MATOP_GET_VECS=89)
      parameter(MATOP_MAT_MULT=90)
      parameter(MATOP_MAT_MULT_SYMBOLIC=91)
      parameter(MATOP_MAT_MULT_NUMERIC=92)
      parameter(MATOP_PTAP=93)
      parameter(MATOP_PTAP_SYMBOLIC=94)
      parameter(MATOP_PTAP_NUMERIC=95)
      parameter(MATOP_MAT_MULT_TRANSPOSE=96)
      parameter(MATOP_MAT_MULT_TRANSPOSE_SYM=97)
      parameter(MATOP_MAT_MULT_TRANSPOSE_NUM=98)
      parameter(MATOP_PTAP_SYMBOLIC_SEQAIJ=99)
      parameter(MATOP_PTAP_NUMERIC_SEQAIJ=100)
      parameter(MATOP_PTAP_SYMBOLIC_MPIAIJ=101)
      parameter(MATOP_PTAP_NUMERIC_MPIAIJ=102)
      parameter(MATOP_SET_VALUES_ROW=105)
!
!  
!
      PetscEnum MATRIX_BINARY_FORMAT_DENSE
      parameter (MATRIX_BINARY_FORMAT_DENSE=-1)
!
!  End of Fortran include file for the Mat package in PETSc
!
#endif
