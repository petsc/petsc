!
!  $Id: petscmat.h,v 1.73 2001/08/06 21:19:26 bsmith Exp $;
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
#define MatOption integer 
#define MatAssemblyType integer
#define MatOrderingType character*(80)
#define MatSORType integer
#define MatInfoType integer
#define MatReuse integer
#define MatOperation integer
#define MatColoringType character*(80)
#define MatInfo double precision
#define MatFactorInfo double precision
#define MatDuplicateOption integer      
#define MatStructure integer
#define MatPartitioningType character*(80)
#define MatNullSpace PetscFortranAddr

#define MatStencil integer
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


#if !defined (PETSC_AVOID_DECLARATIONS)

!
!  Flag for matrix assembly
!
      integer MAT_FLUSH_ASSEMBLY,MAT_FINAL_ASSEMBLY

      parameter(MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0)
!
!  Matrix options; must match those in include/petscmat.h
!
      integer MAT_ROW_ORIENTED,MAT_COLUMN_ORIENTED,MAT_ROWS_SORTED
      integer MAT_COLUMNS_SORTED,MAT_NO_NEW_NONZERO_LOCATIONS
      integer MAT_YES_NEW_NONZERO_LOCATIONS,MAT_SYMMETRIC
      integer MAT_STRUCTURALLY_SYMMETRIC,MAT_NO_NEW_DIAGONALS
      integer MAT_YES_NEW_DIAGONALS,MAT_INODE_LIMIT_1
      integer MAT_INODE_LIMIT_2,MAT_INODE_LIMIT_3,MAT_INODE_LIMIT_4
      integer MAT_INODE_LIMIT_5,MAT_IGNORE_OFF_PROC_ENTRIES
      integer MAT_ROWS_UNSORTED,MAT_COLUMNS_UNSORTED
      integer MAT_NEW_NONZERO_LOCATION_ERR
      integer MAT_NEW_NONZERO_ALLOCATION_ERR,MAT_USE_HASH_TABLE
      integer MAT_KEEP_ZEROED_ROWS,MAT_IGNORE_ZERO_ENTRIES
      integer MAT_USE_INODES,MAT_DO_NOT_USE_INODES
      integer MAT_NOT_SYMMETRIC,MAT_HERMITIAN
      integer MAT_NOT_STRUCTURALLY_SYMMETRIC,MAT_NOT_HERMITIAN
      integer MAT_SYMMETRY_ETERNAL,MAT_NOT_SYMMETRY_ETERNAL

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
      integer MAT_DO_NOT_COPY_VALUES,MAT_COPY_VALUES
      parameter (MAT_DO_NOT_COPY_VALUES=0,MAT_COPY_VALUES=1)
!
!  Flags for PCSetOperators()
!
      integer SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN
      integer SAME_PRECONDITIONER

      parameter (SAME_NONZERO_PATTERN = 0,DIFFERENT_NONZERO_PATTERN = 1)
      parameter (SAME_PRECONDITIONER = 2)

!
!  Note: MAT_INFO_SIZE must equal # elements in MatInfo structure
!  (See petsc/include/petscmat.h)
!
      integer   MAT_INFO_SIZE

      parameter (MAT_INFO_SIZE=14)

      integer MAT_INFO_ROWS_GLOBAL,MAT_INFO_COLUMNS_GLOBAL
      integer MAT_INFO_ROWS_LOCAL,MAT_INFO_COLUMNS_LOCAL
      integer MAT_INFO_BLOCK_SIZE,MAT_INFO_NZ_ALLOCATED
      integer MAT_INFO_NZ_USED,MAT_INFO_NZ_UNNEEDED
      integer MAT_INFO_MEMORY,MAT_INFO_ASSEMBLIES
      integer MAT_INFO_MALLOCS,MAT_INFO_FILL_RATIO_GIVEN
      integer MAT_INFO_FILL_RATIO_NEEDED,MAT_INFO_FACTOR_MALLOCS

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
      integer MAT_INITIAL_MATRIX,MAT_REUSE_MATRIX

      parameter (MAT_INITIAL_MATRIX=0,MAT_REUSE_MATRIX=1)

!
!  MatInfoType
!
      integer MAT_LOCAL,MAT_GLOBAL_MAX,MAT_GLOBAL_SUM

      parameter (MAT_LOCAL=1,MAT_GLOBAL_MAX=2,MAT_GLOBAL_SUM=3)

!
!  Note: MAT_FACTORINFO_SIZE must equal # elements in MatFactorInfo structure
!  (See petsc/include/petscmat.h)
!
      integer   MAT_FACTORINFO_SIZE

      parameter (MAT_FACTORINFO_SIZE=11)

      integer MAT_FACTORINFO_LEVELS
      integer MAT_FACTORINFO_FILL
      integer MAT_FACTORINFO_DIAGONAL_FILL
      integer MAT_FACTORINFO_DT
      integer MAT_FACTORINFO_DTCOL
      integer MAT_FACTORINFO_DTCOUNT
      integer MAT_FACTORINFO_DAMPING
      integer MAT_FACTORINFO_SHIFT
      integer MAT_FACTORINFO_SHIFT_FRACTION
      integer MAT_FACTORINFO_ZERO_PIVOT
      integer MAT_FACTORINFO_PIVOT_IN_BLOCKS

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
      integer SOR_FORWARD_SWEEP,SOR_BACKWARD_SWEEP,SOR_SYMMETRIC_SWEEP
      integer SOR_LOCAL_FORWARD_SWEEP,SOR_LOCAL_BACKWARD_SWEEP
      integer SOR_LOCAL_SYMMETRIC_SWEEP,SOR_ZERO_INITIAL_GUESS
      integer SOR_EISENSTAT,SOR_APPLY_UPPER,SOR_APPLY_LOWER

      parameter (SOR_FORWARD_SWEEP=1,SOR_BACKWARD_SWEEP=2)
      parameter (SOR_SYMMETRIC_SWEEP=3,SOR_LOCAL_FORWARD_SWEEP=4)
      parameter (SOR_LOCAL_BACKWARD_SWEEP=8)
      parameter (SOR_LOCAL_SYMMETRIC_SWEEP=12)
      parameter (SOR_ZERO_INITIAL_GUESS=16,SOR_EISENSTAT=32)
      parameter (SOR_APPLY_UPPER=64,SOR_APPLY_LOWER=128)
!
!  MatOperation
!
      integer MATOP_SET_VALUES
      integer MATOP_GET_ROW
      integer MATOP_RESTORE_ROW
      integer MATOP_MULT
      integer MATOP_MULT_ADD
      integer MATOP_MULT_TRANSPOSE
      integer MATOP_MULT_TRANSPOSE_ADD
      integer MATOP_SOLVE
      integer MATOP_SOLVE_ADD
      integer MATOP_SOLVE_TRANSPOSE
      integer MATOP_SOLVE_TRANSPOSE_ADD
      integer MATOP_LUFACTOR
      integer MATOP_CHOLESKYFACTOR
      integer MATOP_RELAX
      integer MATOP_TRANSPOSE
      integer MATOP_GETINFO
      integer MATOP_EQUAL
      integer MATOP_GET_DIAGONAL 
      integer MATOP_DIAGONAL_SCALE
      integer MATOP_NORM
      integer MATOP_ASSEMBLY_BEGIN
      integer MATOP_ASSEMBLY_END
      integer MATOP_COMPRESS
      integer MATOP_SET_OPTION
      integer MATOP_ZERO_ENTRIES
      integer MATOP_ZERO_ROWS
      integer MATOP_LUFACTOR_SYMBOLIC
      integer MATOP_LUFACTOR_NUMERIC
      integer MATOP_CHOLESKY_FACTOR_SYMBOLIC
      integer MATOP_CHOLESKY_FACTOR_NUMERIC
      integer MATOP_GET_SIZE
      integer MATOP_GET_LOCAL_SIZE
      integer MATOP_GET_OWNERSHIP_RANGE
      integer MATOP_ILUFACTOR_SYMBOLIC
      integer MATOP_ICCFACTOR_SYMBOLIC
      integer MATOP_GET_ARRAY
      integer MATOP_RESTORE_ARRAY

      integer MATOP_CONVERT_SAME_TYPE
      integer MATOP_FORWARD_SOLVE
      integer MATOP_BACKWARD_SOLVE
      integer MATOP_ILUFACTOR
      integer MATOP_ICCFACTOR
      integer MATOP_AXPY
      integer MATOP_GET_SUBMATRICES
      integer MATOP_INCREASE_OVERLAP
      integer MATOP_GET_VALUES
      integer MATOP_COPY
      integer MATOP_PRINT_HELP
      integer MATOP_SCALE
      integer MATOP_SHIFT
      integer MATOP_DIAGONAL_SHIFT
      integer MATOP_ILUDT_FACTOR
      integer MATOP_GET_BLOCK_SIZE

      integer MATOP_GET_ROW_IJ
      integer MATOP_RESTORE_ROW_IJ
      integer MATOP_GET_COLUMN_IJ
      integer MATOP_RESTORE_COLUMN_IJ
      integer MATOP_FDCOLORING_CREATE
      integer MATOP_COLORING_PATCH
      integer MATOP_SET_UNFACTORED
      integer MATOP_PERMUTE
      integer MATOP_SET_VALUES_BLOCKED

      integer MATOP_DESTROY
      integer MATOP_VIEW

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
      parameter(MATOP_GET_SIZE=30)
      parameter(MATOP_GET_LOCAL_SIZE=31)
      parameter(MATOP_GET_OWNERSHIP_RANGE=32)
      parameter(MATOP_ILUFACTOR_SYMBOLIC=33)
      parameter(MATOP_ICCFACTOR_SYMBOLIC=34)
      parameter(MATOP_GET_ARRAY=35)
      parameter(MATOP_RESTORE_ARRAY=36)

      parameter(MATOP_CONVERT_SAME_TYPE=37)
      parameter(MATOP_FORWARD_SOLVE=38)
      parameter(MATOP_BACKWARD_SOLVE=39)
      parameter(MATOP_ILUFACTOR=40)
      parameter(MATOP_ICCFACTOR=41)
      parameter(MATOP_AXPY=42)
      parameter(MATOP_GET_SUBMATRICES=43)
      parameter(MATOP_INCREASE_OVERLAP=44)
      parameter(MATOP_GET_VALUES=45)
      parameter(MATOP_COPY=46)
      parameter(MATOP_PRINT_HELP=47)
      parameter(MATOP_SCALE=48)
      parameter(MATOP_SHIFT=49)
      parameter(MATOP_DIAGONAL_SHIFT=50)
      parameter(MATOP_ILUDT_FACTOR=51)
      parameter(MATOP_GET_BLOCK_SIZE=52)

      parameter(MATOP_GET_ROW_IJ=53)
      parameter(MATOP_RESTORE_ROW_IJ=54)
      parameter(MATOP_GET_COLUMN_IJ=55)
      parameter(MATOP_RESTORE_COLUMN_IJ=56)
      parameter(MATOP_FDCOLORING_CREATE=57)
      parameter(MATOP_COLORING_PATCH=58)
      parameter(MATOP_SET_UNFACTORED=59)
      parameter(MATOP_PERMUTE=60)
      parameter(MATOP_SET_VALUES_BLOCKED=61)


      parameter(MATOP_DESTROY=250)
      parameter(MATOP_VIEW=251)
!
!  
!
      integer MATRIX_BINARY_FORMAT_DENSE
      parameter (MATRIX_BINARY_FORMAT_DENSE=-1)
!
!  End of Fortran include file for the Mat package in PETSc
!
#endif
