C
C  $Id: mat.h,v 1.24 1996/08/30 14:35:15 bsmith Exp curfman $;
C
C  Include file for Fortran use of the Mat package in PETSc
C
#define Mat                 integer
#define MatType             integer
#define MatOption           integer 
#define MatAssemblyType     integer
#define MatReordering       integer
#define MatSORType          integer
#define MatInfoType         integer
#define MatGetSubMatrixCall integer
#define MatOperation        integer
#define MatInfo             Double

C
C  Matrix types
C
      integer MATSAME,MATSEQDENSE,MATSEQAIJ,MATMPIAIJ, 
     *        MATSHELL,MATMPIROWBS,MATSEQBDIAG,
     *        MATMPIBDIAG,MATMPIDENSE,MATSEQBAIJ,
     *        MATMPIBAIJ

      parameter (MATSAME=-1,MATSEQDENSE = 0,MATSEQAIJ = 1,
     *           MATMPIAIJ = 2,MATSHELL = 3, MATMPIROWBS = 4,
     *           MATSEQBDIAG = 5,MATMPIBDIAG = 6,MATMPIDENSE = 7,
     *           MATSEQBAIJ = 8, MATMPIBAIJ = 9)
C
C  Flag for matrix assembly
C
      integer MAT_FLUSH_ASSEMBLY,MAT_FINAL_ASSEMBLY

      parameter( MAT_FLUSH_ASSEMBLY=1,MAT_FINAL_ASSEMBLY=0)
C
C  Matrix options
C
      integer MAT_ROW_ORIENTED,MAT_COLUMN_ORIENTED,MAT_ROWS_SORTED,
     *        MAT_COLUMNS_SORTED,MAT_NO_NEW_NONZERO_LOCATIONS,
     *        MAT_YES_NEW_NONZERO_LOCATIONS,MAT_SYMMETRIC,
     *        MAT_STRUCTURALLY_SYMMETRIC,MAT_NO_NEW_DIAGONALS,
     *        MAT_YES_NEW_DIAGONALS,MAT_INODE_LIMIT_1,
     *        MAT_INODE_LIMIT_2,MAT_INODE_LIMIT_3,MAT_INODE_LIMIT_4,
     *        MAT_INODE_LIMIT_5
      
      parameter (MAT_ROW_ORIENTED=1,MAT_COLUMN_ORIENTED=2,
     *           MAT_ROWS_SORTED=4,MAT_COLUMNS_SORTED=8,
     *           MAT_NO_NEW_NONZERO_LOCATIONS=16,
     *           MAT_YES_NEW_NONZERO_LOCATIONS=32,
     *           MAT_SYMMETRIC=64,
     *           MAT_STRUCTURALLY_SYMMETRIC=65,
     *           MAT_NO_NEW_DIAGONALS=66,
     *           MAT_YES_NEW_DIAGONALS=67,MAT_INODE_LIMIT_1=68,
     *           MAT_INODE_LIMIT_2=69,MAT_INODE_LIMIT_3=70,
     *           MAT_INODE_LIMIT_4=71,
     *           MAT_INODE_LIMIT_5=72)

C
C  MatInfoType
C
      integer MAT_LOCAL,MAT_GLOBAL_MAX,MAT_GLOBAL_SUM

      parameter (MAT_LOCAL=1,MAT_GLOBAL_MAX=2,MAT_GLOBAL_SUM=3)

C
C  Note: MAT_INFO_SIZE must equal # elements in MatInfo structure
C  (See petsc/include/mat.h)
C
      integer MAT_INFO_SIZE
      parameter (MAT_INFO_SIZE=14)

      integer   MAT_INFO_ROWS_GLOBAL,MAT_INFO_COLUMNS_GLOBAL,
     *          MAT_INFO_ROWS_LOCAL,MAT_INFO_COLUMNS_LOCAL,
     *          MAT_INFO_BLOCK_SIZE,MAT_INFO_NZ_ALLOCATED,
     *          MAT_INFO_NZ_USED,MAT_INFO_NZ_UNNEEDED,
     *          MAT_INFO_MEMORY,MAT_INFO_ASSEMBLIES,
     *          MAT_INFO_MALLOCS,MAT_INFO_FILL_RATIO_GIVEN,
     *          MAT_INFO_FILL_RATIO_NEEDED,MAT_FACTOR_MALLOCS

      parameter (MAT_INFO_ROWS_GLOBAL=1,MAT_INFO_COLUMNS_GLOBAL=2,
     *          MAT_INFO_ROWS_LOCAL=3,MAT_INFO_COLUMNS_LOCAL=4,
     *          MAT_INFO_BLOCK_SIZE=5,MAT_INFO_NZ_ALLOCATED=6,
     *          MAT_INFO_NZ_USED=7,MAT_INFO_NZ_UNNEEDED=8,
     *          MAT_INFO_MEMORY=9,MAT_INFO_ASSEMBLIES=10,
     *          MAT_INFO_MALLOCS=11,MAT_INFO_FILL_RATIO_GIVEN=12,
     *          MAT_INFO_FILL_RATIO_NEEDED=13,MAT_FACTOR_MALLOCS=14)

C
C  MatSubMatrixCall
C
      integer MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX

      parameter (MAT_INITIAL_MATRIX=0, MAT_REUSE_MATRIX=1)
C
C  Matrix orderings
C
      integer ORDER_NATURAL,ORDER_ND,ORDER_1WD,
     *        ORDER_RCM,ORDER_QMD,ORDER_ROWLENGTH,ORDER_FLOW,
     *        ORDER_APPLICATION_1,ORDER_APPLICATION_2

      parameter( ORDER_NATURAL=0,ORDER_ND=1,ORDER_1WD=2,
     *           ORDER_RCM=3,ORDER_QMD=4,ORDER_ROWLENGTH=5,
     *           ORDER_FLOW=6,
     *           ORDER_APPLICATION_1=6,ORDER_APPLICATION_2=7)
C
C  Options for SOR and SSOR
C
      integer SOR_FORWARD_SWEEP,SOR_BACKWARD_SWEEP,SOR_SYMMETRIC_SWEEP,
     *        SOR_LOCAL_FORWARD_SWEEP,SOR_LOCAL_BACKWARD_SWEEP,
     *        SOR_LOCAL_SYMMETRIC_SWEEP,SOR_ZERO_INITIAL_GUESS,
     *        SOR_EISENSTAT,SOR_APPLY_UPPER,SOR_APPLY_LOWER

      parameter(SOR_FORWARD_SWEEP=1,SOR_BACKWARD_SWEEP=2,
     *          SOR_SYMMETRIC_SWEEP=3, SOR_LOCAL_FORWARD_SWEEP=4,
     *          SOR_LOCAL_BACKWARD_SWEEP=8,SOR_LOCAL_SYMMETRIC_SWEEP=12,
     *          SOR_ZERO_INITIAL_GUESS=16,SOR_EISENSTAT=32,
     *          SOR_APPLY_UPPER=64,SOR_APPLY_LOWER=128)

C
C  MatOperation
C
      integer MAT_SET_VALUES
      integer MAT_GET_ROW
      integer MAT_RESTORE_ROW
      integer MAT_MULT
      integer MAT_MULT_ADD
      integer MAT_MULT_TRANS
      integer MAT_MULT_TRANS_ADD
      integer MAT_SOLVE
      integer MAT_SOLVE_ADD
      integer MAT_SOLVE_TRANS
      integer MAT_SOLVE_TRANS_ADD
      integer MAT_LUFACTOR
      integer MAT_CHOLESKYFACTOR
      integer MAT_RELAX
      integer MAT_TRANSPOSE
      integer MAT_GETINFO
      integer MAT_EQUAL
      integer MAT_GET_DIAGONAL 
      integer MAT_DIAGONAL_SCALE
      integer MAT_NORM
      integer MAT_ASSEMBLY_BEGIN
      integer MAT_ASSEMBLY_END
      integer MAT_COMPRESS
      integer MAT_SET_OPTION
      integer MAT_ZERO_ENTRIES
      integer MAT_ZERO_ROWS
      integer MAT_LUFACTOR_SYMBOLIC
      integer MAT_LUFACTOR_NUMERIC
      integer MAT_CHOLESKY_FACTOR_SYMBOLIC
      integer MAT_CHOLESKY_FACTOR_NUMERIC
      integer MAT_GET_SIZE
      integer MAT_GET_LOCAL_SIZE
      integer MAT_GET_OWNERSHIP_RANGE
      integer MAT_ILUFACTOR_SYMBOLIC
      integer MAT_INCOMPLETECHOLESKYFACTOR_SY
      integer MAT_GET_ARRAY
      integer MAT_RESTORE_ARRAY
      integer MAT_CONVERT

      integer MAT_CONVERT_SAME_TYPE
      integer MAT_FORWARD_SOLVE
      integer MAT_BACKWARD_SOLVE
      integer MAT_ILUFACTOR
      integer MAT_INCOMPLETECHOLESKYFACTOR
      integer MAT_AXPY
      integer MAT_GET_SUBMATRICES
      integer MAT_INCREASE_OVERLAP
      integer MAT_GET_VALUES
      integer MAT_COPY
      integer MAT_PRINT_HELP
      integer MAT_SCALE
      integer MAT_SHIFT
      integer MAT_DIAGONAL_SHIFT
      integer MAT_ILUDT_FACTOR
      integer MAT_GET_BLOCK_SIZE

      integer MAT_DESTROY
      integer MAT_VIEW

      parameter(MAT_SET_VALUES=0)
      parameter(MAT_GET_ROW=1)
      parameter(MAT_RESTORE_ROW=2)
      parameter(MAT_MULT=3)
      parameter(MAT_MULT_ADD=4)
      parameter(MAT_MULT_TRANS=5)
      parameter(MAT_MULT_TRANS_ADD=6)
      parameter(MAT_SOLVE=7)
      parameter(MAT_SOLVE_ADD=8)
      parameter(MAT_SOLVE_TRANS=9)
      parameter(MAT_SOLVE_TRANS_ADD=10)
      parameter(MAT_LUFACTOR=11)
      parameter(MAT_CHOLESKYFACTOR=12)
      parameter(MAT_RELAX=13)
      parameter(MAT_TRANSPOSE=14)
      parameter(MAT_GETINFO=15)
      parameter(MAT_EQUAL=16)
      parameter(MAT_GET_DIAGONAL=17) 
      parameter(MAT_DIAGONAL_SCALE=18)
      parameter(MAT_NORM=19)
      parameter(MAT_ASSEMBLY_BEGIN=20)
      parameter(MAT_ASSEMBLY_END=21)
      parameter(MAT_COMPRESS=22)
      parameter(MAT_SET_OPTION=23)
      parameter(MAT_ZERO_ENTRIES=24)
      parameter(MAT_ZERO_ROWS=25)
      parameter(MAT_LUFACTOR_SYMBOLIC=26)
      parameter(MAT_LUFACTOR_NUMERIC=27)
      parameter(MAT_CHOLESKY_FACTOR_SYMBOLIC=28)
      parameter(MAT_CHOLESKY_FACTOR_NUMERIC=29)
      parameter(MAT_GET_SIZE=30)
      parameter(MAT_GET_LOCAL_SIZE=31)
      parameter(MAT_GET_OWNERSHIP_RANGE=32)
      parameter(MAT_ILUFACTOR_SYMBOLIC=33)
      parameter(MAT_INCOMPLETECHOLESKYFACTOR_SY=34)
      parameter(MAT_GET_ARRAY=35)
      parameter(MAT_RESTORE_ARRAY=36)
      parameter(MAT_CONVERT=37)

      parameter(MAT_CONVERT_SAME_TYPE=40)
      parameter(MAT_FORWARD_SOLVE=41)
      parameter(MAT_BACKWARD_SOLVE=42)
      parameter(MAT_ILUFACTOR=43)
      parameter(MAT_INCOMPLETECHOLESKYFACTOR=44)
      parameter(MAT_AXPY=45)
      parameter(MAT_GET_SUBMATRICES=46)
      parameter(MAT_INCREASE_OVERLAP=47)
      parameter(MAT_GET_VALUES=48)
      parameter(MAT_COPY=49)
      parameter(MAT_PRINT_HELP=50)
      parameter(MAT_SCALE=51)
      parameter(MAT_SHIFT=52)
      parameter(MAT_DIAGONAL_SHIFT=53)
      parameter(MAT_ILUDT_FACTOR=54)
      parameter(MAT_GET_BLOCK_SIZE=55)
      parameter(MAT_DESTROY=250)
      parameter(MAT_VIEW=251)
C
C  
C
      integer MATRIX_BINARY_FORMAT_DENSE
      parameter (MATRIX_BINARY_FORMAT_DENSE=-1)
C
C  End of Fortran include file for the Mat package in PETSc
C
