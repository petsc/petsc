C
C  $Id: mat.h,v 1.27 1996/09/27 22:09:16 balay Exp curfman $;
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
#define MatColoring         integer
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
     *           ORDER_APPLICATION_1=7,ORDER_APPLICATION_2=8)
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
C MAtColoring
C
      integer COLORING_NATURAL, COLORING_SL, COLORING_LD, COLORING_IF,
     *        COLORING_APPLICATION_1,COLORING_APPLICATION_2

      parameter (COLORING_NATURAL=0, COLORING_SL=1, COLORING_LD=2,
     *          COLORING_IF=3, COLORING_APPLICATION_1=4,
     *          COLORING_APPLICATION_2=5)
C
C  MatOperation
C
      integer MATOP_SET_VALUES
      integer MATOP_GET_ROW
      integer MATOP_RESTORE_ROW
      integer MATOP_MULT
      integer MATOP_MULT_ADD
      integer MATOP_MULT_TRANS
      integer MATOP_MULT_TRANS_ADD
      integer MATOP_SOLVE
      integer MATOP_SOLVE_ADD
      integer MATOP_SOLVE_TRANS
      integer MATOP_SOLVE_TRANS_ADD
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
      integer MATOP_I_CHOLESKYFACTOR_SY
      integer MATOP_GET_ARRAY
      integer MATOP_RESTORE_ARRAY
      integer MATOP_CONVERT

      integer MATOP_CONVERT_SAME_TYPE
      integer MATOP_FORWARD_SOLVE
      integer MATOP_BACKWARD_SOLVE
      integer MATOP_ILUFACTOR
      integer MATOP_INCOMPLETECHOLESKYFACTOR
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

      integer MATOP_DESTROY
      integer MATOP_VIEW

      parameter(MATOP_SET_VALUES=0)
      parameter(MATOP_GET_ROW=1)
      parameter(MATOP_RESTORE_ROW=2)
      parameter(MATOP_MULT=3)
      parameter(MATOP_MULT_ADD=4)
      parameter(MATOP_MULT_TRANS=5)
      parameter(MATOP_MULT_TRANS_ADD=6)
      parameter(MATOP_SOLVE=7)
      parameter(MATOP_SOLVE_ADD=8)
      parameter(MATOP_SOLVE_TRANS=9)
      parameter(MATOP_SOLVE_TRANS_ADD=10)
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
      parameter(MATOP_I_CHOLESKYFACTOR_SY=34)
      parameter(MATOP_GET_ARRAY=35)
      parameter(MATOP_RESTORE_ARRAY=36)
      parameter(MATOP_CONVERT=37)

      parameter(MATOP_CONVERT_SAME_TYPE=40)
      parameter(MATOP_FORWARD_SOLVE=41)
      parameter(MATOP_BACKWARD_SOLVE=42)
      parameter(MATOP_ILUFACTOR=43)
      parameter(MATOP_INCOMPLETECHOLESKYFACTOR=44)
      parameter(MATOP_AXPY=45)
      parameter(MATOP_GET_SUBMATRICES=46)
      parameter(MATOP_INCREASE_OVERLAP=47)
      parameter(MATOP_GET_VALUES=48)
      parameter(MATOP_COPY=49)
      parameter(MATOP_PRINT_HELP=50)
      parameter(MATOP_SCALE=51)
      parameter(MATOP_SHIFT=52)
      parameter(MATOP_DIAGONAL_SHIFT=53)
      parameter(MATOP_ILUDT_FACTOR=54)
      parameter(MATOP_GET_BLOCK_SIZE=55)
      parameter(MATOP_DESTROY=250)
      parameter(MATOP_VIEW=251)
C
C  
C
      integer MATRIX_BINARY_FORMAT_DENSE
      parameter (MATRIX_BINARY_FORMAT_DENSE=-1)
C
C  End of Fortran include file for the Mat package in PETSc
C
