C
C  $Id: mat.h,v 1.16 1996/07/02 18:09:35 bsmith Exp bsmith $;
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
C  MatOperation is too huge. Is it reguired???
C

C
C  
C
      integer MATRIX_BINARY_FORMAT_DENSE
      parameter (MATRIX_BINARY_FORMAT_DENSE=-1)
C
C  End of Fortran include file for the Mat package in PETSc

