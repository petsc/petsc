C
C  $Id: petsc.h,v 1.18 1996/02/12 20:26:22 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the Mat package in PETSc
C
#define Mat             integer
#define MatType         integer
#define MatOption       integer 
#define MatAssemblyType integer
#define MatOrdering     integer
#define MATSORType      integer
#define MatInfoType     integer

C
C  Matrix types
C
      integer MATSAME, MATSEQDENSE, MATSEQAIJ, MATMPIAIJ, MATSHELL, 
     *        MATSEQROW, MATMPIROW, MATMPIROWBS, MATSEQBDIAG, 
     *        MATMPIBDIAG

      parameter(MATSAME=-1,MATSEQDENSE = 0,MATSEQAIJ = 1,MATMPIAIJ = 2, 
     *          MATSHELL = 3, MATSEQROW = 4, MATMPIROW = 5, 
     *          MATMPIROWBS = 6, MATSEQBDIAG = 7, MATMPIBDIAG = 8)
C
C  Flag for matrix assembly
C
      integer FLUSH_ASSEMBLY,FINAL_ASSEMBLY

      parameter( FLUSH_ASSEMBLY=1,FINAL_ASSEMBLY=0)
C
C  Matrix options
C
      integer ROW_ORIENTED,COLUMN_ORIENTED,ROWS_SORTED,
     *        COLUMNS_SORTED,NO_NEW_NONZERO_LOCATIONS,
     *        YES_NEW_NONZERO_LOCATIONS,SYMMETRIC_MATRIX

      parameter( ROW_ORIENTED=1,COLUMN_ORIENTED=2,ROWS_SORTED=4,
     *           COLUMNS_SORTED=8,NO_NEW_NONZERO_LOCATIONS=16,
     *           YES_NEW_NONZERO_LOCATIONS=32,SYMMETRIC_MATRIX=64)
C
C  Matrix orderings
C
      integer ORDER_NATURAL,ORDER_ND,ORDER_1WD,
     *        ORDER_RCM,ORDER_QMD,ORDER_APPLICATION_1,
     *        ORDER_APPLICATION_2

      parameter( ORDER_NATURAL=0,ORDER_ND=1,ORDER_1WD=2,
     *           ORDER_RCM=3,ORDER_QMD=4,ORDER_APPLICATION_1=5,
     *           ORDER_APPLICATION_2=6)
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
C  Flags for MatGetInfo()
C
      integer MAT_LOCAL,MAT_GLOBAL_MAX,MAT_GLOBAL_SUM

      parameter( MAT_LOCAL=1,MAT_GLOBAL_MAX=2,MAT_GLOBAL_SUM=3)
C
C  End of Fortran include file for the Mat package in PETSc

