
C      Include file for for Fortran use of the SNES package
C
       INTEGER NLE_NLS1, NLE_NTR1, NLE_NTR2_DOG, NLE_NTR2_LIN
       INTEGER NLE_NBASIC, NLM_NLS1, NLM_NTR1
       PARAMETER( NLE_NLS1 = 0, NLE_NTR1 = 1, NLE_NTR2_DOG = 2, 
     *            NLE_NTR2_LIN = 3, NLE_NBASIC = 4, NLM_NLS1 = 5, 
     *            NLM_NTR1 = 6 )

       integer nlcreate
       external nlcreate
C
C      Note:  Be sure that the following SpMat types correspond to the
C      definitions in "sparse/sppriv.h"
C
       INTEGER MATROW, MATAIJ, MATBLOCK, MATROWDIST, MATDIAG, MATDENSE
       INTEGER MATPDERIVED, MATBDIAG
       PARAMETER( MATROW = 1, MATAIJ = 3, MATBLOCK = 4, MATROWDIST = 5,
     *            MATDIAG = 6, MATDENSE = 7, MATPDERIVED = 8, 
     *            MATBDIAG = 9 )
C
C      Include file for SLES and KSP (linear solvers)
       include "../solvers/svfort.h"
C
C      End of Fortran include file for the SNES package

