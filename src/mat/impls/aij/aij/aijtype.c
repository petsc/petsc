
#include <petsc-private/matimpl.h>

/*MC
   MATAIJ - MATAIJ = "aij" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJ when constructed with a single process communicator,
   and MATMPIAIJ otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type aij - sets the matrix type to "aij" during a call to MatSetFromOptions()

  Developer Notes: Subclasses include MATAIJCUSP, MATAIJPERM, MATAIJCRL, and also automatically switches over to use inodes when 
   enough exist.

  Level: beginner

.seealso: MatCreateAIJ(), MatCreateSeqAIJ(), MATSEQAIJ,MATMPIAIJ
M*/

/*MC
   MATAIJCRL - MATAIJCRL = "aijcrl" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJCRL when constructed with a single process communicator,
   and MATMPIAIJCRL otherwise.  As a result, for single process communicators, 
   MatSeqAIJSetPreallocation() is supported, and similarly MatMPIAIJSetPreallocation() is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type aijcrl - sets the matrix type to "aijcrl" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIAIJCRL,MATSEQAIJCRL,MATMPIAIJCRL, MATSEQAIJCRL, MATMPIAIJCRL
M*/

