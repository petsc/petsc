/*
   3/99 Modified by Stephen Barnard to support SPAI version 3.0
*/

/*
      Provides an interface to the SPAI Sparse Approximate Inverse Preconditioner
   Code written by Stephen Barnard.

      Note: there is some BAD memory bleeding below!

      This code needs work

   1) get rid of all memory bleeding
   2) fix PETSc/interface so that it gets if the matrix is symmetric from the matrix
      rather than having the sp flag for PC_SPAI
   3) fix to set the block size based on the matrix block size

*/
#include <petsc/private/petscimpl.h>
#include <petscpc.h>

/*@
  PCSPAISetEpsilon -- Set the tolerance for the `PCSPAI` preconditioner

  Input Parameters:
+ pc       - the preconditioner
- epsilon1 - the tolerance (default .4)

  Level: intermediate

  Note:
  `espilon1` must be between 0 and 1. It controls the
  quality of the approximation of M to the inverse of
  A. Higher values of `epsilon1` lead to more work, more
  fill, and usually better preconditioners. In many
  cases the best choice of `epsilon1` is the one that
  divides the total solution time equally between the
  preconditioner and the solver.

.seealso: [](ch_ksp), `PCSPAI`, `PCSetType()`
  @*/
PetscErrorCode PCSPAISetEpsilon(PC pc, PetscReal epsilon1)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetEpsilon_C", (PC, PetscReal), (pc, epsilon1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSPAISetNBSteps - set maximum number of improvement steps per row in
  the `PCSPAI` preconditioner

  Input Parameters:
+ pc       - the preconditioner
- nbsteps1 - number of steps (default 5)

  Note:
  `PCSPAI` constructs to approximation to every column of
  the exact inverse of A in a series of improvement
  steps. The quality of the approximation is determined
  by epsilon. If an approximation achieving an accuracy
  of epsilon is not obtained after `nbsteps1` steps, `PCSPAI` simply
  uses the best approximation constructed so far.

  Level: intermediate

.seealso: [](ch_ksp), `PCSPAI`, `PCSetType()`, `PCSPAISetMaxNew()`
@*/
PetscErrorCode PCSPAISetNBSteps(PC pc, PetscInt nbsteps1)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetNBSteps_C", (PC, PetscInt), (pc, nbsteps1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* added 1/7/99 g.h. */
/*@
  PCSPAISetMax - set the size of various working buffers in the `PCSPAI` preconditioner

  Input Parameters:
+ pc   - the preconditioner
- max1 - size (default is 5000)

  Level: intermediate

.seealso: [](ch_ksp), `PCSPAI`, `PCSetType()`
@*/
PetscErrorCode PCSPAISetMax(PC pc, PetscInt max1)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetMax_C", (PC, PetscInt), (pc, max1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSPAISetMaxNew - set maximum number of new nonzero candidates per step in the `PCSPAI` preconditioner

  Input Parameters:
+ pc      - the preconditioner
- maxnew1 - maximum number (default 5)

  Level: intermediate

.seealso: [](ch_ksp), `PCSPAI`, `PCSetType()`, `PCSPAISetNBSteps()`
@*/
PetscErrorCode PCSPAISetMaxNew(PC pc, PetscInt maxnew1)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetMaxNew_C", (PC, PetscInt), (pc, maxnew1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSPAISetBlockSize - set the block size for the `PCSPAI` preconditioner

  Input Parameters:
+ pc          - the preconditioner
- block_size1 - block size (default 1)

  Level: intermediate

  Notes:
  A block
  size of 1 treats A as a matrix of scalar elements. A
  block size of s > 1 treats A as a matrix of sxs
  blocks. A block size of 0 treats A as a matrix with
  variable sized blocks, which are determined by
  searching for dense square diagonal blocks in A.
  This can be very effective for finite-element
  matrices.

  SPAI will convert A to block form, use a block
  version of the preconditioner algorithm, and then
  convert the result back to scalar form.

  In many cases the a block-size parameter other than 1
  can lead to very significant improvement in
  performance.

  Developer Note:
  This preconditioner could use the matrix block size as the default block size to use

.seealso: [](ch_ksp), `PCSPAI`, `PCSetType()`
@*/
PetscErrorCode PCSPAISetBlockSize(PC pc, PetscInt block_size1)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetBlockSize_C", (PC, PetscInt), (pc, block_size1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSPAISetCacheSize - specify cache size in the `PCSPAI` preconditioner

  Input Parameters:
+ pc         - the preconditioner
- cache_size - cache size {0,1,2,3,4,5} (default 5)

  Level: intermediate

  Note:
  `PCSPAI` uses a hash table to cache messages and avoid
  redundant communication. If suggest always using
  5. This parameter is irrelevant in the serial
  version.

.seealso: [](ch_ksp), `PCSPAI`, `PCSetType()`
@*/
PetscErrorCode PCSPAISetCacheSize(PC pc, PetscInt cache_size)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetCacheSize_C", (PC, PetscInt), (pc, cache_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSPAISetVerbose - verbosity level for the `PCSPAI` preconditioner

  Input Parameters:
+ pc      - the preconditioner
- verbose - level (default 1)

  Level: intermediate

  Note:
  Prints parameters, timings and matrix statistics

.seealso: [](ch_ksp), `PCSPAI`, `PCSetType()`
@*/
PetscErrorCode PCSPAISetVerbose(PC pc, PetscInt verbose)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetVerbose_C", (PC, PetscInt), (pc, verbose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSPAISetSp - specify a symmetric matrix sparsity pattern in the `PCSPAI` preconditioner

  Input Parameters:
+ pc - the preconditioner
- sp - 0 or 1

  Level: intermediate

  Note:
  If A has a symmetric nonzero pattern use `sp` 1 to
  improve performance by eliminating some communication
  in the parallel version. Even if A does not have a
  symmetric nonzero pattern `sp` 1 may well lead to good
  results, but the code will not follow the published
  SPAI algorithm exactly.

.seealso: [](ch_ksp), `PCSPAI`, `PCSetType()`
@*/
PetscErrorCode PCSPAISetSp(PC pc, PetscInt sp)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetSp_C", (PC, PetscInt), (pc, sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
