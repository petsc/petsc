
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
#if !defined(PETSC_SKIP_COMPLEX)
  #define PETSC_SKIP_COMPLEX /* since spai uses I which conflicts with some complex implementations */
#endif

#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/spai/petscspai.h>

/*
    These are the SPAI include files
*/
EXTERN_C_BEGIN
#define SPAI_USE_MPI /* required for setting SPAI_Comm correctly in basics.h */
#include <spai.h>
#include <matrix.h>
EXTERN_C_END

extern PetscErrorCode ConvertMatToMatrix(MPI_Comm, Mat, Mat, matrix **);
extern PetscErrorCode ConvertMatrixToMat(MPI_Comm, matrix *, Mat *);
extern PetscErrorCode ConvertVectorToVec(MPI_Comm, vector *, Vec *);
extern PetscErrorCode MM_to_PETSC(char *, char *, char *);

typedef struct {
  matrix *B;  /* matrix in SPAI format */
  matrix *BT; /* transpose of matrix in SPAI format */
  matrix *M;  /* the approximate inverse in SPAI format */

  Mat PM; /* the approximate inverse PETSc format */

  double epsilon;    /* tolerance */
  int    nbsteps;    /* max number of "improvement" steps per line */
  int    max;        /* max dimensions of is_I, q, etc. */
  int    maxnew;     /* max number of new entries per step */
  int    block_size; /* constant block size */
  int    cache_size; /* one of (1,2,3,4,5,6) indicting size of cache */
  int    verbose;    /* SPAI prints timing and statistics */

  int      sp;        /* symmetric nonzero pattern */
  MPI_Comm comm_spai; /* communicator to be used with spai */
} PC_SPAI;

static PetscErrorCode PCSetUp_SPAI(PC pc)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;
  Mat      AT;

  PetscFunctionBegin;
  init_SPAI();

  if (ispai->sp) {
    PetscCall(ConvertMatToMatrix(ispai->comm_spai, pc->pmat, pc->pmat, &ispai->B));
  } else {
    /* Use the transpose to get the column nonzero structure. */
    PetscCall(MatTranspose(pc->pmat, MAT_INITIAL_MATRIX, &AT));
    PetscCall(ConvertMatToMatrix(ispai->comm_spai, pc->pmat, AT, &ispai->B));
    PetscCall(MatDestroy(&AT));
  }

  /* Destroy the transpose */
  /* Don't know how to do it. PETSc developers? */

  /* construct SPAI preconditioner */
  /* FILE *messages */    /* file for warning messages */
  /* double epsilon */    /* tolerance */
  /* int nbsteps */       /* max number of "improvement" steps per line */
  /* int max */           /* max dimensions of is_I, q, etc. */
  /* int maxnew */        /* max number of new entries per step */
  /* int block_size */    /* block_size == 1 specifies scalar elements
                              block_size == n specifies nxn constant-block elements
                              block_size == 0 specifies variable-block elements */
  /* int cache_size */    /* one of (1,2,3,4,5,6) indicting size of cache. cache_size == 0 indicates no caching */
  /* int    verbose    */ /* verbose == 0 specifies that SPAI is silent
                              verbose == 1 prints timing and matrix statistics */

  PetscCallExternal(bspai, ispai->B, &ispai->M, stdout, ispai->epsilon, ispai->nbsteps, ispai->max, ispai->maxnew, ispai->block_size, ispai->cache_size, ispai->verbose);

  PetscCall(ConvertMatrixToMat(PetscObjectComm((PetscObject)pc), ispai->M, &ispai->PM));

  /* free the SPAI matrices */
  sp_free_matrix(ispai->B);
  sp_free_matrix(ispai->M);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_SPAI(PC pc, Vec xx, Vec y)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  /* Now using PETSc's multiply */
  PetscCall(MatMult(ispai->PM, xx, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApply_SPAI(PC pc, Mat X, Mat Y)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  /* Now using PETSc's multiply */
  PetscCall(MatMatMult(ispai->PM, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_SPAI(PC pc)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ispai->PM));
  PetscCallMPI(MPI_Comm_free(&(ispai->comm_spai)));
  PetscCall(PetscFree(pc->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetEpsilon_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetNBSteps_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetMax_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetMaxNew_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetBlockSize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetCacheSize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetVerbose_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetSp_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_SPAI(PC pc, PetscViewer viewer)
{
  PC_SPAI  *ispai = (PC_SPAI *)pc->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "    epsilon %g\n", ispai->epsilon));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    nbsteps %d\n", ispai->nbsteps));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    max %d\n", ispai->max));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    maxnew %d\n", ispai->maxnew));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    block_size %d\n", ispai->block_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    cache_size %d\n", ispai->cache_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    verbose %d\n", ispai->verbose));
    PetscCall(PetscViewerASCIIPrintf(viewer, "    sp %d\n", ispai->sp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSPAISetEpsilon_SPAI(PC pc, PetscReal epsilon1)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  ispai->epsilon = (double)epsilon1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSPAISetNBSteps_SPAI(PC pc, PetscInt nbsteps1)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  ispai->nbsteps = (int)nbsteps1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* added 1/7/99 g.h. */
static PetscErrorCode PCSPAISetMax_SPAI(PC pc, PetscInt max1)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  ispai->max = (int)max1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSPAISetMaxNew_SPAI(PC pc, PetscInt maxnew1)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  ispai->maxnew = (int)maxnew1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSPAISetBlockSize_SPAI(PC pc, PetscInt block_size1)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  ispai->block_size = (int)block_size1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSPAISetCacheSize_SPAI(PC pc, PetscInt cache_size)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  ispai->cache_size = (int)cache_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSPAISetVerbose_SPAI(PC pc, PetscInt verbose)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  ispai->verbose = (int)verbose;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSPAISetSp_SPAI(PC pc, PetscInt sp)
{
  PC_SPAI *ispai = (PC_SPAI *)pc->data;

  PetscFunctionBegin;
  ispai->sp = (int)sp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSPAISetEpsilon -- Set the tolerance for the `PCSPAI` preconditioner

  Input Parameters:
+ pc - the preconditioner
- eps - epsilon (default .4)

  Note:
    Espilon must be between 0 and 1. It controls the
                 quality of the approximation of M to the inverse of
                 A. Higher values of epsilon lead to more work, more
                 fill, and usually better preconditioners. In many
                 cases the best choice of epsilon is the one that
                 divides the total solution time equally between the
                 preconditioner and the solver.

  Level: intermediate

.seealso: `PCSPAI`, `PCSetType()`
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
+ pc - the preconditioner
- n - number of steps (default 5)

  Note:
    `PCSPAI` constructs to approximation to every column of
                 the exact inverse of A in a series of improvement
                 steps. The quality of the approximation is determined
                 by epsilon. If an approximation achieving an accuracy
                 of epsilon is not obtained after ns steps, SPAI simply
                 uses the best approximation constructed so far.

  Level: intermediate

.seealso: `PCSPAI`, `PCSetType()`, `PCSPAISetMaxNew()`
@*/
PetscErrorCode PCSPAISetNBSteps(PC pc, PetscInt nbsteps1)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetNBSteps_C", (PC, PetscInt), (pc, nbsteps1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* added 1/7/99 g.h. */
/*@
  PCSPAISetMax - set the size of various working buffers in
        the `PCSPAI` preconditioner

  Input Parameters:
+ pc - the preconditioner
- n - size (default is 5000)

  Level: intermediate

.seealso: `PCSPAI`, `PCSetType()`
@*/
PetscErrorCode PCSPAISetMax(PC pc, PetscInt max1)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetMax_C", (PC, PetscInt), (pc, max1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCSPAISetMaxNew - set maximum number of new nonzero candidates per step
   in `PCSPAI` preconditioner

  Input Parameters:
+ pc - the preconditioner
- n - maximum number (default 5)

  Level: intermediate

.seealso: `PCSPAI`, `PCSetType()`, `PCSPAISetNBSteps()`
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
+ pc - the preconditioner
- n - block size (default 1)

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

  Level: intermediate

.seealso: `PCSPAI`, `PCSetType()`
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
+ pc - the preconditioner
- n -  cache size {0,1,2,3,4,5} (default 5)

  Note:
    `PCSPAI` uses a hash table to cache messages and avoid
                 redundant communication. If suggest always using
                 5. This parameter is irrelevant in the serial
                 version.

  Level: intermediate

.seealso: `PCSPAI`, `PCSetType()`
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
+ pc - the preconditioner
- n - level (default 1)

  Note:
    print parameters, timings and matrix statistics

  Level: intermediate

.seealso: `PCSPAI`, `PCSetType()`
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
- n - 0 or 1

  Note:
    If A has a symmetric nonzero pattern use -sp 1 to
                 improve performance by eliminating some communication
                 in the parallel version. Even if A does not have a
                 symmetric nonzero pattern -sp 1 may well lead to good
                 results, but the code will not follow the published
                 SPAI algorithm exactly.

  Level: intermediate

.seealso: `PCSPAI`, `PCSetType()`
@*/
PetscErrorCode PCSPAISetSp(PC pc, PetscInt sp)
{
  PetscFunctionBegin;
  PetscTryMethod(pc, "PCSPAISetSp_C", (PC, PetscInt), (pc, sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_SPAI(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_SPAI  *ispai = (PC_SPAI *)pc->data;
  int       nbsteps1, max1, maxnew1, block_size1, cache_size, verbose, sp;
  double    epsilon1;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SPAI options");
  PetscCall(PetscOptionsReal("-pc_spai_epsilon", "", "PCSPAISetEpsilon", ispai->epsilon, &epsilon1, &flg));
  if (flg) PetscCall(PCSPAISetEpsilon(pc, epsilon1));
  PetscCall(PetscOptionsInt("-pc_spai_nbsteps", "", "PCSPAISetNBSteps", ispai->nbsteps, &nbsteps1, &flg));
  if (flg) PetscCall(PCSPAISetNBSteps(pc, nbsteps1));
  /* added 1/7/99 g.h. */
  PetscCall(PetscOptionsInt("-pc_spai_max", "", "PCSPAISetMax", ispai->max, &max1, &flg));
  if (flg) PetscCall(PCSPAISetMax(pc, max1));
  PetscCall(PetscOptionsInt("-pc_spai_maxnew", "", "PCSPAISetMaxNew", ispai->maxnew, &maxnew1, &flg));
  if (flg) PetscCall(PCSPAISetMaxNew(pc, maxnew1));
  PetscCall(PetscOptionsInt("-pc_spai_block_size", "", "PCSPAISetBlockSize", ispai->block_size, &block_size1, &flg));
  if (flg) PetscCall(PCSPAISetBlockSize(pc, block_size1));
  PetscCall(PetscOptionsInt("-pc_spai_cache_size", "", "PCSPAISetCacheSize", ispai->cache_size, &cache_size, &flg));
  if (flg) PetscCall(PCSPAISetCacheSize(pc, cache_size));
  PetscCall(PetscOptionsInt("-pc_spai_verbose", "", "PCSPAISetVerbose", ispai->verbose, &verbose, &flg));
  if (flg) PetscCall(PCSPAISetVerbose(pc, verbose));
  PetscCall(PetscOptionsInt("-pc_spai_sp", "", "PCSPAISetSp", ispai->sp, &sp, &flg));
  if (flg) PetscCall(PCSPAISetSp(pc, sp));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PCSPAI - Use the Sparse Approximate Inverse method

   Options Database Keys:
+  -pc_spai_epsilon <eps> - set tolerance
.  -pc_spai_nbstep <n> - set nbsteps
.  -pc_spai_max <m> - set max
.  -pc_spai_max_new <m> - set maxnew
.  -pc_spai_block_size <n> - set block size
.  -pc_spai_cache_size <n> - set cache size
.  -pc_spai_sp <m> - set sp
-  -pc_spai_set_verbose <true,false> - verbose output

   Level: beginner

   Note:
    This only works with `MATAIJ` matrices.

   References:
 . * -  Grote and Barnard (SIAM J. Sci. Comput.; vol 18, nr 3)

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCSPAISetEpsilon()`, `PCSPAISetMax()`, `PCSPAISetMaxNew()`, `PCSPAISetBlockSize()`,
          `PCSPAISetVerbose()`, `PCSPAISetSp()`, `PCSPAISetNBSteps()`, `PCSPAISetCacheSize()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_SPAI(PC pc)
{
  PC_SPAI *ispai;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ispai));
  pc->data = ispai;

  pc->ops->destroy         = PCDestroy_SPAI;
  pc->ops->apply           = PCApply_SPAI;
  pc->ops->matapply        = PCMatApply_SPAI;
  pc->ops->applyrichardson = 0;
  pc->ops->setup           = PCSetUp_SPAI;
  pc->ops->view            = PCView_SPAI;
  pc->ops->setfromoptions  = PCSetFromOptions_SPAI;

  ispai->epsilon    = .4;
  ispai->nbsteps    = 5;
  ispai->max        = 5000;
  ispai->maxnew     = 5;
  ispai->block_size = 1;
  ispai->cache_size = 5;
  ispai->verbose    = 0;

  ispai->sp = 1;
  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)pc), &(ispai->comm_spai)));

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetEpsilon_C", PCSPAISetEpsilon_SPAI));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetNBSteps_C", PCSPAISetNBSteps_SPAI));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetMax_C", PCSPAISetMax_SPAI));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetMaxNew_C", PCSPAISetMaxNew_SPAI));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetBlockSize_C", PCSPAISetBlockSize_SPAI));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetCacheSize_C", PCSPAISetCacheSize_SPAI));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetVerbose_C", PCSPAISetVerbose_SPAI));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCSPAISetSp_C", PCSPAISetSp_SPAI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Converts from a PETSc matrix to an SPAI matrix
*/
PetscErrorCode ConvertMatToMatrix(MPI_Comm comm, Mat A, Mat AT, matrix **B)
{
  matrix                  *M;
  int                      i, j, col;
  int                      row_indx;
  int                      len, pe, local_indx, start_indx;
  int                     *mapping;
  const int               *cols;
  const double            *vals;
  int                      n, mnl, nnl, nz, rstart, rend;
  PetscMPIInt              size, rank;
  struct compressed_lines *rows;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatGetSize(A, &n, &n));
  PetscCall(MatGetLocalSize(A, &mnl, &nnl));

  /*
    not sure why a barrier is required. commenting out
  PetscCallMPI(MPI_Barrier(comm));
  */

  M = new_matrix((SPAI_Comm)comm);

  M->n              = n;
  M->bs             = 1;
  M->max_block_size = 1;

  M->mnls          = (int *)malloc(sizeof(int) * size);
  M->start_indices = (int *)malloc(sizeof(int) * size);
  M->pe            = (int *)malloc(sizeof(int) * n);
  M->block_sizes   = (int *)malloc(sizeof(int) * n);
  for (i = 0; i < n; i++) M->block_sizes[i] = 1;

  PetscCallMPI(MPI_Allgather(&mnl, 1, MPI_INT, M->mnls, 1, MPI_INT, comm));

  M->start_indices[0] = 0;
  for (i = 1; i < size; i++) M->start_indices[i] = M->start_indices[i - 1] + M->mnls[i - 1];

  M->mnl            = M->mnls[M->myid];
  M->my_start_index = M->start_indices[M->myid];

  for (i = 0; i < size; i++) {
    start_indx = M->start_indices[i];
    for (j = 0; j < M->mnls[i]; j++) M->pe[start_indx + j] = i;
  }

  if (AT) {
    M->lines = new_compressed_lines(M->mnls[rank], 1);
  } else {
    M->lines = new_compressed_lines(M->mnls[rank], 0);
  }

  rows = M->lines;

  /* Determine the mapping from global indices to pointers */
  PetscCall(PetscMalloc1(M->n, &mapping));
  pe         = 0;
  local_indx = 0;
  for (i = 0; i < M->n; i++) {
    if (local_indx >= M->mnls[pe]) {
      pe++;
      local_indx = 0;
    }
    mapping[i] = local_indx + M->start_indices[pe];
    local_indx++;
  }

  /************** Set up the row structure *****************/

  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    row_indx = i - rstart;
    PetscCall(MatGetRow(A, i, &nz, &cols, &vals));
    /* allocate buffers */
    rows->ptrs[row_indx] = (int *)malloc(nz * sizeof(int));
    rows->A[row_indx]    = (double *)malloc(nz * sizeof(double));
    /* copy the matrix */
    for (j = 0; j < nz; j++) {
      col = cols[j];
      len = rows->len[row_indx]++;

      rows->ptrs[row_indx][len] = mapping[col];
      rows->A[row_indx][len]    = vals[j];
    }
    rows->slen[row_indx] = rows->len[row_indx];

    PetscCall(MatRestoreRow(A, i, &nz, &cols, &vals));
  }

  /************** Set up the column structure *****************/

  if (AT) {
    for (i = rstart; i < rend; i++) {
      row_indx = i - rstart;
      PetscCall(MatGetRow(AT, i, &nz, &cols, &vals));
      /* allocate buffers */
      rows->rptrs[row_indx] = (int *)malloc(nz * sizeof(int));
      /* copy the matrix (i.e., the structure) */
      for (j = 0; j < nz; j++) {
        col = cols[j];
        len = rows->rlen[row_indx]++;

        rows->rptrs[row_indx][len] = mapping[col];
      }
      PetscCall(MatRestoreRow(AT, i, &nz, &cols, &vals));
    }
  }

  PetscCall(PetscFree(mapping));

  order_pointers(M);
  M->maxnz = calc_maxnz(M);
  *B       = M;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Converts from an SPAI matrix B  to a PETSc matrix PB.
   This assumes that the SPAI matrix B is stored in
   COMPRESSED-ROW format.
*/
PetscErrorCode ConvertMatrixToMat(MPI_Comm comm, matrix *B, Mat *PB)
{
  PetscMPIInt size, rank;
  int         m, n, M, N;
  int         d_nz, o_nz;
  int        *d_nnz, *o_nnz;
  int         i, k, global_row, global_col, first_diag_col, last_diag_col;
  PetscScalar val;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  m = n = B->mnls[rank];
  d_nz = o_nz = 0;

  /* Determine preallocation for MatCreateAIJ */
  PetscCall(PetscMalloc1(m, &d_nnz));
  PetscCall(PetscMalloc1(m, &o_nnz));
  for (i = 0; i < m; i++) d_nnz[i] = o_nnz[i] = 0;
  first_diag_col = B->start_indices[rank];
  last_diag_col  = first_diag_col + B->mnls[rank];
  for (i = 0; i < B->mnls[rank]; i++) {
    for (k = 0; k < B->lines->len[i]; k++) {
      global_col = B->lines->ptrs[i][k];
      if ((global_col >= first_diag_col) && (global_col < last_diag_col)) d_nnz[i]++;
      else o_nnz[i]++;
    }
  }

  M = N = B->n;
  /* Here we only know how to create AIJ format */
  PetscCall(MatCreate(comm, PB));
  PetscCall(MatSetSizes(*PB, m, n, M, N));
  PetscCall(MatSetType(*PB, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(*PB, d_nz, d_nnz));
  PetscCall(MatMPIAIJSetPreallocation(*PB, d_nz, d_nnz, o_nz, o_nnz));

  for (i = 0; i < B->mnls[rank]; i++) {
    global_row = B->start_indices[rank] + i;
    for (k = 0; k < B->lines->len[i]; k++) {
      global_col = B->lines->ptrs[i][k];

      val = B->lines->A[i][k];
      PetscCall(MatSetValues(*PB, 1, &global_row, 1, &global_col, &val, ADD_VALUES));
    }
  }

  PetscCall(PetscFree(d_nnz));
  PetscCall(PetscFree(o_nnz));

  PetscCall(MatAssemblyBegin(*PB, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*PB, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Converts from an SPAI vector v  to a PETSc vec Pv.
*/
PetscErrorCode ConvertVectorToVec(MPI_Comm comm, vector *v, Vec *Pv)
{
  PetscMPIInt size, rank;
  int         m, M, i, *mnls, *start_indices, *global_indices;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  m = v->mnl;
  M = v->n;

  PetscCall(VecCreateMPI(comm, m, M, Pv));

  PetscCall(PetscMalloc1(size, &mnls));
  PetscCallMPI(MPI_Allgather(&v->mnl, 1, MPI_INT, mnls, 1, MPI_INT, comm));

  PetscCall(PetscMalloc1(size, &start_indices));

  start_indices[0] = 0;
  for (i = 1; i < size; i++) start_indices[i] = start_indices[i - 1] + mnls[i - 1];

  PetscCall(PetscMalloc1(v->mnl, &global_indices));
  for (i = 0; i < v->mnl; i++) global_indices[i] = start_indices[rank] + i;

  PetscCall(PetscFree(mnls));
  PetscCall(PetscFree(start_indices));

  PetscCall(VecSetValues(*Pv, v->mnl, global_indices, v->v, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(*Pv));
  PetscCall(VecAssemblyEnd(*Pv));

  PetscCall(PetscFree(global_indices));
  PetscFunctionReturn(PETSC_SUCCESS);
}
