/*
    This file creates an MPI parallel KSP from a sequential PC that lives on MPI rank 0.
    It is intended to allow using PETSc MPI parallel linear solvers from non-MPI codes.

    That program may use OpenMP to compute the right-hand side and matrix for the linear system

    The code uses MPI_COMM_WORLD below but maybe it should be PETSC_COMM_WORLD

    The resulting KSP and PC can only be controlled via the options database, though some common commands
    could be passed through the server.

*/
#include <petsc/private/pcimpl.h> /*I "petscksp.h" I*/
#include <petsc/private/kspimpl.h>
#include <petscts.h>
#include <petsctao.h>
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
  #include <pthread.h>
#endif

#define PC_MPI_MAX_RANKS  256
#define PC_MPI_COMM_WORLD MPI_COMM_WORLD

typedef struct {
  KSP       ksps[PC_MPI_MAX_RANKS];                               /* The addresses of the MPI parallel KSP on each process, NULL when not on a process. */
  PetscInt  sendcount[PC_MPI_MAX_RANKS], displ[PC_MPI_MAX_RANKS]; /* For scatter/gather of rhs/solution */
  PetscInt  NZ[PC_MPI_MAX_RANKS], NZdispl[PC_MPI_MAX_RANKS];      /* For scatter of nonzero values in matrix (and nonzero column indices initially */
  PetscInt  mincntperrank;                                        /* minimum number of desired matrix rows per active rank in MPI parallel KSP solve */
  PetscBool alwaysuseserver;                                      /* for debugging use the server infrastructure even if only one MPI process is used for the solve */
} PC_MPI;

typedef enum {
  PCMPI_EXIT, /* exit the PC server loop, means the controlling sequential program is done */
  PCMPI_CREATE,
  PCMPI_SET_MAT,           /* set original matrix (or one with different nonzero pattern) */
  PCMPI_UPDATE_MAT_VALUES, /* update current matrix with new nonzero values */
  PCMPI_SOLVE,
  PCMPI_VIEW,
  PCMPI_DESTROY /* destroy a PC that is no longer needed */
} PCMPICommand;

static MPI_Comm      PCMPIComms[PC_MPI_MAX_RANKS];
static PetscBool     PCMPICommSet = PETSC_FALSE;
static PetscInt      PCMPISolveCounts[PC_MPI_MAX_RANKS], PCMPIKSPCounts[PC_MPI_MAX_RANKS], PCMPIMatCounts[PC_MPI_MAX_RANKS], PCMPISolveCountsSeq = 0, PCMPIKSPCountsSeq = 0;
static PetscInt      PCMPIIterations[PC_MPI_MAX_RANKS], PCMPISizes[PC_MPI_MAX_RANKS], PCMPIIterationsSeq = 0, PCMPISizesSeq = 0;
static PetscLogEvent EventServerDist, EventServerDistMPI;
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
static pthread_mutex_t *PCMPIServerLocks;
#else
static void *PCMPIServerLocks;
#endif

static PetscErrorCode PCMPICommsCreate(void)
{
  MPI_Comm    comm = PC_MPI_COMM_WORLD;
  PetscMPIInt size, rank, i;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size <= PC_MPI_MAX_RANKS, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for using more than PC_MPI_MAX_RANKS MPI ranks in an MPI linear solver server solve");
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* comm for size 1 is useful only for debugging */
  for (i = 0; i < size; i++) {
    PetscMPIInt color = rank < i + 1 ? 0 : MPI_UNDEFINED;
    PetscCallMPI(MPI_Comm_split(comm, color, 0, &PCMPIComms[i]));
    PCMPISolveCounts[i] = 0;
    PCMPIKSPCounts[i]   = 0;
    PCMPIIterations[i]  = 0;
    PCMPISizes[i]       = 0;
  }
  PCMPICommSet = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPICommsDestroy(void)
{
  MPI_Comm    comm = PC_MPI_COMM_WORLD;
  PetscMPIInt size, rank, i;

  PetscFunctionBegin;
  if (!PCMPICommSet) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  for (i = 0; i < size; i++) {
    if (PCMPIComms[i] != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_free(&PCMPIComms[i]));
  }
  PCMPICommSet = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPICreate(PC pc)
{
  PC_MPI     *km   = pc ? (PC_MPI *)pc->data : NULL;
  MPI_Comm    comm = PC_MPI_COMM_WORLD;
  KSP         ksp;
  PetscInt    N[2], mincntperrank = 0;
  PetscMPIInt size;
  Mat         sA;
  char       *cprefix = NULL;
  PetscMPIInt len     = 0;

  PetscFunctionBegin;
  PCMPIServerInSolve = PETSC_TRUE;
  if (!PCMPICommSet) PetscCall(PCMPICommsCreate());
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (pc) {
    if (size == 1) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Warning: Running KSP type of MPI on a one rank MPI run, this will be less efficient then not using this type\n"));
    PetscCall(PCGetOperators(pc, &sA, &sA));
    PetscCall(MatGetSize(sA, &N[0], &N[1]));
  }
  PetscCallMPI(MPI_Bcast(N, 2, MPIU_INT, 0, comm));

  /* choose a suitable sized MPI_Comm for the problem to be solved on */
  if (km) mincntperrank = km->mincntperrank;
  PetscCallMPI(MPI_Bcast(&mincntperrank, 1, MPI_INT, 0, comm));
  comm = PCMPIComms[PetscMin(size, PetscMax(1, N[0] / mincntperrank)) - 1];
  if (comm == MPI_COMM_NULL) {
    ksp                = NULL;
    PCMPIServerInSolve = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogStagePush(PCMPIStage));
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetNestLevel(ksp, 1));
  PetscCall(PetscObjectSetTabLevel((PetscObject)ksp, 1));
  PetscCall(PetscLogStagePop());
  PetscCallMPI(MPI_Gather(&ksp, 1, MPI_AINT, pc ? km->ksps : NULL, 1, MPI_AINT, 0, comm));
  if (pc) {
    size_t      slen;
    const char *prefix = NULL;
    char       *found  = NULL;

    PetscCallMPI(MPI_Comm_size(comm, &size));
    PCMPIKSPCounts[size - 1]++;
    /* Created KSP gets prefix of PC minus the mpi_linear_solver_server_ portion */
    PetscCall(PCGetOptionsPrefix(pc, &prefix));
    PetscCheck(prefix, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCMPI missing required prefix");
    PetscCall(PetscStrallocpy(prefix, &cprefix));
    PetscCall(PetscStrstr(cprefix, "mpi_linear_solver_server_", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCMPI missing mpi_linear_solver_server_ portion of prefix");
    *found = 0;
    PetscCall(PetscStrlen(cprefix, &slen));
    PetscCall(PetscMPIIntCast(slen, &len));
  }
  PetscCallMPI(MPI_Bcast(&len, 1, MPI_INT, 0, comm));
  if (len) {
    if (!pc) PetscCall(PetscMalloc1(len + 1, &cprefix));
    PetscCallMPI(MPI_Bcast(cprefix, len + 1, MPI_CHAR, 0, comm));
    PetscCall(KSPSetOptionsPrefix(ksp, cprefix));
  }
  PetscCall(PetscFree(cprefix));
  PCMPIServerInSolve = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPISetMat(PC pc)
{
  PC_MPI            *km = pc ? (PC_MPI *)pc->data : NULL;
  Mat                A;
  PetscInt           m, n, j, bs;
  Mat                sA;
  MPI_Comm           comm = PC_MPI_COMM_WORLD;
  KSP                ksp;
  PetscLayout        layout;
  const PetscInt    *IA = NULL, *JA = NULL, *ia, *ja;
  const PetscInt    *range;
  PetscInt          *NZ = NULL, sendcounti[PC_MPI_MAX_RANKS], displi[PC_MPI_MAX_RANKS], *NZdispl = NULL, nz;
  PetscMPIInt        size, i;
  const PetscScalar *a                = NULL, *sa;
  PetscInt           matproperties[8] = {0}, rstart, rend;
  char              *cprefix;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Scatter(pc ? km->ksps : NULL, 1, MPI_AINT, &ksp, 1, MPI_AINT, 0, comm));
  if (!ksp) PetscFunctionReturn(PETSC_SUCCESS);
  PCMPIServerInSolve = PETSC_TRUE;
  PetscCall(PetscLogEventBegin(EventServerDist, NULL, NULL, NULL, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  if (pc) {
    PetscBool   isset, issymmetric, ishermitian, isspd, isstructurallysymmetric;
    const char *prefix;
    size_t      clen;

    PetscCallMPI(MPI_Comm_size(comm, &size));
    PCMPIMatCounts[size - 1]++;
    PetscCall(PCGetOperators(pc, &sA, &sA));
    PetscCall(MatGetSize(sA, &matproperties[0], &matproperties[1]));
    PetscCall(MatGetBlockSize(sA, &bs));
    matproperties[2] = bs;
    PetscCall(MatIsSymmetricKnown(sA, &isset, &issymmetric));
    matproperties[3] = !isset ? 0 : (issymmetric ? 1 : 2);
    PetscCall(MatIsHermitianKnown(sA, &isset, &ishermitian));
    matproperties[4] = !isset ? 0 : (ishermitian ? 1 : 2);
    PetscCall(MatIsSPDKnown(sA, &isset, &isspd));
    matproperties[5] = !isset ? 0 : (isspd ? 1 : 2);
    PetscCall(MatIsStructurallySymmetricKnown(sA, &isset, &isstructurallysymmetric));
    matproperties[6] = !isset ? 0 : (isstructurallysymmetric ? 1 : 2);
    /* Created Mat gets prefix of input Mat PLUS the mpi_linear_solver_server_ portion */
    PetscCall(MatGetOptionsPrefix(sA, &prefix));
    PetscCall(PetscStrallocpy(prefix, &cprefix));
    PetscCall(PetscStrlen(cprefix, &clen));
    matproperties[7] = (PetscInt)clen;
  }
  PetscCallMPI(MPI_Bcast(matproperties, PETSC_STATIC_ARRAY_LENGTH(matproperties), MPIU_INT, 0, comm));

  /* determine ownership ranges of matrix columns */
  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscLayoutSetBlockSize(layout, matproperties[2]));
  PetscCall(PetscLayoutSetSize(layout, matproperties[1]));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetLocalSize(layout, &n));
  PetscCall(PetscLayoutDestroy(&layout));

  /* determine ownership ranges of matrix rows */
  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscLayoutSetBlockSize(layout, matproperties[2]));
  PetscCall(PetscLayoutSetSize(layout, matproperties[0]));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetLocalSize(layout, &m));
  PetscCall(PetscLayoutGetRange(layout, &rstart, &rend));

  PetscCall(PetscLogEventBegin(EventServerDistMPI, NULL, NULL, NULL, NULL));
  /* copy over the matrix nonzero structure and values */
  if (pc) {
    PetscCall(MatGetRowIJ(sA, 0, PETSC_FALSE, PETSC_FALSE, NULL, &IA, &JA, NULL));
    if (!PCMPIServerUseShmget) {
      NZ      = km->NZ;
      NZdispl = km->NZdispl;
      PetscCall(PetscLayoutGetRanges(layout, &range));
      for (i = 0; i < size; i++) {
        sendcounti[i] = 1 + range[i + 1] - range[i];
        NZ[i]         = IA[range[i + 1]] - IA[range[i]];
      }
      displi[0]  = 0;
      NZdispl[0] = 0;
      for (j = 1; j < size; j++) {
        displi[j]  = displi[j - 1] + sendcounti[j - 1] - 1;
        NZdispl[j] = NZdispl[j - 1] + NZ[j - 1];
      }
    }
    PetscCall(MatSeqAIJGetArrayRead(sA, &sa));
  }
  PetscCall(PetscLayoutDestroy(&layout));

  PetscCall(MatCreate(comm, &A));
  if (matproperties[7] > 0) {
    PetscMPIInt ni;

    PetscCall(PetscMPIIntCast(matproperties[7] + 1, &ni));
    if (!pc) PetscCall(PetscMalloc1(matproperties[7] + 1, &cprefix));
    PetscCallMPI(MPI_Bcast(cprefix, ni, MPI_CHAR, 0, comm));
    PetscCall(MatSetOptionsPrefix(A, cprefix));
    PetscCall(PetscFree(cprefix));
  }
  PetscCall(MatAppendOptionsPrefix(A, "mpi_linear_solver_server_"));
  PetscCall(MatSetSizes(A, m, n, matproperties[0], matproperties[1]));
  PetscCall(MatSetType(A, MATMPIAIJ));

  if (!PCMPIServerUseShmget) {
    PetscCallMPI(MPI_Scatter(NZ, 1, MPIU_INT, &nz, 1, MPIU_INT, 0, comm));
    PetscCall(PetscMalloc3(n + 1, &ia, nz, &ja, nz, &a));
    PetscCallMPI(MPIU_Scatterv(IA, sendcounti, displi, MPIU_INT, (void *)ia, n + 1, MPIU_INT, 0, comm));
    PetscCallMPI(MPIU_Scatterv(JA, NZ, NZdispl, MPIU_INT, (void *)ja, nz, MPIU_INT, 0, comm));
    PetscCallMPI(MPIU_Scatterv(sa, NZ, NZdispl, MPIU_SCALAR, (void *)a, nz, MPIU_SCALAR, 0, comm));
  } else {
    const void           *addr[3] = {(const void **)IA, (const void **)JA, (const void **)sa};
    PCMPIServerAddresses *addresses;

    PetscCall(PetscNew(&addresses));
    addresses->n = 3;
    PetscCall(PetscShmgetMapAddresses(comm, addresses->n, addr, addresses->addr));
    ia = rstart + (PetscInt *)addresses->addr[0];
    ja = ia[0] + (PetscInt *)addresses->addr[1];
    a  = ia[0] + (PetscScalar *)addresses->addr[2];
    PetscCall(PetscObjectContainerCompose((PetscObject)A, "PCMPIServerAddresses", (void *)addresses, PCMPIServerAddressesDestroy));
  }

  if (pc) {
    PetscCall(MatSeqAIJRestoreArrayRead(sA, &sa));
    PetscCall(MatRestoreRowIJ(sA, 0, PETSC_FALSE, PETSC_FALSE, NULL, &IA, &JA, NULL));
  }
  PetscCall(PetscLogEventEnd(EventServerDistMPI, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogStagePush(PCMPIStage));
  PetscCall(MatMPIAIJSetPreallocationCSR(A, ia, ja, a));
  PetscCall(MatSetBlockSize(A, matproperties[2]));

  if (matproperties[3]) PetscCall(MatSetOption(A, MAT_SYMMETRIC, matproperties[3] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[4]) PetscCall(MatSetOption(A, MAT_HERMITIAN, matproperties[4] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[5]) PetscCall(MatSetOption(A, MAT_SPD, matproperties[5] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[6]) PetscCall(MatSetOption(A, MAT_STRUCTURALLY_SYMMETRIC, matproperties[6] == 1 ? PETSC_TRUE : PETSC_FALSE));

  if (!PCMPIServerUseShmget) PetscCall(PetscFree3(ia, ja, a));
  PetscCall(KSPSetOperators(ksp, A, A));
  if (!ksp->vec_sol) PetscCall(MatCreateVecs(A, &ksp->vec_sol, &ksp->vec_rhs));
  PetscCall(PetscLogStagePop());
  if (pc && !PCMPIServerUseShmget) { /* needed for scatterv/gatherv of rhs and solution */
    const PetscInt *range;

    PetscCall(VecGetOwnershipRanges(ksp->vec_sol, &range));
    for (i = 0; i < size; i++) {
      km->sendcount[i] = range[i + 1] - range[i];
      km->displ[i]     = range[i];
    }
  }
  PetscCall(MatDestroy(&A));
  PetscCall(PetscLogEventEnd(EventServerDist, NULL, NULL, NULL, NULL));
  PetscCall(KSPSetFromOptions(ksp));
  PCMPIServerInSolve = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPIUpdateMatValues(PC pc)
{
  PC_MPI            *km = pc ? (PC_MPI *)pc->data : NULL;
  KSP                ksp;
  Mat                sA, A;
  MPI_Comm           comm = PC_MPI_COMM_WORLD;
  const PetscInt    *ia, *IA;
  const PetscScalar *a;
  PetscCount         nz;
  const PetscScalar *sa = NULL;
  PetscMPIInt        size;
  PetscInt           rstart, matproperties[4] = {0, 0, 0, 0};

  PetscFunctionBegin;
  if (pc) {
    PetscCall(PCGetOperators(pc, &sA, &sA));
    PetscCall(MatSeqAIJGetArrayRead(sA, &sa));
    PetscCall(MatGetRowIJ(sA, 0, PETSC_FALSE, PETSC_FALSE, NULL, &IA, NULL, NULL));
  }
  PetscCallMPI(MPI_Scatter(pc ? km->ksps : NULL, 1, MPI_AINT, &ksp, 1, MPI_AINT, 0, comm));
  if (!ksp) PetscFunctionReturn(PETSC_SUCCESS);
  PCMPIServerInSolve = PETSC_TRUE;
  PetscCall(PetscLogEventBegin(EventServerDist, NULL, NULL, NULL, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PCMPIMatCounts[size - 1]++;
  PetscCall(KSPGetOperators(ksp, NULL, &A));
  PetscCall(PetscLogEventBegin(EventServerDistMPI, NULL, NULL, NULL, NULL));
  if (!PCMPIServerUseShmget) {
    PetscInt petsc_nz;

    PetscCall(MatMPIAIJGetNumberNonzeros(A, &nz));
    PetscCall(PetscIntCast(nz, &petsc_nz));
    PetscCall(PetscMalloc1(nz, &a));
    PetscCallMPI(MPIU_Scatterv(sa, pc ? km->NZ : NULL, pc ? km->NZdispl : NULL, MPIU_SCALAR, (void *)a, petsc_nz, MPIU_SCALAR, 0, comm));
  } else {
    PetscCall(MatGetOwnershipRange(A, &rstart, NULL));
    PCMPIServerAddresses *addresses;
    PetscCall(PetscObjectContainerQuery((PetscObject)A, "PCMPIServerAddresses", (void **)&addresses));
    ia = rstart + (PetscInt *)addresses->addr[0];
    a  = ia[0] + (PetscScalar *)addresses->addr[2];
  }
  PetscCall(PetscLogEventEnd(EventServerDistMPI, NULL, NULL, NULL, NULL));
  if (pc) {
    PetscBool isset, issymmetric, ishermitian, isspd, isstructurallysymmetric;

    PetscCall(MatSeqAIJRestoreArrayRead(sA, &sa));
    PetscCall(MatRestoreRowIJ(sA, 0, PETSC_FALSE, PETSC_FALSE, NULL, &IA, NULL, NULL));

    PetscCall(MatIsSymmetricKnown(sA, &isset, &issymmetric));
    matproperties[0] = !isset ? 0 : (issymmetric ? 1 : 2);
    PetscCall(MatIsHermitianKnown(sA, &isset, &ishermitian));
    matproperties[1] = !isset ? 0 : (ishermitian ? 1 : 2);
    PetscCall(MatIsSPDKnown(sA, &isset, &isspd));
    matproperties[2] = !isset ? 0 : (isspd ? 1 : 2);
    PetscCall(MatIsStructurallySymmetricKnown(sA, &isset, &isstructurallysymmetric));
    matproperties[3] = !isset ? 0 : (isstructurallysymmetric ? 1 : 2);
  }
  PetscCall(MatUpdateMPIAIJWithArray(A, a));
  if (!PCMPIServerUseShmget) PetscCall(PetscFree(a));
  PetscCallMPI(MPI_Bcast(matproperties, 4, MPIU_INT, 0, comm));
  /* if any of these properties was previously set and is now not set this will result in incorrect properties in A since there is no way to unset a property */
  if (matproperties[0]) PetscCall(MatSetOption(A, MAT_SYMMETRIC, matproperties[0] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[1]) PetscCall(MatSetOption(A, MAT_HERMITIAN, matproperties[1] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[2]) PetscCall(MatSetOption(A, MAT_SPD, matproperties[2] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[3]) PetscCall(MatSetOption(A, MAT_STRUCTURALLY_SYMMETRIC, matproperties[3] == 1 ? PETSC_TRUE : PETSC_FALSE));
  PetscCall(PetscLogEventEnd(EventServerDist, NULL, NULL, NULL, NULL));
  PCMPIServerInSolve = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPISolve(PC pc, Vec B, Vec X)
{
  PC_MPI            *km = pc ? (PC_MPI *)pc->data : NULL;
  KSP                ksp;
  MPI_Comm           comm = PC_MPI_COMM_WORLD;
  const PetscScalar *sb   = NULL, *x;
  PetscScalar       *b, *sx = NULL;
  PetscInt           its, n;
  PetscMPIInt        size;
  void              *addr[2];

  PetscFunctionBegin;
  PetscCallMPI(MPI_Scatter(pc ? km->ksps : &ksp, 1, MPI_AINT, &ksp, 1, MPI_AINT, 0, comm));
  if (!ksp) PetscFunctionReturn(PETSC_SUCCESS);
  PCMPIServerInSolve = PETSC_TRUE;
  PetscCall(PetscLogEventBegin(EventServerDist, NULL, NULL, NULL, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  /* scatterv rhs */
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (pc) {
    PetscInt N;

    PCMPISolveCounts[size - 1]++;
    PetscCall(MatGetSize(pc->pmat, &N, NULL));
    PCMPISizes[size - 1] += N;
  }
  PetscCall(VecGetLocalSize(ksp->vec_rhs, &n));
  PetscCall(PetscLogEventBegin(EventServerDistMPI, NULL, NULL, NULL, NULL));
  if (!PCMPIServerUseShmget) {
    PetscCall(VecGetArray(ksp->vec_rhs, &b));
    if (pc) PetscCall(VecGetArrayRead(B, &sb));
    PetscCallMPI(MPIU_Scatterv(sb, pc ? km->sendcount : NULL, pc ? km->displ : NULL, MPIU_SCALAR, b, n, MPIU_SCALAR, 0, comm));
    if (pc) PetscCall(VecRestoreArrayRead(B, &sb));
    PetscCall(VecRestoreArray(ksp->vec_rhs, &b));
    // TODO: scatter initial guess if needed
  } else {
    PetscInt rstart;

    if (pc) PetscCall(VecGetArrayRead(B, &sb));
    if (pc) PetscCall(VecGetArray(X, &sx));
    const void *inaddr[2] = {(const void **)sb, (const void **)sx};
    if (pc) PetscCall(VecRestoreArray(X, &sx));
    if (pc) PetscCall(VecRestoreArrayRead(B, &sb));

    PetscCall(PetscShmgetMapAddresses(comm, 2, inaddr, addr));
    PetscCall(VecGetOwnershipRange(ksp->vec_rhs, &rstart, NULL));
    PetscCall(VecPlaceArray(ksp->vec_rhs, rstart + (PetscScalar *)addr[0]));
    PetscCall(VecPlaceArray(ksp->vec_sol, rstart + (PetscScalar *)addr[1]));
  }
  PetscCall(PetscLogEventEnd(EventServerDistMPI, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventEnd(EventServerDist, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogStagePush(PCMPIStage));
  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(PetscLogStagePop());
  PetscCall(PetscLogEventBegin(EventServerDist, NULL, NULL, NULL, NULL));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PCMPIIterations[size - 1] += its;
  // TODO: send iterations up to outer KSP

  if (PCMPIServerUseShmget) PetscCall(PetscShmgetUnmapAddresses(2, addr));

  /* gather solution */
  PetscCall(PetscLogEventBegin(EventServerDistMPI, NULL, NULL, NULL, NULL));
  if (!PCMPIServerUseShmget) {
    PetscCall(VecGetArrayRead(ksp->vec_sol, &x));
    if (pc) PetscCall(VecGetArray(X, &sx));
    PetscCallMPI(MPIU_Gatherv(x, n, MPIU_SCALAR, sx, pc ? km->sendcount : NULL, pc ? km->displ : NULL, MPIU_SCALAR, 0, comm));
    if (pc) PetscCall(VecRestoreArray(X, &sx));
    PetscCall(VecRestoreArrayRead(ksp->vec_sol, &x));
  } else {
    PetscCallMPI(MPI_Barrier(comm));
    PetscCall(VecResetArray(ksp->vec_rhs));
    PetscCall(VecResetArray(ksp->vec_sol));
  }
  PetscCall(PetscLogEventEnd(EventServerDistMPI, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventEnd(EventServerDist, NULL, NULL, NULL, NULL));
  PCMPIServerInSolve = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPIDestroy(PC pc)
{
  PC_MPI  *km = pc ? (PC_MPI *)pc->data : NULL;
  KSP      ksp;
  MPI_Comm comm = PC_MPI_COMM_WORLD;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Scatter(pc ? km->ksps : NULL, 1, MPI_AINT, &ksp, 1, MPI_AINT, 0, comm));
  if (!ksp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStagePush(PCMPIStage));
  PCMPIServerInSolve = PETSC_TRUE;
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscLogStagePop());
  PCMPIServerInSolve = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPIServerBroadcastRequest(PCMPICommand request)
{
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
  PetscMPIInt dummy1 = 1, dummy2;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
  if (PCMPIServerUseShmget) {
    for (PetscMPIInt i = 1; i < PetscGlobalSize; i++) pthread_mutex_unlock(&PCMPIServerLocks[i]);
  }
#endif
  PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, MPI_COMM_WORLD));
  /* next line ensures the sender has already taken the lock */
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
  if (PCMPIServerUseShmget) {
    PetscCallMPI(MPI_Reduce(&dummy1, &dummy2, 1, MPI_INT, MPI_SUM, 0, PC_MPI_COMM_WORLD));
    for (PetscMPIInt i = 1; i < PetscGlobalSize; i++) pthread_mutex_lock(&PCMPIServerLocks[i]);
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCMPIServerBegin - starts a server that runs on the `rank != 0` MPI processes waiting to process requests for
  parallel `KSP` solves and management of parallel `KSP` objects.

  Logically Collective on all MPI processes except rank 0

  Options Database Keys:
+ -mpi_linear_solver_server                   - causes the PETSc program to start in MPI linear solver server mode where only the first MPI rank runs user code
. -mpi_linear_solver_server_view              - displays information about all the linear systems solved by the MPI linear solver server at the conclusion of the program
- -mpi_linear_solver_server_use_shared_memory - use shared memory when communicating matrices and vectors to server processes (default where supported)

  Level: developer

  Note:
  This is normally started automatically in `PetscInitialize()` when the option is provided

  See `PCMPI` for information on using the solver with a `KSP` object

  See `PetscShmgetAllocateArray()` for instructions on how to ensure the shared memory is available on your machine.

  Developer Notes:
  When called on MPI rank 0 this sets `PETSC_COMM_WORLD` to `PETSC_COMM_SELF` to allow a main program
  written with `PETSC_COMM_WORLD` to run correctly on the single rank while all the ranks
  (that would normally be sharing `PETSC_COMM_WORLD`) to run the solver server.

  Can this be integrated into the `PetscDevice` abstraction that is currently being developed?

  Conceivably `PCREDISTRIBUTE` could be organized in a similar manner to simplify its usage

  This could be implemented directly at the `KSP` level instead of using the `PCMPI` wrapper object

  The code could be extended to allow an MPI + OpenMP application to use the linear solver server concept across all shared-memory
  nodes with a single MPI process per node for the user application but multiple MPI processes per node for the linear solver.

  The concept could also be extended for users's callbacks for `SNES`, `TS`, and `Tao` where the `SNESSolve()` for example, runs on
  all MPI processes but the user callback only runs on one MPI process per node.

  PETSc could also be extended with an MPI-less API that provides access to PETSc's solvers without any reference to MPI, essentially remove
  the `MPI_Comm` argument from PETSc calls.

.seealso: [](sec_pcmpi), `PCMPIServerEnd()`, `PCMPI`, `KSPCheckPCMPI()`
@*/
PetscErrorCode PCMPIServerBegin(void)
{
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL, "Starting MPI Linear Solver Server\n"));
  if (PetscDefined(USE_SINGLE_LIBRARY)) {
    PetscCall(VecInitializePackage());
    PetscCall(MatInitializePackage());
    PetscCall(DMInitializePackage());
    PetscCall(PCInitializePackage());
    PetscCall(KSPInitializePackage());
    PetscCall(SNESInitializePackage());
    PetscCall(TSInitializePackage());
    PetscCall(TaoInitializePackage());
  }
  PetscCall(PetscLogStageRegister("PCMPI", &PCMPIStage));
  PetscCall(PetscLogEventRegister("ServerDist", PC_CLASSID, &EventServerDist));
  PetscCall(PetscLogEventRegister("ServerDistMPI", PC_CLASSID, &EventServerDistMPI));

  if (!PetscDefined(HAVE_SHMGET)) PCMPIServerUseShmget = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mpi_linear_solver_server_use_shared_memory", &PCMPIServerUseShmget, NULL));

  PetscCallMPI(MPI_Comm_rank(PC_MPI_COMM_WORLD, &rank));
  if (PCMPIServerUseShmget) {
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    if (size > 1) {
      pthread_mutex_t *locks;

      if (rank == 0) {
        PCMPIServerActive = PETSC_TRUE;
        PetscCall(PetscShmgetAllocateArray(size, sizeof(pthread_mutex_t), (void **)&locks));
      }
      PetscCall(PetscShmgetMapAddresses(PETSC_COMM_WORLD, 1, (const void **)&locks, (void **)&PCMPIServerLocks));
      if (rank == 0) {
        pthread_mutexattr_t attr;

        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);

        for (int i = 1; i < size; i++) {
          pthread_mutex_init(&PCMPIServerLocks[i], &attr);
          pthread_mutex_lock(&PCMPIServerLocks[i]);
        }
      }
      PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    }
#endif
  }
  if (rank == 0) {
    PETSC_COMM_WORLD  = PETSC_COMM_SELF;
    PCMPIServerActive = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  while (PETSC_TRUE) {
    PCMPICommand request = PCMPI_CREATE;
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
    PetscMPIInt dummy1 = 1, dummy2;
#endif

    // TODO: can we broadcast the number of active ranks here so only the correct subset of processes waits on the later scatters?
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
    if (PCMPIServerUseShmget) pthread_mutex_lock(&PCMPIServerLocks[PetscGlobalRank]);
#endif
    PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, PC_MPI_COMM_WORLD));
#if defined(PETSC_HAVE_PTHREAD_MUTEX)
    if (PCMPIServerUseShmget) {
      /* next line ensures PetscGlobalRank has locked before rank 0 can take the lock back */
      PetscCallMPI(MPI_Reduce(&dummy1, &dummy2, 1, MPI_INT, MPI_SUM, 0, PC_MPI_COMM_WORLD));
      pthread_mutex_unlock(&PCMPIServerLocks[PetscGlobalRank]);
    }
#endif
    switch (request) {
    case PCMPI_CREATE:
      PetscCall(PCMPICreate(NULL));
      break;
    case PCMPI_SET_MAT:
      PetscCall(PCMPISetMat(NULL));
      break;
    case PCMPI_UPDATE_MAT_VALUES:
      PetscCall(PCMPIUpdateMatValues(NULL));
      break;
    case PCMPI_VIEW:
      // PetscCall(PCMPIView(NULL));
      break;
    case PCMPI_SOLVE:
      PetscCall(PCMPISolve(NULL, NULL, NULL));
      break;
    case PCMPI_DESTROY:
      PetscCall(PCMPIDestroy(NULL));
      break;
    case PCMPI_EXIT:
      if (PCMPIServerUseShmget) PetscCall(PetscShmgetUnmapAddresses(1, (void **)&PCMPIServerLocks));
      PetscCall(PetscFinalize());
      exit(0); /* not sure if this is a good idea, but cannot return because it will run users main program */
      break;
    default:
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCMPIServerEnd - ends a server that runs on the rank != 0 MPI processes waiting to process requests for
  parallel KSP solves and management of parallel `KSP` objects.

  Logically Collective on all MPI ranks except 0

  Level: developer

  Note:
  This is normally called automatically in `PetscFinalize()`

.seealso: [](sec_pcmpi), `PCMPIServerBegin()`, `PCMPI`, `KSPCheckPCMPI()`
@*/
PetscErrorCode PCMPIServerEnd(void)
{
  PetscFunctionBegin;
  if (PetscGlobalRank == 0) {
    PetscViewer       viewer = NULL;
    PetscViewerFormat format;

    PetscCall(PetscShmgetAddressesFinalize());
    PetscCall(PCMPIServerBroadcastRequest(PCMPI_EXIT));
    if (PCMPIServerUseShmget) PetscCall(PetscShmgetUnmapAddresses(1, (void **)&PCMPIServerLocks));
    PETSC_COMM_WORLD = MPI_COMM_WORLD; /* could use PC_MPI_COMM_WORLD */
    PetscOptionsBegin(PETSC_COMM_SELF, NULL, "MPI linear solver server options", NULL);
    PetscCall(PetscOptionsViewer("-mpi_linear_solver_server_view", "View information about system solved with the server", "PCMPI", &viewer, &format, NULL));
    PetscOptionsEnd();
    if (viewer) {
      PetscBool isascii;

      PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
      if (isascii) {
        PetscMPIInt size;
        PetscMPIInt i;

        PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
        PetscCall(PetscViewerASCIIPrintf(viewer, "MPI linear solver server statistics:\n"));
        PetscCall(PetscViewerASCIIPrintf(viewer, "    Ranks        KSPSolve()s     Mats        KSPs       Avg. Size      Avg. Its\n"));
        if (PCMPIKSPCountsSeq) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Sequential         %" PetscInt_FMT "                         %" PetscInt_FMT "            %" PetscInt_FMT "           %" PetscInt_FMT "\n", PCMPISolveCountsSeq, PCMPIKSPCountsSeq, PCMPISizesSeq / PCMPISolveCountsSeq, PCMPIIterationsSeq / PCMPISolveCountsSeq));
        }
        for (i = 0; i < size; i++) {
          if (PCMPIKSPCounts[i]) {
            PetscCall(PetscViewerASCIIPrintf(viewer, "     %d               %" PetscInt_FMT "            %" PetscInt_FMT "           %" PetscInt_FMT "            %" PetscInt_FMT "            %" PetscInt_FMT "\n", i + 1, PCMPISolveCounts[i], PCMPIMatCounts[i], PCMPIKSPCounts[i], PCMPISizes[i] / PCMPISolveCounts[i], PCMPIIterations[i] / PCMPISolveCounts[i]));
          }
        }
        PetscCall(PetscViewerASCIIPrintf(viewer, "MPI linear solver server %susing shared memory\n", PCMPIServerUseShmget ? "" : "not "));
      }
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscCall(PCMPICommsDestroy());
  PCMPIServerActive = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    This version is used in the trivial case when the MPI parallel solver server is running on just the original MPI rank 0
    because, for example, the problem is small. This version is more efficient because it does not require copying any data
*/
static PetscErrorCode PCSetUp_Seq(PC pc)
{
  PC_MPI     *km = (PC_MPI *)pc->data;
  Mat         sA;
  const char *prefix;
  char       *found = NULL, *cprefix;

  PetscFunctionBegin;
  PCMPIServerInSolve = PETSC_TRUE;
  PetscCall(PCGetOperators(pc, NULL, &sA));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(KSPCreate(PETSC_COMM_SELF, &km->ksps[0]));
  PetscCall(KSPSetNestLevel(km->ksps[0], 1));
  PetscCall(PetscObjectSetTabLevel((PetscObject)km->ksps[0], 1));

  /* Created KSP gets prefix of PC minus the mpi_linear_solver_server_ portion */
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCheck(prefix, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCMPI missing required prefix");
  PetscCall(PetscStrallocpy(prefix, &cprefix));
  PetscCall(PetscStrstr(cprefix, "mpi_linear_solver_server_", &found));
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCMPI missing mpi_linear_solver_server_ portion of prefix");
  *found = 0;
  PetscCall(KSPSetOptionsPrefix(km->ksps[0], cprefix));
  PetscCall(PetscFree(cprefix));

  PetscCall(KSPSetOperators(km->ksps[0], sA, sA));
  PetscCall(KSPSetFromOptions(km->ksps[0]));
  PetscCall(KSPSetUp(km->ksps[0]));
  PetscCall(PetscInfo(pc, "MPI parallel linear solver system is being solved directly on rank 0 due to its small size\n"));
  PCMPIKSPCountsSeq++;
  PCMPIServerInSolve = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_Seq(PC pc, Vec b, Vec x)
{
  PC_MPI  *km = (PC_MPI *)pc->data;
  PetscInt its, n;
  Mat      A;

  PetscFunctionBegin;
  PCMPIServerInSolve = PETSC_TRUE;
  PetscCall(KSPSolve(km->ksps[0], b, x));
  PetscCall(KSPGetIterationNumber(km->ksps[0], &its));
  PCMPISolveCountsSeq++;
  PCMPIIterationsSeq += its;
  PetscCall(KSPGetOperators(km->ksps[0], NULL, &A));
  PetscCall(MatGetSize(A, &n, NULL));
  PCMPISizesSeq += n;
  PCMPIServerInSolve = PETSC_FALSE;
  /*
    do not keep reference to previous rhs and solution since destroying them in the next KSPSolve()
    my use PetscFree() instead of PCMPIArrayDeallocate()
  */
  PetscCall(VecDestroy(&km->ksps[0]->vec_rhs));
  PetscCall(VecDestroy(&km->ksps[0]->vec_sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Seq(PC pc, PetscViewer viewer)
{
  PC_MPI *km = (PC_MPI *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Running MPI linear solver server directly on rank 0 due to its small size\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Desired minimum number of nonzeros per rank for MPI parallel solve %" PetscInt_FMT "\n", km->mincntperrank));
  PetscCall(PetscViewerASCIIPrintf(viewer, "*** Use -mpi_linear_solver_server_view to statistics on all the solves ***\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Seq(PC pc)
{
  PC_MPI *km = (PC_MPI *)pc->data;
  Mat     A, B;
  Vec     x, b;

  PetscFunctionBegin;
  PCMPIServerInSolve = PETSC_TRUE;
  /* since matrices and vectors are shared with outer KSP we need to ensure they are not destroyed with PetscFree() */
  PetscCall(KSPGetOperators(km->ksps[0], &A, &B));
  PetscCall(PetscObjectReference((PetscObject)A));
  PetscCall(PetscObjectReference((PetscObject)B));
  PetscCall(KSPGetSolution(km->ksps[0], &x));
  PetscCall(PetscObjectReference((PetscObject)x));
  PetscCall(KSPGetRhs(km->ksps[0], &b));
  PetscCall(PetscObjectReference((PetscObject)b));
  PetscCall(KSPDestroy(&km->ksps[0]));
  PetscCall(PetscFree(pc->data));
  PCMPIServerInSolve = PETSC_FALSE;
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PCSetUp_MPI - Trigger the creation of the MPI parallel PC and copy parts of the matrix and
     right-hand side to the parallel PC
*/
static PetscErrorCode PCSetUp_MPI(PC pc)
{
  PC_MPI     *km = (PC_MPI *)pc->data;
  PetscMPIInt rank, size;
  PetscBool   newmatrix = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  PetscCheck(rank == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PCMPI can only be used from 0th rank of MPI_COMM_WORLD. Perhaps a missing -mpi_linear_solver_server?");
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  if (!pc->setupcalled) {
    if (!km->alwaysuseserver) {
      PetscInt n;
      Mat      sA;
      /* short circuit for small systems */
      PetscCall(PCGetOperators(pc, &sA, &sA));
      PetscCall(MatGetSize(sA, &n, NULL));
      if (n < 2 * km->mincntperrank - 1 || size == 1) {
        pc->ops->setup   = NULL;
        pc->ops->apply   = PCApply_Seq;
        pc->ops->destroy = PCDestroy_Seq;
        pc->ops->view    = PCView_Seq;
        PetscCall(PCSetUp_Seq(pc));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }

    PetscCall(PCMPIServerBroadcastRequest(PCMPI_CREATE));
    PetscCall(PCMPICreate(pc));
    newmatrix = PETSC_TRUE;
  }
  if (pc->flag == DIFFERENT_NONZERO_PATTERN) newmatrix = PETSC_TRUE;

  if (newmatrix) {
    PetscCall(PetscInfo(pc, "New matrix or matrix has changed nonzero structure\n"));
    PetscCall(PCMPIServerBroadcastRequest(PCMPI_SET_MAT));
    PetscCall(PCMPISetMat(pc));
  } else {
    PetscCall(PetscInfo(pc, "Matrix has only changed nonzero values\n"));
    PetscCall(PCMPIServerBroadcastRequest(PCMPI_UPDATE_MAT_VALUES));
    PetscCall(PCMPIUpdateMatValues(pc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_MPI(PC pc, Vec b, Vec x)
{
  PetscFunctionBegin;
  PetscCall(PCMPIServerBroadcastRequest(PCMPI_SOLVE));
  PetscCall(PCMPISolve(pc, b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_MPI(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCMPIServerBroadcastRequest(PCMPI_DESTROY));
  PetscCall(PCMPIDestroy(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PCView_MPI - Cannot call view on the MPI parallel KSP because other ranks do not have access to the viewer, use options database
*/
static PetscErrorCode PCView_MPI(PC pc, PetscViewer viewer)
{
  PC_MPI     *km = (PC_MPI *)pc->data;
  MPI_Comm    comm;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)km->ksps[0], &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Size of MPI communicator used for MPI parallel KSP solve %d\n", size));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Desired minimum number of matrix rows on each MPI process for MPI parallel solve %" PetscInt_FMT "\n", km->mincntperrank));
  PetscCall(PetscViewerASCIIPrintf(viewer, "*** Use -mpi_linear_solver_server_view to view statistics on all the solves ***\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_MPI(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_MPI *km = (PC_MPI *)pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPI linear solver server options");
  PetscCall(PetscOptionsInt("-minimum_count_per_rank", "Desired minimum number of nonzeros per rank", "None", km->mincntperrank, &km->mincntperrank, NULL));
  PetscCall(PetscOptionsBool("-always_use_server", "Use the server even if only one rank is used for the solve (for debugging)", "None", km->alwaysuseserver, &km->alwaysuseserver, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCMPI - Calls an MPI parallel `KSP` to solve a linear system from user code running on one process

   Options Database Keys for the Server:
+  -mpi_linear_solver_server - causes the PETSc program to start in MPI linear solver server mode where only the first MPI rank runs user code
.  -mpi_linear_solver_server_view - displays information about all the linear systems solved by the MPI linear solver server
-  -mpi_linear_solver_server_use_shared_memory <true, false> - use shared memory to distribute matrix and right hand side, defaults to true

   Options Database Keys for a specific `KSP` object
+  -[any_ksp_prefix]_mpi_linear_solver_server_minimum_count_per_rank - sets the minimum size of the linear system per MPI rank that the solver will strive for
-  -[any_ksp_prefix]_mpi_linear_solver_server_always_use_server - use the server solver code even if the particular system is only solved on the process (for debugging and testing purposes)

   Level: developer

   Notes:
   This cannot be used with vectors or matrices that are created using arrays provided by the user, such as `VecCreateWithArray()` or
   `MatCreateSeqAIJWithArrays()`

   The options database prefix for the actual solver is any prefix provided before use to the original `KSP` with `KSPSetOptionsPrefix()`, mostly commonly no prefix is used.

   It can be particularly useful for user OpenMP code or potentially user GPU code.

   When the program is running with a single MPI process then it directly uses the provided matrix and right-hand side
   and does not need to distribute the matrix and vector to the various MPI processes; thus it incurs no extra overhead over just using the `KSP` directly.

   The solver options for actual solving `KSP` and `PC` must be controlled via the options database, calls to set options directly on the user level `KSP` and `PC` have no effect
   because they are not the actual solver objects.

   When `-log_view` is used with this solver the events within the parallel solve are logging in their own stage. Some of the logging in the other
   stages will be confusing since the event times are only recorded on the 0th MPI rank, thus the percent of time in the events will be misleading.

   Developer Note:
   This `PCType` is never directly selected by the user, it is set when the option `-mpi_linear_solver_server` is used and the `PC` is at the outer most nesting of
   a `KSP`. The outer most `KSP` object is automatically set to `KSPPREONLY` and thus is not directly visible to the user.

.seealso: [](sec_pcmpi), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `PC`, `PCMPIServerBegin()`, `PCMPIServerEnd()`, `KSPCheckPCMPI()`
M*/
PETSC_EXTERN PetscErrorCode PCCreate_MPI(PC pc)
{
  PC_MPI *km;
  char   *found = NULL;

  PetscFunctionBegin;
  PetscCall(PetscStrstr(((PetscObject)pc)->prefix, "mpi_linear_solver_server_", &found));
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCMPI object prefix does not have mpi_linear_solver_server_");

  /* material from PCSetType() */
  PetscTryTypeMethod(pc, destroy);
  pc->ops->destroy = NULL;
  pc->data         = NULL;

  PetscCall(PetscFunctionListDestroy(&((PetscObject)pc)->qlist));
  PetscCall(PetscMemzero(pc->ops, sizeof(struct _PCOps)));
  pc->modifysubmatrices  = NULL;
  pc->modifysubmatricesP = NULL;
  pc->setupcalled        = PETSC_FALSE;

  PetscCall(PetscNew(&km));
  pc->data = (void *)km;

  km->mincntperrank = 10000;

  pc->ops->setup          = PCSetUp_MPI;
  pc->ops->apply          = PCApply_MPI;
  pc->ops->destroy        = PCDestroy_MPI;
  pc->ops->view           = PCView_MPI;
  pc->ops->setfromoptions = PCSetFromOptions_MPI;
  PetscCall(PetscObjectChangeTypeName((PetscObject)pc, PCMPI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMPIGetKSP - Gets the `KSP` created by the `PCMPI`

  Not Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. innerksp - the inner `KSP`

  Level: advanced

.seealso: [](ch_ksp), `KSP`, `PCMPI`, `PCREDISTRIBUTE`
@*/
PetscErrorCode PCMPIGetKSP(PC pc, KSP *innerksp)
{
  PC_MPI *red = (PC_MPI *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(innerksp, 2);
  *innerksp = red->ksps[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}
