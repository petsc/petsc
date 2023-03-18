/*
    This file creates an MPI parallel KSP from a sequential PC that lives on MPI rank 0.
    It is intended to allow using PETSc MPI parallel linear solvers from non-MPI codes.

    That program may use OpenMP to compute the right hand side and matrix for the linear system

    The code uses MPI_COMM_WORLD below but maybe it should be PETSC_COMM_WORLD

    The resulting KSP and PC can only be controlled via the options database, though some common commands
    could be passed through the server.

*/
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <petsc.h>

#define PC_MPI_MAX_RANKS  256
#define PC_MPI_COMM_WORLD MPI_COMM_WORLD

typedef struct {
  KSP         ksps[PC_MPI_MAX_RANKS];                               /* The addresses of the MPI parallel KSP on each rank, NULL when not on a rank. */
  PetscMPIInt sendcount[PC_MPI_MAX_RANKS], displ[PC_MPI_MAX_RANKS]; /* For scatter/gather of rhs/solution */
  PetscMPIInt NZ[PC_MPI_MAX_RANKS], NZdispl[PC_MPI_MAX_RANKS];      /* For scatter of nonzero values in matrix (and nonzero column indices initially */
  PetscInt    mincntperrank;                                        /* minimum number of desired nonzeros per active rank in MPI parallel KSP solve */
  PetscBool   alwaysuseserver;                                      /* for debugging use the server infrastructure even if only one MPI rank is used for the solve */
} PC_MPI;

typedef enum {
  PCMPI_EXIT, /* exit the PC server loop, means the controlling sequential program is done */
  PCMPI_CREATE,
  PCMPI_SET_MAT,           /* set original matrix (or one with different nonzero pattern) */
  PCMPI_UPDATE_MAT_VALUES, /* update current matrix with new nonzero values */
  PCMPI_SOLVE,
  PCMPI_VIEW,
  PCMPI_DESTROY /* destroy a KSP that is no longer needed */
} PCMPICommand;

static MPI_Comm  PCMPIComms[PC_MPI_MAX_RANKS];
static PetscBool PCMPICommSet = PETSC_FALSE;
static PetscInt  PCMPISolveCounts[PC_MPI_MAX_RANKS], PCMPIKSPCounts[PC_MPI_MAX_RANKS], PCMPIMatCounts[PC_MPI_MAX_RANKS], PCMPISolveCountsSeq = 0, PCMPIKSPCountsSeq = 0;
static PetscInt  PCMPIIterations[PC_MPI_MAX_RANKS], PCMPISizes[PC_MPI_MAX_RANKS], PCMPIIterationsSeq = 0, PCMPISizesSeq = 0;

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

PetscErrorCode PCMPICommsDestroy(void)
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
  char       *prefix;
  PetscMPIInt len = 0;

  PetscFunctionBegin;
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
    ksp = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogStagePush(PCMPIStage));
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(PetscLogStagePop());
  PetscCallMPI(MPI_Gather(&ksp, 1, MPI_AINT, pc ? km->ksps : NULL, 1, MPI_AINT, 0, comm));
  if (pc) {
    size_t slen;

    PetscCallMPI(MPI_Comm_size(comm, &size));
    PCMPIKSPCounts[size - 1]++;
    PetscCall(PCGetOptionsPrefix(pc, (const char **)&prefix));
    PetscCall(PetscStrlen(prefix, &slen));
    len = (PetscMPIInt)slen;
  }
  PetscCallMPI(MPI_Bcast(&len, 1, MPI_INT, 0, comm));
  if (len) {
    if (!pc) PetscCall(PetscMalloc1(len + 1, &prefix));
    PetscCallMPI(MPI_Bcast(prefix, len + 1, MPI_CHAR, 0, comm));
    PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  }
  PetscCall(KSPAppendOptionsPrefix(ksp, "mpi_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPISetMat(PC pc)
{
  PC_MPI            *km = pc ? (PC_MPI *)pc->data : NULL;
  Mat                A;
  PetscInt           m, n, *ia, *ja, j, bs;
  Mat                sA;
  MPI_Comm           comm = PC_MPI_COMM_WORLD;
  KSP                ksp;
  PetscLayout        layout;
  const PetscInt    *IA = NULL, *JA = NULL;
  const PetscInt    *range;
  PetscMPIInt       *NZ = NULL, sendcounti[PC_MPI_MAX_RANKS], displi[PC_MPI_MAX_RANKS], *NZdispl = NULL, nz, size, i;
  PetscScalar       *a;
  const PetscScalar *sa               = NULL;
  PetscInt           matproperties[7] = {0, 0, 0, 0, 0, 0, 0};

  PetscFunctionBegin;
  PetscCallMPI(MPI_Scatter(pc ? km->ksps : NULL, 1, MPI_AINT, &ksp, 1, MPI_AINT, 0, comm));
  if (!ksp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  if (pc) {
    PetscBool isset, issymmetric, ishermitian, isspd, isstructurallysymmetric;

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
  }
  PetscCallMPI(MPI_Bcast(matproperties, 7, MPIU_INT, 0, comm));

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

  /* copy over the matrix nonzero structure and values */
  if (pc) {
    NZ      = km->NZ;
    NZdispl = km->NZdispl;
    PetscCall(PetscLayoutGetRanges(layout, &range));
    PetscCall(MatGetRowIJ(sA, 0, PETSC_FALSE, PETSC_FALSE, NULL, &IA, &JA, NULL));
    for (i = 0; i < size; i++) {
      sendcounti[i] = (PetscMPIInt)(1 + range[i + 1] - range[i]);
      NZ[i]         = (PetscMPIInt)(IA[range[i + 1]] - IA[range[i]]);
    }
    displi[0]  = 0;
    NZdispl[0] = 0;
    for (j = 1; j < size; j++) {
      displi[j]  = displi[j - 1] + sendcounti[j - 1] - 1;
      NZdispl[j] = NZdispl[j - 1] + NZ[j - 1];
    }
    PetscCall(MatSeqAIJGetArrayRead(sA, &sa));
  }
  PetscCall(PetscLayoutDestroy(&layout));
  PetscCallMPI(MPI_Scatter(NZ, 1, MPI_INT, &nz, 1, MPI_INT, 0, comm));

  PetscCall(PetscMalloc3(n + 1, &ia, nz, &ja, nz, &a));
  PetscCallMPI(MPI_Scatterv(IA, sendcounti, displi, MPIU_INT, ia, n + 1, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Scatterv(JA, NZ, NZdispl, MPIU_INT, ja, nz, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Scatterv(sa, NZ, NZdispl, MPIU_SCALAR, a, nz, MPIU_SCALAR, 0, comm));

  if (pc) {
    PetscCall(MatSeqAIJRestoreArrayRead(sA, &sa));
    PetscCall(MatRestoreRowIJ(sA, 0, PETSC_FALSE, PETSC_FALSE, NULL, &IA, &JA, NULL));
  }

  for (j = 1; j < n + 1; j++) ia[j] -= ia[0];
  ia[0] = 0;
  PetscCall(PetscLogStagePush(PCMPIStage));
  PetscCall(MatCreateMPIAIJWithArrays(comm, m, n, matproperties[0], matproperties[1], ia, ja, a, &A));
  PetscCall(MatSetBlockSize(A, matproperties[2]));
  PetscCall(MatSetOptionsPrefix(A, "mpi_"));
  if (matproperties[3]) PetscCall(MatSetOption(A, MAT_SYMMETRIC, matproperties[3] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[4]) PetscCall(MatSetOption(A, MAT_HERMITIAN, matproperties[4] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[5]) PetscCall(MatSetOption(A, MAT_SPD, matproperties[5] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[6]) PetscCall(MatSetOption(A, MAT_STRUCTURALLY_SYMMETRIC, matproperties[6] == 1 ? PETSC_TRUE : PETSC_FALSE));

  PetscCall(PetscFree3(ia, ja, a));
  PetscCall(KSPSetOperators(ksp, A, A));
  if (!ksp->vec_sol) PetscCall(MatCreateVecs(A, &ksp->vec_sol, &ksp->vec_rhs));
  PetscCall(PetscLogStagePop());
  if (pc) { /* needed for scatterv/gatherv of rhs and solution */
    const PetscInt *range;

    PetscCall(VecGetOwnershipRanges(ksp->vec_sol, &range));
    for (i = 0; i < size; i++) {
      km->sendcount[i] = (PetscMPIInt)(range[i + 1] - range[i]);
      km->displ[i]     = (PetscMPIInt)range[i];
    }
  }
  PetscCall(MatDestroy(&A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMPIUpdateMatValues(PC pc)
{
  PC_MPI            *km = pc ? (PC_MPI *)pc->data : NULL;
  KSP                ksp;
  Mat                sA, A;
  MPI_Comm           comm = PC_MPI_COMM_WORLD;
  PetscScalar       *a;
  PetscCount         nz;
  const PetscScalar *sa = NULL;
  PetscMPIInt        size;
  PetscInt           matproperties[4] = {0, 0, 0, 0};

  PetscFunctionBegin;
  if (pc) {
    PetscCall(PCGetOperators(pc, &sA, &sA));
    PetscCall(MatSeqAIJGetArrayRead(sA, &sa));
  }
  PetscCallMPI(MPI_Scatter(pc ? km->ksps : NULL, 1, MPI_AINT, &ksp, 1, MPI_AINT, 0, comm));
  if (!ksp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PCMPIMatCounts[size - 1]++;
  PetscCall(KSPGetOperators(ksp, NULL, &A));
  PetscCall(MatMPIAIJGetNumberNonzeros(A, &nz));
  PetscCall(PetscMalloc1(nz, &a));
  PetscCallMPI(MPI_Scatterv(sa, pc ? km->NZ : NULL, pc ? km->NZdispl : NULL, MPIU_SCALAR, a, nz, MPIU_SCALAR, 0, comm));
  if (pc) {
    PetscBool isset, issymmetric, ishermitian, isspd, isstructurallysymmetric;

    PetscCall(MatSeqAIJRestoreArrayRead(sA, &sa));

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
  PetscCall(PetscFree(a));
  PetscCallMPI(MPI_Bcast(matproperties, 4, MPIU_INT, 0, comm));
  /* if any of these properties was previously set and is now not set this will result in incorrect properties in A since there is no way to unset a property */
  if (matproperties[0]) PetscCall(MatSetOption(A, MAT_SYMMETRIC, matproperties[0] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[1]) PetscCall(MatSetOption(A, MAT_HERMITIAN, matproperties[1] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[2]) PetscCall(MatSetOption(A, MAT_SPD, matproperties[2] == 1 ? PETSC_TRUE : PETSC_FALSE));
  if (matproperties[3]) PetscCall(MatSetOption(A, MAT_STRUCTURALLY_SYMMETRIC, matproperties[3] == 1 ? PETSC_TRUE : PETSC_FALSE));
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

  PetscFunctionBegin;
  PetscCallMPI(MPI_Scatter(pc ? km->ksps : &ksp, 1, MPI_AINT, &ksp, 1, MPI_AINT, 0, comm));
  if (!ksp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  /* TODO: optimize code to not require building counts/displ every time */

  /* scatterv rhs */
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (pc) {
    PetscInt N;

    PCMPISolveCounts[size - 1]++;
    PetscCall(MatGetSize(pc->pmat, &N, NULL));
    ;
    PCMPISizes[size - 1] += N;
    PetscCall(VecGetArrayRead(B, &sb));
  }
  PetscCall(VecGetLocalSize(ksp->vec_rhs, &n));
  PetscCall(VecGetArray(ksp->vec_rhs, &b));
  PetscCallMPI(MPI_Scatterv(sb, pc ? km->sendcount : NULL, pc ? km->displ : NULL, MPIU_SCALAR, b, n, MPIU_SCALAR, 0, comm));
  PetscCall(VecRestoreArray(ksp->vec_rhs, &b));
  if (pc) PetscCall(VecRestoreArrayRead(B, &sb));

  PetscCall(PetscLogStagePush(PCMPIStage));
  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(PetscLogStagePop());
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PCMPIIterations[size - 1] += its;

  /* gather solution */
  PetscCall(VecGetArrayRead(ksp->vec_sol, &x));
  if (pc) PetscCall(VecGetArray(X, &sx));
  PetscCallMPI(MPI_Gatherv(x, n, MPIU_SCALAR, sx, pc ? km->sendcount : NULL, pc ? km->displ : NULL, MPIU_SCALAR, 0, comm));
  if (pc) PetscCall(VecRestoreArray(X, &sx));
  PetscCall(VecRestoreArrayRead(ksp->vec_sol, &x));
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
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     PCMPIServerBegin - starts a server that runs on the rank != 0 MPI processes waiting to process requests for
     parallel `KSP` solves and management of parallel `KSP` objects.

     Logically Collective on all MPI ranks except 0

   Options Database Keys:
+   -mpi_linear_solver_server - causes the PETSc program to start in MPI linear solver server mode where only the first MPI rank runs user code
-   -mpi_linear_solver_server_view - displays information about the linear systems solved by the MPI linear solver server

     Level: developer

     Note:
      This is normally started automatically in `PetscInitialize()` when the option is provided

     Developer Notes:
       When called on rank zero this sets `PETSC_COMM_WORLD` to `PETSC_COMM_SELF` to allow a main program
       written with `PETSC_COMM_WORLD` to run correctly on the single rank while all the ranks
       (that would normally be sharing `PETSC_COMM_WORLD`) to run the solver server.

       Can this be integrated into the `PetscDevice` abstraction that is currently being developed?

.seealso: `PCMPIServerEnd()`, `PCMPI`
@*/
PetscErrorCode PCMPIServerBegin(void)
{
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL, "Starting MPI Linear Solver Server"));
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

  PetscCallMPI(MPI_Comm_rank(PC_MPI_COMM_WORLD, &rank));
  if (rank == 0) {
    PETSC_COMM_WORLD = PETSC_COMM_SELF;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  while (PETSC_TRUE) {
    PCMPICommand request = PCMPI_CREATE;
    PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, PC_MPI_COMM_WORLD));
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
      This is normally ended automatically in `PetscFinalize()` when the option is provided

.seealso: `PCMPIServerBegin()`, `PCMPI`
@*/
PetscErrorCode PCMPIServerEnd(void)
{
  PCMPICommand request = PCMPI_EXIT;

  PetscFunctionBegin;
  if (PetscGlobalRank == 0) {
    PetscViewer       viewer = NULL;
    PetscViewerFormat format;

    PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, PC_MPI_COMM_WORLD));
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
      }
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscCall(PCMPICommsDestroy());
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

  PetscFunctionBegin;
  PetscCall(PCGetOperators(pc, NULL, &sA));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(KSPCreate(PETSC_COMM_SELF, &km->ksps[0]));
  PetscCall(KSPSetOptionsPrefix(km->ksps[0], prefix));
  PetscCall(KSPAppendOptionsPrefix(km->ksps[0], "mpi_"));
  PetscCall(KSPSetOperators(km->ksps[0], sA, sA));
  PetscCall(KSPSetFromOptions(km->ksps[0]));
  PetscCall(KSPSetUp(km->ksps[0]));
  PetscCall(PetscInfo((PetscObject)pc, "MPI parallel linear solver system is being solved directly on rank 0 due to its small size\n"));
  PCMPIKSPCountsSeq++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_Seq(PC pc, Vec b, Vec x)
{
  PC_MPI  *km = (PC_MPI *)pc->data;
  PetscInt its, n;
  Mat      A;

  PetscFunctionBegin;
  PetscCall(KSPSolve(km->ksps[0], b, x));
  PetscCall(KSPGetIterationNumber(km->ksps[0], &its));
  PCMPISolveCountsSeq++;
  PCMPIIterationsSeq += its;
  PetscCall(KSPGetOperators(km->ksps[0], NULL, &A));
  PetscCall(MatGetSize(A, &n, NULL));
  PCMPISizesSeq += n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Seq(PC pc, PetscViewer viewer)
{
  PC_MPI     *km = (PC_MPI *)pc->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Running MPI linear solver server directly on rank 0 due to its small size\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Desired minimum number of nonzeros per rank for MPI parallel solve %d\n", (int)km->mincntperrank));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  if (prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "*** Use -%smpi_ksp_view to see the MPI KSP parameters ***\n", prefix));
  else PetscCall(PetscViewerASCIIPrintf(viewer, "*** Use -mpi_ksp_view to see the MPI KSP parameters ***\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "*** Use -mpi_linear_solver_server_view to statistics on all the solves ***\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Seq(PC pc)
{
  PC_MPI *km = (PC_MPI *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&km->ksps[0]));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PCSetUp_MPI - Trigger the creation of the MPI parallel PC and copy parts of the matrix and
     right hand side to the parallel PC
*/
static PetscErrorCode PCSetUp_MPI(PC pc)
{
  PC_MPI      *km = (PC_MPI *)pc->data;
  PetscMPIInt  rank, size;
  PCMPICommand request;
  PetscBool    newmatrix = PETSC_FALSE;

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

    request = PCMPI_CREATE;
    PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, MPI_COMM_WORLD));
    PetscCall(PCMPICreate(pc));
    newmatrix = PETSC_TRUE;
  }
  if (pc->flag == DIFFERENT_NONZERO_PATTERN) newmatrix = PETSC_TRUE;

  if (newmatrix) {
    PetscCall(PetscInfo((PetscObject)pc, "New matrix or matrix has changed nonzero structure\n"));
    request = PCMPI_SET_MAT;
    PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, MPI_COMM_WORLD));
    PetscCall(PCMPISetMat(pc));
  } else {
    PetscCall(PetscInfo((PetscObject)pc, "Matrix has only changed nozero values\n"));
    request = PCMPI_UPDATE_MAT_VALUES;
    PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, MPI_COMM_WORLD));
    PetscCall(PCMPIUpdateMatValues(pc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_MPI(PC pc, Vec b, Vec x)
{
  PCMPICommand request = PCMPI_SOLVE;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, MPI_COMM_WORLD));
  PetscCall(PCMPISolve(pc, b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCDestroy_MPI(PC pc)
{
  PCMPICommand request = PCMPI_DESTROY;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Bcast(&request, 1, MPIU_ENUM, 0, MPI_COMM_WORLD));
  PetscCall(PCMPIDestroy(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PCView_MPI - Cannot call view on the MPI parallel KSP because other ranks do not have access to the viewer
*/
PetscErrorCode PCView_MPI(PC pc, PetscViewer viewer)
{
  PC_MPI     *km = (PC_MPI *)pc->data;
  MPI_Comm    comm;
  PetscMPIInt size;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)km->ksps[0], &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Size of MPI communicator used for MPI parallel KSP solve %d\n", size));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Desired minimum number of nonzeros per rank for MPI parallel solve %d\n", (int)km->mincntperrank));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  if (prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "*** Use -%smpi_ksp_view to see the MPI KSP parameters ***\n", prefix));
  else PetscCall(PetscViewerASCIIPrintf(viewer, "*** Use -mpi_ksp_view to see the MPI KSP parameters ***\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "*** Use -mpi_linear_solver_server_view to statistics on all the solves ***\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCSetFromOptions_MPI(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_MPI *km = (PC_MPI *)pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPI linear solver server options");
  PetscCall(PetscOptionsInt("-pc_mpi_minimum_count_per_rank", "Desired minimum number of nonzeros per rank", "None", km->mincntperrank, &km->mincntperrank, NULL));
  PetscCall(PetscOptionsBool("-pc_mpi_always_use_server", "Use the server even if only one rank is used for the solve (for debugging)", "None", km->alwaysuseserver, &km->alwaysuseserver, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCMPI - Calls an MPI parallel `KSP` to solve a linear system from user code running on one process

   Options Database Keys:
+  -mpi_linear_solver_server - causes the PETSc program to start in MPI linear solver server mode where only the first MPI rank runs user code
.  -mpi_linear_solver_server_view - displays information about the linear systems solved by the MPI linear solver server
.  -pc_mpi_minimum_count_per_rank - sets the minimum size of the linear system per MPI rank that the solver will strive for
-  -pc_mpi_always_use_server - use the server solver code even if the particular system is only solved on the process, this option is only for debugging and testing purposes

   Level: beginner

   Notes:
   The options database prefix for the MPI parallel `KSP` and `PC` is -mpi_

   The MPI linear solver server will not support scaling user code to utilize extremely large numbers of MPI ranks but should give reasonable speedup for
   potentially 4 to 8 MPI ranks depending on the linear system being solved, solver algorithm, and the hardware.

   It can be particularly useful for user OpenMP code or potentially user GPU code.

   When the program is running with a single MPI rank then this directly uses the provided matrix and right hand side (still in a `KSP` with the options prefix of -mpi)
   and does not need to distribute the matrix and vector to the various MPI ranks; thus it incurs no extra overhead over just using the `KSP` directly.

   The solver options for `KSP` and `PC` must be controlled via the options database, calls to set options directly on the user level `KSP` and `PC` have no effect
   because they are not the actual solver objects.

   When `-log_view` is used with this solver the events within the parallel solve are logging in their own stage. Some of the logging in the other
   stages will be confusing since the event times are only recorded on the 0th MPI rank, thus the percent of time in the events will be misleading.

.seealso: `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `PC`, `PCMPIServerBegin()`, `PCMPIServerEnd()`
M*/
PETSC_EXTERN PetscErrorCode PCCreate_MPI(PC pc)
{
  PC_MPI *km;

  PetscFunctionBegin;
  PetscCall(PetscNew(&km));
  pc->data = (void *)km;

  km->mincntperrank = 10000;

  pc->ops->setup          = PCSetUp_MPI;
  pc->ops->apply          = PCApply_MPI;
  pc->ops->destroy        = PCDestroy_MPI;
  pc->ops->view           = PCView_MPI;
  pc->ops->setfromoptions = PCSetFromOptions_MPI;
  PetscFunctionReturn(PETSC_SUCCESS);
}
