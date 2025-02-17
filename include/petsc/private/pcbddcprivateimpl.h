/* prototypes of all BDDC private functions */
#pragma once

#include <petsc/private/pcbddcstructsimpl.h>

/* main functions */
PETSC_INTERN PetscErrorCode PCBDDCAnalyzeInterface(PC);
PETSC_INTERN PetscErrorCode PCBDDCConstraintsSetUp(PC);

/* load or dump customization */
PETSC_EXTERN PetscErrorCode PCBDDCLoadOrViewCustomization(PC, PetscBool, const char *);

/* reset functions */
PETSC_EXTERN PetscErrorCode PCBDDCResetTopography(PC);
PETSC_EXTERN PetscErrorCode PCBDDCResetSolvers(PC);
PETSC_EXTERN PetscErrorCode PCBDDCResetCustomization(PC);

/* graph */
PETSC_EXTERN PetscErrorCode    PCBDDCGraphCreate(PCBDDCGraph *);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphDestroy(PCBDDCGraph *);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphInit(PCBDDCGraph, ISLocalToGlobalMapping, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphReset(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphResetCSR(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphResetCoords(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphSetUp(PCBDDCGraph, PetscInt, IS, IS, PetscInt, IS[], IS);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphComputeConnectedComponents(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphComputeConnectedComponentsLocal(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphASCIIView(PCBDDCGraph, PetscInt, PetscViewer);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphGetCandidatesIS(PCBDDCGraph, PetscInt *, IS *[], PetscInt *, IS *[], IS *);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphRestoreCandidatesIS(PCBDDCGraph, PetscInt *, IS *[], PetscInt *, IS *[], IS *);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphGetDirichletDofs(PCBDDCGraph, IS *);
PETSC_EXTERN PetscErrorCode    PCBDDCGraphGetDirichletDofsB(PCBDDCGraph, IS *);
PETSC_EXTERN PetscCtxDestroyFn PCBDDCDestroyGraphCandidatesIS;

/* interface for scaling operator */
PETSC_INTERN PetscErrorCode PCBDDCScalingSetUp(PC);
PETSC_INTERN PetscErrorCode PCBDDCScalingDestroy(PC);
PETSC_INTERN PetscErrorCode PCBDDCScalingRestriction(PC, Vec, Vec);
PETSC_INTERN PetscErrorCode PCBDDCScalingExtension(PC, Vec, Vec);

/* nullspace correction */
PETSC_INTERN PetscErrorCode PCBDDCNullSpaceAssembleCorrection(PC, PetscBool, PetscBool);

/* utils */
PETSC_INTERN PetscErrorCode PCBDDCComputeLocalMatrix(PC, Mat);
PETSC_INTERN PetscErrorCode PCBDDCSetUpLocalWorkVectors(PC);
PETSC_INTERN PetscErrorCode PCBDDCSetUpSolvers(PC);
PETSC_INTERN PetscErrorCode PCBDDCSetUpLocalScatters(PC);
PETSC_INTERN PetscErrorCode PCBDDCSetUpLocalSolvers(PC, PetscBool, PetscBool);
PETSC_INTERN PetscErrorCode PCBDDCSetUpCorrection(PC, Mat *);
PETSC_INTERN PetscErrorCode PCBDDCSetUpCoarseSolver(PC, Mat);
PETSC_INTERN PetscErrorCode PCBDDCComputePrimalNumbering(PC, PetscInt *, PetscInt **);
PETSC_INTERN PetscErrorCode PCBDDCScatterCoarseDataBegin(PC, InsertMode, ScatterMode);
PETSC_INTERN PetscErrorCode PCBDDCScatterCoarseDataEnd(PC, InsertMode, ScatterMode);
PETSC_INTERN PetscErrorCode PCBDDCApplyInterfacePreconditioner(PC, PetscBool);
PETSC_INTERN PetscErrorCode PCBDDCOrthonormalizeVecs(PetscInt *, Vec[]);
PETSC_INTERN PetscErrorCode PCBDDCSetUseExactDirichlet(PC, PetscBool);
PETSC_INTERN PetscErrorCode PCBDDCSetLevel(PC, PetscInt);
PETSC_INTERN PetscErrorCode PCBDDCGlobalToLocal(VecScatter, Vec, Vec, IS, IS *);
PETSC_INTERN PetscErrorCode PCBDDCAdaptiveSelection(PC);
PETSC_INTERN PetscErrorCode PCBDDCConsistencyCheckIS(PC, MPI_Op, IS *);
PETSC_INTERN PetscErrorCode PCBDDCComputeLocalTopologyInfo(PC);
PETSC_INTERN PetscErrorCode PCBDDCDetectDisconnectedComponents(PC, PetscBool, PetscInt *, IS *[], IS *);
PETSC_INTERN PetscErrorCode PCBDDCReuseSolversBenignAdapt(PCBDDCReuseSolvers, Vec, Vec, PetscBool, PetscBool);
PETSC_INTERN PetscErrorCode PCBDDCComputeNoNetFlux(Mat, Mat, PetscBool, IS, PCBDDCGraph, MatNullSpace *);
PETSC_INTERN PetscErrorCode PCBDDCNullSpaceCreate(MPI_Comm, PetscBool, PetscInt, Vec[], MatNullSpace *);
PETSC_INTERN PetscErrorCode PCBDDCNedelecSupport(PC);
PETSC_INTERN PetscErrorCode PCBDDCAddPrimalVerticesLocalIS(PC, IS);
PETSC_INTERN PetscErrorCode PCBDDCComputeFakeChange(PC, PetscBool, PCBDDCGraph, PCBDDCSubSchurs, Mat *, IS *, IS *, PetscBool *);
PETSC_INTERN PetscErrorCode MatCreateSubMatrixUnsorted(Mat, IS, IS, Mat *);
PETSC_INTERN PetscErrorCode MatSeqAIJCompress(Mat, Mat *);
PETSC_INTERN PetscErrorCode MatNullSpacePropagateAny_Private(Mat, IS, Mat);

/* benign subspace trick */
PETSC_INTERN PetscErrorCode PCBDDCBenignPopOrPushB0(PC, PetscBool);
PETSC_INTERN PetscErrorCode PCBDDCBenignGetOrSetP0(PC, Vec, PetscBool);
PETSC_INTERN PetscErrorCode PCBDDCBenignDetectSaddlePoint(PC, PetscBool, IS *);
PETSC_INTERN PetscErrorCode PCBDDCBenignCheck(PC, IS);
PETSC_INTERN PetscErrorCode PCBDDCBenignShellMat(PC, PetscBool);
PETSC_INTERN PetscErrorCode PCBDDCBenignRemoveInterior(PC, Vec, Vec);

/* feti-dp */
PETSC_INTERN PetscErrorCode PCBDDCCreateFETIDPMatContext(PC, FETIDPMat_ctx *);
PETSC_INTERN PetscErrorCode PCBDDCSetupFETIDPMatContext(FETIDPMat_ctx);
PETSC_INTERN PetscErrorCode PCBDDCCreateFETIDPPCContext(PC, FETIDPPC_ctx *);
PETSC_INTERN PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat, FETIDPPC_ctx);
PETSC_INTERN PetscErrorCode FETIDPPCApply(PC, Vec, Vec);
PETSC_INTERN PetscErrorCode FETIDPPCApplyTranspose(PC, Vec, Vec);
PETSC_INTERN PetscErrorCode FETIDPPCView(PC, PetscViewer);
PETSC_INTERN PetscErrorCode PCBDDCDestroyFETIDPPC(PC);
PETSC_INTERN PetscErrorCode FETIDPMatMult(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode FETIDPMatMultTranspose(Mat, Vec, Vec);

PETSC_INTERN PetscErrorCode PCBDDCDestroyFETIDPMat(Mat);

/* interface to SubSchurs */
PETSC_INTERN PetscErrorCode PCBDDCInitSubSchurs(PC);
PETSC_INTERN PetscErrorCode PCBDDCSetUpSubSchurs(PC);

/* sub schurs API */
PETSC_INTERN PetscErrorCode PCBDDCSubSchursCreate(PCBDDCSubSchurs *);
PETSC_INTERN PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs, const char *, IS, IS, PCBDDCGraph, ISLocalToGlobalMapping, PetscBool, PetscBool);
PETSC_INTERN PetscErrorCode PCBDDCSubSchursReset(PCBDDCSubSchurs);
PETSC_INTERN PetscErrorCode PCBDDCSubSchursDestroy(PCBDDCSubSchurs *);
PETSC_INTERN PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs, Mat, Mat, PetscBool, PetscInt[], PetscInt[], PetscInt, Vec, PetscBool, PetscBool, PetscBool, PetscInt, PetscInt[], IS[], Mat, IS);
