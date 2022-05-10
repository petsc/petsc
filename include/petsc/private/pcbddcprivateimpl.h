/* prototypes of all BDDC private functions */
#if !defined(__pcbddc_private_h)
#define __pcbddc_private_h

#include <petsc/private/pcbddcstructsimpl.h>

/* main functions */
PETSC_EXTERN PetscErrorCode PCBDDCAnalyzeInterface(PC);
PETSC_EXTERN PetscErrorCode PCBDDCConstraintsSetUp(PC);

/* reset functions */
PETSC_EXTERN PetscErrorCode PCBDDCResetTopography(PC);
PETSC_EXTERN PetscErrorCode PCBDDCResetSolvers(PC);
PETSC_EXTERN PetscErrorCode PCBDDCResetCustomization(PC);

/* graph */
PETSC_EXTERN PetscErrorCode PCBDDCGraphCreate(PCBDDCGraph*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphDestroy(PCBDDCGraph*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphInit(PCBDDCGraph,ISLocalToGlobalMapping,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PCBDDCGraphReset(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphResetCSR(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphResetCoords(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphSetUp(PCBDDCGraph,PetscInt,IS,IS,PetscInt,IS[],IS);
PETSC_EXTERN PetscErrorCode PCBDDCGraphComputeConnectedComponents(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphComputeConnectedComponentsLocal(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphASCIIView(PCBDDCGraph,PetscInt,PetscViewer);
PETSC_EXTERN PetscErrorCode PCBDDCGraphGetCandidatesIS(PCBDDCGraph,PetscInt*,IS*[],PetscInt*,IS*[],IS*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphRestoreCandidatesIS(PCBDDCGraph,PetscInt*,IS*[],PetscInt*,IS*[],IS*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphGetDirichletDofs(PCBDDCGraph,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphGetDirichletDofsB(PCBDDCGraph,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCDestroyGraphCandidatesIS(void*);

/* interface for scaling operator */
PETSC_EXTERN PetscErrorCode PCBDDCScalingSetUp(PC);
PETSC_EXTERN PetscErrorCode PCBDDCScalingDestroy(PC);
PETSC_EXTERN PetscErrorCode PCBDDCScalingRestriction(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCBDDCScalingExtension(PC,Vec,Vec);

/* nullspace correction */
PETSC_EXTERN PetscErrorCode PCBDDCNullSpaceAssembleCorrection(PC,PetscBool,PetscBool);

/* utils */
PETSC_EXTERN PetscErrorCode PCBDDCComputeLocalMatrix(PC,Mat);
PETSC_EXTERN PetscErrorCode PCBDDCSetUpLocalWorkVectors(PC);
PETSC_EXTERN PetscErrorCode PCBDDCSetUpSolvers(PC);
PETSC_EXTERN PetscErrorCode PCBDDCSetUpLocalScatters(PC);
PETSC_EXTERN PetscErrorCode PCBDDCSetUpLocalSolvers(PC,PetscBool,PetscBool);
PETSC_EXTERN PetscErrorCode PCBDDCSetUpCorrection(PC,PetscScalar**);
PETSC_EXTERN PetscErrorCode PCBDDCSetUpCoarseSolver(PC,PetscScalar*);
PETSC_EXTERN PetscErrorCode PCBDDCComputePrimalNumbering(PC,PetscInt*,PetscInt**);
PETSC_EXTERN PetscErrorCode PCBDDCScatterCoarseDataBegin(PC,InsertMode,ScatterMode);
PETSC_EXTERN PetscErrorCode PCBDDCScatterCoarseDataEnd(PC,InsertMode,ScatterMode);
PETSC_EXTERN PetscErrorCode PCBDDCApplyInterfacePreconditioner(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCBDDCOrthonormalizeVecs(PetscInt*,Vec[]);
PETSC_EXTERN PetscErrorCode PCBDDCSetUseExactDirichlet(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCBDDCSetLevel(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCBDDCGlobalToLocal(VecScatter,Vec,Vec,IS,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCAdaptiveSelection(PC);
PETSC_EXTERN PetscErrorCode PCBDDCConsistencyCheckIS(PC,MPI_Op,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCComputeLocalTopologyInfo(PC);
PETSC_EXTERN PetscErrorCode MatCreateSubMatrixUnsorted(Mat,IS,IS,Mat*);
PETSC_EXTERN PetscErrorCode PCBDDCDetectDisconnectedComponents(PC,PetscBool,PetscInt*,IS*[],IS*);
PETSC_EXTERN PetscErrorCode MatSeqAIJCompress(Mat,Mat*);
PETSC_EXTERN PetscErrorCode PCBDDCReuseSolversBenignAdapt(PCBDDCReuseSolvers,Vec,Vec,PetscBool,PetscBool);
PETSC_EXTERN PetscErrorCode PCBDDCComputeNoNetFlux(Mat,Mat,PetscBool,IS,PCBDDCGraph,MatNullSpace*);
PETSC_EXTERN PetscErrorCode PCBDDCNullSpaceCreate(MPI_Comm,PetscBool,PetscInt,Vec[],MatNullSpace*);
PETSC_EXTERN PetscErrorCode PCBDDCNedelecSupport(PC);
PETSC_EXTERN PetscErrorCode PCBDDCAddPrimalVerticesLocalIS(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCComputeFakeChange(PC,PetscBool,PCBDDCGraph,PCBDDCSubSchurs,Mat*,IS*,IS*,PetscBool*);

/* benign subspace trick */
PETSC_EXTERN PetscErrorCode PCBDDCBenignPopOrPushB0(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCBDDCBenignGetOrSetP0(PC,Vec,PetscBool);
PETSC_EXTERN PetscErrorCode PCBDDCBenignDetectSaddlePoint(PC,PetscBool,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCBenignCheck(PC,IS);
PETSC_EXTERN PetscErrorCode PCBDDCBenignShellMat(PC,PetscBool);
PETSC_EXTERN PetscErrorCode PCBDDCBenignRemoveInterior(PC,Vec,Vec);

/* feti-dp */
PETSC_EXTERN PetscErrorCode PCBDDCCreateFETIDPMatContext(PC,FETIDPMat_ctx*);
PETSC_EXTERN PetscErrorCode PCBDDCSetupFETIDPMatContext(FETIDPMat_ctx);
PETSC_EXTERN PetscErrorCode PCBDDCCreateFETIDPPCContext(PC,FETIDPPC_ctx*);
PETSC_EXTERN PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat,FETIDPPC_ctx);
PETSC_EXTERN PetscErrorCode FETIDPPCApply(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode FETIDPPCApplyTranspose(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode FETIDPPCView(PC,PetscViewer);
PETSC_EXTERN PetscErrorCode PCBDDCDestroyFETIDPPC(PC);
PETSC_EXTERN PetscErrorCode FETIDPMatMult(Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode FETIDPMatMultTranspose(Mat,Vec,Vec);

PETSC_EXTERN PetscErrorCode PCBDDCDestroyFETIDPMat(Mat);

/* interface to SubSchurs */
PETSC_EXTERN PetscErrorCode PCBDDCInitSubSchurs(PC);
PETSC_EXTERN PetscErrorCode PCBDDCSetUpSubSchurs(PC);

/* sub schurs API */
PETSC_EXTERN PetscErrorCode PCBDDCSubSchursCreate(PCBDDCSubSchurs*);
PETSC_EXTERN PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs,const char*,IS,IS,PCBDDCGraph,ISLocalToGlobalMapping,PetscBool,PetscBool);
PETSC_EXTERN PetscErrorCode PCBDDCSubSchursReset(PCBDDCSubSchurs);
PETSC_EXTERN PetscErrorCode PCBDDCSubSchursDestroy(PCBDDCSubSchurs*);
PETSC_EXTERN PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs,Mat,Mat,PetscBool,PetscInt[],PetscInt[],PetscInt,Vec,PetscBool,PetscBool,PetscBool,PetscInt,PetscInt[],IS[],Mat,IS);

#endif

