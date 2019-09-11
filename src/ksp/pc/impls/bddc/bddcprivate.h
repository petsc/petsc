/* prototypes of all BDDC private functions */
#if !defined(__pcbddc_private_h)
#define __pcbddc_private_h

#include <../src/ksp/pc/impls/bddc/bddcstructs.h>

/* main functions */
PetscErrorCode PCBDDCAnalyzeInterface(PC);
PetscErrorCode PCBDDCConstraintsSetUp(PC);

/* reset functions */
PetscErrorCode PCBDDCResetTopography(PC);
PetscErrorCode PCBDDCResetSolvers(PC);
PetscErrorCode PCBDDCResetCustomization(PC);

/* graph */
PetscErrorCode PCBDDCGraphCreate(PCBDDCGraph*);
PetscErrorCode PCBDDCGraphDestroy(PCBDDCGraph*);
PetscErrorCode PCBDDCGraphInit(PCBDDCGraph,ISLocalToGlobalMapping,PetscInt,PetscInt);
PetscErrorCode PCBDDCGraphReset(PCBDDCGraph);
PetscErrorCode PCBDDCGraphResetCSR(PCBDDCGraph);
PetscErrorCode PCBDDCGraphResetCoords(PCBDDCGraph);
PetscErrorCode PCBDDCGraphSetUp(PCBDDCGraph,PetscInt,IS,IS,PetscInt,IS[],IS);
PetscErrorCode PCBDDCGraphComputeConnectedComponents(PCBDDCGraph);
PetscErrorCode PCBDDCGraphComputeConnectedComponentsLocal(PCBDDCGraph);
PetscErrorCode PCBDDCGraphASCIIView(PCBDDCGraph,PetscInt,PetscViewer);
PetscErrorCode PCBDDCGraphGetCandidatesIS(PCBDDCGraph,PetscInt*,IS*[],PetscInt*,IS*[],IS*);
PetscErrorCode PCBDDCGraphRestoreCandidatesIS(PCBDDCGraph,PetscInt*,IS*[],PetscInt*,IS*[],IS*);
PetscErrorCode PCBDDCGraphGetDirichletDofs(PCBDDCGraph,IS*);
PetscErrorCode PCBDDCGraphGetDirichletDofsB(PCBDDCGraph,IS*);

/* interface for scaling operator */
PetscErrorCode PCBDDCScalingSetUp(PC);
PetscErrorCode PCBDDCScalingDestroy(PC);
PetscErrorCode PCBDDCScalingRestriction(PC,Vec,Vec);
PetscErrorCode PCBDDCScalingExtension(PC,Vec,Vec);

/* nullspace correction */
PetscErrorCode PCBDDCNullSpaceAssembleCorrection(PC,PetscBool,PetscBool);

/* utils */
PetscErrorCode PCBDDCComputeLocalMatrix(PC,Mat);
PetscErrorCode PCBDDCSetUpLocalWorkVectors(PC);
PetscErrorCode PCBDDCSetUpSolvers(PC);
PetscErrorCode PCBDDCSetUpLocalScatters(PC);
PetscErrorCode PCBDDCSetUpLocalSolvers(PC,PetscBool,PetscBool);
PetscErrorCode PCBDDCSetUpCorrection(PC,PetscScalar**);
PetscErrorCode PCBDDCSetUpCoarseSolver(PC,PetscScalar*);
PetscErrorCode PCBDDCComputePrimalNumbering(PC,PetscInt*,PetscInt**);
PetscErrorCode PCBDDCScatterCoarseDataBegin(PC,InsertMode,ScatterMode);
PetscErrorCode PCBDDCScatterCoarseDataEnd(PC,InsertMode,ScatterMode);
PetscErrorCode PCBDDCApplyInterfacePreconditioner(PC,PetscBool);
PetscErrorCode PCBDDCOrthonormalizeVecs(PetscInt*,Vec[]);
PetscErrorCode PCBDDCSetUseExactDirichlet(PC,PetscBool);
PetscErrorCode PCBDDCSetLevel(PC,PetscInt);
PetscErrorCode PCBDDCGlobalToLocal(VecScatter,Vec,Vec,IS,IS*);
PetscErrorCode PCBDDCAdaptiveSelection(PC);
PetscErrorCode PCBDDCConsistencyCheckIS(PC,MPI_Op,IS*);
PetscErrorCode PCBDDCComputeLocalTopologyInfo(PC);
PetscErrorCode MatCreateSubMatrixUnsorted(Mat,IS,IS,Mat*);
PetscErrorCode PCBDDCDetectDisconnectedComponents(PC,PetscBool,PetscInt*,IS*[],IS*);
PetscErrorCode MatSeqAIJCompress(Mat,Mat*);
PetscErrorCode PCBDDCReuseSolversBenignAdapt(PCBDDCReuseSolvers,Vec,Vec,PetscBool,PetscBool);
PetscErrorCode PCBDDCComputeNoNetFlux(Mat,Mat,PetscBool,IS,PCBDDCGraph,MatNullSpace*);
PetscErrorCode PCBDDCNullSpaceCreate(MPI_Comm,PetscBool,PetscInt,Vec[],MatNullSpace*);
PetscErrorCode PCBDDCNedelecSupport(PC);
PetscErrorCode PCBDDCAddPrimalVerticesLocalIS(PC,IS);

/* benign subspace trick */
PetscErrorCode PCBDDCBenignPopOrPushB0(PC,PetscBool);
PetscErrorCode PCBDDCBenignGetOrSetP0(PC,Vec,PetscBool);
PetscErrorCode PCBDDCBenignDetectSaddlePoint(PC,PetscBool,IS*);
PetscErrorCode PCBDDCBenignCheck(PC,IS);
PetscErrorCode PCBDDCBenignShellMat(PC,PetscBool);
PetscErrorCode PCBDDCBenignRemoveInterior(PC,Vec,Vec);

/* feti-dp */
PetscErrorCode PCBDDCCreateFETIDPMatContext(PC,FETIDPMat_ctx*);
PetscErrorCode PCBDDCSetupFETIDPMatContext(FETIDPMat_ctx);
PetscErrorCode PCBDDCCreateFETIDPPCContext(PC,FETIDPPC_ctx*);
PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat,FETIDPPC_ctx);
PetscErrorCode FETIDPPCApply(PC,Vec,Vec);
PetscErrorCode FETIDPPCApplyTranspose(PC,Vec,Vec);
PetscErrorCode FETIDPPCView(PC,PetscViewer);
PetscErrorCode PCBDDCDestroyFETIDPPC(PC);
PetscErrorCode FETIDPMatMult(Mat,Vec,Vec);
PetscErrorCode FETIDPMatMultTranspose(Mat,Vec,Vec);

PetscErrorCode PCBDDCDestroyFETIDPMat(Mat);

/* interface to SubSchurs */
PetscErrorCode PCBDDCInitSubSchurs(PC);
PetscErrorCode PCBDDCSetUpSubSchurs(PC);

/* sub schurs */
PetscErrorCode PCBDDCSubSchursCreate(PCBDDCSubSchurs*);
PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs,const char*,IS,IS,PCBDDCGraph,ISLocalToGlobalMapping,PetscBool);
PetscErrorCode PCBDDCSubSchursReset(PCBDDCSubSchurs);
PetscErrorCode PCBDDCSubSchursDestroy(PCBDDCSubSchurs*);
PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs,Mat,Mat,PetscBool,PetscInt[],PetscInt[],PetscInt,Vec,PetscBool,PetscBool,PetscBool,PetscInt,PetscInt[],IS[],Mat,IS);

#endif

