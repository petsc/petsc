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
PETSC_EXTERN PetscErrorCode PCBDDCGraphCreate(PCBDDCGraph*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphDestroy(PCBDDCGraph*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphInit(PCBDDCGraph,ISLocalToGlobalMapping,PetscInt);
PETSC_EXTERN PetscErrorCode PCBDDCGraphReset(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphResetCSR(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphSetUp(PCBDDCGraph,PetscInt,IS,IS,PetscInt,IS[],IS);
PETSC_EXTERN PetscErrorCode PCBDDCGraphComputeConnectedComponents(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphComputeConnectedComponentsLocal(PCBDDCGraph);
PETSC_EXTERN PetscErrorCode PCBDDCGraphASCIIView(PCBDDCGraph,PetscInt,PetscViewer);
PETSC_EXTERN PetscErrorCode PCBDDCGraphGetCandidatesIS(PCBDDCGraph,PetscInt*,IS*[],PetscInt*,IS*[],IS*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphGetDirichletDofs(PCBDDCGraph,IS*);
PETSC_EXTERN PetscErrorCode PCBDDCGraphGetDirichletDofsB(PCBDDCGraph,IS*);

/* interface for scaling operator */
PetscErrorCode PCBDDCScalingSetUp(PC);
PetscErrorCode PCBDDCScalingDestroy(PC);
PetscErrorCode PCBDDCScalingRestriction(PC,Vec,Vec);
PetscErrorCode PCBDDCScalingExtension(PC,Vec,Vec);

/* nullspace stuffs */
PetscErrorCode PCBDDCNullSpaceAssembleCorrection(PC,PetscBool,IS);
PetscErrorCode PCBDDCNullSpaceAdaptGlobal(PC);
PetscErrorCode PCBDDCNullSpaceAssembleCoarse(PC,Mat,MatNullSpace*);

/* utils */
PetscErrorCode PCBDDCComputeLocalMatrix(PC,Mat);
PetscErrorCode PCBDDCSetUpLocalWorkVectors(PC);
PetscErrorCode PCBDDCSetUpSolvers(PC);
PetscErrorCode PCBDDCSetUpLocalScatters(PC);
PetscErrorCode PCBDDCSetUpLocalSolvers(PC,PetscBool,PetscBool);
PetscErrorCode PCBDDCSetUpCorrection(PC,PetscScalar**);
PetscErrorCode PCBDDCSetUpCoarseSolver(PC,PetscScalar*);
PetscErrorCode PCBDDCSubsetNumbering(IS,IS,PetscInt*,IS*);
PetscErrorCode PCBDDCComputePrimalNumbering(PC,PetscInt*,PetscInt**);
PetscErrorCode PCBDDCScatterCoarseDataBegin(PC,InsertMode,ScatterMode);
PetscErrorCode PCBDDCScatterCoarseDataEnd(PC,InsertMode,ScatterMode);
PetscErrorCode PCBDDCApplyInterfacePreconditioner(PC,PetscBool);
PetscErrorCode PCBDDCOrthonormalizeVecs(PetscInt,Vec[]);
PetscErrorCode PCBDDCSetUseExactDirichlet(PC,PetscBool);
PetscErrorCode PCBDDCSetLevel(PC,PetscInt);
PetscErrorCode PCBDDCGlobalToLocal(VecScatter,Vec,Vec,IS,IS*);
PetscErrorCode PCBDDCAdaptiveSelection(PC);
PetscErrorCode MatGetSubMatrixUnsorted(Mat,IS,IS,Mat*);

/* feti-dp */
PetscErrorCode PCBDDCCreateFETIDPMatContext(PC,FETIDPMat_ctx*);
PetscErrorCode PCBDDCSetupFETIDPMatContext(FETIDPMat_ctx);
PetscErrorCode PCBDDCCreateFETIDPPCContext(PC,FETIDPPC_ctx*);
PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat,FETIDPPC_ctx);
PetscErrorCode FETIDPPCApply(PC,Vec,Vec);
PetscErrorCode PCBDDCDestroyFETIDPPC(PC);
PetscErrorCode FETIDPMatMult(Mat,Vec,Vec);
PetscErrorCode PCBDDCDestroyFETIDPMat(Mat);

/* interface to SubSchurs */
PetscErrorCode PCBDDCInitSubSchurs(PC);
PetscErrorCode PCBDDCSetUpSubSchurs(PC);

/* sub schurs */
PetscErrorCode PCBDDCSubSchursCreate(PCBDDCSubSchurs*);
PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs,IS,IS,PCBDDCGraph,ISLocalToGlobalMapping);
PetscErrorCode PCBDDCSubSchursDestroy(PCBDDCSubSchurs*);
PetscErrorCode PCBDDCSubSchursReset(PCBDDCSubSchurs);
PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs,Mat,Mat,PetscInt[],PetscInt[],PetscInt,PetscBool,PetscBool,PetscBool);

#endif

