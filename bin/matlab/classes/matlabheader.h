
/*
    This is used by bin/matlab/classes/PetscInitialize() to define to Matlab all the functions available in the 
   PETSc shared library. We cannot simply use the regular PETSc include files because they are too complicated for 
   Matlab to parse.

*/
typedef struct mxArray_tag mxArray;

int PetscInitializeNoPointers(int,char **,const char*,const char*);
int PetscInitialized(int*);
typedef int MPI_Comm;
int PetscGetPETSC_COMM_SELF(MPI_Comm*);
int PetscFinalize(void);

typedef int PetscLogStage;

int PetscLogStageRegister(const char*,PetscLogStage*);
int PetscLogStagePush(PetscLogStage);
int PetscLogStagePop(void);

typedef int PetscBool;
typedef int InsertMode;
typedef int CopyMode;
typedef long int PetscPointer;

typedef PetscPointer PetscViewer;
typedef int PetscFileMode;
int PetscViewerCreate(MPI_Comm,PetscViewer*);
int PetscViewerSetType(PetscViewer,const char*);
int PetscViewerASCIIOpen(MPI_Comm,const char*,PetscViewer*);
int PetscViewerFileSetMode(PetscViewer,PetscFileMode);
int PetscViewerFileSetName(PetscViewer,const char*);
int PetscViewerView(PetscViewer,PetscViewer);
int PetscViewerBinaryOpen(MPI_Comm,const char*,PetscFileMode,PetscViewer*);
int PetscViewerDestroy(PetscViewer*);

int PetscOptionsView(PetscViewer);
int PetscOptionsSetValue(const char*,const char*);
int PetscOptionsClearValue(const char*);
int PetscOptionsInsertString(const char*);
char* PetscOptionsGetStringMatlab(const char*,const char*);

typedef PetscPointer PetscObject;
int PetscObjectsView(PetscViewer);
char* PetscObjectsGetObjectMatlab(const char*,PetscObject*);

typedef PetscPointer IS;
int ISCreate(MPI_Comm,IS *);
int ISSetType(IS,const char*);
int ISGeneralSetIndices(IS,int,const int[],CopyMode);
int ISGetSize(IS,int*);
int ISGetIndicesCopy(IS,int*);
int ISView(IS,PetscViewer);
int ISDestroy(IS*);

typedef PetscPointer Vec;
int VecCreate(MPI_Comm,Vec *);
int VecSetType(Vec,const char*);
int VecSetFromOptions(Vec);
int VecSetSizes(Vec,int,int);
int VecSet(Vec,double);
int VecGetLocalSize(Vec,int*);
int VecSetValues(Vec,int,int*,double*,InsertMode);
int VecGetValues(Vec,int,int*,double*);
int VecAssemblyBegin(Vec);
int VecAssemblyEnd(Vec);
int VecCopy(Vec,Vec);
int VecDuplicate(Vec,Vec*);
int VecView(Vec,PetscViewer);
int VecDestroy(Vec*);

typedef PetscPointer Mat;
typedef int MatAssemblyType;
typedef int MatStructure;
typedef struct {
    int k,j,i,c;
} MatStencil;

int MatCreate(MPI_Comm,Mat *);
int MatSetType(Mat,const char*);
int MatSetFromOptions(Mat);
int MatSetUp(Mat);
int MatSetSizes(Mat,int,int,int,int);
int MatGetLocalSize(Mat,int*,int*);
int MatSetValues(Mat,int,int*,int,int*,double*,InsertMode);
int MatAssemblyBegin(Mat,MatAssemblyType);
int MatAssemblyEnd(Mat,MatAssemblyType);
int MatView(Mat,PetscViewer);
int MatLoad(Mat,PetscViewer);
int MatDestroy(Mat*);
int MatSetValuesStencil(Mat,int,MatStencil*,int,MatStencil*,double*,InsertMode);
int MatSeqAIJFromMatlab(mxArray*,Mat);
mxArray* MatSeqAIJToMatlab(Mat);

typedef PetscPointer DM;
typedef int DMDABoundaryType;
typedef int DMDAStencilType;
int DMCreate(MPI_Comm,DM*);
int DMSetType(DM,const char*);
int DMDASetDim(DM,int);
int DMDASetSizes(DM,int,int,int);
int DMSetVecType(DM,const char*);
int DMSetFromOptions(DM);
int DMDestroy(DM*);
int DMView(DM,PetscViewer);
int DMSetFunctionMatlab(DM,const char*);
int DMSetJacobianMatlab(DM,const char*);
int DMDASetBoundaryType(DM, DMDABoundaryType,DMDABoundaryType,DMDABoundaryType);
int DMDASetDof(DM, int);
int DMSetUp(DM);
int DMDASetStencilWidth(DM, int);
int DMDASetStencilType(DM, DMDAStencilType);
int DMCreateGlobalVector(DM,Vec*);
int DMCreateMatrix(DM,const char*,Mat*);
int DMDAGetInfo(DM,int*,int*,int*,int*,int*,int*,int*,int*,int*,DMDABoundaryType*,DMDAStencilType*);

typedef PetscPointer PC;
int PCCreate(MPI_Comm,PC *);
int PCSetType(PC,const char*);
int PCSetDM(PC,DM);
int PCSetFromOptions(PC);
int PCSetOperators(PC,Mat,Mat,MatStructure);
int PCSetUp(PC);
int PCView(PC,PetscViewer);
int PCFieldSplitSetIS(PC,const char*,IS);
int PCDestroy(PC*);

typedef PetscPointer KSP;
int KSPCreate(MPI_Comm,KSP *);
int KSPSetType(KSP,const char*);
int KSPSetDM(KSP,DM);
int KSPGetPC(KSP,PC*);
int KSPSetFromOptions(KSP);
int KSPSetOperators(KSP,Mat,Mat,MatStructure);
int KSPSolve(KSP,Vec,Vec);
int KSPSetUp(KSP);
int KSPGetSolution(KSP,Vec*);
int KSPView(KSP,PetscViewer);
int KSPDestroy(KSP*);

typedef PetscPointer SNES;
int SNESCreate(MPI_Comm,SNES *);
int SNESSetType(SNES,const char*);
int SNESSetDM(SNES,DM);
int SNESSetFromOptions(SNES);
int SNESSetFunctionMatlab(SNES,Vec,const char*,mxArray*);
int SNESSetJacobianMatlab(SNES,Mat,Mat,const char*,mxArray*);
int SNESSolve(SNES,Vec,Vec);
int SNESSetUp(SNES);
int SNESVISetVariableBounds(SNES,Vec,Vec);
int SNESVIGetActiveSetIS(SNES,Vec,Vec,IS*);
int SNESView(SNES,PetscViewer);
int SNESDestroy(SNES*);
int SNESMonitorSetMatlab(SNES,const char*,mxArray*);
int SNESSetConvergenceHistory(SNES,double[],int[],int,PetscBool);
mxArray* SNESGetConvergenceHistoryMatlab(SNES);
int SNESGetKSP(SNES,KSP*);
int SNESVISetRedundancyCheckMatlab(SNES,const char*,mxArray*);

typedef PetscPointer TS;
int TSCreate(MPI_Comm,TS *);
int TSSetType(TS,const char*);
int TSSetProblemType(TS,int);
int TSSetDM(TS,DM);
int TSSetFromOptions(TS);
int TSSolve(TS,Vec,double*);
int TSSetUp(TS);
int TSView(TS,PetscViewer);
int TSDestroy(TS*);
int TSSetFunctionMatlab(TS,const char*,mxArray*);
int TSSetJacobianMatlab(TS,Mat,Mat,const char*,mxArray*);
int TSMonitorSetMatlab(TS,const char*,mxArray*);
