
/*
    This is used by bin/matlab/classes/PetscInitialize() to define to Matlab all the functions available in the 
   PETSc shared library. We cannot simply use the regular PETSc include files because they are too complicated for 
   Matlab to parse.

*/

/* Matlab cannot handle char ***, so lie to it about the argument */
int PetscInitializeNonPointers(int,char **,const char*,const char*);
int PetscFinalize(void);

typedef int MPI_Comm;
typedef int InsertMode;

typedef int PetscPointer;

typedef PetscPointer PetscViewer;
int PetscViewerCreate(MPI_Comm,PetscViewer*);
int PetscViewerSetType(PetscViewer,const char*);
int PetscViewerView(PetscViewer,PetscViewer);
int PetscViewerDestroy(PetscViewer);

typedef PetscPointer IS;
int ISCreate(MPI_Comm,IS *);
int ISSetType(IS,const char*);
int ISGeneralSetIndices(IS,int,const int*);
int ISView(IS,PetscViewer);
int ISDestroy(IS);

typedef PetscPointer Vec;
int VecCreate(MPI_Comm,Vec *);
int VecSetType(Vec,const char*);
int VecSetFromOptions(Vec);
int VecSetSizes(Vec,int,int);
int VecGetLocalSize(Vec,int*);
int VecSetValues(Vec,int,int*,double*,InsertMode);
int VecGetValues(Vec,int,int*,double*);
int VecAssemblyBegin(Vec);
int VecAssemblyEnd(Vec);
int VecCopy(Vec,Vec);
int VecDuplicate(Vec,Vec*);
int VecView(Vec,PetscViewer);
int VecDestroy(Vec);

typedef PetscPointer Mat;
typedef int MatAssemblyType;
typedef int MatStructure;
int MatCreate(MPI_Comm,Mat *);
int MatSetType(Mat,const char*);
int MatSetFromOptions(Mat);
int MatSetSizes(Mat,int,int,int,int);
int MatSetValues(Vec,int,int*,int,int*,double*,InsertMode);
int MatAssemblyBegin(Mat,MatAssemblyType);
int MatAssemblyEnd(Mat,MatAssemblyType);
int MatView(Mat,PetscViewer);
int MatDestroy(Mat);

typedef PetscPointer DA;
typedef int DAPeriodicType;
typedef int DAStencilType;
int DACreate(MPI_Comm,DA*);
int DASetType(DA,const char*);
int DASetDim(DA,int);
int DASetSizes(DA,int,int,int);
int DASetVecType(DA,const char*);
int DASetFromOptions(DA);
int DADestroy(DA);
int DAView(DA,PetscViewer);
int DASetPeriodicity(DA, DAPeriodicType);
int DASetDof(DA, int);
int DASetStencilWidth(DA, int);
int DASetStencilType(DA, DAStencilType);

typedef PetscPointer KSP;
int KSPCreate(MPI_Comm,KSP *);
int KSPSetType(KSP,const char*);
int KSPSetDM(KSP,DA);
int KSPSetFromOptions(KSP);
int KSPSetOperators(KSP,Mat,Mat,MatStructure);
int KSPSolve(KSP,Vec,Vec);
int KSPSetUp(KSP);
int KSPView(KSP,PetscViewer);
int KSPDestroy(KSP);

typedef PetscPointer SNES;
int SNESCreate(MPI_Comm,SNES *);
int SNESSetType(SNES,const char*);
int SNESSetDM(SNES,DA);
int SNESSetFromOptions(SNES);
int SNESSetFunctionMatlab(SNES,Vec,const char*,mxArray*);
int SNESSetJacobianMatlab(SNES,Mat,Mat,const char*,mxArray*);
int SNESSolve(SNES,Vec,Vec);
int SNESSetUp(SNES);
int SNESView(SNES,PetscViewer);
int SNESDestroy(SNES);

typedef PetscPointer TS;
int TSCreate(MPI_Comm,TS *);
int TSSetType(TS,const char*);
int TSSetDM(TS,DA);
int TSSetFromOptions(TS);
int TSSetIFunctionMatlab(TS,const char*,mxArray*);
int TSSetIJacobianMatlab(TS,Mat,Mat,const char*,mxArray*);
int TSSolve(TS,Vec);
int TSSetUp(TS);
int TSView(TS,PetscViewer);
int TSDestroy(TS);

