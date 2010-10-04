
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
int VecSetSizes(Vec,int,int);
int VecSetValues(Vec,int,int*,double*,InsertMode);
int VecGetValues(Vec,int,int*,double*);
int VecAssemblyBegin(Vec);
int VecAssemblyEnd(Vec);
int VecDuplicate(Vec,Vec*);
int VecView(Vec,PetscViewer);
int VecDestroy(Vec);

typedef PetscPointer Mat;
typedef int MatAssemblyType;
typedef int MatStructure;
int MatCreate(MPI_Comm,Mat *);
int MatSetType(Mat,const char*);
int MatSetSizes(Mat,int,int,int,int);
int MatSetValues(Vec,int,int*,int,int*,double*,InsertMode);
int MatAssemblyBegin(Mat,MatAssemblyType);
int MatAssemblyEnd(Mat,MatAssemblyType);
int MatView(Mat,PetscViewer);
int MatDestroy(Mat);

typedef PetscPointer KSP;
int KSPCreate(MPI_Comm,KSP *);
int KSPSetType(KSP,const char*);
int KSPSetOperators(KSP,Mat,Mat,MatStructure);
int KSPSolve(KSP,Vec,Vec);
int KSPSetUp(KSP);
int KSPView(KSP,PetscViewer);
int KSPDestroy(KSP);

typedef PetscPointer SNES;
int SNESCreate(MPI_Comm,SNES *);
int SNESSetType(SNES,const char*);
int SNESSetFunction(SNES,Vec,Vec);
int SNESSolve(SNES,Vec,Vec);
int SNESSetUp(SNES);
int SNESView(SNES,PetscViewer);
int SNESDestroy(SNES);
