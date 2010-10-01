
/*
    This is used by bin/matlab/classes/PetscInitialize() to define to Matlab all the functions available in the 
   PETSc shared library. We cannot simply use the regular PETSc include files because they are too complicated for 
   Matlab to parse.

*/

/* Matlab cannot handle char ***, so lie to it about the argument */
int PetscInitialize(int*,int *,const char*,const char*);
int PetscFinalize(void);

typedef int MPI_Comm;
typedef int InsertMode;

typedef int PetscViewer;
int PetscViewerCreate(MPI_Comm,PetscViewer*);
int PetscViewerSetType(PetscViewer,const char*);
int PetscViewerView(PetscViewer,PetscViewer);
int PetscViewerDestroy(PetscViewer);

typedef int Vec;
int VecCreate(MPI_Comm,Vec *);
int VecSetType(Vec,const char*);
int VecSetSizes(Vec,int,int);
int VecSetValues(Vec,int,int*,double*,InsertMode);
int VecGetValues(Vec,int,int*,double*);
int VecAssemblyBegin(Vec);
int VecAssemblyEnd(Vec);
int VecView(Vec,PetscViewer);
int VecDestroy(Vec);

