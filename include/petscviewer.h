
#if !defined(__VIEWER_PACKAGE)
#define __VIEWER_PACKAGE

#include "petsc.h"

typedef struct _Viewer*      Viewer;
#define VIEWER_COOKIE        PETSC_COOKIE+1
#define MATLAB_VIEWER        0
#define FILE_VIEWER          1
#define FILES_VIEWER         2

extern int ViewerFileOpen(char*,Viewer *);
extern int ViewerSyncFileOpen(char*,MPI_Comm,Viewer *);
extern int ViewerMatlabOpen(char*,int,Viewer *);
extern FILE *ViewerFileGetPointer(Viewer);

extern Viewer STDOUT_VIEWER;  
extern Viewer STDERR_VIEWER;
extern Viewer SYNC_STDOUT_VIEWER;

extern int ViewerMatlabPutArray(Viewer,int,int,Scalar*);
extern int ViewMatlabPutSparse(Viewer,int,int,int,Scalar*,int*,int *);

extern int PetscView(PetscObject,Viewer);
extern int ViewerInitialize();

#endif
