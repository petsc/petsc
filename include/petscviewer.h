
#if !defined(__VIEWER_PACKAGE)
#define __VIEWER_PACKAGE

#include "petsc.h"

typedef struct _Viewer*      Viewer;
#define VIEWER_COOKIE        PETSC_COOKIE+1
#define MATLAB_VIEWER        0
#define FILE_VIEWER          1
#define FILES_VIEWER         2

#define FILE_FORMAT_DEFAULT  0
#define FILE_FORMAT_MATLAB   1
#define FILE_FORMAT_IMPL     2

extern int ViewerFileOpen(char*,Viewer *);
extern int ViewerFileOpenSync(char*,MPI_Comm,Viewer *);
extern int ViewerFileSetFormat(Viewer,int,char *);
extern int ViewerMatlabOpen(char*,int,Viewer *);

extern Viewer STDOUT_VIEWER;  
extern Viewer STDERR_VIEWER;
extern Viewer SYNC_STDOUT_VIEWER;

extern int PetscView(PetscObject,Viewer);

#endif
