/* $Id: viewer.h,v 1.14 1995/09/06 03:06:53 bsmith Exp bsmith $ */

#if !defined(__VIEWER_PACKAGE)
#define __VIEWER_PACKAGE

#include "petsc.h"

typedef struct _Viewer*            Viewer;
#define VIEWER_COOKIE              PETSC_COOKIE+1
#define MATLAB_VIEWER              0
#define ASCII_FILE_VIEWER          1
#define ASCII_FILES_VIEWER         2
#define BINARY_FILE_VIEWER         3

#define FILE_FORMAT_DEFAULT  0
#define FILE_FORMAT_MATLAB   1
#define FILE_FORMAT_IMPL     2
#define FILE_FORMAT_INFO     3

/* file types */
typedef enum { BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE} ViewerBinaryType;

extern int ViewerFileOpen(MPI_Comm,char*,Viewer *);
extern int ViewerFileOpenBinary(MPI_Comm,char*,ViewerBinaryType,Viewer *);
extern int ViewerFileSetFormat(Viewer,int,char *);
extern int ViewerMatlabOpen(char*,int,Viewer *);
extern int ViewerDestroy(Viewer);

extern Viewer STDOUT_VIEWER_SELF;  
extern Viewer STDERR_VIEWER_SELF;
extern Viewer STDOUT_VIEWER_WORLD;

#endif
