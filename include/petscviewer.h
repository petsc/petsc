/* $Id: viewer.h,v 1.18 1995/11/09 22:33:28 bsmith Exp bsmith $ */

#if !defined(__VIEWER_PACKAGE)
#define __VIEWER_PACKAGE

#include "petsc.h"

typedef struct _Viewer*            Viewer;
#define VIEWER_COOKIE              PETSC_COOKIE+1
#define MATLAB_VIEWER              0
#define ASCII_FILE_VIEWER          1
#define ASCII_FILES_VIEWER         2
#define BINARY_FILE_VIEWER         3

#define FILE_FORMAT_DEFAULT       0
#define FILE_FORMAT_MATLAB        1
#define FILE_FORMAT_IMPL          2
#define FILE_FORMAT_INFO          3
#define FILE_FORMAT_INFO_DETAILED 4

extern int ViewerFileOpenASCII(MPI_Comm,char*,Viewer *);
typedef enum { BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE} ViewerBinaryType;
extern int ViewerFileOpenBinary(MPI_Comm,char*,ViewerBinaryType,Viewer *);
extern int ViewerMatlabOpen(MPI_Comm,char*,int,Viewer *);

extern int ViewerDestroy(Viewer);

extern int ViewerFileSetFormat(Viewer,int,char *);

extern Viewer STDOUT_VIEWER_SELF;  
extern Viewer STDERR_VIEWER_SELF;
extern Viewer STDOUT_VIEWER_WORLD;

#endif
