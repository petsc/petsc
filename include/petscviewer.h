/* $Id: viewer.h,v 1.28 1996/04/12 00:07:43 curfman Exp bsmith $ */

#if !defined(__VIEWER_PACKAGE)
#define __VIEWER_PACKAGE

#include "petsc.h"

typedef struct _Viewer*            Viewer;
#define VIEWER_COOKIE              PETSC_COOKIE+1
typedef enum { MATLAB_VIEWER,ASCII_FILE_VIEWER, ASCII_FILES_VIEWER, 
               BINARY_FILE_VIEWER, STRING_VIEWER, DRAW_VIEWER} ViewerType;

extern int ViewerFileOpenASCII(MPI_Comm,char*,Viewer *);
typedef enum { BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE} ViewerBinaryType;
extern int ViewerFileOpenBinary(MPI_Comm,char*,ViewerBinaryType,Viewer *);
extern int ViewerMatlabOpen(MPI_Comm,char*,int,Viewer *);
extern int ViewerStringOpen(MPI_Comm,char *,int, Viewer *);
extern int ViewerDrawOpenX(MPI_Comm,char *,char *,int,int,int,int,Viewer*);

extern int ViewerGetType(Viewer,ViewerType*);
extern int ViewerDestroy(Viewer);

extern int ViewerASCIIGetPointer(Viewer,FILE**);
extern int ViewerBinaryGetDescriptor(Viewer,int*);
extern int ViewerBinaryGetInfoPointer(Viewer,FILE **);

#define ASCII_FORMAT_DEFAULT       0
#define ASCII_FORMAT_MATLAB        1
#define ASCII_FORMAT_IMPL          2
#define ASCII_FORMAT_INFO          3
#define ASCII_FORMAT_INFO_DETAILED 4
#define ASCII_FORMAT_COMMON        5
#define BINARY_FORMAT_DEFAULT      0
#define BINARY_FORMAT_NATIVE       1
extern int ViewerSetFormat(Viewer,int,char *);
extern int ViewerGetFormat(Viewer,int*);

extern int ViewerFlush(Viewer);
extern int ViewerStringSPrintf(Viewer,char *,...);

extern Viewer STDOUT_VIEWER_SELF;  
extern Viewer STDERR_VIEWER_SELF;
extern Viewer STDOUT_VIEWER_WORLD;

#endif
