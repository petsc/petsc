/* $Id: viewer.h,v 1.31 1996/07/08 22:24:30 bsmith Exp bsmith $ */

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
extern int ViewerDrawOpenVRML(MPI_Comm,char *,char *,Viewer*);

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
#define VIEWER_FORMAT_DRAW_BASIC   0
#define VIEWER_FORMAT_DRAW_LG      1

extern int    ViewerSetFormat(Viewer,int,char *);
extern int    ViewerPushFormat(Viewer,int,char *);
extern int    ViewerPopFormat(Viewer);
extern int    ViewerGetFormat(Viewer,int*);

extern int    ViewerFlush(Viewer);
extern int    ViewerStringSPrintf(Viewer,char *,...);

extern Viewer VIEWER_STDOUT_SELF;  
extern Viewer VIEWER_STDERR_SELF;
extern Viewer VIEWER_STDOUT_WORLD;
extern Viewer VIEWER_DRAWX_WORLD_PRIVATE;
extern Viewer VIEWER_DRAWX_SELF_PRIVATE; 

extern int    ViewerInitializeDrawXWorld_Private();
extern int    ViewerInitializeDrawXSelf_Private();
extern int    ViewerDestroyDrawX_Private();

#define VIEWER_DRAWX_WORLD \
              (ViewerInitializeDrawXWorld_Private(),VIEWER_DRAWX_WORLD_PRIVATE) 
#define VIEWER_DRAWX_SELF \
              (ViewerInitializeDrawXSelf_Private(),VIEWER_DRAWX_SELF_PRIVATE) 

#endif
