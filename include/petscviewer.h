/* $Id: viewer.h,v 1.36 1996/09/14 13:00:18 bsmith Exp bsmith $ */

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

#define VIEWER_FORMAT_ASCII_DEFAULT       0
#define VIEWER_FORMAT_ASCII_MATLAB        1
#define VIEWER_FORMAT_ASCII_IMPL          2
#define VIEWER_FORMAT_ASCII_INFO          3
#define VIEWER_FORMAT_ASCII_INFO_LONG 4
#define VIEWER_FORMAT_ASCII_COMMON        5
#define VIEWER_FORMAT_BINARY_DEFAULT      0
#define VIEWER_FORMAT_BINARY_NATIVE       1
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
extern Viewer VIEWER_DRAWX_WORLD_PRIVATE_0;
extern Viewer VIEWER_DRAWX_WORLD_PRIVATE_1;
extern Viewer VIEWER_DRAWX_WORLD_PRIVATE_2;
extern Viewer VIEWER_DRAWX_SELF_PRIVATE; 
extern Viewer VIEWER_MATLAB_WORLD_PRIVATE;
extern Viewer VIEWER_MATLAB_SELF_PRIVATE;  /* not yet used */

extern int    ViewerInitializeDrawXWorld_Private_0();
extern int    ViewerInitializeDrawXWorld_Private_1();
extern int    ViewerInitializeDrawXWorld_Private_2();
extern int    ViewerInitializeDrawXSelf_Private();

#define VIEWER_DRAWX_WORLD_0 \
              (ViewerInitializeDrawXWorld_Private_0(),VIEWER_DRAWX_WORLD_PRIVATE_0) 
#define VIEWER_DRAWX_WORLD_1 \
              (ViewerInitializeDrawXWorld_Private_1(),VIEWER_DRAWX_WORLD_PRIVATE_1) 
#define VIEWER_DRAWX_WORLD_2 \
              (ViewerInitializeDrawXWorld_Private_2(),VIEWER_DRAWX_WORLD_PRIVATE_2) 

#define VIEWER_DRAWX_SELF \
              (ViewerInitializeDrawXSelf_Private(),VIEWER_DRAWX_SELF_PRIVATE) 
#define VIEWER_DRAWX_WORLD VIEWER_DRAWX_WORLD_0
#define VIEWER_MATLAB_WORLD \
        (ViewerInitializeMatlabWorld_Private(),VIEWER_MATLAB_WORLD_PRIVATE) 

#endif
