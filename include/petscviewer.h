/* $Id: viewer.h,v 1.67 1999/03/19 21:25:01 bsmith Exp bsmith $ */
/*
     Viewers are objects where other objects can be looked at or stored.
*/

#if !defined(__VIEWER_H)
#define __VIEWER_H

typedef struct _p_Viewer*            Viewer;

/*
    petsc.h must be included AFTER the definition of Viewer for ADIC to 
   process correctly.
*/
#include "petsc.h"

#define VIEWER_COOKIE              PETSC_COOKIE+1
typedef char* ViewerType;

#define SOCKET_VIEWER       "socket"
#define ASCII_VIEWER        "ascii"
#define BINARY_VIEWER       "binary"
#define STRING_VIEWER       "string"
#define DRAW_VIEWER         "draw"
#define AMS_VIEWER          "ams"

extern FList ViewerList;
extern int ViewerRegisterAll(char *);
extern int ViewerRegisterDestroy(void);

extern int ViewerRegister_Private(char*,char*,char*,int(*)(Viewer));
#if defined(USE_DYNAMIC_LIBRARIES)
#define ViewerRegister(a,b,c,d) ViewerRegister_Private(a,b,c,0)
#else
#define ViewerRegister(a,b,c,d) ViewerRegister_Private(a,b,c,d)
#endif
extern int ViewerCreate(MPI_Comm,Viewer*);
extern int ViewerSetFromOptions(Viewer);


extern int ViewerASCIIOpen(MPI_Comm,const char[],Viewer*);
typedef enum {BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE} ViewerBinaryType;
extern int ViewerBinaryOpen(MPI_Comm,const char[],ViewerBinaryType,Viewer*);
extern int ViewerSocketOpen(MPI_Comm,const char[],int,Viewer*);
extern int ViewerStringOpen(MPI_Comm,char[],int, Viewer*);
extern int ViewerDrawOpen(MPI_Comm,const char[],const char[],int,int,int,int,Viewer*);
extern int ViewerAMSSetCommName(Viewer,const char[]);

extern int ViewerGetOutputname(Viewer,char**);  
extern int ViewerGetType(Viewer,ViewerType*);
extern int ViewerSetType(Viewer,ViewerType);
extern int ViewerDestroy(Viewer);

#define VIEWER_FORMAT_ASCII_DEFAULT       0
#define VIEWER_FORMAT_ASCII_MATLAB        1
#define VIEWER_FORMAT_ASCII_IMPL          2
#define VIEWER_FORMAT_ASCII_INFO          3
#define VIEWER_FORMAT_ASCII_INFO_LONG     4
#define VIEWER_FORMAT_ASCII_COMMON        5
#define VIEWER_FORMAT_ASCII_SYMMODU       6
#define VIEWER_FORMAT_ASCII_INDEX         7
#define VIEWER_FORMAT_ASCII_DENSE         8

#define VIEWER_FORMAT_BINARY_DEFAULT      9
#define VIEWER_FORMAT_BINARY_NATIVE       10

#define VIEWER_FORMAT_DRAW_BASIC          11
#define VIEWER_FORMAT_DRAW_LG             12
#define VIEWER_FORMAT_DRAW_CONTOUR        13

#define VIEWER_FORMAT_NATIVE              14

extern int    ViewerSetFormat(Viewer,int,char[]);
extern int    ViewerPushFormat(Viewer,int,char[]);
extern int    ViewerPopFormat(Viewer);
extern int    ViewerGetFormat(Viewer,int*);
extern int    ViewerFlush(Viewer);

/*
   Operations explicit to a particular class of viewers
*/
extern int ViewerASCIIGetPointer(Viewer,FILE**);
extern int ViewerASCIIPrintf(Viewer,const char[],...);
extern int ViewerASCIIPushTab(Viewer);
extern int ViewerASCIIPopTab(Viewer);
extern int ViewerBinaryGetDescriptor(Viewer,int*);
extern int ViewerBinaryGetInfoPointer(Viewer,FILE **);
extern int ViewerBinarySetType(Viewer,ViewerBinaryType);
extern int ViewerStringSPrintf(Viewer,char *,...);
extern int ViewerStringSetString(Viewer,char[],int);
extern int ViewerDrawClear(Viewer);
extern int ViewerDrawSetInfo(Viewer,const char[],const char[],int,int,int,int);
extern int ViewerSocketSetConnection(Viewer,const char[],int);

extern int ViewerSetFilename(Viewer,const char[]);

/*
     These are all the default viewers that do not have 
   to be explicitly opened
*/
extern Viewer VIEWER_STDOUT_SELF;  
extern Viewer VIEWER_STDERR_SELF;
extern Viewer VIEWER_STDOUT_WORLD;
extern Viewer VIEWER_STDOUT_(MPI_Comm);
extern int    VIEWER_STDOUT_Destroy(MPI_Comm);
extern Viewer VIEWER_STDERR_WORLD;
extern Viewer VIEWER_STDERR_(MPI_Comm);
extern int    VIEWER_STDERR_Destroy(MPI_Comm);
extern Viewer VIEWER_DRAW_WORLD_PRIVATE_0;
extern Viewer VIEWER_DRAW_WORLD_PRIVATE_1;
extern Viewer VIEWER_DRAW_WORLD_PRIVATE_2;
extern Viewer VIEWER_DRAW_SELF_PRIVATE; 
extern Viewer VIEWER_SOCKET_WORLD_PRIVATE;
extern Viewer VIEWER_SOCKET_SELF_PRIVATE;  /* not yet used */

extern int    ViewerInitializeDrawWorld_Private_0(void);
extern int    ViewerInitializeDrawWorld_Private_1(void);
extern int    ViewerInitializeDrawWorld_Private_2(void);
extern int    ViewerInitializeDrawSelf_Private(void);
extern int    ViewerInitializeSocketWorld_Private(void);
extern Viewer VIEWER_DRAW_(MPI_Comm);
extern int    VIEWER_DRAW_Destroy(MPI_Comm);
extern Viewer VIEWER_SOCKET_(MPI_Comm);
extern int    VIEWER_SOCKET_Destroy(MPI_Comm);

#define VIEWER_DRAW_WORLD_0 \
              (ViewerInitializeDrawWorld_Private_0(),VIEWER_DRAW_WORLD_PRIVATE_0) 
#define VIEWER_DRAW_WORLD_1 \
              (ViewerInitializeDrawWorld_Private_1(),VIEWER_DRAW_WORLD_PRIVATE_1) 
#define VIEWER_DRAW_WORLD_2 \
              (ViewerInitializeDrawWorld_Private_2(),VIEWER_DRAW_WORLD_PRIVATE_2) 

#define VIEWER_DRAW_SELF \
              (ViewerInitializeDrawSelf_Private(),VIEWER_DRAW_SELF_PRIVATE) 
#define VIEWER_DRAW_WORLD VIEWER_DRAW_WORLD_0

#define VIEWER_SOCKET_WORLD \
        (ViewerInitializeSocketWorld_Private(),VIEWER_SOCKET_WORLD_PRIVATE) 

/*
    Viewer based on the ALICE Memory Snooper
*/
#if defined(HAVE_AMS)
#include "ams.h"
extern int    ViewerAMSGetAMSComm(Viewer,AMS_Comm *);
extern int    ViewerAMSOpen(MPI_Comm,const char[],Viewer*);
extern Viewer VIEWER_AMS_(MPI_Comm);
extern int    VIEWER_AMS_Destroy(MPI_Comm);
extern Viewer VIEWER_AMS_WORLD_PRIVATE;
extern int    ViewerInitializeAMSWorld_Private(void);
#define VIEWER_AMS_WORLD (ViewerInitializeAMSWorld_Private(),VIEWER_AMS_WORLD_PRIVATE) 
#endif

/* 
    Viewer utility routines used by PETSc that are not normally used
   by users.
*/
extern int  ViewerSocketPutScalar_Private(Viewer,int,int,Scalar*);
extern int  ViewerSocketPutDouble_Private(Viewer,int,int,double*);
extern int  ViewerSocketPutInt_Private(Viewer,int,int*);
extern int  ViewerSocketPutSparse_Private(Viewer,int,int,int,Scalar*,int*,int *);
extern int  ViewerInitializeASCII_Private(void);
extern int  ViewerDestroyASCII_Private(void);
extern int  ViewerDestroyDraw_Private(void);
extern int  ViewerDestroySocket_Private(void);
extern int  ViewerDestroyAMS_Private(void);

/*
    Manages sets of viewers
*/
typedef struct _p_Viewers* Viewers;
extern int ViewersCreate(MPI_Comm,Viewers*);
extern int ViewersDestroy(Viewers);
extern int ViewersGetViewer(Viewers,int,Viewer*);

#endif

