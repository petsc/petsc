/* $Id: petscviewer.h,v 1.75 2000/05/08 15:09:50 balay Exp bsmith $ */
/*
     Viewers are objects where other objects can be looked at or stored.
*/

#if !defined(__PETSCVIEWER_H)
#define __PETSCVIEWER_H

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
EXTERN int ViewerRegisterAll(char *);
EXTERN int ViewerRegisterDestroy(void);

EXTERN int ViewerRegister(char*,char*,char*,int(*)(Viewer));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define ViewerRegisterDynamic(a,b,c,d) ViewerRegister(a,b,c,0)
#else
#define ViewerRegisterDynamic(a,b,c,d) ViewerRegister(a,b,c,d)
#endif
EXTERN int ViewerCreate(MPI_Comm,Viewer*);
EXTERN int ViewerSetFromOptions(Viewer);


EXTERN int ViewerASCIIOpen(MPI_Comm,const char[],Viewer*);
typedef enum {BINARY_RDONLY,BINARY_WRONLY,BINARY_CREATE} ViewerBinaryType;
EXTERN int ViewerBinaryOpen(MPI_Comm,const char[],ViewerBinaryType,Viewer*);
EXTERN int ViewerSocketOpen(MPI_Comm,const char[],int,Viewer*);
EXTERN int ViewerStringOpen(MPI_Comm,char[],int,Viewer*);
EXTERN int ViewerDrawOpen(MPI_Comm,const char[],const char[],int,int,int,int,Viewer*);
EXTERN int ViewerAMSSetCommName(Viewer,const char[]);

EXTERN int ViewerGetOutputname(Viewer,char**);  
EXTERN int ViewerGetType(Viewer,ViewerType*);
EXTERN int ViewerSetType(Viewer,ViewerType);
EXTERN int ViewerDestroy(Viewer);
EXTERN int ViewerGetSingleton(Viewer,Viewer*);
EXTERN int ViewerRestoreSingleton(Viewer,Viewer*);

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
#define VIEWER_FORMAT_DRAW_PORTS          15

#define VIEWER_FORMAT_NATIVE              14

EXTERN int    ViewerSetFormat(Viewer,int,char[]);
EXTERN int    ViewerPushFormat(Viewer,int,char[]);
EXTERN int    ViewerPopFormat(Viewer);
EXTERN int    ViewerGetFormat(Viewer,int*);
EXTERN int    ViewerFlush(Viewer);

/*
   Operations explicit to a particular class of viewers
*/
EXTERN int ViewerASCIIGetPointer(Viewer,FILE**);
EXTERN int ViewerASCIIPrintf(Viewer,const char[],...);
EXTERN int ViewerASCIISynchronizedPrintf(Viewer,const char[],...);
EXTERN int ViewerASCIIPushTab(Viewer);
EXTERN int ViewerASCIIPopTab(Viewer);
EXTERN int ViewerASCIIUseTabs(Viewer,PetscTruth);
EXTERN int ViewerBinaryGetDescriptor(Viewer,int*);
EXTERN int ViewerBinaryGetInfoPointer(Viewer,FILE **);
EXTERN int ViewerBinarySetType(Viewer,ViewerBinaryType);
EXTERN int ViewerStringSPrintf(Viewer,char *,...);
EXTERN int ViewerStringSetString(Viewer,char[],int);
EXTERN int ViewerDrawClear(Viewer);
EXTERN int ViewerDrawSetInfo(Viewer,const char[],const char[],int,int,int,int);
EXTERN int ViewerSocketSetConnection(Viewer,const char[],int);

EXTERN int ViewerSetFilename(Viewer,const char[]);
EXTERN int ViewerGetFilename(Viewer,char**);

/*
     These are all the default viewers that do not have 
   to be explicitly opened
*/
EXTERN Viewer VIEWER_STDOUT_(MPI_Comm);
EXTERN int    VIEWER_STDOUT_Destroy(MPI_Comm);
EXTERN Viewer VIEWER_STDERR_(MPI_Comm);
EXTERN int    VIEWER_STDERR_Destroy(MPI_Comm);
EXTERN Viewer VIEWER_DRAW_(MPI_Comm);
EXTERN int    VIEWER_DRAW_Destroy(MPI_Comm);
EXTERN Viewer VIEWER_SOCKET_(MPI_Comm);
EXTERN int    VIEWER_SOCKET_Destroy(MPI_Comm);

#define VIEWER_STDOUT_SELF  VIEWER_STDOUT_(PETSC_COMM_SELF)
#define VIEWER_STDOUT_WORLD VIEWER_STDOUT_(PETSC_COMM_WORLD)
#define VIEWER_STDERR_SELF  VIEWER_STDERR_(PETSC_COMM_SELF)
#define VIEWER_STDERR_WORLD VIEWER_STDERR_(PETSC_COMM_WORLD)

#define VIEWER_DRAW_SELF    VIEWER_DRAW_(PETSC_COMM_SELF)
#define VIEWER_DRAW_WORLD   VIEWER_DRAW_(PETSC_COMM_WORLD)
#define VIEWER_SOCKET_WORLD VIEWER_SOCKET_(PETSC_COMM_WORLD)
#define VIEWER_SOCKET_SELF  VIEWER_SOCKET_(PETSC_COMM_SELF)
/*
    Viewer based on the ALICE Memory Snooper
*/
#if defined(PETSC_HAVE_AMS)
#include "ams.h"
EXTERN int    ViewerAMSGetAMSComm(Viewer,AMS_Comm *);
EXTERN int    ViewerAMSOpen(MPI_Comm,const char[],Viewer*);
EXTERN Viewer VIEWER_AMS_(MPI_Comm);
EXTERN int    VIEWER_AMS_Destroy(MPI_Comm);
#define VIEWER_AMS_WORLD VIEWER_AMS_(PETSC_COMM_WORLD)
#endif

/* 
    Viewer utility routines used by PETSc that are not normally used
   by users.
*/
EXTERN int  ViewerSocketPutScalar_Private(Viewer,int,int,Scalar*);
EXTERN int  ViewerSocketPutReal_Private(Viewer,int,int,double*);
EXTERN int  ViewerSocketPutInt_Private(Viewer,int,int*);
EXTERN int  ViewerSocketPutSparse_Private(Viewer,int,int,int,Scalar*,int*,int *);

EXTERN int  ViewerDestroyAMS_Private(void);

/*
    Manages sets of viewers
*/
typedef struct _p_Viewers* Viewers;
EXTERN int ViewersCreate(MPI_Comm,Viewers*);
EXTERN int ViewersDestroy(Viewers);
EXTERN int ViewersGetViewer(Viewers,int,Viewer*);

#endif




