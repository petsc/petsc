/* $Id: petscviewer.h,v 1.77 2000/07/27 15:46:05 bsmith Exp bsmith $ */
/*
     PetscViewers are objects where other objects can be looked at or stored.
*/

#if !defined(__PETSCVIEWER_H)
#define __PETSCVIEWER_H

typedef struct _p_PetscViewer*            PetscViewer;

/*
    petsc.h must be included AFTER the definition of PetscViewer for ADIC to 
   process correctly.
*/
#include "petsc.h"

#define PETSC_VIEWER_COOKIE              PETSC_COOKIE+1
typedef char*PetscViewerType;

#define PETSC_VIEWER_SOCKET       "socket"
#define PETSC_VIEWER_ASCII        "ascii"
#define PETSC_BINARY_VIEWER       "binary"
#define PETSC_VIEWER_STRING       "string"
#define PETSC_DRAW_VIEWER         "draw"
#define PETSC_VIEWER_AMS          "ams"

extern PetscFList PetscViewerList;
EXTERN int PetscViewerRegisterAll(char *);
EXTERN int PetscViewerRegisterDestroy(void);

EXTERN int PetscViewerRegister(char*,char*,char*,int(*)(PetscViewer));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscViewerRegisterDynamic(a,b,c,d) PetscViewerRegister(a,b,c,0)
#else
#define PetscViewerRegisterDynamic(a,b,c,d) PetscViewerRegister(a,b,c,d)
#endif
EXTERN int PetscViewerCreate(MPI_Comm,PetscViewer*);
EXTERN int PetscViewerSetFromOptions(PetscViewer);


EXTERN int PetscViewerASCIIOpen(MPI_Comm,const char[],PetscViewer*);
typedef enum {PETSC_BINARY_RDONLY,PETSC_BINARY_WRONLY,PETSC_BINARY_CREATE} PetscViewerBinaryType;
EXTERN int PetscViewerBinaryOpen(MPI_Comm,const char[],PetscViewerBinaryType,PetscViewer*);
EXTERN int PetscViewerSocketOpen(MPI_Comm,const char[],int,PetscViewer*);
EXTERN int PetscViewerStringOpen(MPI_Comm,char[],int,PetscViewer*);
EXTERN int PetscViewerDrawOpen(MPI_Comm,const char[],const char[],int,int,int,int,PetscViewer*);
EXTERN int PetscViewerAMSSetCommName(PetscViewer,const char[]);

EXTERN int PetscViewerGetOutputname(PetscViewer,char**);  
EXTERN int PetscViewerGetType(PetscViewer,PetscViewerType*);
EXTERN int PetscViewerSetType(PetscViewer,PetscViewerType);
EXTERN int PetscViewerDestroy(PetscViewer);
EXTERN int PetscViewerGetSingleton(PetscViewer,PetscViewer*);
EXTERN int PetscViewerRestoreSingleton(PetscViewer,PetscViewer*);

#define PETSC_VIEWER_FORMAT_ASCII_DEFAULT       0
#define PETSC_VIEWER_FORMAT_ASCII_MATLAB        1
#define PETSC_VIEWER_FORMAT_ASCII_IMPL          2
#define PETSC_VIEWER_FORMAT_ASCII_INFO          3
#define PETSC_VIEWER_FORMAT_ASCII_INFO_LONG     4
#define PETSC_VIEWER_FORMAT_ASCII_COMMON        5
#define PETSC_VIEWER_FORMAT_ASCII_SYMMODU       6
#define PETSC_VIEWER_FORMAT_ASCII_INDEX         7
#define PETSC_VIEWER_FORMAT_ASCII_DENSE         8

#define PETSC_VIEWER_FORMAT_BINARY_DEFAULT      9
#define PETSC_VIEWER_FORMAT_BINARY_NATIVE       10

#define PETSC_VIEWER_FORMAT_DRAW_BASIC          11
#define PETSC_VIEWER_FORMAT_DRAW_LG             12
#define PETSC_VIEWER_FORMAT_DRAW_CONTOUR        13
#define PETSC_VIEWER_FORMAT_DRAW_PORTS          15

#define PETSC_VIEWER_FORMAT_NATIVE              14

EXTERN int    PetscViewerSetFormat(PetscViewer,int,char[]);
EXTERN int    PetscViewerPushFormat(PetscViewer,int,char[]);
EXTERN int    PetscViewerPopFormat(PetscViewer);
EXTERN int    PetscViewerGetFormat(PetscViewer,int*);
EXTERN int    PetscViewerFlush(PetscViewer);

/*
   Operations explicit to a particular class of viewers
*/
EXTERN int PetscViewerASCIIGetPointer(PetscViewer,FILE**);
EXTERN int PetscViewerASCIIPrintf(PetscViewer,const char[],...);
EXTERN int PetscViewerASCIISynchronizedPrintf(PetscViewer,const char[],...);
EXTERN int PetscViewerASCIIPushTab(PetscViewer);
EXTERN int PetscViewerASCIIPopTab(PetscViewer);
EXTERN int PetscViewerASCIIUseTabs(PetscViewer,PetscTruth);
EXTERN int PetscViewerASCIISetTab(PetscViewer,int);
EXTERN int PetscViewerBinaryGetDescriptor(PetscViewer,int*);
EXTERN int PetscViewerBinaryGetInfoPointer(PetscViewer,FILE **);
EXTERN int PetscViewerBinarySetType(PetscViewer,PetscViewerBinaryType);
EXTERN int PetscViewerStringSPrintf(PetscViewer,char *,...);
EXTERN int PetscViewerStringSetString(PetscViewer,char[],int);
EXTERN int PetscViewerDrawClear(PetscViewer);
EXTERN int PetscViewerDrawSetInfo(PetscViewer,const char[],const char[],int,int,int,int);
EXTERN int PetscViewerSocketSetConnection(PetscViewer,const char[],int);

EXTERN int PetscViewerSetFilename(PetscViewer,const char[]);
EXTERN int PetscViewerGetFilename(PetscViewer,char**);

/*
     These are all the default viewers that do not have 
   to be explicitly opened
*/
EXTERN PetscViewer PETSC_VIEWER_STDOUT_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_STDERR_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_DRAW_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_SOCKET_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_BINARY_(MPI_Comm);

#define PETSC_VIEWER_STDOUT_SELF  PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)
#define PETSC_VIEWER_STDOUT_WORLD PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD)
#define PETSC_VIEWER_STDERR_SELF  PETSC_VIEWER_STDERR_(PETSC_COMM_SELF)
#define PETSC_VIEWER_STDERR_WORLD PETSC_VIEWER_STDERR_(PETSC_COMM_WORLD)
#define PETSC_VIEWER_DRAW_SELF    PETSC_VIEWER_DRAW_(PETSC_COMM_SELF)
#define PETSC_VIEWER_DRAW_WORLD   PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD)
#define PETSC_VIEWER_SOCKET_WORLD PETSC_VIEWER_SOCKET_(PETSC_COMM_WORLD)
#define PETSC_VIEWER_SOCKET_SELF  PETSC_VIEWER_SOCKET_(PETSC_COMM_SELF)
#define PETSC_VIEWER_BINARY_WORLD PETSC_VIEWER_BINARY_(PETSC_COMM_WORLD)
#define PETSC_VIEWER_BINARY_SELF  PETSC_VIEWER_BINARY_(PETSC_COMM_SELF)

/*
    PetscViewer based on the ALICE Memory Snooper
*/
#if defined(PETSC_HAVE_AMS)
#include "ams.h"
EXTERN int         PetscViewerAMSGetAMSComm(PetscViewer,AMS_Comm *);
EXTERN int         PetscViewerAMSOpen(MPI_Comm,const char[],PetscViewer*);
EXTERN PetscViewer PETSC_VIEWER_AMS_(MPI_Comm);
EXTERN int         PETSC_VIEWER_AMS_Destroy(MPI_Comm);
#define PETSC_VIEWER_AMS_WORLD PETSC_VIEWER_AMS_(PETSC_COMM_WORLD)
#endif

/* 
    PetscViewer utility routines used by PETSc that are not normally used
   by users.
*/
EXTERN int  PetscViewerSocketPutScalar(PetscViewer,int,int,Scalar*);
EXTERN int  PetscViewerSocketPutReal(PetscViewer,int,int,double*);
EXTERN int  PetscViewerSocketPutInt(PetscViewer,int,int*);
EXTERN int  PetscViewerSocketPutSparse_Private(PetscViewer,int,int,int,Scalar*,int*,int *);
EXTERN int  PetscViewerDestroyAMS_Private(void);

/*
    Manages sets of viewers
*/
typedef struct _p_PetscViewers* PetscViewers;
EXTERN int PetscViewersCreate(MPI_Comm,PetscViewers*);
EXTERN int PetscViewersDestroy(PetscViewers);
EXTERN int PetscViewersGetViewer(PetscViewers,int,PetscViewer*);

#endif




