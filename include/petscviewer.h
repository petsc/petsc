/* $Id: petscviewer.h,v 1.85 2001/08/06 21:13:28 bsmith Exp $ */
/*
     PetscViewers are objects where other objects can be looked at or stored.
*/

#if !defined(__PETSCVIEWER_H)
#define __PETSCVIEWER_H

extern int PETSC_VIEWER_COOKIE;

/*S
     PetscViewer - Abstract PETSc object that helps view (in ASCII, binary, graphically etc)
         other PETSc objects

   Level: beginner

  Concepts: viewing

.seealso:  PetscViewerCreate(), PetscViewerSetType(), PetscViewerType
S*/
typedef struct _p_PetscViewer* PetscViewer;

/*
    petsc.h must be included AFTER the definition of PetscViewer for ADIC to 
   process correctly.
*/
#include "petsc.h"

/*E
    PetscViewerType - String with the name of a PETSc PETScViewer

   Level: beginner

.seealso: PetscViewerSetType(), PetscViewer
E*/
typedef char* PetscViewerType;
#define PETSC_VIEWER_SOCKET       "socket"
#define PETSC_VIEWER_ASCII        "ascii"
#define PETSC_VIEWER_BINARY       "binary"
#define PETSC_VIEWER_STRING       "string"
#define PETSC_VIEWER_DRAW         "draw"
#define PETSC_VIEWER_AMS          "ams"
#define PETSC_VIEWER_VU           "vu"
#define PETSC_VIEWER_MATHEMATICA  "mathematica"
#define PETSC_VIEWER_SILO         "silo"

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
EXTERN int PetscViewerMathematicaOpen(MPI_Comm, int, const char[], const char[], PetscViewer *);
EXTERN int PetscViewerSiloOpen(MPI_Comm, const char[], PetscViewer *);

EXTERN int PetscViewerGetType(PetscViewer,PetscViewerType*);
EXTERN int PetscViewerSetType(PetscViewer,PetscViewerType);
EXTERN int PetscViewerDestroy(PetscViewer);
EXTERN int PetscViewerGetSingleton(PetscViewer,PetscViewer*);
EXTERN int PetscViewerRestoreSingleton(PetscViewer,PetscViewer*);


/*E
    PetscViewerFormat - Way a viewer presents the object

   Level: beginner

.seealso: PetscViewerSetFormat(), PetscViewer, PetscViewerType, PetscViewerPushFormat(), PetscViewerPopFormat()
E*/
typedef enum { 
  PETSC_VIEWER_ASCII_DEFAULT,
  PETSC_VIEWER_ASCII_MATLAB, 
  PETSC_VIEWER_ASCII_MATHEMATICA,
  PETSC_VIEWER_ASCII_IMPL,
  PETSC_VIEWER_ASCII_INFO,
  PETSC_VIEWER_ASCII_INFO_LONG,
  PETSC_VIEWER_ASCII_COMMON,
  PETSC_VIEWER_ASCII_SYMMODU,
  PETSC_VIEWER_ASCII_INDEX,
  PETSC_VIEWER_ASCII_DENSE,
  PETSC_VIEWER_BINARY_DEFAULT,
  PETSC_VIEWER_BINARY_NATIVE,
  PETSC_VIEWER_DRAW_BASIC,
  PETSC_VIEWER_DRAW_LG,
  PETSC_VIEWER_DRAW_CONTOUR, 
  PETSC_VIEWER_DRAW_PORTS,
  PETSC_VIEWER_NATIVE,
  PETSC_VIEWER_NOFORMAT} PetscViewerFormat;

EXTERN int PetscViewerSetFormat(PetscViewer,PetscViewerFormat);
EXTERN int PetscViewerPushFormat(PetscViewer,PetscViewerFormat);
EXTERN int PetscViewerPopFormat(PetscViewer);
EXTERN int PetscViewerGetFormat(PetscViewer,PetscViewerFormat*);
EXTERN int PetscViewerFlush(PetscViewer);

/*
   Operations explicit to a particular class of viewers
*/

/*E
  PetscViewerFormat - Access mode for a file.

  Level: beginner

.seealso: PetscViewerASCIISetMode()
E*/
typedef enum {FILE_MODE_READ, FILE_MODE_WRITE, FILE_MODE_APPEND, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE} PetscFileMode;

EXTERN int PetscViewerASCIIGetPointer(PetscViewer,FILE**);
EXTERN int PetscViewerASCIISetMode(PetscViewer,PetscFileMode);
EXTERN int PetscViewerASCIIPrintf(PetscViewer,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN int PetscViewerASCIISynchronizedPrintf(PetscViewer,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN int PetscViewerASCIIPushTab(PetscViewer);
EXTERN int PetscViewerASCIIPopTab(PetscViewer);
EXTERN int PetscViewerASCIIUseTabs(PetscViewer,PetscTruth);
EXTERN int PetscViewerASCIISetTab(PetscViewer,int);
EXTERN int PetscViewerBinaryGetDescriptor(PetscViewer,int*);
EXTERN int PetscViewerBinaryGetInfoPointer(PetscViewer,FILE **);
EXTERN int PetscViewerBinarySetType(PetscViewer,PetscViewerBinaryType);
EXTERN int PetscViewerStringSPrintf(PetscViewer,char *,...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN int PetscViewerStringSetString(PetscViewer,char[],int);
EXTERN int PetscViewerDrawClear(PetscViewer);
EXTERN int PetscViewerDrawSetInfo(PetscViewer,const char[],const char[],int,int,int,int);
EXTERN int PetscViewerSocketSetConnection(PetscViewer,const char[],int);
EXTERN int PetscViewerBinarySkipInfo(PetscViewer);

EXTERN int PetscViewerSetFilename(PetscViewer,const char[]);
EXTERN int PetscViewerGetFilename(PetscViewer,char**);

EXTERN int PetscPLAPACKInitializePackage(char *);
EXTERN int PetscPLAPACKFinalizePackage(void);

EXTERN int PetscViewerVUGetPointer(PetscViewer, FILE**);
EXTERN int PetscViewerVUSetMode(PetscViewer, PetscFileMode);
EXTERN int PetscViewerVUSetVecSeen(PetscViewer, PetscTruth);
EXTERN int PetscViewerVUGetVecSeen(PetscViewer, PetscTruth *);
EXTERN int PetscViewerVUPrintDeferred(PetscViewer, const char [], ...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN int PetscViewerVUFlushDeferred(PetscViewer);

EXTERN int PetscViewerMathematicaInitializePackage(char *);
EXTERN int PetscViewerMathematicaFinalizePackage(void);
EXTERN int PetscViewerMathematicaGetName(PetscViewer, const char **);
EXTERN int PetscViewerMathematicaSetName(PetscViewer, const char []);
EXTERN int PetscViewerMathematicaClearName(PetscViewer);
EXTERN int PetscViewerMathematicaSkipPackets(PetscViewer, int);
#if defined(PETSC_HAVE_MATHEMATICA)
EXTERN int PetscViewerMathematicaGetLink(PetscViewer, MLINK *);
#endif

EXTERN int PetscViewerSiloGetName(PetscViewer, char **);
EXTERN int PetscViewerSiloSetName(PetscViewer, const char []);
EXTERN int PetscViewerSiloClearName(PetscViewer);
EXTERN int PetscViewerSiloGetMeshName(PetscViewer, char **);
EXTERN int PetscViewerSiloSetMeshName(PetscViewer, const char []);
EXTERN int PetscViewerSiloClearMeshName(PetscViewer);

/*
     These are all the default viewers that do not have 
   to be explicitly opened
*/
EXTERN PetscViewer PETSC_VIEWER_STDOUT_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_STDERR_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_DRAW_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_SOCKET_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_BINARY_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE;

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
#define PETSC_VIEWER_MATHEMATICA_WORLD (PetscViewerInitializeMathematicaWorld_Private(),PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE) 

/*
    PetscViewer based on the ALICE Memory Snooper
*/
#if defined(PETSC_HAVE_AMS)
#include "ams.h"
EXTERN int         PetscViewerAMSGetAMSComm(PetscViewer,AMS_Comm *);
EXTERN int         PetscViewerAMSOpen(MPI_Comm,const char[],PetscViewer*);
EXTERN int         PetscViewerAMSLock(PetscViewer);
EXTERN PetscViewer PETSC_VIEWER_AMS_(MPI_Comm);
EXTERN int         PETSC_VIEWER_AMS_Destroy(MPI_Comm);
#define PETSC_VIEWER_AMS_WORLD PETSC_VIEWER_AMS_(PETSC_COMM_WORLD)
#endif

/* 
    PetscViewer utility routines used by PETSc that are not normally used
   by users.
*/
EXTERN int  PetscViewerSocketPutScalar(PetscViewer,int,int,PetscScalar*);
EXTERN int  PetscViewerSocketPutReal(PetscViewer,int,int,PetscReal*);
EXTERN int  PetscViewerSocketPutInt(PetscViewer,int,int*);
EXTERN int  PetscViewerSocketPutSparse_Private(PetscViewer,int,int,int,PetscScalar*,int*,int *);
EXTERN int  PetscViewerDestroyAMS_Private(void);

/*S
     PetscViewers - Abstract collection of PetscViewers

   Level: intermediate

  Concepts: viewing

.seealso:  PetscViewerCreate(), PetscViewerSetType(), PetscViewerType, PetscViewer, PetscViewersCreate(),
           PetscViewersGetViewer()
S*/
typedef struct _p_PetscViewers* PetscViewers;
EXTERN int PetscViewersCreate(MPI_Comm,PetscViewers*);
EXTERN int PetscViewersDestroy(PetscViewers);
EXTERN int PetscViewersGetViewer(PetscViewers,int,PetscViewer*);

#endif
