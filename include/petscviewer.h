/*
     PetscViewers are objects where other objects can be looked at or stored.
*/

#if !defined(__PETSCVIEWER_H)
#define __PETSCVIEWER_H

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
extern "C" {
#endif

/*S
     PetscViewer - Abstract PETSc object that helps view (in ASCII, binary, graphically etc)
         other PETSc objects

   Level: beginner

  Concepts: viewing

.seealso:  PetscViewerCreate(), PetscViewerSetType(), PetscViewerType
S*/
typedef struct _p_PetscViewer* PetscViewer;

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
}
#endif

#include "petscsys.h"

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
extern "C" {
#endif

extern PETSC_DLLEXPORT PetscCookie PETSC_VIEWER_COOKIE;

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
}
#endif


/*
    petscsys.h must be included AFTER the definition of PetscViewer for ADIC to 
   process correctly.
*/
PETSC_EXTERN_CXX_BEGIN
/*E
    PetscViewerType - String with the name of a PETSc PETScViewer

   Level: beginner

.seealso: PetscViewerSetType(), PetscViewer
E*/
#define PetscViewerType char*
#define PETSC_VIEWER_SOCKET       "socket"
#define PETSC_VIEWER_ASCII        "ascii"
#define PETSC_VIEWER_BINARY       "binary"
#define PETSC_VIEWER_STRING       "string"
#define PETSC_VIEWER_DRAW         "draw"
#define PETSC_VIEWER_VU           "vu"
#define PETSC_VIEWER_MATHEMATICA  "mathematica"
#define PETSC_VIEWER_SILO         "silo"
#define PETSC_VIEWER_NETCDF       "netcdf"
#define PETSC_VIEWER_HDF5         "hdf5"
#define PETSC_VIEWER_MATLAB       "matlab"

extern PetscFList PetscViewerList;
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerRegisterAll(const char *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerRegisterDestroy(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerInitializePackage(const char[]);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerRegister(const char*,const char*,const char*,PetscErrorCode (*)(PetscViewer));

/*MC
   PetscViewerRegisterDynamic - Adds a viewer

   Synopsis:
   PetscErrorCode PetscViewerRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(PetscViewer))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined viewer
.  path - path (either absolute or relative) the library containing this viewer
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   PetscViewerRegisterDynamic() may be called multiple times to add several user-defined viewers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   PetscViewerRegisterDynamic("my_viewer_type",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyViewerCreate",MyViewerCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PetscViewerSetType(viewer,"my_viewer_type")
   or at runtime via the option
$     -viewer_type my_viewer_type

  Concepts: registering^Viewers

.seealso: PetscViewerRegisterAll(), PetscViewerRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscViewerRegisterDynamic(a,b,c,d) PetscViewerRegister(a,b,c,0)
#else
#define PetscViewerRegisterDynamic(a,b,c,d) PetscViewerRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerCreate(MPI_Comm,PetscViewer*);
PetscPolymorphicSubroutine(PetscViewerCreate,(PetscViewer *v),(PETSC_COMM_SELF,v))
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSetFromOptions(PetscViewer);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIOpen(MPI_Comm,const char[],PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryCreate(MPI_Comm,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryOpen(MPI_Comm,const char[],PetscFileMode,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetFlowControl(PetscViewer,PetscInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinarySetMPIIO(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetMPIIO(PetscViewer,PetscTruth*);
#if defined(PETSC_HAVE_MPIIO)
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetMPIIODescriptor(PetscViewer,MPI_File*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetMPIIOOffset(PetscViewer,MPI_Offset*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryAddMPIIOOffset(PetscViewer,MPI_Offset);
#endif

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSocketOpen(MPI_Comm,const char[],int,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerStringOpen(MPI_Comm,char[],PetscInt,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDrawOpen(MPI_Comm,const char[],const char[],int,int,int,int,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaOpen(MPI_Comm, int, const char[], const char[], PetscViewer *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloOpen(MPI_Comm, const char[], PetscViewer *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMatlabOpen(MPI_Comm,const char[],PetscFileMode,PetscViewer*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerGetType(PetscViewer,const PetscViewerType*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSetType(PetscViewer,const PetscViewerType);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDestroy(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerGetSingleton(PetscViewer,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerRestoreSingleton(PetscViewer,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerGetSubcomm(PetscViewer,MPI_Comm,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerRestoreSubcomm(PetscViewer,MPI_Comm,PetscViewer*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSetUp(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerView(PetscViewer,PetscViewer);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSetOptionsPrefix(PetscViewer,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerAppendOptionsPrefix(PetscViewer,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerGetOptionsPrefix(PetscViewer,const char*[]);

/*E
    PetscViewerFormat - Way a viewer presents the object

   Level: beginner

   The values below are also listed in finclude/petscviewer.h. If another values is added below it
   must also be added there.

.seealso: PetscViewerSetFormat(), PetscViewer, PetscViewerType, PetscViewerPushFormat(), PetscViewerPopFormat()
E*/
typedef enum { 
  PETSC_VIEWER_DEFAULT,
  PETSC_VIEWER_ASCII_MATLAB, 
  PETSC_VIEWER_ASCII_MATHEMATICA,
  PETSC_VIEWER_ASCII_IMPL,
  PETSC_VIEWER_ASCII_INFO,
  PETSC_VIEWER_ASCII_INFO_DETAIL,
  PETSC_VIEWER_ASCII_COMMON,
  PETSC_VIEWER_ASCII_SYMMODU,
  PETSC_VIEWER_ASCII_INDEX,
  PETSC_VIEWER_ASCII_DENSE,
  PETSC_VIEWER_ASCII_MATRIXMARKET,
  PETSC_VIEWER_ASCII_VTK,
  PETSC_VIEWER_ASCII_VTK_CELL,
  PETSC_VIEWER_ASCII_VTK_COORDS,
  PETSC_VIEWER_ASCII_PCICE,
  PETSC_VIEWER_ASCII_PYLITH,
  PETSC_VIEWER_ASCII_PYLITH_LOCAL,
  PETSC_VIEWER_ASCII_PYTHON,
  PETSC_VIEWER_ASCII_FACTOR_INFO,
  PETSC_VIEWER_DRAW_BASIC,
  PETSC_VIEWER_DRAW_LG,
  PETSC_VIEWER_DRAW_CONTOUR, 
  PETSC_VIEWER_DRAW_PORTS,
  PETSC_VIEWER_NATIVE,
  PETSC_VIEWER_NOFORMAT
  } PetscViewerFormat;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSetFormat(PetscViewer,PetscViewerFormat);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerPushFormat(PetscViewer,PetscViewerFormat);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerPopFormat(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerGetFormat(PetscViewer,PetscViewerFormat*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerFlush(PetscViewer);

/*S
     PetscViewerASCIIMonitor - Context for the default KSP, SNES and TS monitors that print
                                  ASCII strings of residual norms etc.


   Level: advanced

  Concepts: viewing, monitoring

.seealso:  PetscViewerCreate(), PetscViewerSetType(), PetscViewerType, KSPMonitorSet(), SNESMonitorSet(), TSMonitorSet(),
           KSPMonitorDefault(), SNESMonitorDefault()

S*/
struct _p_PetscViewerASCIIMonitor {
  PetscViewer viewer;
  PetscInt    tabs;
};
typedef struct _p_PetscViewerASCIIMonitor* PetscViewerASCIIMonitor;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIMonitorCreate(MPI_Comm,const char *,PetscInt,PetscViewerASCIIMonitor*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIMonitorDestroy(PetscViewerASCIIMonitor);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIMonitorPrintf(PetscViewerASCIIMonitor,const char[],...);

/*
   Operations explicit to a particular class of viewers
*/

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIGetPointer(PetscViewer,FILE**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerFileGetMode(PetscViewer,PetscFileMode*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetMode(PetscViewer,PetscFileMode);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIPrintf(PetscViewer,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIISynchronizedPrintf(PetscViewer,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIPushTab(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIPopTab(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIUseTabs(PetscViewer,PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIISetTab(PetscViewer,PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetDescriptor(PetscViewer,int*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetInfoPointer(PetscViewer,FILE **);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryRead(PetscViewer,void*,PetscInt,PetscDataType);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryWrite(PetscViewer,void*,PetscInt,PetscDataType,PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerStringSPrintf(PetscViewer,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerStringSetString(PetscViewer,char[],PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDrawClear(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDrawSetInfo(PetscViewer,const char[],const char[],int,int,int,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSocketSetConnection(PetscViewer,const char[],PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinarySkipInfo(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinarySetSkipOptions(PetscViewer,PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetSkipOptions(PetscViewer,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryReadStringArray(PetscViewer,char***);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryWriteStringArray(PetscViewer,char**);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetName(PetscViewer,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerFileGetName(PetscViewer,char**);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscPLAPACKInitializePackage(MPI_Comm com);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscPLAPACKFinalizePackage(void);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerVUGetPointer(PetscViewer, FILE**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerVUSetVecSeen(PetscViewer, PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerVUGetVecSeen(PetscViewer, PetscTruth *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerVUPrintDeferred(PetscViewer, const char [], ...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerVUFlushDeferred(PetscViewer);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaInitializePackage(const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaFinalizePackage(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaGetName(PetscViewer, const char **);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSetName(PetscViewer, const char []);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaClearName(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSkipPackets(PetscViewer, int);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloGetName(PetscViewer, char **);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloSetName(PetscViewer, const char []);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloClearName(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloGetMeshName(PetscViewer, char **);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloSetMeshName(PetscViewer, const char []);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloClearMeshName(PetscViewer);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerNetcdfOpen(MPI_Comm,const char[],PetscFileMode,PetscViewer*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerNetcdfGetID(PetscViewer, int *);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerHDF5WriteSDS(PetscViewer,float *,int,int *,int);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerHDF5Open(MPI_Comm,const char[],PetscFileMode,PetscViewer*);
#ifdef PETSC_HAVE_HDF5
#include <hdf5.h>
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerHDF5GetFileId(PetscViewer,hid_t*);
#endif

/*
     These are all the default viewers that do not have 
   to be explicitly opened
*/
EXTERN PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_STDOUT_(MPI_Comm);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIGetStdout(MPI_Comm,PetscViewer*);
EXTERN PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_STDERR_(MPI_Comm);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIGetStderr(MPI_Comm,PetscViewer*);
EXTERN PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_DRAW_(MPI_Comm);
EXTERN PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_SOCKET_(MPI_Comm);
EXTERN PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_BINARY_(MPI_Comm);
EXTERN PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_MATLAB_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE;

#define PETSC_VIEWER_STDERR_SELF  PETSC_VIEWER_STDERR_(PETSC_COMM_SELF)
#define PETSC_VIEWER_STDERR_WORLD PETSC_VIEWER_STDERR_(PETSC_COMM_WORLD)

/*MC
  PETSC_VIEWER_STDOUT_WORLD  - same as PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD)

  Level: beginner
M*/
#define PETSC_VIEWER_STDOUT_WORLD PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD)

/*MC
  PETSC_VIEWER_STDOUT_SELF  - same as PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)

  Level: beginner
M*/
#define PETSC_VIEWER_STDOUT_SELF  PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)

/*MC
  PETSC_VIEWER_DRAW_WORLD  - same as PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD)

  Level: intermediate
M*/
#define PETSC_VIEWER_DRAW_WORLD   PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD)

/*MC
  PETSC_VIEWER_DRAW_SELF  - same as PETSC_VIEWER_DRAW_(PETSC_COMM_SELF)

  Level: intermediate
M*/
#define PETSC_VIEWER_DRAW_SELF    PETSC_VIEWER_DRAW_(PETSC_COMM_SELF)

/*MC
  PETSC_VIEWER_SOCKET_WORLD  - same as PETSC_VIEWER_SOCKET_(PETSC_COMM_WORLD)

  Level: intermediate
M*/
#define PETSC_VIEWER_SOCKET_WORLD PETSC_VIEWER_SOCKET_(PETSC_COMM_WORLD)

/*MC
  PETSC_VIEWER_SOCKET_SELF  - same as PETSC_VIEWER_SOCKET_(PETSC_COMM_SELF)

  Level: intermediate
M*/
#define PETSC_VIEWER_SOCKET_SELF  PETSC_VIEWER_SOCKET_(PETSC_COMM_SELF)

/*MC
  PETSC_VIEWER_BINARY_WORLD  - same as PETSC_VIEWER_BINARY_(PETSC_COMM_WORLD)

  Level: intermediate
M*/
#define PETSC_VIEWER_BINARY_WORLD PETSC_VIEWER_BINARY_(PETSC_COMM_WORLD)

/*MC
  PETSC_VIEWER_BINARY_SELF  - same as PETSC_VIEWER_BINARY_(PETSC_COMM_SELF)

  Level: intermediate
M*/
#define PETSC_VIEWER_BINARY_SELF  PETSC_VIEWER_BINARY_(PETSC_COMM_SELF)

/*MC
  PETSC_VIEWER_MATLAB_WORLD  - same as PETSC_VIEWER_MATLAB_(PETSC_COMM_WORLD)

  Level: intermediate
M*/
#define PETSC_VIEWER_MATLAB_WORLD PETSC_VIEWER_MATLAB_(PETSC_COMM_WORLD)

/*MC
  PETSC_VIEWER_MATLAB_SELF  - same as PETSC_VIEWER_MATLAB_(PETSC_COMM_SELF)

  Level: intermediate
M*/
#define PETSC_VIEWER_MATLAB_SELF  PETSC_VIEWER_MATLAB_(PETSC_COMM_SELF)

#define PETSC_VIEWER_MATHEMATICA_WORLD (PetscViewerInitializeMathematicaWorld_Private(),PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE) 

/*
   petscViewer writes to Matlab .mat file
*/
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMatlabPutArray(PetscViewer,int,int,PetscScalar*,char*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMatlabGetArray(PetscViewer,int,int,PetscScalar*,char*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerMatlabPutVariable(PetscViewer,const char*,void*);

/*S
     PetscViewers - Abstract collection of PetscViewers. It is just an expandable array of viewers. 

   Level: intermediate

  Concepts: viewing

.seealso:  PetscViewerCreate(), PetscViewerSetType(), PetscViewerType, PetscViewer, PetscViewersCreate(),
           PetscViewersGetViewer()
S*/
typedef struct _n_PetscViewers* PetscViewers;
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewersCreate(MPI_Comm,PetscViewers*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewersDestroy(PetscViewers);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewersGetViewer(PetscViewers,PetscInt,PetscViewer*);

PETSC_EXTERN_CXX_END
#endif
