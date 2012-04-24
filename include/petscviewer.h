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

extern  PetscClassId PETSC_VIEWER_CLASSID;

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
}
#endif


/*
    petscsys.h must be included AFTER the definition of PetscViewer for ADIC to 
   process correctly.
*/
PETSC_EXTERN_CXX_BEGIN
/*J
    PetscViewerType - String with the name of a PETSc PETScViewer

   Level: beginner

.seealso: PetscViewerSetType(), PetscViewer
J*/
#define PetscViewerType char*
#define PETSCVIEWERSOCKET       "socket"
#define PETSCVIEWERASCII        "ascii"
#define PETSCVIEWERBINARY       "binary"
#define PETSCVIEWERSTRING       "string"
#define PETSCVIEWERDRAW         "draw"
#define PETSCVIEWERVU           "vu"
#define PETSCVIEWERMATHEMATICA  "mathematica"
#define PETSCVIEWERNETCDF       "netcdf"
#define PETSCVIEWERHDF5         "hdf5"
#define PETSCVIEWERVTK          "vtk"
#define PETSCVIEWERMATLAB       "matlab"
#define PETSCVIEWERAMS          "ams"

extern PetscFList PetscViewerList;
extern PetscErrorCode  PetscViewerRegisterAll(const char *);
extern PetscErrorCode  PetscViewerRegisterDestroy(void);
extern PetscErrorCode  PetscViewerInitializePackage(const char[]);

extern PetscErrorCode  PetscViewerRegister(const char*,const char*,const char*,PetscErrorCode (*)(PetscViewer));

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

extern PetscErrorCode  PetscViewerCreate(MPI_Comm,PetscViewer*);
extern PetscErrorCode  PetscViewerSetFromOptions(PetscViewer);
extern PetscErrorCode  PetscViewerASCIIOpenWithFILE(MPI_Comm,FILE*,PetscViewer*);

extern PetscErrorCode  PetscViewerASCIIOpen(MPI_Comm,const char[],PetscViewer*);
extern PetscErrorCode  PetscViewerASCIISetFILE(PetscViewer,FILE*);
extern PetscErrorCode  PetscViewerBinaryOpen(MPI_Comm,const char[],PetscFileMode,PetscViewer*);
extern PetscErrorCode  PetscViewerBinaryGetFlowControl(PetscViewer,PetscInt*);
extern PetscErrorCode  PetscViewerBinarySetFlowControl(PetscViewer,PetscInt);
extern PetscErrorCode  PetscViewerBinarySetMPIIO(PetscViewer);
extern PetscErrorCode  PetscViewerBinaryGetMPIIO(PetscViewer,PetscBool *);
#if defined(PETSC_HAVE_MPIIO)
extern PetscErrorCode  PetscViewerBinaryGetMPIIODescriptor(PetscViewer,MPI_File*);
extern PetscErrorCode  PetscViewerBinaryGetMPIIOOffset(PetscViewer,MPI_Offset*);
extern PetscErrorCode  PetscViewerBinaryAddMPIIOOffset(PetscViewer,MPI_Offset);
#endif

extern PetscErrorCode  PetscViewerSocketOpen(MPI_Comm,const char[],int,PetscViewer*);
extern PetscErrorCode  PetscViewerStringOpen(MPI_Comm,char[],PetscInt,PetscViewer*);
extern PetscErrorCode  PetscViewerDrawOpen(MPI_Comm,const char[],const char[],int,int,int,int,PetscViewer*);
extern PetscErrorCode  PetscViewerMathematicaOpen(MPI_Comm, int, const char[], const char[], PetscViewer *);
extern PetscErrorCode  PetscViewerSiloOpen(MPI_Comm, const char[], PetscViewer *);
extern PetscErrorCode  PetscViewerMatlabOpen(MPI_Comm,const char[],PetscFileMode,PetscViewer*);

extern PetscErrorCode  PetscViewerGetType(PetscViewer,const PetscViewerType*);
extern PetscErrorCode  PetscViewerSetType(PetscViewer,const PetscViewerType);
extern PetscErrorCode  PetscViewerDestroy(PetscViewer*);
extern PetscErrorCode  PetscViewerGetSingleton(PetscViewer,PetscViewer*);
extern PetscErrorCode  PetscViewerRestoreSingleton(PetscViewer,PetscViewer*);
extern PetscErrorCode  PetscViewerGetSubcomm(PetscViewer,MPI_Comm,PetscViewer*);
extern PetscErrorCode  PetscViewerRestoreSubcomm(PetscViewer,MPI_Comm,PetscViewer*);

extern PetscErrorCode  PetscViewerSetUp(PetscViewer);
extern PetscErrorCode  PetscViewerView(PetscViewer,PetscViewer);

extern PetscErrorCode  PetscViewerSetOptionsPrefix(PetscViewer,const char[]);
extern PetscErrorCode  PetscViewerAppendOptionsPrefix(PetscViewer,const char[]);
extern PetscErrorCode  PetscViewerGetOptionsPrefix(PetscViewer,const char*[]);

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
  PETSC_VIEWER_ASCII_PYTHON,
  PETSC_VIEWER_ASCII_FACTOR_INFO,
  PETSC_VIEWER_ASCII_LATEX,
  PETSC_VIEWER_DRAW_BASIC,
  PETSC_VIEWER_DRAW_LG,
  PETSC_VIEWER_DRAW_CONTOUR,
  PETSC_VIEWER_DRAW_PORTS,
  PETSC_VIEWER_VTK_VTS,
  PETSC_VIEWER_NATIVE,
  PETSC_VIEWER_NOFORMAT
  } PetscViewerFormat;
extern const char *const PetscViewerFormats[];

extern PetscErrorCode  PetscViewerSetFormat(PetscViewer,PetscViewerFormat);
extern PetscErrorCode  PetscViewerPushFormat(PetscViewer,PetscViewerFormat);
extern PetscErrorCode  PetscViewerPopFormat(PetscViewer);
extern PetscErrorCode  PetscViewerGetFormat(PetscViewer,PetscViewerFormat*);
extern PetscErrorCode  PetscViewerFlush(PetscViewer);

/*
   Operations explicit to a particular class of viewers
*/

extern PetscErrorCode  PetscViewerASCIIGetPointer(PetscViewer,FILE**);
extern PetscErrorCode  PetscViewerFileGetMode(PetscViewer,PetscFileMode*);
extern PetscErrorCode  PetscViewerFileSetMode(PetscViewer,PetscFileMode);
extern PetscErrorCode  PetscViewerASCIIPrintf(PetscViewer,const char[],...);
extern PetscErrorCode  PetscViewerASCIISynchronizedPrintf(PetscViewer,const char[],...);
extern PetscErrorCode  PetscViewerASCIISynchronizedAllow(PetscViewer,PetscBool);
extern PetscErrorCode  PetscViewerASCIIPushTab(PetscViewer);
extern PetscErrorCode  PetscViewerASCIIPopTab(PetscViewer);
extern PetscErrorCode  PetscViewerASCIIUseTabs(PetscViewer,PetscBool );
extern PetscErrorCode  PetscViewerASCIISetTab(PetscViewer,PetscInt);
extern PetscErrorCode  PetscViewerASCIIAddTab(PetscViewer,PetscInt);
extern PetscErrorCode  PetscViewerASCIISubtractTab(PetscViewer,PetscInt);
extern PetscErrorCode  PetscViewerBinaryGetDescriptor(PetscViewer,int*);
extern PetscErrorCode  PetscViewerBinaryGetInfoPointer(PetscViewer,FILE **);
extern PetscErrorCode  PetscViewerBinaryRead(PetscViewer,void*,PetscInt,PetscDataType);
extern PetscErrorCode  PetscViewerBinaryWrite(PetscViewer,void*,PetscInt,PetscDataType,PetscBool );
extern PetscErrorCode  PetscViewerStringSPrintf(PetscViewer,const char[],...);
extern PetscErrorCode  PetscViewerStringSetString(PetscViewer,char[],PetscInt);
extern PetscErrorCode  PetscViewerDrawClear(PetscViewer);
extern PetscErrorCode  PetscViewerDrawSetHold(PetscViewer,PetscBool);
extern PetscErrorCode  PetscViewerDrawGetHold(PetscViewer,PetscBool*);
extern PetscErrorCode  PetscViewerDrawSetPause(PetscViewer,PetscReal);
extern PetscErrorCode  PetscViewerDrawGetPause(PetscViewer,PetscReal*);
extern PetscErrorCode  PetscViewerDrawSetInfo(PetscViewer,const char[],const char[],int,int,int,int);
extern PetscErrorCode  PetscViewerDrawResize(PetscViewer,int,int);
extern PetscErrorCode  PetscViewerDrawSetBounds(PetscViewer,PetscInt,const PetscReal*);
extern PetscErrorCode  PetscViewerDrawGetBounds(PetscViewer,PetscInt*,const PetscReal**);
extern PetscErrorCode  PetscViewerSocketSetConnection(PetscViewer,const char[],int);
extern PetscErrorCode  PetscViewerBinarySkipInfo(PetscViewer);
extern PetscErrorCode  PetscViewerBinarySetSkipOptions(PetscViewer,PetscBool );
extern PetscErrorCode  PetscViewerBinaryGetSkipOptions(PetscViewer,PetscBool *);
extern PetscErrorCode  PetscViewerBinarySetSkipHeader(PetscViewer,PetscBool);
extern PetscErrorCode  PetscViewerBinaryGetSkipHeader(PetscViewer,PetscBool*);
extern PetscErrorCode  PetscViewerBinaryReadStringArray(PetscViewer,char***);
extern PetscErrorCode  PetscViewerBinaryWriteStringArray(PetscViewer,char**);

extern PetscErrorCode  PetscViewerFileSetName(PetscViewer,const char[]);
extern PetscErrorCode  PetscViewerFileGetName(PetscViewer,const char**);

extern PetscErrorCode  PetscPLAPACKInitializePackage(MPI_Comm com);
extern PetscErrorCode  PetscPLAPACKFinalizePackage(void);

extern PetscErrorCode  PetscViewerVUGetPointer(PetscViewer, FILE**);
extern PetscErrorCode  PetscViewerVUSetVecSeen(PetscViewer, PetscBool );
extern PetscErrorCode  PetscViewerVUGetVecSeen(PetscViewer, PetscBool  *);
extern PetscErrorCode  PetscViewerVUPrintDeferred(PetscViewer, const char [], ...);
extern PetscErrorCode  PetscViewerVUFlushDeferred(PetscViewer);

extern PetscErrorCode  PetscViewerMathematicaInitializePackage(const char[]);
extern PetscErrorCode  PetscViewerMathematicaFinalizePackage(void);
extern PetscErrorCode  PetscViewerMathematicaGetName(PetscViewer, const char **);
extern PetscErrorCode  PetscViewerMathematicaSetName(PetscViewer, const char []);
extern PetscErrorCode  PetscViewerMathematicaClearName(PetscViewer);
extern PetscErrorCode  PetscViewerMathematicaSkipPackets(PetscViewer, int);

extern PetscErrorCode  PetscViewerSiloGetName(PetscViewer, char **);
extern PetscErrorCode  PetscViewerSiloSetName(PetscViewer, const char []);
extern PetscErrorCode  PetscViewerSiloClearName(PetscViewer);
extern PetscErrorCode  PetscViewerSiloGetMeshName(PetscViewer, char **);
extern PetscErrorCode  PetscViewerSiloSetMeshName(PetscViewer, const char []);
extern PetscErrorCode  PetscViewerSiloClearMeshName(PetscViewer);

extern PetscErrorCode  PetscViewerNetcdfOpen(MPI_Comm,const char[],PetscFileMode,PetscViewer*);
extern PetscErrorCode  PetscViewerNetcdfGetID(PetscViewer, int *);

extern PetscErrorCode  PetscViewerHDF5WriteSDS(PetscViewer,float *,int,int *,int);

extern PetscErrorCode  PetscViewerHDF5Open(MPI_Comm,const char[],PetscFileMode,PetscViewer*);
extern PetscErrorCode  PetscViewerHDF5PushGroup(PetscViewer,const char *);
extern PetscErrorCode  PetscViewerHDF5PopGroup(PetscViewer);
extern PetscErrorCode  PetscViewerHDF5GetGroup(PetscViewer, const char **);
extern PetscErrorCode  PetscViewerHDF5IncrementTimestep(PetscViewer);
extern PetscErrorCode  PetscViewerHDF5SetTimestep(PetscViewer,PetscInt);
extern PetscErrorCode  PetscViewerHDF5GetTimestep(PetscViewer,PetscInt*);
#ifdef PETSC_HAVE_HDF5
#include <hdf5.h>
extern PetscErrorCode  PetscViewerHDF5GetFileId(PetscViewer,hid_t*);
extern PetscErrorCode  PetscViewerHDF5OpenGroup(PetscViewer, hid_t *, hid_t *);
#endif

typedef enum {PETSC_VTK_POINT_FIELD, PETSC_VTK_POINT_VECTOR_FIELD, PETSC_VTK_CELL_FIELD, PETSC_VTK_CELL_VECTOR_FIELD} PetscViewerVTKFieldType;
typedef PetscErrorCode (*PetscViewerVTKWriteFunction)(PetscObject,PetscViewer);
extern PetscErrorCode PetscViewerVTKAddField(PetscViewer,PetscObject,PetscViewerVTKWriteFunction,PetscViewerVTKFieldType,PetscObject);
extern PetscErrorCode PetscViewerVTKOpen(MPI_Comm,const char[],PetscFileMode,PetscViewer*);

/*
     These are all the default viewers that do not have 
   to be explicitly opened
*/
extern PetscViewer  PETSC_VIEWER_STDOUT_(MPI_Comm);
extern PetscErrorCode  PetscViewerASCIIGetStdout(MPI_Comm,PetscViewer*);
extern PetscViewer  PETSC_VIEWER_STDERR_(MPI_Comm);
extern PetscErrorCode  PetscViewerASCIIGetStderr(MPI_Comm,PetscViewer*);
extern PetscViewer  PETSC_VIEWER_DRAW_(MPI_Comm);
extern PetscViewer  PETSC_VIEWER_SOCKET_(MPI_Comm);
extern PetscViewer  PETSC_VIEWER_BINARY_(MPI_Comm);
extern PetscViewer  PETSC_VIEWER_MATLAB_(MPI_Comm);
extern PetscViewer PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE;

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

#define PetscViewerFlowControlStart(viewer,mcnt,cnt)  (PetscViewerBinaryGetFlowControl(viewer,mcnt) || PetscViewerBinaryGetFlowControl(viewer,cnt))
#define PetscViewerFlowControlStepMaster(viewer,i,mcnt,cnt) ((i >= mcnt) ?  (mcnt += cnt,MPI_Bcast(&mcnt,1,MPIU_INT,0,((PetscObject)viewer)->comm)) : 0)
#define PetscViewerFlowControlEndMaster(viewer,mcnt) (mcnt = 0,MPI_Bcast(&mcnt,1,MPIU_INT,0,((PetscObject)viewer)->comm))
#define PetscViewerFlowControlStepWorker(viewer,rank,mcnt) 0; while (1) { PetscErrorCode _ierr; \
    if (rank < mcnt) break;				\
  _ierr = MPI_Bcast(&mcnt,1,MPIU_INT,0,((PetscObject)viewer)->comm);CHKERRQ(_ierr);\
  }
#define PetscViewerFlowControlEndWorker(viewer,mcnt) 0; while (1) { PetscErrorCode _ierr; \
  _ierr = MPI_Bcast(&mcnt,1,MPIU_INT,0,((PetscObject)viewer)->comm);CHKERRQ(_ierr);\
    if (mcnt == 0) break;				\
  }

/*
   petscViewer writes to MATLAB .mat file
*/
extern PetscErrorCode  PetscViewerMatlabPutArray(PetscViewer,int,int,const PetscScalar*,const char*);
extern PetscErrorCode  PetscViewerMatlabGetArray(PetscViewer,int,int,PetscScalar*,const char*);
extern PetscErrorCode  PetscViewerMatlabPutVariable(PetscViewer,const char*,void*);

/*S
     PetscViewers - Abstract collection of PetscViewers. It is just an expandable array of viewers. 

   Level: intermediate

  Concepts: viewing

.seealso:  PetscViewerCreate(), PetscViewerSetType(), PetscViewerType, PetscViewer, PetscViewersCreate(),
           PetscViewersGetViewer()
S*/
typedef struct _n_PetscViewers* PetscViewers;
extern PetscErrorCode  PetscViewersCreate(MPI_Comm,PetscViewers*);
extern PetscErrorCode  PetscViewersDestroy(PetscViewers*);
extern PetscErrorCode  PetscViewersGetViewer(PetscViewers,PetscInt,PetscViewer*);

#if defined(PETSC_HAVE_AMS)
#include <ams.h>
extern PetscErrorCode PetscViewerAMSSetCommName(PetscViewer,const char[]);
extern PetscErrorCode PetscViewerAMSGetAMSComm(PetscViewer,AMS_Comm *);
extern PetscErrorCode PetscViewerAMSOpen(MPI_Comm,const char[],PetscViewer*);
extern PetscErrorCode PetscViewerAMSLock(PetscViewer);
extern PetscViewer    PETSC_VIEWER_AMS_(MPI_Comm);
extern PetscErrorCode PETSC_VIEWER_AMS_Destroy(MPI_Comm);
#define PETSC_VIEWER_AMS_WORLD PETSC_VIEWER_AMS_(PETSC_COMM_WORLD)
#endif


PETSC_EXTERN_CXX_END
#endif
