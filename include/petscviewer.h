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

#include "petsc.h"

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
extern "C" {
#endif

extern PetscCookie PETSC_VIEWER_COOKIE;

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
}
#endif


/*
    petsc.h must be included AFTER the definition of PetscViewer for ADIC to 
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
#define PETSC_VIEWER_HDF4         "hdf4"
#define PETSC_VIEWER_MATLAB       "matlab"

extern PetscFList PetscViewerList;
EXTERN PetscErrorCode PetscViewerRegisterAll(const char *);
EXTERN PetscErrorCode PetscViewerRegisterDestroy(void);

EXTERN PetscErrorCode PetscViewerRegister(const char*,const char*,const char*,PetscErrorCode (*)(PetscViewer));

/*MC
   PetscViewerRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   PetscErrorCode PetscViewerRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(PetscViewer))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   PetscViewerRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   PetscViewerRegisterDynamic("my_viewer_type",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyViewerCreate",MyViewerCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PetscViewerSetType(ksp,"my_viewer_type")
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

EXTERN PetscErrorCode PetscViewerCreate(MPI_Comm,PetscViewer*);
PetscPolymorphicSubroutine(PetscViewerCreate,(PetscViewer *v),(PETSC_COMM_SELF,v))
EXTERN PetscErrorCode PetscViewerSetFromOptions(PetscViewer);

EXTERN PetscErrorCode PetscViewerASCIIOpen(MPI_Comm,const char[],PetscViewer*);

/*E
  PetscViewerFileType - Indicates how the file should be opened for the viewer

  Level: beginner

.seealso: PetscViewerSetFileName(), PetscViewerSetFileType(), PetscViewerBinaryOpen(), PetscViewerASCIIOpen(),
          PetscViewerMatlabOpen()
E*/
typedef enum {PETSC_FILE_RDONLY,PETSC_FILE_WRONLY,PETSC_FILE_CREATE} PetscViewerFileType;

/*M
    PETSC_FILE_RDONLY - File is open to be read from only, not written to

    Level: beginner

.seealso: PetscViewerFileType, PETSC_FILE_WRONLY, PETSC_FILE_CREATE, PetscViewerSetFileName(), PetscViewerSetFileType(), 
          PetscViewerBinaryOpen(), PetscViewerASCIIOpen(), PetscViewerMatlabOpen()

M*/

/*M
    PETSC_FILE_WRONLY - File is open to be appended to.

    Level: beginner

.seealso: PetscViewerFileType, PETSC_FILE_RDONLY, PETSC_FILE_CREATE, PetscViewerSetFileName(), PetscViewerSetFileType(), 
          PetscViewerBinaryOpen(), PetscViewerASCIIOpen(), PetscViewerMatlabOpen()

M*/

/*M
    PETSC_FILE_CREATE - Create the file, or delete it and open an empty file if it already existed

    Level: beginner

.seealso: PetscViewerFileType, PETSC_FILE_RDONLY, PETSC_FILE_WRONLY, PetscViewerSetFileName(), PetscViewerSetFileType(), 
          PetscViewerBinaryOpen(), PetscViewerASCIIOpen(), PetscViewerMatlabOpen()

M*/

EXTERN PetscErrorCode PetscViewerBinaryOpen(MPI_Comm,const char[],PetscViewerFileType,PetscViewer*);
EXTERN PetscErrorCode PetscViewerSocketOpen(MPI_Comm,const char[],int,PetscViewer*);
EXTERN PetscErrorCode PetscViewerStringOpen(MPI_Comm,char[],PetscInt,PetscViewer*);
EXTERN PetscErrorCode PetscViewerDrawOpen(MPI_Comm,const char[],const char[],int,int,int,int,PetscViewer*);
EXTERN PetscErrorCode PetscViewerMathematicaOpen(MPI_Comm, int, const char[], const char[], PetscViewer *);
EXTERN PetscErrorCode PetscViewerSiloOpen(MPI_Comm, const char[], PetscViewer *);
EXTERN PetscErrorCode PetscViewerMatlabOpen(MPI_Comm,const char[],PetscViewerFileType,PetscViewer*);

EXTERN PetscErrorCode PetscViewerGetType(PetscViewer,PetscViewerType*);
EXTERN PetscErrorCode PetscViewerSetType(PetscViewer,const PetscViewerType);
EXTERN PetscErrorCode PetscViewerDestroy(PetscViewer);
EXTERN PetscErrorCode PetscViewerGetSingleton(PetscViewer,PetscViewer*);
EXTERN PetscErrorCode PetscViewerRestoreSingleton(PetscViewer,PetscViewer*);


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
  PETSC_VIEWER_ASCII_INFO_DETAIL,
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
  PETSC_VIEWER_NOFORMAT,
  PETSC_VIEWER_ASCII_FACTOR_INFO} PetscViewerFormat;

EXTERN PetscErrorCode PetscViewerSetFormat(PetscViewer,PetscViewerFormat);
EXTERN PetscErrorCode PetscViewerPushFormat(PetscViewer,PetscViewerFormat);
EXTERN PetscErrorCode PetscViewerPopFormat(PetscViewer);
EXTERN PetscErrorCode PetscViewerGetFormat(PetscViewer,PetscViewerFormat*);
EXTERN PetscErrorCode PetscViewerFlush(PetscViewer);

/*
   Operations explicit to a particular class of viewers
*/

/*E
  PetscViewerFormat - Access mode for a file.

  Level: beginner

.seealso: PetscViewerASCIISetMode()
E*/
typedef enum {FILE_MODE_READ, FILE_MODE_WRITE, FILE_MODE_APPEND, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE} PetscFileMode;

EXTERN PetscErrorCode PetscViewerASCIIGetPointer(PetscViewer,FILE**);
EXTERN PetscErrorCode PetscViewerASCIISetMode(PetscViewer,PetscFileMode);
EXTERN PetscErrorCode PetscViewerASCIIPrintf(PetscViewer,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode PetscViewerASCIISynchronizedPrintf(PetscViewer,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode PetscViewerASCIIPushTab(PetscViewer);
EXTERN PetscErrorCode PetscViewerASCIIPopTab(PetscViewer);
EXTERN PetscErrorCode PetscViewerASCIIUseTabs(PetscViewer,PetscTruth);
EXTERN PetscErrorCode PetscViewerASCIISetTab(PetscViewer,PetscInt);
EXTERN PetscErrorCode PetscViewerBinaryGetDescriptor(PetscViewer,int*);
EXTERN PetscErrorCode PetscViewerBinaryGetInfoPointer(PetscViewer,FILE **);
EXTERN PetscErrorCode PetscViewerSetFileType(PetscViewer,PetscViewerFileType);
EXTERN PetscErrorCode PetscViewerStringSPrintf(PetscViewer,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode PetscViewerStringSetString(PetscViewer,char[],PetscInt);
EXTERN PetscErrorCode PetscViewerDrawClear(PetscViewer);
EXTERN PetscErrorCode PetscViewerDrawSetInfo(PetscViewer,const char[],const char[],int,int,int,int);
EXTERN PetscErrorCode PetscViewerSocketSetConnection(PetscViewer,const char[],PetscInt);
EXTERN PetscErrorCode PetscViewerBinarySkipInfo(PetscViewer);
EXTERN PetscErrorCode PetscViewerBinaryLoadInfo(PetscViewer);


EXTERN PetscErrorCode PetscViewerSetFilename(PetscViewer,const char[]);
EXTERN PetscErrorCode PetscViewerGetFilename(PetscViewer,char**);

EXTERN PetscErrorCode PetscPLAPACKInitializePackage(char *);
EXTERN PetscErrorCode PetscPLAPACKFinalizePackage(void);

EXTERN PetscErrorCode PetscViewerVUGetPointer(PetscViewer, FILE**);
EXTERN PetscErrorCode PetscViewerVUSetMode(PetscViewer, PetscFileMode);
EXTERN PetscErrorCode PetscViewerVUSetVecSeen(PetscViewer, PetscTruth);
EXTERN PetscErrorCode PetscViewerVUGetVecSeen(PetscViewer, PetscTruth *);
EXTERN PetscErrorCode PetscViewerVUPrintDeferred(PetscViewer, const char [], ...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode PetscViewerVUFlushDeferred(PetscViewer);

EXTERN PetscErrorCode PetscViewerMathematicaInitializePackage(char *);
EXTERN PetscErrorCode PetscViewerMathematicaFinalizePackage(void);
EXTERN PetscErrorCode PetscViewerMathematicaGetName(PetscViewer, const char **);
EXTERN PetscErrorCode PetscViewerMathematicaSetName(PetscViewer, const char []);
EXTERN PetscErrorCode PetscViewerMathematicaClearName(PetscViewer);
EXTERN PetscErrorCode PetscViewerMathematicaSkipPackets(PetscViewer, int);

EXTERN PetscErrorCode PetscViewerSiloGetName(PetscViewer, char **);
EXTERN PetscErrorCode PetscViewerSiloSetName(PetscViewer, const char []);
EXTERN PetscErrorCode PetscViewerSiloClearName(PetscViewer);
EXTERN PetscErrorCode PetscViewerSiloGetMeshName(PetscViewer, char **);
EXTERN PetscErrorCode PetscViewerSiloSetMeshName(PetscViewer, const char []);
EXTERN PetscErrorCode PetscViewerSiloClearMeshName(PetscViewer);

EXTERN PetscErrorCode PetscViewerNetcdfOpen(MPI_Comm,const char[],PetscViewerFileType,PetscViewer*);
EXTERN PetscErrorCode PetscViewerNetcdfGetID(PetscViewer, int *);

EXTERN PetscErrorCode PetscViewerHDF4Open(MPI_Comm,const char[],PetscViewerFileType,PetscViewer*);
EXTERN PetscErrorCode PetscViewerHDF4WriteSDS(PetscViewer viewer, float *xf, int d, int *dims, int bs);

/*
     These are all the default viewers that do not have 
   to be explicitly opened
*/
EXTERN PetscViewer PETSC_VIEWER_STDOUT_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_STDERR_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_DRAW_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_SOCKET_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_BINARY_(MPI_Comm);
EXTERN PetscViewer PETSC_VIEWER_MATLAB_(MPI_Comm);
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
EXTERN PetscErrorCode PetscViewerMatlabPutArray(PetscViewer,int,int,PetscScalar*,char*);
EXTERN PetscErrorCode PetscViewerMatlabGetArray(PetscViewer,int,int,PetscScalar*,char*);
EXTERN PetscErrorCode PetscViewerMatlabPutVariable(PetscViewer,const char*,void*);

/* 
    PetscViewer utility routines used by PETSc that are not normally used
   by users.
*/
EXTERN PetscErrorCode  PetscViewerSocketPutScalar(PetscViewer,PetscInt,PetscInt,PetscScalar*);
EXTERN PetscErrorCode  PetscViewerSocketPutReal(PetscViewer,PetscInt,PetscInt,PetscReal*);
EXTERN PetscErrorCode  PetscViewerSocketPutInt(PetscViewer,PetscInt,PetscInt*);
EXTERN PetscErrorCode  PetscViewerSocketPutSparse_Private(PetscViewer,PetscInt,PetscInt,PetscInt,PetscScalar*,PetscInt*,PetscInt *);

/*S
     PetscViewers - Abstract collection of PetscViewers

   Level: intermediate

  Concepts: viewing

.seealso:  PetscViewerCreate(), PetscViewerSetType(), PetscViewerType, PetscViewer, PetscViewersCreate(),
           PetscViewersGetViewer()
S*/
typedef struct _p_PetscViewers* PetscViewers;
EXTERN PetscErrorCode PetscViewersCreate(MPI_Comm,PetscViewers*);
EXTERN PetscErrorCode PetscViewersDestroy(PetscViewers);
EXTERN PetscErrorCode PetscViewersGetViewer(PetscViewers,int,PetscViewer*);

PETSC_EXTERN_CXX_END
#endif
