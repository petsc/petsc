#define PETSC_DESIRE_FEATURE_TEST_MACROS
/*
   This file defines the initialization of PETSc, including PetscInitialize()
*/
#include <petsc/private/petscimpl.h> /*I  "petscsys.h"   I*/
#include <petscviewer.h>
#include <petsc/private/garbagecollector.h>

#if !defined(PETSC_HAVE_WINDOWS_COMPILERS)
  #include <petsc/private/valgrind/valgrind.h>
#endif

#if defined(PETSC_HAVE_FORTRAN)
  #include <petsc/private/fortranimpl.h>
#endif

#if PetscDefined(USE_COVERAGE)
EXTERN_C_BEGIN
  #if defined(PETSC_HAVE___GCOV_DUMP)
    #define __gcov_flush(x) __gcov_dump(x)
  #endif
void __gcov_flush(void);
EXTERN_C_END
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
PETSC_INTERN PetscFPT PetscFPTData;
PetscFPT              PetscFPTData = 0;
#endif

#if PetscDefined(HAVE_SAWS)
  #include <petscviewersaws.h>
#endif

/* -----------------------------------------------------------------------------------------*/

PETSC_INTERN FILE *petsc_history;

PETSC_INTERN PetscErrorCode PetscInitialize_DynamicLibraries(void);
PETSC_INTERN PetscErrorCode PetscFinalize_DynamicLibraries(void);
PETSC_INTERN PetscErrorCode PetscSequentialPhaseBegin_Private(MPI_Comm, int);
PETSC_INTERN PetscErrorCode PetscSequentialPhaseEnd_Private(MPI_Comm, int);
PETSC_INTERN PetscErrorCode PetscCloseHistoryFile(FILE **);

/* user may set these BEFORE calling PetscInitialize() */
MPI_Comm PETSC_COMM_WORLD = MPI_COMM_NULL;
#if PetscDefined(HAVE_MPI_INIT_THREAD)
PetscMPIInt PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_FUNNELED;
#else
PetscMPIInt PETSC_MPI_THREAD_REQUIRED = 0;
#endif

PetscMPIInt Petsc_Counter_keyval      = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_InnerComm_keyval    = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_OuterComm_keyval    = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_ShmComm_keyval      = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_CreationIdx_keyval  = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_Garbage_HMap_keyval = MPI_KEYVAL_INVALID;

PetscMPIInt Petsc_SharedWD_keyval  = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_SharedTmp_keyval = MPI_KEYVAL_INVALID;

/*
     Declare and set all the string names of the PETSc enums
*/
const char *const PetscBools[]     = {"FALSE", "TRUE", "PetscBool", "PETSC_", NULL};
const char *const PetscCopyModes[] = {"COPY_VALUES", "OWN_POINTER", "USE_POINTER", "PetscCopyMode", "PETSC_", NULL};

PetscBool PetscPreLoadingUsed = PETSC_FALSE;
PetscBool PetscPreLoadingOn   = PETSC_FALSE;

PetscInt PetscHotRegionDepth;

PetscBool PETSC_RUNNING_ON_VALGRIND = PETSC_FALSE;

#if defined(PETSC_HAVE_THREADSAFETY)
PetscSpinlock PetscViewerASCIISpinLockOpen;
PetscSpinlock PetscViewerASCIISpinLockStdout;
PetscSpinlock PetscViewerASCIISpinLockStderr;
PetscSpinlock PetscCommSpinLock;
#endif

/*
      PetscInitializeNoPointers - Calls PetscInitialize() from C/C++ without the pointers to argc and args

   Collective

   Level: advanced

    Notes:
    this is called only by the PETSc Julia interface. Even though it might start MPI it sets the flag to
     indicate that it did NOT start MPI so that the PetscFinalize() does not end MPI, thus allowing PetscInitialize() to
     be called multiple times from Julia without the problem of trying to initialize MPI more than once.

     Developer Note: Turns off PETSc signal handling to allow Julia to manage signals

.seealso: `PetscInitialize()`, `PetscInitializeFortran()`, `PetscInitializeNoArguments()`
*/
PetscErrorCode PetscInitializeNoPointers(int argc, char **args, const char *filename, const char *help)
{
  int    myargc = argc;
  char **myargs = args;

  PetscFunctionBegin;
  PetscCall(PetscInitialize(&myargc, &myargs, filename, help));
  PetscCall(PetscPopSignalHandler());
  PetscBeganMPI = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
      Used by Julia interface to get communicator
*/
PetscErrorCode PetscGetPETSC_COMM_SELF(MPI_Comm *comm)
{
  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscValidPointer(comm, 1);
  *comm = PETSC_COMM_SELF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
      PetscInitializeNoArguments - Calls `PetscInitialize()` from C/C++ without
        the command line arguments.

   Collective

   Level: advanced

.seealso: `PetscInitialize()`, `PetscInitializeFortran()`
@*/
PetscErrorCode PetscInitializeNoArguments(void)
{
  int    argc = 0;
  char **args = NULL;

  PetscFunctionBegin;
  PetscCall(PetscInitialize(&argc, &args, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
      PetscInitialized - Determine whether PETSc is initialized.

   Level: beginner

.seealso: `PetscInitialize()`, `PetscInitializeNoArguments()`, `PetscInitializeFortran()`
@*/
PetscErrorCode PetscInitialized(PetscBool *isInitialized)
{
  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscValidBoolPointer(isInitialized, 1);
  *isInitialized = PetscInitializeCalled;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
      PetscFinalized - Determine whether `PetscFinalize()` has been called yet

   Level: developer

.seealso: `PetscInitialize()`, `PetscInitializeNoArguments()`, `PetscInitializeFortran()`
@*/
PetscErrorCode PetscFinalized(PetscBool *isFinalized)
{
  PetscFunctionBegin;
  if (!PetscFinalizeCalled) PetscValidBoolPointer(isFinalized, 1);
  *isFinalized = PetscFinalizeCalled;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscOptionsCheckInitial_Private(const char[]);

/*
       This function is the MPI reduction operation used to compute the sum of the
   first half of the datatype and the max of the second half.
*/
MPI_Op MPIU_MAXSUM_OP               = 0;
MPI_Op Petsc_Garbage_SetIntersectOp = 0;

PETSC_INTERN void MPIAPI MPIU_MaxSum_Local(void *in, void *out, int *cnt, MPI_Datatype *datatype)
{
  PetscInt *xin = (PetscInt *)in, *xout = (PetscInt *)out, i, count = *cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_2INT) {
    PetscErrorCode ierr = (*PetscErrorPrintf)("Can only handle MPIU_2INT data types");
    (void)ierr;
    PETSCABORT(MPI_COMM_SELF, PETSC_ERR_ARG_WRONG);
  }

  for (i = 0; i < count; i++) {
    xout[2 * i] = PetscMax(xout[2 * i], xin[2 * i]);
    xout[2 * i + 1] += xin[2 * i + 1];
  }
  PetscFunctionReturnVoid();
}

/*
    Returns the max of the first entry owned by this processor and the
sum of the second entry.

    The reason sizes[2*i] contains lengths sizes[2*i+1] contains flag of 1 if length is nonzero
is so that the MPIU_MAXSUM_OP() can set TWO values, if we passed in only sizes[i] with lengths
there would be no place to store the both needed results.
*/
PetscErrorCode PetscMaxSum(MPI_Comm comm, const PetscInt sizes[], PetscInt *max, PetscInt *sum)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_REDUCE_SCATTER_BLOCK)
  {
    struct {
      PetscInt max, sum;
    } work;
    PetscCallMPI(MPI_Reduce_scatter_block((void *)sizes, &work, 1, MPIU_2INT, MPIU_MAXSUM_OP, comm));
    *max = work.max;
    *sum = work.sum;
  }
#else
  {
    PetscMPIInt size, rank;
    struct {
      PetscInt max, sum;
    } *work;
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(PetscMalloc1(size, &work));
    PetscCall(MPIU_Allreduce((void *)sizes, work, size, MPIU_2INT, MPIU_MAXSUM_OP, comm));
    *max = work[rank].max;
    *sum = work[rank].sum;
    PetscCall(PetscFree(work));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ----------------------------------------------------------------------------*/

#if defined(PETSC_HAVE_REAL___FLOAT128) || defined(PETSC_HAVE_REAL___FP16)
  #if defined(PETSC_HAVE_REAL___FLOAT128)
    #include <quadmath.h>
  #endif
MPI_Op MPIU_SUM___FP16___FLOAT128 = 0;
  #if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
MPI_Op MPIU_SUM = 0;
  #endif

PETSC_EXTERN void MPIAPI PetscSum_Local(void *in, void *out, PetscMPIInt *cnt, MPI_Datatype *datatype)
{
  PetscInt i, count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal *)in, *xout = (PetscReal *)out;
    for (i = 0; i < count; i++) xout[i] += xin[i];
  }
  #if defined(PETSC_HAVE_COMPLEX)
  else if (*datatype == MPIU_COMPLEX) {
    PetscComplex *xin = (PetscComplex *)in, *xout = (PetscComplex *)out;
    for (i = 0; i < count; i++) xout[i] += xin[i];
  }
  #endif
  #if defined(PETSC_HAVE_REAL___FLOAT128)
  else if (*datatype == MPIU___FLOAT128) {
    __float128 *xin = (__float128 *)in, *xout = (__float128 *)out;
    for (i = 0; i < count; i++) xout[i] += xin[i];
    #if defined(PETSC_HAVE_COMPLEX)
  } else if (*datatype == MPIU___COMPLEX128) {
    __complex128 *xin = (__complex128 *)in, *xout = (__complex128 *)out;
    for (i = 0; i < count; i++) xout[i] += xin[i];
    #endif
  }
  #endif
  #if defined(PETSC_HAVE_REAL___FP16)
  else if (*datatype == MPIU___FP16) {
    __fp16 *xin = (__fp16 *)in, *xout = (__fp16 *)out;
    for (i = 0; i < count; i++) xout[i] += xin[i];
  }
  #endif
  else {
  #if !defined(PETSC_HAVE_REAL___FLOAT128) && !defined(PETSC_HAVE_REAL___FP16)
    PetscCallAbort(MPI_COMM_SElF, (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_COMPLEX data types"));
  #elif !defined(PETSC_HAVE_REAL___FP16)
    PetscCallAbort(MPI_COMM_SELF, (*PetscErrorPrintf)("Can only handle MPIU_REAL, MPIU_COMPLEX, MPIU___FLOAT128, or MPIU___COMPLEX128 data types"));
  #elif !defined(PETSC_HAVE_REAL___FLOAT128)
    PetscCallAbort(MPI_COMM_SELF, (*PetscErrorPrintf)("Can only handle MPIU_REAL, MPIU_COMPLEX, or MPIU___FP16 data types"));
  #else
    PetscCallAbort(MPI_COMM_SELF, (*PetscErrorPrintf)("Can only handle MPIU_REAL, MPIU_COMPLEX, MPIU___FLOAT128, MPIU___COMPLEX128, or MPIU___FP16 data types"));
  #endif
    PETSCABORT(MPI_COMM_SELF, PETSC_ERR_ARG_WRONG);
  }
  PetscFunctionReturnVoid();
}
#endif

#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
MPI_Op MPIU_MAX = 0;
MPI_Op MPIU_MIN = 0;

PETSC_EXTERN void MPIAPI PetscMax_Local(void *in, void *out, PetscMPIInt *cnt, MPI_Datatype *datatype)
{
  PetscInt i, count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal *)in, *xout = (PetscReal *)out;
    for (i = 0; i < count; i++) xout[i] = PetscMax(xout[i], xin[i]);
  }
  #if defined(PETSC_HAVE_COMPLEX)
  else if (*datatype == MPIU_COMPLEX) {
    PetscComplex *xin = (PetscComplex *)in, *xout = (PetscComplex *)out;
    for (i = 0; i < count; i++) xout[i] = PetscRealPartComplex(xout[i]) < PetscRealPartComplex(xin[i]) ? xin[i] : xout[i];
  }
  #endif
  else {
    PetscCallAbort(MPI_COMM_SELF, (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_COMPLEX data types"));
    PETSCABORT(MPI_COMM_SELF, PETSC_ERR_ARG_WRONG);
  }
  PetscFunctionReturnVoid();
}

PETSC_EXTERN void MPIAPI PetscMin_Local(void *in, void *out, PetscMPIInt *cnt, MPI_Datatype *datatype)
{
  PetscInt i, count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal *)in, *xout = (PetscReal *)out;
    for (i = 0; i < count; i++) xout[i] = PetscMin(xout[i], xin[i]);
  }
  #if defined(PETSC_HAVE_COMPLEX)
  else if (*datatype == MPIU_COMPLEX) {
    PetscComplex *xin = (PetscComplex *)in, *xout = (PetscComplex *)out;
    for (i = 0; i < count; i++) xout[i] = PetscRealPartComplex(xout[i]) > PetscRealPartComplex(xin[i]) ? xin[i] : xout[i];
  }
  #endif
  else {
    PetscCallAbort(MPI_COMM_SELF, (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_SCALAR data (i.e. double or complex) types"));
    PETSCABORT(MPI_COMM_SELF, PETSC_ERR_ARG_WRONG);
  }
  PetscFunctionReturnVoid();
}
#endif

/*
   Private routine to delete internal tag/name counter storage when a communicator is freed.

   This is called by MPI, not by users. This is called by MPI_Comm_free() when the communicator that has this  data as an attribute is freed.

   Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_Counter_Attr_Delete_Fn(MPI_Comm comm, PetscMPIInt keyval, void *count_val, void *extra_state)
{
  PetscCommCounter      *counter = (PetscCommCounter *)count_val;
  struct PetscCommStash *comms   = counter->comms, *pcomm;

  PetscFunctionBegin;
  PetscCallMPI(PetscInfo(NULL, "Deleting counter data in an MPI_Comm %ld\n", (long)comm));
  PetscCallMPI(PetscFree(counter->iflags));
  while (comms) {
    PetscCallMPI(MPI_Comm_free(&comms->comm));
    pcomm = comms;
    comms = comms->next;
    PetscCall(PetscFree(pcomm));
  }
  PetscCallMPI(PetscFree(counter));
  PetscFunctionReturn(MPI_SUCCESS);
}

/*
  This is invoked on the outer comm as a result of either PetscCommDestroy() (via MPI_Comm_delete_attr) or when the user
  calls MPI_Comm_free().

  This is the only entry point for breaking the links between inner and outer comms.

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_InnerComm_Attr_Delete_Fn(MPI_Comm comm, PetscMPIInt keyval, void *attr_val, void *extra_state)
{
  union
  {
    MPI_Comm comm;
    void    *ptr;
  } icomm;

  PetscFunctionBegin;
  if (keyval != Petsc_InnerComm_keyval) SETERRMPI(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Unexpected keyval");
  icomm.ptr = attr_val;
  if (PetscDefined(USE_DEBUG)) {
    /* Error out if the inner/outer comms are not correctly linked through their Outer/InnterComm attributes */
    PetscMPIInt flg;
    union
    {
      MPI_Comm comm;
      void    *ptr;
    } ocomm;
    PetscCallMPI(MPI_Comm_get_attr(icomm.comm, Petsc_OuterComm_keyval, &ocomm, &flg));
    if (!flg) SETERRMPI(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Inner comm does not have OuterComm attribute");
    if (ocomm.comm != comm) SETERRMPI(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Inner comm's OuterComm attribute does not point to outer PETSc comm");
  }
  PetscCallMPI(MPI_Comm_delete_attr(icomm.comm, Petsc_OuterComm_keyval));
  PetscCallMPI(PetscInfo(NULL, "User MPI_Comm %ld is being unlinked from inner PETSc comm %ld\n", (long)comm, (long)icomm.comm));
  PetscFunctionReturn(MPI_SUCCESS);
}

/*
 * This is invoked on the inner comm when Petsc_InnerComm_Attr_Delete_Fn calls MPI_Comm_delete_attr().  It should not be reached any other way.
 */
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_OuterComm_Attr_Delete_Fn(MPI_Comm comm, PetscMPIInt keyval, void *attr_val, void *extra_state)
{
  PetscFunctionBegin;
  PetscCallMPI(PetscInfo(NULL, "Removing reference to PETSc communicator embedded in a user MPI_Comm %ld\n", (long)comm));
  PetscFunctionReturn(MPI_SUCCESS);
}

PETSC_EXTERN PetscMPIInt MPIAPI Petsc_ShmComm_Attr_Delete_Fn(MPI_Comm, PetscMPIInt, void *, void *);

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
PETSC_EXTERN PetscMPIInt PetscDataRep_extent_fn(MPI_Datatype, MPI_Aint *, void *);
PETSC_EXTERN PetscMPIInt PetscDataRep_read_conv_fn(void *, MPI_Datatype, PetscMPIInt, void *, MPI_Offset, void *);
PETSC_EXTERN PetscMPIInt PetscDataRep_write_conv_fn(void *, MPI_Datatype, PetscMPIInt, void *, MPI_Offset, void *);
#endif

PetscMPIInt PETSC_MPI_ERROR_CLASS = MPI_ERR_LASTCODE, PETSC_MPI_ERROR_CODE;

PETSC_INTERN int    PetscGlobalArgc;
PETSC_INTERN char **PetscGlobalArgs;
int                 PetscGlobalArgc = 0;
char              **PetscGlobalArgs = NULL;
PetscSegBuffer      PetscCitationsList;

PetscErrorCode PetscCitationsInitialize(void)
{
  PetscFunctionBegin;
  PetscCall(PetscSegBufferCreate(1, 10000, &PetscCitationsList));

  PetscCall(PetscCitationsRegister("@TechReport{petsc-user-ref,\n\
  Author = {Satish Balay and Shrirang Abhyankar and Mark~F. Adams and Steven Benson and Jed Brown\n\
    and Peter Brune and Kris Buschelman and Emil Constantinescu and Lisandro Dalcin and Alp Dener\n\
    and Victor Eijkhout and Jacob Faibussowitsch and William~D. Gropp and V\'{a}clav Hapla and Tobin Isaac and Pierre Jolivet\n\
    and Dmitry Karpeev and Dinesh Kaushik and Matthew~G. Knepley and Fande Kong and Scott Kruger\n\
    and Dave~A. May and Lois Curfman McInnes and Richard Tran Mills and Lawrence Mitchell and Todd Munson\n\
    and Jose~E. Roman and Karl Rupp and Patrick Sanan and Jason Sarich and Barry~F. Smith\n\
    and Stefano Zampini and Hong Zhang and Hong Zhang and Junchao Zhang},\n\
  Title = {{PETSc/TAO} Users Manual},\n\
  Number = {ANL-21/39 - Revision 3.19},\n\
  Institution = {Argonne National Laboratory},\n\
  Year = {2023}\n}\n",
                                   NULL));

  PetscCall(PetscCitationsRegister("@InProceedings{petsc-efficient,\n\
  Author = {Satish Balay and William D. Gropp and Lois Curfman McInnes and Barry F. Smith},\n\
  Title = {Efficient Management of Parallelism in Object Oriented Numerical Software Libraries},\n\
  Booktitle = {Modern Software Tools in Scientific Computing},\n\
  Editor = {E. Arge and A. M. Bruaset and H. P. Langtangen},\n\
  Pages = {163--202},\n\
  Publisher = {Birkh{\\\"{a}}user Press},\n\
  Year = {1997}\n}\n",
                                   NULL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static char programname[PETSC_MAX_PATH_LEN] = ""; /* HP includes entire path in name */

PetscErrorCode PetscSetProgramName(const char name[])
{
  PetscFunctionBegin;
  PetscCall(PetscStrncpy(programname, name, sizeof(programname)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscGetProgramName - Gets the name of the running program.

    Not Collective

    Input Parameter:
.   len - length of the string name

    Output Parameter:
.   name - the name of the running program, provide a string of length `PETSC_MAX_PATH_LEN`

   Level: advanced

@*/
PetscErrorCode PetscGetProgramName(char name[], size_t len)
{
  PetscFunctionBegin;
  PetscCall(PetscStrncpy(name, programname, len));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscGetArgs - Allows you to access the raw command line arguments anywhere
     after PetscInitialize() is called but before `PetscFinalize()`.

   Not Collective

   Output Parameters:
+  argc - count of number of command line arguments
-  args - the command line arguments

   Level: intermediate

   Notes:
      This is usually used to pass the command line arguments into other libraries
   that are called internally deep in PETSc or the application.

      The first argument contains the program name as is normal for C arguments.

.seealso: `PetscFinalize()`, `PetscInitializeFortran()`, `PetscGetArguments()`
@*/
PetscErrorCode PetscGetArgs(int *argc, char ***args)
{
  PetscFunctionBegin;
  PetscCheck(PetscInitializeCalled || !PetscFinalizeCalled, PETSC_COMM_SELF, PETSC_ERR_ORDER, "You must call after PetscInitialize() but before PetscFinalize()");
  *argc = PetscGlobalArgc;
  *args = PetscGlobalArgs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscGetArguments - Allows you to access the  command line arguments anywhere
     after `PetscInitialize()` is called but before `PetscFinalize()`.

   Not Collective

   Output Parameter:
.  args - the command line arguments

   Level: intermediate

   Notes:
      This does NOT start with the program name and IS null terminated (final arg is void)

.seealso: `PetscFinalize()`, `PetscInitializeFortran()`, `PetscGetArgs()`, `PetscFreeArguments()`
@*/
PetscErrorCode PetscGetArguments(char ***args)
{
  PetscInt i, argc = PetscGlobalArgc;

  PetscFunctionBegin;
  PetscCheck(PetscInitializeCalled || !PetscFinalizeCalled, PETSC_COMM_SELF, PETSC_ERR_ORDER, "You must call after PetscInitialize() but before PetscFinalize()");
  if (!argc) {
    *args = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMalloc1(argc, args));
  for (i = 0; i < argc - 1; i++) PetscCall(PetscStrallocpy(PetscGlobalArgs[i + 1], &(*args)[i]));
  (*args)[argc - 1] = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscFreeArguments - Frees the memory obtained with `PetscGetArguments()`

   Not Collective

   Output Parameter:
.  args - the command line arguments

   Level: intermediate

.seealso: `PetscFinalize()`, `PetscInitializeFortran()`, `PetscGetArgs()`, `PetscGetArguments()`
@*/
PetscErrorCode PetscFreeArguments(char **args)
{
  PetscFunctionBegin;
  if (args) {
    PetscInt i = 0;

    while (args[i]) PetscCall(PetscFree(args[i++]));
    PetscCall(PetscFree(args));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(HAVE_SAWS)
  #include <petscconfiginfo.h>

PETSC_INTERN PetscErrorCode PetscInitializeSAWs(const char help[])
{
  PetscFunctionBegin;
  if (!PetscGlobalRank) {
    char      cert[PETSC_MAX_PATH_LEN], root[PETSC_MAX_PATH_LEN], *intro, programname[64], *appline, *options, version[64];
    int       port;
    PetscBool flg, rootlocal = PETSC_FALSE, flg2, selectport = PETSC_FALSE;
    size_t    applinelen, introlen;
    char      sawsurl[256];

    PetscCall(PetscOptionsHasName(NULL, NULL, "-saws_log", &flg));
    if (flg) {
      char sawslog[PETSC_MAX_PATH_LEN];

      PetscCall(PetscOptionsGetString(NULL, NULL, "-saws_log", sawslog, sizeof(sawslog), NULL));
      if (sawslog[0]) {
        PetscCallSAWs(SAWs_Set_Use_Logfile, (sawslog));
      } else {
        PetscCallSAWs(SAWs_Set_Use_Logfile, (NULL));
      }
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-saws_https", cert, sizeof(cert), &flg));
    if (flg) PetscCallSAWs(SAWs_Set_Use_HTTPS, (cert));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-saws_port_auto_select", &selectport, NULL));
    if (selectport) {
      PetscCallSAWs(SAWs_Get_Available_Port, (&port));
      PetscCallSAWs(SAWs_Set_Port, (port));
    } else {
      PetscCall(PetscOptionsGetInt(NULL, NULL, "-saws_port", &port, &flg));
      if (flg) PetscCallSAWs(SAWs_Set_Port, (port));
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-saws_root", root, sizeof(root), &flg));
    if (flg) {
      PetscCallSAWs(SAWs_Set_Document_Root, (root));
      PetscCall(PetscStrcmp(root, ".", &rootlocal));
    } else {
      PetscCall(PetscOptionsHasName(NULL, NULL, "-saws_options", &flg));
      if (flg) {
        PetscCall(PetscStrreplace(PETSC_COMM_WORLD, "${PETSC_DIR}/share/petsc/saws", root, sizeof(root)));
        PetscCallSAWs(SAWs_Set_Document_Root, (root));
      }
    }
    PetscCall(PetscOptionsHasName(NULL, NULL, "-saws_local", &flg2));
    if (flg2) {
      char jsdir[PETSC_MAX_PATH_LEN];
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_SUP, "-saws_local option requires -saws_root option");
      PetscCall(PetscSNPrintf(jsdir, sizeof(jsdir), "%s/js", root));
      PetscCall(PetscTestDirectory(jsdir, 'r', &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "-saws_local option requires js directory in root directory");
      PetscCallSAWs(SAWs_Push_Local_Header, ());
    }
    PetscCall(PetscGetProgramName(programname, sizeof(programname)));
    PetscCall(PetscStrlen(help, &applinelen));
    introlen = 4096 + applinelen;
    applinelen += 1024;
    PetscCall(PetscMalloc(applinelen, &appline));
    PetscCall(PetscMalloc(introlen, &intro));

    if (rootlocal) {
      PetscCall(PetscSNPrintf(appline, applinelen, "%s.c.html", programname));
      PetscCall(PetscTestFile(appline, 'r', &rootlocal));
    }
    PetscCall(PetscOptionsGetAll(NULL, &options));
    if (rootlocal && help) {
      PetscCall(PetscSNPrintf(appline, applinelen, "<center> Running <a href=\"%s.c.html\">%s</a> %s</center><br><center><pre>%s</pre></center><br>\n", programname, programname, options, help));
    } else if (help) {
      PetscCall(PetscSNPrintf(appline, applinelen, "<center>Running %s %s</center><br><center><pre>%s</pre></center><br>", programname, options, help));
    } else {
      PetscCall(PetscSNPrintf(appline, applinelen, "<center> Running %s %s</center><br>\n", programname, options));
    }
    PetscCall(PetscFree(options));
    PetscCall(PetscGetVersion(version, sizeof(version)));
    PetscCall(PetscSNPrintf(intro, introlen,
                            "<body>\n"
                            "<center><h2> <a href=\"https://petsc.org/\">PETSc</a> Application Web server powered by <a href=\"https://bitbucket.org/saws/saws\">SAWs</a> </h2></center>\n"
                            "<center>This is the default PETSc application dashboard, from it you can access any published PETSc objects or logging data</center><br><center>%s configured with %s</center><br>\n"
                            "%s",
                            version, petscconfigureoptions, appline));
    PetscCallSAWs(SAWs_Push_Body, ("index.html", 0, intro));
    PetscCall(PetscFree(intro));
    PetscCall(PetscFree(appline));
    if (selectport) {
      PetscBool silent;

      /* another process may have grabbed the port so keep trying */
      while (SAWs_Initialize()) {
        PetscCallSAWs(SAWs_Get_Available_Port, (&port));
        PetscCallSAWs(SAWs_Set_Port, (port));
      }

      PetscCall(PetscOptionsGetBool(NULL, NULL, "-saws_port_auto_select_silent", &silent, NULL));
      if (!silent) {
        PetscCallSAWs(SAWs_Get_FullURL, (sizeof(sawsurl), sawsurl));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Point your browser to %s for SAWs\n", sawsurl));
      }
    } else {
      PetscCallSAWs(SAWs_Initialize, ());
    }
    PetscCall(PetscCitationsRegister("@TechReport{ saws,\n"
                                     "  Author = {Matt Otten and Jed Brown and Barry Smith},\n"
                                     "  Title  = {Scientific Application Web Server (SAWs) Users Manual},\n"
                                     "  Institution = {Argonne National Laboratory},\n"
                                     "  Year   = 2013\n}\n",
                                     NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/* Things must be done before MPI_Init() when MPI is not yet initialized, and can be shared between C init and Fortran init */
PETSC_INTERN PetscErrorCode PetscPreMPIInit_Private(void)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_HWLOC_SOLARIS_BUG)
  /* see MPI.py for details on this bug */
  (void)setenv("HWLOC_COMPONENTS", "-x86", 1);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(HAVE_ADIOS)
  #include <adios.h>
  #include <adios_read.h>
int64_t Petsc_adios_group;
#endif
#if PetscDefined(HAVE_OPENMP)
  #include <omp.h>
PetscInt PetscNumOMPThreads;
#endif

#include <petsc/private/deviceimpl.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
// REMOVE ME
cudaStream_t PetscDefaultCudaStream = NULL;
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
// REMOVE ME
hipStream_t PetscDefaultHipStream = NULL;
#endif

#if PetscDefined(HAVE_DLFCN_H)
  #include <dlfcn.h>
#endif
#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#endif
#if PetscDefined(HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode PetscViennaCLInit(void);
PetscBool                   PetscViennaCLSynchronize = PETSC_FALSE;
#endif

PetscBool PetscCIEnabled = PETSC_FALSE, PetscCIEnabledPortableErrorOutput = PETSC_FALSE;

/*
  PetscInitialize_Common  - shared code between C and Fortran initialization

  prog:     program name
  file:     optional PETSc database file name. Might be in Fortran string format when 'ftn' is true
  help:     program help message
  ftn:      is it called from Fortran initialization (petscinitializef_)?
  readarguments,len: used when fortran is true
*/
PETSC_INTERN PetscErrorCode PetscInitialize_Common(const char *prog, const char *file, const char *help, PetscBool ftn, PetscBool readarguments, PetscInt len)
{
  PetscMPIInt size;
  PetscBool   flg = PETSC_TRUE;
  char        hostname[256];

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(PETSC_SUCCESS);
  /* these must be initialized in a routine, not as a constant declaration */
  PETSC_STDOUT = stdout;
  PETSC_STDERR = stderr;

  /* PetscCall can be used from now */
  PetscErrorHandlingInitialized = PETSC_TRUE;

  /*
      The checking over compatible runtime libraries is complicated by the MPI ABI initiative
      https://wiki.mpich.org/mpich/index.php/ABI_Compatibility_Initiative which started with
        MPICH v3.1 (Released February 2014)
        IBM MPI v2.1 (December 2014)
        Intel MPI Library v5.0 (2014)
        Cray MPT v7.0.0 (June 2014)
      As of July 31, 2017 the ABI number still appears to be 12, that is all of the versions
      listed above and since that time are compatible.

      Unfortunately the MPI ABI initiative has not defined a way to determine the ABI number
      at compile time or runtime. Thus we will need to systematically track the allowed versions
      and how they are represented in the mpi.h and MPI_Get_library_version() output in order
      to perform the checking.

      Currently we only check for pre MPI ABI versions (and packages that do not follow the MPI ABI).

      Questions:

        Should the checks for ABI incompatibility be only on the major version number below?
        Presumably the output to stderr will be removed before a release.
  */

#if defined(PETSC_HAVE_MPI_GET_LIBRARY_VERSION)
  {
    char        mpilibraryversion[MPI_MAX_LIBRARY_VERSION_STRING];
    PetscMPIInt mpilibraryversionlength;

    PetscCallMPI(MPI_Get_library_version(mpilibraryversion, &mpilibraryversionlength));
    /* check for MPICH versions before MPI ABI initiative */
  #if defined(MPICH_VERSION)
    #if MPICH_NUMVERSION < 30100000
    {
      char     *ver, *lf;
      PetscBool flg = PETSC_FALSE;

      PetscCall(PetscStrstr(mpilibraryversion, "MPICH Version:", &ver));
      if (ver) {
        PetscCall(PetscStrchr(ver, '\n', &lf));
        if (lf) {
          *lf = 0;
          PetscCall(PetscStrendswith(ver, MPICH_VERSION, &flg));
        }
      }
      if (!flg) {
        PetscCall(PetscInfo(NULL, "PETSc warning --- MPICH library version \n%s does not match what PETSc was compiled with %s.\n", mpilibraryversion, MPICH_VERSION));
        flg = PETSC_TRUE;
      }
    }
    #endif
      /* check for OpenMPI version, it is not part of the MPI ABI initiative (is it part of another initiative that needs to be handled?) */
  #elif defined(OMPI_MAJOR_VERSION)
    {
      char     *ver, bs[MPI_MAX_LIBRARY_VERSION_STRING], *bsf;
      PetscBool flg                                              = PETSC_FALSE;
    #define PSTRSZ 2
      char      ompistr1[PSTRSZ][MPI_MAX_LIBRARY_VERSION_STRING] = {"Open MPI", "FUJITSU MPI"};
      char      ompistr2[PSTRSZ][MPI_MAX_LIBRARY_VERSION_STRING] = {"v", "Library "};
      int       i;
      for (i = 0; i < PSTRSZ; i++) {
        PetscCall(PetscStrstr(mpilibraryversion, ompistr1[i], &ver));
        if (ver) {
          PetscCall(PetscSNPrintf(bs, MPI_MAX_LIBRARY_VERSION_STRING, "%s%d.%d", ompistr2[i], OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION));
          PetscCall(PetscStrstr(ver, bs, &bsf));
          if (bsf) flg = PETSC_TRUE;
          break;
        }
      }
      if (!flg) {
        PetscCall(PetscInfo(NULL, "PETSc warning --- Open MPI library version \n%s does not match what PETSc was compiled with %d.%d.\n", mpilibraryversion, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION));
        flg = PETSC_TRUE;
      }
    }
  #endif
  }
#endif

#if defined(PETSC_HAVE_DLADDR) && !(defined(__cray__) && defined(__clang__))
  /* These symbols are currently in the OpenMPI and MPICH libraries; they may not always be, in that case the test will simply not detect the problem */
  PetscCheck(!dlsym(RTLD_DEFAULT, "ompi_mpi_init") || !dlsym(RTLD_DEFAULT, "MPID_Abort"), PETSC_COMM_SELF, PETSC_ERR_MPI_LIB_INCOMP, "Application was linked against both OpenMPI and MPICH based MPI libraries and will not run correctly");
#endif

  /* on Windows - set printf to default to printing 2 digit exponents */
#if defined(PETSC_HAVE__SET_OUTPUT_FORMAT)
  _set_output_format(_TWO_DIGIT_EXPONENT);
#endif

  PetscCall(PetscOptionsCreateDefault());

  PetscFinalizeCalled = PETSC_FALSE;

  PetscCall(PetscSetProgramName(prog));
  PetscCall(PetscSpinlockCreate(&PetscViewerASCIISpinLockOpen));
  PetscCall(PetscSpinlockCreate(&PetscViewerASCIISpinLockStdout));
  PetscCall(PetscSpinlockCreate(&PetscViewerASCIISpinLockStderr));
  PetscCall(PetscSpinlockCreate(&PetscCommSpinLock));

  if (PETSC_COMM_WORLD == MPI_COMM_NULL) PETSC_COMM_WORLD = MPI_COMM_WORLD;
  PetscCallMPI(MPI_Comm_set_errhandler(PETSC_COMM_WORLD, MPI_ERRORS_RETURN));

  if (PETSC_MPI_ERROR_CLASS == MPI_ERR_LASTCODE) {
    PetscCallMPI(MPI_Add_error_class(&PETSC_MPI_ERROR_CLASS));
    PetscCallMPI(MPI_Add_error_code(PETSC_MPI_ERROR_CLASS, &PETSC_MPI_ERROR_CODE));
  }

  /* Done after init due to a bug in MPICH-GM? */
  PetscCall(PetscErrorPrintfInitialize());

  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &PetscGlobalRank));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &PetscGlobalSize));

  MPIU_BOOL        = MPI_INT;
  MPIU_ENUM        = MPI_INT;
  MPIU_FORTRANADDR = (sizeof(void *) == sizeof(int)) ? MPI_INT : MPIU_INT64;
  if (sizeof(size_t) == sizeof(unsigned)) MPIU_SIZE_T = MPI_UNSIGNED;
  else if (sizeof(size_t) == sizeof(unsigned long)) MPIU_SIZE_T = MPI_UNSIGNED_LONG;
#if defined(PETSC_SIZEOF_LONG_LONG)
  else if (sizeof(size_t) == sizeof(unsigned long long)) MPIU_SIZE_T = MPI_UNSIGNED_LONG_LONG;
#endif
  else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS, "Could not find MPI type for size_t");

    /*
     Initialized the global complex variable; this is because with
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
#if defined(PETSC_HAVE_COMPLEX)
  {
  #if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_REAL___FLOAT128)
    PetscComplex ic(0.0, 1.0);
    PETSC_i = ic;
  #else
    PETSC_i = _Complex_I;
  #endif
  }
#endif /* PETSC_HAVE_COMPLEX */

  /*
     Create the PETSc MPI reduction operator that sums of the first
     half of the entries and maxes the second half.
  */
  PetscCallMPI(MPI_Op_create(MPIU_MaxSum_Local, 1, &MPIU_MAXSUM_OP));

#if defined(PETSC_HAVE_REAL___FLOAT128)
  PetscCallMPI(MPI_Type_contiguous(2, MPI_DOUBLE, &MPIU___FLOAT128));
  PetscCallMPI(MPI_Type_commit(&MPIU___FLOAT128));
  #if defined(PETSC_HAVE_COMPLEX)
  PetscCallMPI(MPI_Type_contiguous(4, MPI_DOUBLE, &MPIU___COMPLEX128));
  PetscCallMPI(MPI_Type_commit(&MPIU___COMPLEX128));
  #endif
#endif
#if defined(PETSC_HAVE_REAL___FP16)
  PetscCallMPI(MPI_Type_contiguous(2, MPI_CHAR, &MPIU___FP16));
  PetscCallMPI(MPI_Type_commit(&MPIU___FP16));
#endif

#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
  PetscCallMPI(MPI_Op_create(PetscSum_Local, 1, &MPIU_SUM));
  PetscCallMPI(MPI_Op_create(PetscMax_Local, 1, &MPIU_MAX));
  PetscCallMPI(MPI_Op_create(PetscMin_Local, 1, &MPIU_MIN));
#elif defined(PETSC_HAVE_REAL___FLOAT128) || defined(PETSC_HAVE_REAL___FP16)
  PetscCallMPI(MPI_Op_create(PetscSum_Local, 1, &MPIU_SUM___FP16___FLOAT128));
#endif

  PetscCallMPI(MPI_Type_contiguous(2, MPIU_SCALAR, &MPIU_2SCALAR));
  PetscCallMPI(MPI_Op_create(PetscGarbageKeySortedIntersect, 1, &Petsc_Garbage_SetIntersectOp));
  PetscCallMPI(MPI_Type_commit(&MPIU_2SCALAR));

  /* create datatypes used by MPIU_MAXLOC, MPIU_MINLOC and PetscSplitReduction_Op */
#if !defined(PETSC_HAVE_MPIUNI)
  {
    PetscMPIInt  blockSizes[2]   = {1, 1};
    MPI_Aint     blockOffsets[2] = {offsetof(struct petsc_mpiu_real_int, v), offsetof(struct petsc_mpiu_real_int, i)};
    MPI_Datatype blockTypes[2]   = {MPIU_REAL, MPIU_INT}, tmpStruct;

    PetscCallMPI(MPI_Type_create_struct(2, blockSizes, blockOffsets, blockTypes, &tmpStruct));
    PetscCallMPI(MPI_Type_create_resized(tmpStruct, 0, sizeof(struct petsc_mpiu_real_int), &MPIU_REAL_INT));
    PetscCallMPI(MPI_Type_free(&tmpStruct));
    PetscCallMPI(MPI_Type_commit(&MPIU_REAL_INT));
  }
  {
    PetscMPIInt  blockSizes[2]   = {1, 1};
    MPI_Aint     blockOffsets[2] = {offsetof(struct petsc_mpiu_scalar_int, v), offsetof(struct petsc_mpiu_scalar_int, i)};
    MPI_Datatype blockTypes[2]   = {MPIU_SCALAR, MPIU_INT}, tmpStruct;

    PetscCallMPI(MPI_Type_create_struct(2, blockSizes, blockOffsets, blockTypes, &tmpStruct));
    PetscCallMPI(MPI_Type_create_resized(tmpStruct, 0, sizeof(struct petsc_mpiu_scalar_int), &MPIU_SCALAR_INT));
    PetscCallMPI(MPI_Type_free(&tmpStruct));
    PetscCallMPI(MPI_Type_commit(&MPIU_SCALAR_INT));
  }
#endif

#if defined(PETSC_USE_64BIT_INDICES)
  PetscCallMPI(MPI_Type_contiguous(2, MPIU_INT, &MPIU_2INT));
  PetscCallMPI(MPI_Type_commit(&MPIU_2INT));
#endif
  PetscCallMPI(MPI_Type_contiguous(4, MPI_INT, &MPI_4INT));
  PetscCallMPI(MPI_Type_commit(&MPI_4INT));
  PetscCallMPI(MPI_Type_contiguous(4, MPIU_INT, &MPIU_4INT));
  PetscCallMPI(MPI_Type_commit(&MPIU_4INT));

  /*
     Attributes to be set on PETSc communicators
  */
  PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_Counter_Attr_Delete_Fn, &Petsc_Counter_keyval, (void *)0));
  PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_InnerComm_Attr_Delete_Fn, &Petsc_InnerComm_keyval, (void *)0));
  PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_OuterComm_Attr_Delete_Fn, &Petsc_OuterComm_keyval, (void *)0));
  PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_ShmComm_Attr_Delete_Fn, &Petsc_ShmComm_keyval, (void *)0));
  PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &Petsc_CreationIdx_keyval, (void *)0));
  PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &Petsc_Garbage_HMap_keyval, (void *)0));

#if defined(PETSC_HAVE_FORTRAN)
  if (ftn) PetscCall(PetscInitFortran_Private(readarguments, file, len));
  else
#endif
    PetscCall(PetscOptionsInsert(NULL, &PetscGlobalArgc, &PetscGlobalArgs, file));

  /* call a second time so it can look in the options database */
  PetscCall(PetscErrorPrintfInitialize());

  /*
     Check system options and print help
  */
  PetscCall(PetscOptionsCheckInitial_Private(help));

  /*
    Creates the logging data structures; this is enabled even if logging is not turned on
    This is the last thing we do before returning to the user code to prevent having the
    logging numbers contaminated by any startup time associated with MPI
  */
#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogInitialize());
#endif

  /*
   Initialize PetscDevice and PetscDeviceContext

   Note to any future devs thinking of moving this, proper initialization requires:
   1. MPI initialized
   2. Options DB initialized
   3. Petsc error handling initialized, specifically signal handlers. This expects to set up
      its own SIGSEV handler via the push/pop interface.
   4. Logging initialized
  */
  PetscCall(PetscDeviceInitializeFromOptions_Internal(PETSC_COMM_WORLD));

#if PetscDefined(HAVE_VIENNACL)
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-log_summary", &flg));
  if (!flg) PetscCall(PetscOptionsHasName(NULL, NULL, "-log_view", &flg));
  if (!flg) PetscCall(PetscOptionsGetBool(NULL, NULL, "-viennacl_synchronize", &flg, NULL));
  PetscViennaCLSynchronize = flg;
  PetscCall(PetscViennaCLInit());
#endif

  PetscCall(PetscCitationsInitialize());

#if defined(PETSC_HAVE_SAWS)
  PetscCall(PetscInitializeSAWs(ftn ? NULL : help));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-stack_view", &flg));
  if (flg) PetscCall(PetscStackViewSAWs());
#endif

  /*
     Load the dynamic libraries (on machines that support them), this registers all
     the solvers etc. (On non-dynamic machines this initializes the PetscDraw and PetscViewer classes)
  */
  PetscCall(PetscInitialize_DynamicLibraries());

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscInfo(NULL, "PETSc successfully started: number of processors = %d\n", size));
  PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
  PetscCall(PetscInfo(NULL, "Running on machine: %s\n", hostname));
#if defined(PETSC_HAVE_OPENMP)
  {
    PetscBool omp_view_flag;
    char     *threads = getenv("OMP_NUM_THREADS");

    if (threads) {
      PetscCall(PetscInfo(NULL, "Number of OpenMP threads %s (as given by OMP_NUM_THREADS)\n", threads));
      (void)sscanf(threads, "%" PetscInt_FMT, &PetscNumOMPThreads);
    } else {
      PetscNumOMPThreads = (PetscInt)omp_get_max_threads();
      PetscCall(PetscInfo(NULL, "Number of OpenMP threads %" PetscInt_FMT " (as given by omp_get_max_threads())\n", PetscNumOMPThreads));
    }
    PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "OpenMP options", "Sys");
    PetscCall(PetscOptionsInt("-omp_num_threads", "Number of OpenMP threads to use (can also use environmental variable OMP_NUM_THREADS", "None", PetscNumOMPThreads, &PetscNumOMPThreads, &flg));
    PetscCall(PetscOptionsName("-omp_view", "Display OpenMP number of threads", NULL, &omp_view_flag));
    PetscOptionsEnd();
    if (flg) {
      PetscCall(PetscInfo(NULL, "Number of OpenMP theads %" PetscInt_FMT " (given by -omp_num_threads)\n", PetscNumOMPThreads));
      omp_set_num_threads((int)PetscNumOMPThreads);
    }
    if (omp_view_flag) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OpenMP: number of threads %" PetscInt_FMT "\n", PetscNumOMPThreads));
  }
#endif

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
  /*
      Tell MPI about our own data representation converter, this would/should be used if extern32 is not supported by the MPI

      Currently not used because it is not supported by MPICH.
  */
  if (!PetscBinaryBigEndian()) PetscCallMPI(MPI_Register_datarep((char *)"petsc", PetscDataRep_read_conv_fn, PetscDataRep_write_conv_fn, PetscDataRep_extent_fn, NULL));
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
  PetscCall(PetscFPTCreate(10000));
#endif

#if defined(PETSC_HAVE_HWLOC)
  {
    PetscViewer viewer;
    PetscCall(PetscOptionsGetViewer(PETSC_COMM_WORLD, NULL, NULL, "-process_view", &viewer, NULL, &flg));
    if (flg) {
      PetscCall(PetscProcessPlacementView(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
#endif

  flg = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-viewfromoptions", &flg, NULL));
  if (!flg) PetscCall(PetscOptionsPushGetViewerOff(PETSC_TRUE));

#if defined(PETSC_HAVE_ADIOS)
  PetscCallExternal(adios_init_noxml, PETSC_COMM_WORLD);
  PetscCallExternal(adios_declare_group, &Petsc_adios_group, "PETSc", "", adios_stat_default);
  PetscCallExternal(adios_select_method, Petsc_adios_group, "MPI", "", "");
  PetscCallExternal(adios_read_init_method, ADIOS_READ_METHOD_BP, PETSC_COMM_WORLD, "");
#endif

#if defined(__VALGRIND_H)
  PETSC_RUNNING_ON_VALGRIND = RUNNING_ON_VALGRIND ? PETSC_TRUE : PETSC_FALSE;
  #if defined(PETSC_USING_DARWIN) && defined(PETSC_BLASLAPACK_SDOT_RETURNS_DOUBLE)
  if (PETSC_RUNNING_ON_VALGRIND) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "WARNING: Running valgrind with the MacOS native BLAS and LAPACK can fail. If it fails suggest configuring with --download-fblaslapack or --download-f2cblaslapack"));
  #endif
#endif
  /*
      Set flag that we are completely initialized
  */
  PetscInitializeCalled = PETSC_TRUE;

  PetscCall(PetscOptionsHasName(NULL, NULL, "-python", &flg));
  if (flg) PetscCall(PetscPythonInitialize(NULL, NULL));

  PetscCall(PetscOptionsHasName(NULL, NULL, "-mpi_linear_solver_server", &flg));
  if (PetscDefined(USE_SINGLE_LIBRARY) && flg) PetscCall(PCMPIServerBegin());
  else PetscCheck(!flg, PETSC_COMM_WORLD, PETSC_ERR_SUP, "PETSc configured using -with-single-library=0; -mpi_linear_solver_server not supported in that case");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscInitialize - Initializes the PETSc database and MPI.
   `PetscInitialize()` calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of
   your program -- usually the very first line!

   Collective on `MPI_COMM_WORLD` or `PETSC_COMM_WORLD` if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, append ":yaml" to filename to specify YAML options format.
          Use NULL or empty string to not check for code specific file.
          Also checks ~/.petscrc, .petscrc and petscrc.
          Use -skip_petscrc in the code specific file (or command line) to skip ~/.petscrc, .petscrc and petscrc files.
-  help - [optional] Help message to print, use NULL for no message

   If you wish PETSc code to run ONLY on a subcommunicator of `MPI_COMM_WORLD`, create that
   communicator first and assign it to `PETSC_COMM_WORLD` BEFORE calling `PetscInitialize()`. Thus if you are running a
   four process job and two processes will run PETSc and have `PetscInitialize()` and PetscFinalize() and two process will not,
   then do this. If ALL processes in the job are using `PetscInitialize()` and `PetscFinalize()` then you don't need to do this, even
   if different subcommunicators of the job are doing different things with PETSc.

   Options Database Keys:
+  -help [intro] - prints help method for each option; if intro is given the program stops after printing the introductory help message
.  -start_in_debugger [noxterm,dbx,xdb,gdb,...] - Starts program in debugger
.  -on_error_attach_debugger [noxterm,dbx,xdb,gdb,...] - Starts debugger when error detected
.  -on_error_emacs <machinename> - causes emacsclient to jump to error file
.  -on_error_abort - calls `abort()` when error detected (no traceback)
.  -on_error_mpiabort - calls `MPI_abort()` when error detected
.  -error_output_stdout - prints PETSc error messages to stdout instead of the default stderr
.  -error_output_none - does not print the error messages (but handles errors in the same way as if this was not called)
.  -debugger_ranks [rank1,rank2,...] - Indicates ranks to start in debugger
.  -debugger_pause [sleeptime] (in seconds) - Pauses debugger
.  -stop_for_debugger - Print message on how to attach debugger manually to
                        process and wait (-debugger_pause) seconds for attachment
.  -malloc - Indicates use of PETSc error-checking malloc (on by default for debug version of libraries) (deprecated, use -malloc_debug)
.  -malloc no - Indicates not to use error-checking malloc (deprecated, use -malloc_debug no)
.  -malloc_debug - check for memory corruption at EVERY malloc or free, see `PetscMallocSetDebug()`
.  -malloc_dump - prints a list of all unfreed memory at the end of the run
.  -malloc_test - like -malloc_dump -malloc_debug, but only active for debugging builds, ignored in optimized build. May want to set in PETSC_OPTIONS environmental variable
.  -malloc_view - show a list of all allocated memory during `PetscFinalize()`
.  -malloc_view_threshold <t> - only list memory allocations of size greater than t with -malloc_view
.  -malloc_requested_size - malloc logging will record the requested size rather than size after alignment
.  -fp_trap - Stops on floating point exceptions
.  -no_signal_handler - Indicates not to trap error signals
.  -shared_tmp - indicates /tmp directory is shared by all processors
.  -not_shared_tmp - each processor has own /tmp
.  -tmp - alternative name of /tmp directory
.  -get_total_flops - returns total flops done by all processors
-  -memory_view - Print memory usage at end of run

   Options Database Keys for Option Database:
+  -skip_petscrc - skip the default option files ~/.petscrc, .petscrc, petscrc
.  -options_monitor - monitor all set options to standard output for the whole program run
-  -options_monitor_cancel - cancel options monitoring hard-wired using `PetscOptionsMonitorSet()`

   Options -options_monitor_{all,cancel} are
   position-independent and apply to all options set since the PETSc start.
   They can be used also in option files.

   See `PetscOptionsMonitorSet()` to do monitoring programmatically.

   Options Database Keys for Profiling:
   See Users-Manual: ch_profiling for details.
+  -info [filename][:[~]<list,of,classnames>[:[~]self]] - Prints verbose information. See `PetscInfo()`.
.  -log_sync - Enable barrier synchronization for all events. This option is useful to debug imbalance within each event,
        however it slows things down and gives a distorted view of the overall runtime.
.  -log_trace [filename] - Print traces of all PETSc calls to the screen (useful to determine where a program
        hangs without running in the debugger).  See `PetscLogTraceBegin()`.
.  -log_view [:filename:format] - Prints summary of flop and timing information to screen or file, see `PetscLogView()`.
.  -log_view_memory - Includes in the summary from -log_view the memory used in each event, see `PetscLogView()`.
.  -log_view_gpu_time - Includes in the summary from -log_view the time used in each GPU kernel, see `PetscLogView().
.  -log_summary [filename] - (Deprecated, use -log_view) Prints summary of flop and timing information to screen. If the filename is specified the
        summary is written to the file.  See PetscLogView().
.  -log_exclude: <vec,mat,pc,ksp,snes> - excludes subset of object classes from logging
.  -log_all [filename] - Logs extensive profiling information  See `PetscLogDump()`.
.  -log [filename] - Logs basic profiline information  See `PetscLogDump()`.
.  -log_mpe [filename] - Creates a logfile viewable by the utility Jumpshot (in MPICH distribution)
.  -viewfromoptions on,off - Enable or disable `XXXSetFromOptions()` calls, for applications with many small solves turn this off
-  -check_pointer_intensity 0,1,2 - if pointers are checked for validity (debug version only), using 0 will result in faster code

    Only one of -log_trace, -log_view, -log_all, -log, or -log_mpe may be used at a time

   Options Database Keys for SAWs:
+  -saws_port <portnumber> - port number to publish SAWs data, default is 8080
.  -saws_port_auto_select - have SAWs select a new unique port number where it publishes the data, the URL is printed to the screen
                            this is useful when you are running many jobs that utilize SAWs at the same time
.  -saws_log <filename> - save a log of all SAWs communication
.  -saws_https <certificate file> - have SAWs use HTTPS instead of HTTP
-  -saws_root <directory> - allow SAWs to have access to the given directory to search for requested resources and files

   Environmental Variables:
+   `PETSC_TMP` - alternative tmp directory
.   `PETSC_SHARED_TMP` - tmp is shared by all processes
.   `PETSC_NOT_SHARED_TMP` - each process has its own private tmp
.   `PETSC_OPTIONS` - a string containing additional options for petsc in the form of command line "-key value" pairs
.   `PETSC_OPTIONS_YAML` - (requires configuring PETSc to use libyaml) a string containing additional options for petsc in the form of a YAML document
.   `PETSC_VIEWER_SOCKET_PORT` - socket number to use for socket viewer
-   `PETSC_VIEWER_SOCKET_MACHINE` - machine to use for socket viewer to connect to

   Level: beginner

   Note:
   If for some reason you must call `MPI_Init()` separately, call
   it before `PetscInitialize()`.

   Fortran Notes:
   In Fortran this routine can be called with
.vb
       call PetscInitialize(ierr)
       call PetscInitialize(file,ierr) or
       call PetscInitialize(file,help,ierr)
.ve

   If your main program is C but you call Fortran code that also uses PETSc you need to call `PetscInitializeFortran()` soon after
   calling `PetscInitialize()`.

   Options Database Key for Developers:
.  -checkfunctionlist - automatically checks that function lists associated with objects are correctly cleaned up. Produces messages of the form:
    "function name: MatInodeGetInodeSizes_C" if they are not cleaned up. This flag is always set for the test harness (in framework.py)

.seealso: `PetscFinalize()`, `PetscInitializeFortran()`, `PetscGetArgs()`, `PetscInitializeNoArguments()`, `PetscLogGpuTime()`
@*/
PetscErrorCode PetscInitialize(int *argc, char ***args, const char file[], const char help[])
{
  PetscMPIInt flag;
  const char *prog = "Unknown Name", *mpienv;

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Initialized(&flag));
  if (!flag) {
    PetscCheck(PETSC_COMM_WORLD == MPI_COMM_NULL, PETSC_COMM_SELF, PETSC_ERR_SUP, "You cannot set PETSC_COMM_WORLD if you have not initialized MPI first");
    PetscCall(PetscPreMPIInit_Private());
#if defined(PETSC_HAVE_MPI_INIT_THREAD)
    {
      PetscMPIInt PETSC_UNUSED provided;
      PetscCallMPI(MPI_Init_thread(argc, args, PETSC_MPI_THREAD_REQUIRED, &provided));
    }
#else
    PetscCallMPI(MPI_Init(argc, args));
#endif
    if (PetscDefined(HAVE_MPIUNI)) {
      mpienv = getenv("PMI_SIZE");
      if (!mpienv) mpienv = getenv("OMPI_COMM_WORLD_SIZE");
      if (mpienv) {
        PetscInt isize;
        PetscCall(PetscOptionsStringToInt(mpienv, &isize));
        if (isize != 1) printf("You are using an MPI-uni (sequential) install of PETSc but trying to launch parallel jobs; you need full MPI version of PETSc\n");
        PetscCheck(isize == 1, MPI_COMM_SELF, PETSC_ERR_MPI, "You are using an MPI-uni (sequential) install of PETSc but trying to launch parallel jobs; you need full MPI version of PETSc");
      }
    }
    PetscBeganMPI = PETSC_TRUE;
  }

  if (argc && *argc) prog = **args;
  if (argc && args) {
    PetscGlobalArgc = *argc;
    PetscGlobalArgs = *args;
  }
  PetscCall(PetscInitialize_Common(prog, file, help, PETSC_FALSE, PETSC_FALSE, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(USE_LOG)
PETSC_INTERN PetscObject *PetscObjects;
PETSC_INTERN PetscInt     PetscObjectsCounts;
PETSC_INTERN PetscInt     PetscObjectsMaxCounts;
PETSC_INTERN PetscBool    PetscObjectsLog;
#endif

/*
    Frees all the MPI types and operations that PETSc may have created
*/
PetscErrorCode PetscFreeMPIResources(void)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_REAL___FLOAT128)
  PetscCallMPI(MPI_Type_free(&MPIU___FLOAT128));
  #if defined(PETSC_HAVE_COMPLEX)
  PetscCallMPI(MPI_Type_free(&MPIU___COMPLEX128));
  #endif
#endif
#if defined(PETSC_HAVE_REAL___FP16)
  PetscCallMPI(MPI_Type_free(&MPIU___FP16));
#endif

#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
  PetscCallMPI(MPI_Op_free(&MPIU_SUM));
  PetscCallMPI(MPI_Op_free(&MPIU_MAX));
  PetscCallMPI(MPI_Op_free(&MPIU_MIN));
#elif defined(PETSC_HAVE_REAL___FLOAT128) || defined(PETSC_HAVE_REAL___FP16)
  PetscCallMPI(MPI_Op_free(&MPIU_SUM___FP16___FLOAT128));
#endif

  PetscCallMPI(MPI_Type_free(&MPIU_2SCALAR));
  PetscCallMPI(MPI_Type_free(&MPIU_REAL_INT));
  PetscCallMPI(MPI_Type_free(&MPIU_SCALAR_INT));
#if defined(PETSC_USE_64BIT_INDICES)
  PetscCallMPI(MPI_Type_free(&MPIU_2INT));
#endif
  PetscCallMPI(MPI_Type_free(&MPI_4INT));
  PetscCallMPI(MPI_Type_free(&MPIU_4INT));
  PetscCallMPI(MPI_Op_free(&MPIU_MAXSUM_OP));
  PetscCallMPI(MPI_Op_free(&Petsc_Garbage_SetIntersectOp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogFinalize(void);
#endif

/*@C
   PetscFinalize - Checks for options to be called at the conclusion
   of the program. `MPI_Finalize()` is called only if the user had not
   called `MPI_Init()` before calling `PetscInitialize()`.

   Collective on `PETSC_COMM_WORLD`

   Options Database Keys:
+  -options_view - Calls `PetscOptionsView()`
.  -options_left - Prints unused options that remain in the database
.  -objects_dump [all] - Prints list of objects allocated by the user that have not been freed, the option all cause all outstanding objects to be listed
.  -mpidump - Calls PetscMPIDump()
.  -malloc_dump <optional filename> - Calls `PetscMallocDump()`, displays all memory allocated that has not been freed
.  -malloc_info - Prints total memory usage
-  -malloc_view <optional filename> - Prints list of all memory allocated and where

   Level: beginner

   Note:
   See `PetscInitialize()` for other runtime options.

.seealso: `PetscInitialize()`, `PetscOptionsView()`, `PetscMallocDump()`, `PetscMPIDump()`, `PetscEnd()`
@*/
PetscErrorCode PetscFinalize(void)
{
  PetscMPIInt rank;
  PetscInt    nopt;
  PetscBool   flg1 = PETSC_FALSE, flg2 = PETSC_FALSE, flg3 = PETSC_FALSE;
  PetscBool   flg;
#if defined(PETSC_USE_LOG)
  char mname[PETSC_MAX_PATH_LEN];
#endif

  PetscFunctionBegin;
  PetscCheck(PetscInitializeCalled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscInitialize() must be called before PetscFinalize()");
  PetscCall(PetscInfo(NULL, "PetscFinalize() called\n"));

  PetscCall(PetscOptionsHasName(NULL, NULL, "-mpi_linear_solver_server", &flg));
  if (PetscDefined(USE_SINGLE_LIBRARY) && flg) PetscCall(PCMPIServerEnd());

  /* Clean up Garbage automatically on COMM_SELF and COMM_WORLD at finalize */
  {
    union
    {
      MPI_Comm comm;
      void    *ptr;
    } ucomm;
    PetscMPIInt flg;
    void       *tmp;

    PetscCallMPI(MPI_Comm_get_attr(PETSC_COMM_SELF, Petsc_InnerComm_keyval, &ucomm, &flg));
    if (flg) PetscCallMPI(MPI_Comm_get_attr(ucomm.comm, Petsc_Garbage_HMap_keyval, &tmp, &flg));
    if (flg) PetscCall(PetscGarbageCleanup(PETSC_COMM_SELF));
    PetscCallMPI(MPI_Comm_get_attr(PETSC_COMM_WORLD, Petsc_InnerComm_keyval, &ucomm, &flg));
    if (flg) PetscCallMPI(MPI_Comm_get_attr(ucomm.comm, Petsc_Garbage_HMap_keyval, &tmp, &flg));
    if (flg) PetscCall(PetscGarbageCleanup(PETSC_COMM_WORLD));
  }

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
#if defined(PETSC_HAVE_ADIOS)
  PetscCallExternal(adios_read_finalize_method, ADIOS_READ_METHOD_BP_AGGREGATE);
  PetscCallExternal(adios_finalize, rank);
#endif
  PetscCall(PetscOptionsHasName(NULL, NULL, "-citations", &flg));
  if (flg) {
    char *cits, filename[PETSC_MAX_PATH_LEN];
    FILE *fd = PETSC_STDOUT;

    PetscCall(PetscOptionsGetString(NULL, NULL, "-citations", filename, sizeof(filename), NULL));
    if (filename[0]) PetscCall(PetscFOpen(PETSC_COMM_WORLD, filename, "w", &fd));
    PetscCall(PetscSegBufferGet(PetscCitationsList, 1, &cits));
    cits[0] = 0;
    PetscCall(PetscSegBufferExtractAlloc(PetscCitationsList, &cits));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "If you publish results based on this computation please cite the following:\n"));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "===========================================================================\n"));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "%s", cits));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fd, "===========================================================================\n"));
    PetscCall(PetscFClose(PETSC_COMM_WORLD, fd));
    PetscCall(PetscFree(cits));
  }
  PetscCall(PetscSegBufferDestroy(&PetscCitationsList));

#if defined(PETSC_HAVE_SSL) && defined(PETSC_USE_SOCKET_VIEWER)
  /* TextBelt is run for testing purposes only, please do not use this feature often */
  {
    PetscInt nmax = 2;
    char   **buffs;
    PetscCall(PetscMalloc1(2, &buffs));
    PetscCall(PetscOptionsGetStringArray(NULL, NULL, "-textbelt", buffs, &nmax, &flg1));
    if (flg1) {
      PetscCheck(nmax, PETSC_COMM_WORLD, PETSC_ERR_USER, "-textbelt requires either the phone number or number,\"message\"");
      if (nmax == 1) {
        size_t len = 128;
        PetscCall(PetscMalloc1(len, &buffs[1]));
        PetscCall(PetscGetProgramName(buffs[1], 32));
        PetscCall(PetscStrlcat(buffs[1], " has completed", len));
      }
      PetscCall(PetscTextBelt(PETSC_COMM_WORLD, buffs[0], buffs[1], NULL));
      PetscCall(PetscFree(buffs[0]));
      PetscCall(PetscFree(buffs[1]));
    }
    PetscCall(PetscFree(buffs));
  }
  {
    PetscInt nmax = 2;
    char   **buffs;
    PetscCall(PetscMalloc1(2, &buffs));
    PetscCall(PetscOptionsGetStringArray(NULL, NULL, "-tellmycell", buffs, &nmax, &flg1));
    if (flg1) {
      PetscCheck(nmax, PETSC_COMM_WORLD, PETSC_ERR_USER, "-tellmycell requires either the phone number or number,\"message\"");
      if (nmax == 1) {
        size_t len = 128;
        PetscCall(PetscMalloc1(len, &buffs[1]));
        PetscCall(PetscGetProgramName(buffs[1], 32));
        PetscCall(PetscStrlcat(buffs[1], " has completed", len));
      }
      PetscCall(PetscTellMyCell(PETSC_COMM_WORLD, buffs[0], buffs[1], NULL));
      PetscCall(PetscFree(buffs[0]));
      PetscCall(PetscFree(buffs[1]));
    }
    PetscCall(PetscFree(buffs));
  }
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
  PetscCall(PetscFPTDestroy());
#endif

#if defined(PETSC_HAVE_SAWS)
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-saw_options", &flg, NULL));
  if (flg) PetscCall(PetscOptionsSAWsDestroy());
#endif

#if defined(PETSC_HAVE_X)
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-x_virtual", &flg1, NULL));
  if (flg1) {
    /*  this is a crude hack, but better than nothing */
    PetscCall(PetscPOpen(PETSC_COMM_WORLD, NULL, "pkill -9 Xvfb", "r", NULL));
  }
#endif

#if !defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_info", &flg2, NULL));
  if (!flg2) {
    flg2 = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-memory_view", &flg2, NULL));
  }
  if (flg2) PetscCall(PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD, "Summary of Memory Usage in PETSc\n"));
#endif

#if defined(PETSC_USE_LOG)
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-get_total_flops", &flg1, NULL));
  if (flg1) {
    PetscLogDouble flops = 0;
    PetscCallMPI(MPI_Reduce(&petsc_TotalFlops, &flops, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total flops over all processors %g\n", flops));
  }
#endif

#if defined(PETSC_USE_LOG)
  #if defined(PETSC_HAVE_MPE)
  mname[0] = 0;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_mpe", mname, sizeof(mname), &flg1));
  if (flg1) {
    if (mname[0]) PetscCall(PetscLogMPEDump(mname));
    else PetscCall(PetscLogMPEDump(0));
  }
  #endif
#endif

  /*
     Free all objects registered with PetscObjectRegisterDestroy() such as PETSC_VIEWER_XXX_().
  */
  PetscCall(PetscObjectRegisterDestroyAll());

#if defined(PETSC_USE_LOG)
  PetscCall(PetscOptionsPushGetViewerOff(PETSC_FALSE));
  PetscCall(PetscLogViewFromOptions());
  PetscCall(PetscOptionsPopGetViewerOff());

  mname[0] = 0;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_summary", mname, sizeof(mname), &flg1));
  if (flg1) {
    PetscViewer viewer;
    PetscCall((*PetscHelpPrintf)(PETSC_COMM_WORLD, "\n\n WARNING:   -log_summary is being deprecated; switch to -log_view\n\n\n"));
    if (mname[0]) {
      PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, mname, &viewer));
      PetscCall(PetscLogView(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    } else {
      viewer = PETSC_VIEWER_STDOUT_WORLD;
      PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT));
      PetscCall(PetscLogView(viewer));
      PetscCall(PetscViewerPopFormat(viewer));
    }
  }

  /*
     Free any objects created by the last block of code.
  */
  PetscCall(PetscObjectRegisterDestroyAll());

  mname[0] = 0;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_all", mname, sizeof(mname), &flg1));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log", mname, sizeof(mname), &flg2));
  if (flg1 || flg2) PetscCall(PetscLogDump(mname));
#endif

  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_signal_handler", &flg1, NULL));
  if (!flg1) PetscCall(PetscPopSignalHandler());
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mpidump", &flg1, NULL));
  if (flg1) PetscCall(PetscMPIDump(stdout));
  flg1 = PETSC_FALSE;
  flg2 = PETSC_FALSE;
  /* preemptive call to avoid listing this option in options table as unused */
  PetscCall(PetscOptionsHasName(NULL, NULL, "-malloc_dump", &flg1));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-objects_dump", &flg1));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-options_view", &flg2, NULL));

  if (flg2) {
    PetscViewer viewer;
    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
    PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
    PetscCall(PetscOptionsView(NULL, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* to prevent PETSc -options_left from warning */
  PetscCall(PetscOptionsHasName(NULL, NULL, "-nox", &flg1));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-nox_warning", &flg1));

  flg3 = PETSC_FALSE; /* default value is required */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-options_left", &flg3, &flg1));
  if (PetscUnlikelyDebug(!flg1)) flg3 = PETSC_TRUE;
  if (flg3) {
    if (!flg2 && flg1) { /* have not yet printed the options */
      PetscViewer viewer;
      PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
      PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
      PetscCall(PetscOptionsView(NULL, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    PetscCall(PetscOptionsAllUsed(NULL, &nopt));
    if (nopt) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "WARNING! There are options you set that were not used!\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "WARNING! could be spelling mistake, etc!\n"));
      if (nopt == 1) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "There is one unused database option. It is:\n"));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "There are %" PetscInt_FMT " unused database options. They are:\n", nopt));
      }
    } else if (flg3 && flg1) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "There are no unused options.\n"));
    }
    PetscCall(PetscOptionsLeft(NULL));
  }

#if defined(PETSC_HAVE_SAWS)
  if (!PetscGlobalRank) {
    PetscCall(PetscStackSAWsViewOff());
    PetscCallSAWs(SAWs_Finalize, ());
  }
#endif

#if defined(PETSC_USE_LOG)
  /*
       List all objects the user may have forgot to free
  */
  if (PetscObjectsLog) {
    PetscCall(PetscOptionsHasName(NULL, NULL, "-objects_dump", &flg1));
    if (flg1) {
      MPI_Comm local_comm;
      char     string[64];

      PetscCall(PetscOptionsGetString(NULL, NULL, "-objects_dump", string, sizeof(string), NULL));
      PetscCallMPI(MPI_Comm_dup(PETSC_COMM_WORLD, &local_comm));
      PetscCall(PetscSequentialPhaseBegin_Private(local_comm, 1));
      PetscCall(PetscObjectsDump(stdout, (string[0] == 'a') ? PETSC_TRUE : PETSC_FALSE));
      PetscCall(PetscSequentialPhaseEnd_Private(local_comm, 1));
      PetscCallMPI(MPI_Comm_free(&local_comm));
    }
  }
#endif

#if defined(PETSC_USE_LOG)
  PetscObjectsCounts    = 0;
  PetscObjectsMaxCounts = 0;
  PetscCall(PetscFree(PetscObjects));
#endif

  /*
     Destroy any packages that registered a finalize
  */
  PetscCall(PetscRegisterFinalizeAll());

#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogFinalize());
#endif

  /*
     Print PetscFunctionLists that have not been properly freed
  */
  if (PetscPrintFunctionList) PetscCall(PetscFunctionListPrintAll());

  if (petsc_history) {
    PetscCall(PetscCloseHistoryFile(&petsc_history));
    petsc_history = NULL;
  }
  PetscCall(PetscOptionsHelpPrintedDestroy(&PetscOptionsHelpPrintedSingleton));
  PetscCall(PetscInfoDestroy());

#if !defined(PETSC_HAVE_THREADSAFETY)
  if (!(PETSC_RUNNING_ON_VALGRIND)) {
    char  fname[PETSC_MAX_PATH_LEN];
    char  sname[PETSC_MAX_PATH_LEN];
    FILE *fd;
    int   err;

    flg2 = PETSC_FALSE;
    flg3 = PETSC_FALSE;
    if (PetscDefined(USE_DEBUG)) PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_test", &flg2, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_debug", &flg3, NULL));
    fname[0] = 0;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-malloc_dump", fname, sizeof(fname), &flg1));
    if (flg1 && fname[0]) {
      PetscCall(PetscSNPrintf(sname, sizeof(sname), "%s_%d", fname, rank));
      fd = fopen(sname, "w");
      PetscCheck(fd, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open log file: %s", sname);
      PetscCall(PetscMallocDump(fd));
      err = fclose(fd);
      PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
    } else if (flg1 || flg2 || flg3) {
      MPI_Comm local_comm;

      PetscCallMPI(MPI_Comm_dup(PETSC_COMM_WORLD, &local_comm));
      PetscCall(PetscSequentialPhaseBegin_Private(local_comm, 1));
      PetscCall(PetscMallocDump(stdout));
      PetscCall(PetscSequentialPhaseEnd_Private(local_comm, 1));
      PetscCallMPI(MPI_Comm_free(&local_comm));
    }
    fname[0] = 0;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-malloc_view", fname, sizeof(fname), &flg1));
    if (flg1 && fname[0]) {
      PetscCall(PetscSNPrintf(sname, sizeof(sname), "%s_%d", fname, rank));
      fd = fopen(sname, "w");
      PetscCheck(fd, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open log file: %s", sname);
      PetscCall(PetscMallocView(fd));
      err = fclose(fd);
      PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
    } else if (flg1) {
      MPI_Comm local_comm;

      PetscCallMPI(MPI_Comm_dup(PETSC_COMM_WORLD, &local_comm));
      PetscCall(PetscSequentialPhaseBegin_Private(local_comm, 1));
      PetscCall(PetscMallocView(stdout));
      PetscCall(PetscSequentialPhaseEnd_Private(local_comm, 1));
      PetscCallMPI(MPI_Comm_free(&local_comm));
    }
  }
#endif

  /*
     Close any open dynamic libraries
  */
  PetscCall(PetscFinalize_DynamicLibraries());

  /* Can be destroyed only after all the options are used */
  PetscCall(PetscOptionsDestroyDefault());

  PetscGlobalArgc = 0;
  PetscGlobalArgs = NULL;

#if defined(PETSC_HAVE_KOKKOS)
  if (PetscBeganKokkos) {
    PetscCall(PetscKokkosFinalize_Private());
    PetscBeganKokkos       = PETSC_FALSE;
    PetscKokkosInitialized = PETSC_FALSE;
  }
#endif

#if defined(PETSC_HAVE_NVSHMEM)
  if (PetscBeganNvshmem) {
    PetscCall(PetscNvshmemFinalize());
    PetscBeganNvshmem = PETSC_FALSE;
  }
#endif

  PetscCall(PetscFreeMPIResources());

  /*
     Destroy any known inner MPI_Comm's and attributes pointing to them
     Note this will not destroy any new communicators the user has created.

     If all PETSc objects were not destroyed those left over objects will have hanging references to
     the MPI_Comms that were freed; but that is ok because those PETSc objects will never be used again
 */
  {
    PetscCommCounter *counter;
    PetscMPIInt       flg;
    MPI_Comm          icomm;
    union
    {
      MPI_Comm comm;
      void    *ptr;
    } ucomm;
    PetscCallMPI(MPI_Comm_get_attr(PETSC_COMM_SELF, Petsc_InnerComm_keyval, &ucomm, &flg));
    if (flg) {
      icomm = ucomm.comm;
      PetscCallMPI(MPI_Comm_get_attr(icomm, Petsc_Counter_keyval, &counter, &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");

      PetscCallMPI(MPI_Comm_delete_attr(PETSC_COMM_SELF, Petsc_InnerComm_keyval));
      PetscCallMPI(MPI_Comm_delete_attr(icomm, Petsc_Counter_keyval));
      PetscCallMPI(MPI_Comm_free(&icomm));
    }
    PetscCallMPI(MPI_Comm_get_attr(PETSC_COMM_WORLD, Petsc_InnerComm_keyval, &ucomm, &flg));
    if (flg) {
      icomm = ucomm.comm;
      PetscCallMPI(MPI_Comm_get_attr(icomm, Petsc_Counter_keyval, &counter, &flg));
      PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT, "Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");

      PetscCallMPI(MPI_Comm_delete_attr(PETSC_COMM_WORLD, Petsc_InnerComm_keyval));
      PetscCallMPI(MPI_Comm_delete_attr(icomm, Petsc_Counter_keyval));
      PetscCallMPI(MPI_Comm_free(&icomm));
    }
  }

  PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Counter_keyval));
  PetscCallMPI(MPI_Comm_free_keyval(&Petsc_InnerComm_keyval));
  PetscCallMPI(MPI_Comm_free_keyval(&Petsc_OuterComm_keyval));
  PetscCallMPI(MPI_Comm_free_keyval(&Petsc_ShmComm_keyval));
  PetscCallMPI(MPI_Comm_free_keyval(&Petsc_CreationIdx_keyval));
  PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Garbage_HMap_keyval));

  // Free keyvals which may be silently created by some routines
  if (Petsc_SharedWD_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_SharedWD_keyval));
  if (Petsc_SharedTmp_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_SharedTmp_keyval));

  PetscCall(PetscSpinlockDestroy(&PetscViewerASCIISpinLockOpen));
  PetscCall(PetscSpinlockDestroy(&PetscViewerASCIISpinLockStdout));
  PetscCall(PetscSpinlockDestroy(&PetscViewerASCIISpinLockStderr));
  PetscCall(PetscSpinlockDestroy(&PetscCommSpinLock));

  if (PetscBeganMPI) {
    PetscMPIInt flag;
    PetscCallMPI(MPI_Finalized(&flag));
    PetscCheck(!flag, PETSC_COMM_SELF, PETSC_ERR_LIB, "MPI_Finalize() has already been called, even though MPI_Init() was called by PetscInitialize()");
    /* wait until the very last moment to disable error handling */
    PetscErrorHandlingInitialized = PETSC_FALSE;
    PetscCallMPI(MPI_Finalize());
  } else PetscErrorHandlingInitialized = PETSC_FALSE;

  /*

     Note: In certain cases PETSC_COMM_WORLD is never MPI_Comm_free()ed because
   the communicator has some outstanding requests on it. Specifically if the
   flag PETSC_HAVE_BROKEN_REQUEST_FREE is set (for IBM MPI implementation). See
   src/vec/utils/vpscat.c. Due to this the memory allocated in PetscCommDuplicate()
   is never freed as it should be. Thus one may obtain messages of the form
   [ 1] 8 bytes PetscCommDuplicate() line 645 in src/sys/mpiu.c indicating the
   memory was not freed.

*/
  PetscCall(PetscMallocClear());
  PetscCall(PetscStackReset());

  PetscInitializeCalled = PETSC_FALSE;
  PetscFinalizeCalled   = PETSC_TRUE;
#if defined(PETSC_USE_COVERAGE)
  /*
     flush gcov, otherwise during CI the flushing continues into the next pipeline resulting in git not being able to delete directories since the
     gcov files are still being added to the directories as git tries to remove the directories.
   */
  __gcov_flush();
#endif
  /* To match PetscFunctionBegin() at the beginning of this function */
  PetscStackClearTop;
  return PETSC_SUCCESS;
}

#if defined(PETSC_MISSING_LAPACK_lsame_)
PETSC_EXTERN int lsame_(char *a, char *b)
{
  if (*a == *b) return 1;
  if (*a + 32 == *b) return 1;
  if (*a - 32 == *b) return 1;
  return 0;
}
#endif

#if defined(PETSC_MISSING_LAPACK_lsame)
PETSC_EXTERN int lsame(char *a, char *b)
{
  if (*a == *b) return 1;
  if (*a + 32 == *b) return 1;
  if (*a - 32 == *b) return 1;
  return 0;
}
#endif
