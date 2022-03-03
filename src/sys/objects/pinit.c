#define PETSC_DESIRE_FEATURE_TEST_MACROS
/*
   This file defines the initialization of PETSc, including PetscInitialize()
*/
#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/
#include <petscviewer.h>

#if !defined(PETSC_HAVE_WINDOWS_COMPILERS)
#include <petsc/private/valgrind/valgrind.h>
#endif

#if defined(PETSC_HAVE_FORTRAN)
#include <petsc/private/fortranimpl.h>
#endif

#if defined(PETSC_USE_GCOV)
EXTERN_C_BEGIN
void  __gcov_flush(void);
EXTERN_C_END
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
PETSC_INTERN PetscFPT PetscFPTData;
PetscFPT PetscFPTData = 0;
#endif

#if PetscDefined(HAVE_SAWS)
#include <petscviewersaws.h>
#endif

/* -----------------------------------------------------------------------------------------*/

PETSC_INTERN FILE *petsc_history;

PETSC_INTERN PetscErrorCode PetscInitialize_DynamicLibraries(void);
PETSC_INTERN PetscErrorCode PetscFinalize_DynamicLibraries(void);
PETSC_INTERN PetscErrorCode PetscFunctionListPrintAll(void);
PETSC_INTERN PetscErrorCode PetscSequentialPhaseBegin_Private(MPI_Comm,int);
PETSC_INTERN PetscErrorCode PetscSequentialPhaseEnd_Private(MPI_Comm,int);
PETSC_INTERN PetscErrorCode PetscCloseHistoryFile(FILE**);

/* user may set these BEFORE calling PetscInitialize() */
MPI_Comm PETSC_COMM_WORLD = MPI_COMM_NULL;
#if PetscDefined(HAVE_MPI_INIT_THREAD)
PetscMPIInt PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_FUNNELED;
#else
PetscMPIInt PETSC_MPI_THREAD_REQUIRED = 0;
#endif

PetscMPIInt Petsc_Counter_keyval   = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_InnerComm_keyval = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_OuterComm_keyval = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_ShmComm_keyval   = MPI_KEYVAL_INVALID;

/*
     Declare and set all the string names of the PETSc enums
*/
const char *const PetscBools[]     = {"FALSE","TRUE","PetscBool","PETSC_",NULL};
const char *const PetscCopyModes[] = {"COPY_VALUES","OWN_POINTER","USE_POINTER","PetscCopyMode","PETSC_",NULL};

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

.seealso: PetscInitialize(), PetscInitializeFortran(), PetscInitializeNoArguments()
*/
PetscErrorCode  PetscInitializeNoPointers(int argc,char **args,const char *filename,const char *help)
{
  PetscErrorCode ierr;
  int            myargc   = argc;
  char           **myargs = args;

  PetscFunctionBegin;
  ierr = PetscInitialize(&myargc,&myargs,filename,help);if (ierr) PetscFunctionReturn(ierr);
  CHKERRQ(PetscPopSignalHandler());
  PetscBeganMPI = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*
      Used by Julia interface to get communicator
*/
PetscErrorCode  PetscGetPETSC_COMM_SELF(MPI_Comm *comm)
{
  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscValidPointer(comm,1);
  *comm = PETSC_COMM_SELF;
  PetscFunctionReturn(0);
}

/*@C
      PetscInitializeNoArguments - Calls PetscInitialize() from C/C++ without
        the command line arguments.

   Collective

   Level: advanced

.seealso: PetscInitialize(), PetscInitializeFortran()
@*/
PetscErrorCode  PetscInitializeNoArguments(void)
{
  PetscErrorCode ierr;
  int            argc   = 0;
  char           **args = NULL;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&args,NULL,NULL);
  PetscFunctionReturn(ierr);
}

/*@
      PetscInitialized - Determine whether PETSc is initialized.

   Level: beginner

.seealso: PetscInitialize(), PetscInitializeNoArguments(), PetscInitializeFortran()
@*/
PetscErrorCode PetscInitialized(PetscBool *isInitialized)
{
  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscValidBoolPointer(isInitialized,1);
  *isInitialized = PetscInitializeCalled;
  PetscFunctionReturn(0);
}

/*@
      PetscFinalized - Determine whether PetscFinalize() has been called yet

   Level: developer

.seealso: PetscInitialize(), PetscInitializeNoArguments(), PetscInitializeFortran()
@*/
PetscErrorCode  PetscFinalized(PetscBool  *isFinalized)
{
  PetscFunctionBegin;
  if (!PetscFinalizeCalled) PetscValidBoolPointer(isFinalized,1);
  *isFinalized = PetscFinalizeCalled;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscOptionsCheckInitial_Private(const char []);

/*
       This function is the MPI reduction operation used to compute the sum of the
   first half of the datatype and the max of the second half.
*/
MPI_Op MPIU_MAXSUM_OP = 0;

PETSC_INTERN void MPIAPI MPIU_MaxSum_Local(void *in,void *out,int *cnt,MPI_Datatype *datatype)
{
  PetscInt *xin = (PetscInt*)in,*xout = (PetscInt*)out,i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_2INT) {
    (*PetscErrorPrintf)("Can only handle MPIU_2INT data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }

  for (i=0; i<count; i++) {
    xout[2*i]    = PetscMax(xout[2*i],xin[2*i]);
    xout[2*i+1] += xin[2*i+1];
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
PetscErrorCode  PetscMaxSum(MPI_Comm comm,const PetscInt sizes[],PetscInt *max,PetscInt *sum)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_REDUCE_SCATTER_BLOCK)
  {
    struct {PetscInt max,sum;} work;
    CHKERRMPI(MPI_Reduce_scatter_block((void*)sizes,&work,1,MPIU_2INT,MPIU_MAXSUM_OP,comm));
    *max = work.max;
    *sum = work.sum;
  }
#else
  {
    PetscMPIInt    size,rank;
    struct {PetscInt max,sum;} *work;
    CHKERRMPI(MPI_Comm_size(comm,&size));
    CHKERRMPI(MPI_Comm_rank(comm,&rank));
    CHKERRQ(PetscMalloc1(size,&work));
    CHKERRMPI(MPIU_Allreduce((void*)sizes,work,size,MPIU_2INT,MPIU_MAXSUM_OP,comm));
    *max = work[rank].max;
    *sum = work[rank].sum;
    CHKERRQ(PetscFree(work));
  }
#endif
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/

#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
MPI_Op MPIU_SUM = 0;

PETSC_EXTERN void MPIAPI PetscSum_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscInt i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal*)in,*xout = (PetscReal*)out;
    for (i=0; i<count; i++) xout[i] += xin[i];
  }
#if defined(PETSC_HAVE_COMPLEX)
  else if (*datatype == MPIU_COMPLEX) {
    PetscComplex *xin = (PetscComplex*)in,*xout = (PetscComplex*)out;
    for (i=0; i<count; i++) xout[i] += xin[i];
  }
#endif
  else {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_COMPLEX data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  PetscFunctionReturnVoid();
}
#endif

#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
MPI_Op MPIU_MAX = 0;
MPI_Op MPIU_MIN = 0;

PETSC_EXTERN void MPIAPI PetscMax_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscInt i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal*)in,*xout = (PetscReal*)out;
    for (i=0; i<count; i++) xout[i] = PetscMax(xout[i],xin[i]);
  }
#if defined(PETSC_HAVE_COMPLEX)
  else if (*datatype == MPIU_COMPLEX) {
    PetscComplex *xin = (PetscComplex*)in,*xout = (PetscComplex*)out;
    for (i=0; i<count; i++) {
      xout[i] = PetscRealPartComplex(xout[i])<PetscRealPartComplex(xin[i]) ? xin[i] : xout[i];
    }
  }
#endif
  else {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_COMPLEX data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  PetscFunctionReturnVoid();
}

PETSC_EXTERN void MPIAPI PetscMin_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscInt    i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal*)in,*xout = (PetscReal*)out;
    for (i=0; i<count; i++) xout[i] = PetscMin(xout[i],xin[i]);
  }
#if defined(PETSC_HAVE_COMPLEX)
  else if (*datatype == MPIU_COMPLEX) {
    PetscComplex *xin = (PetscComplex*)in,*xout = (PetscComplex*)out;
    for (i=0; i<count; i++) {
      xout[i] = PetscRealPartComplex(xout[i])>PetscRealPartComplex(xin[i]) ? xin[i] : xout[i];
    }
  }
#endif
  else {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_SCALAR data (i.e. double or complex) types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  PetscFunctionReturnVoid();
}
#endif

/*
   Private routine to delete internal tag/name counter storage when a communicator is freed.

   This is called by MPI, not by users. This is called by MPI_Comm_free() when the communicator that has this  data as an attribute is freed.

   Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_Counter_Attr_Delete_Fn(MPI_Comm comm,PetscMPIInt keyval,void *count_val,void *extra_state)
{
  PetscCommCounter      *counter=(PetscCommCounter*)count_val;
  struct PetscCommStash *comms = counter->comms, *pcomm;

  PetscFunctionBegin;
  CHKERRMPI(PetscInfo(NULL,"Deleting counter data in an MPI_Comm %ld\n",(long)comm));
  CHKERRMPI(PetscFree(counter->iflags));
  while (comms) {
    CHKERRMPI(MPI_Comm_free(&comms->comm));
    pcomm = comms;
    comms = comms->next;
    CHKERRQ(PetscFree(pcomm));
  }
  CHKERRMPI(PetscFree(counter));
  PetscFunctionReturn(MPI_SUCCESS);
}

/*
  This is invoked on the outer comm as a result of either PetscCommDestroy() (via MPI_Comm_delete_attr) or when the user
  calls MPI_Comm_free().

  This is the only entry point for breaking the links between inner and outer comms.

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_InnerComm_Attr_Delete_Fn(MPI_Comm comm,PetscMPIInt keyval,void *attr_val,void *extra_state)
{
  union {MPI_Comm comm; void *ptr;} icomm;

  PetscFunctionBegin;
  if (keyval != Petsc_InnerComm_keyval) SETERRMPI(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unexpected keyval");
  icomm.ptr = attr_val;
  if (PetscDefined(USE_DEBUG)) {
    /* Error out if the inner/outer comms are not correctly linked through their Outer/InnterComm attributes */
    PetscMPIInt flg;
    union {MPI_Comm comm; void *ptr;} ocomm;
    CHKERRMPI(MPI_Comm_get_attr(icomm.comm,Petsc_OuterComm_keyval,&ocomm,&flg));
    if (!flg) SETERRMPI(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner comm does not have OuterComm attribute");
    if (ocomm.comm != comm) SETERRMPI(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner comm's OuterComm attribute does not point to outer PETSc comm");
  }
  CHKERRMPI(MPI_Comm_delete_attr(icomm.comm,Petsc_OuterComm_keyval));
  CHKERRMPI(PetscInfo(NULL,"User MPI_Comm %ld is being unlinked from inner PETSc comm %ld\n",(long)comm,(long)icomm.comm));
  PetscFunctionReturn(MPI_SUCCESS);
}

/*
 * This is invoked on the inner comm when Petsc_InnerComm_Attr_Delete_Fn calls MPI_Comm_delete_attr().  It should not be reached any other way.
 */
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_OuterComm_Attr_Delete_Fn(MPI_Comm comm,PetscMPIInt keyval,void *attr_val,void *extra_state)
{
  PetscFunctionBegin;
  CHKERRMPI(PetscInfo(NULL,"Removing reference to PETSc communicator embedded in a user MPI_Comm %ld\n",(long)comm));
  PetscFunctionReturn(MPI_SUCCESS);
}

PETSC_EXTERN PetscMPIInt MPIAPI Petsc_ShmComm_Attr_Delete_Fn(MPI_Comm,PetscMPIInt,void *,void *);

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
PETSC_EXTERN PetscMPIInt PetscDataRep_extent_fn(MPI_Datatype,MPI_Aint*,void*);
PETSC_EXTERN PetscMPIInt PetscDataRep_read_conv_fn(void*, MPI_Datatype,PetscMPIInt,void*,MPI_Offset,void*);
PETSC_EXTERN PetscMPIInt PetscDataRep_write_conv_fn(void*, MPI_Datatype,PetscMPIInt,void*,MPI_Offset,void*);
#endif

PetscMPIInt PETSC_MPI_ERROR_CLASS=MPI_ERR_LASTCODE,PETSC_MPI_ERROR_CODE;

PETSC_INTERN int  PetscGlobalArgc;
PETSC_INTERN char **PetscGlobalArgs;
int  PetscGlobalArgc   = 0;
char **PetscGlobalArgs = NULL;
PetscSegBuffer PetscCitationsList;

PetscErrorCode PetscCitationsInitialize(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSegBufferCreate(1,10000,&PetscCitationsList));
  CHKERRQ(PetscCitationsRegister("@TechReport{petsc-user-ref,\n  Author = {Satish Balay and Shrirang Abhyankar and Mark F. Adams and Jed Brown \n            and Peter Brune and Kris Buschelman and Lisandro Dalcin and\n            Victor Eijkhout and William D. Gropp and Dmitry Karpeyev and\n            Dinesh Kaushik and Matthew G. Knepley and Dave A. May and Lois Curfman McInnes\n            and Richard Tran Mills and Todd Munson and Karl Rupp and Patrick Sanan\n            and Barry F. Smith and Stefano Zampini and Hong Zhang and Hong Zhang},\n  Title = {{PETS}c Users Manual},\n  Number = {ANL-95/11 - Revision 3.11},\n  Institution = {Argonne National Laboratory},\n  Year = {2019}\n}\n",NULL));
  CHKERRQ(PetscCitationsRegister("@InProceedings{petsc-efficient,\n  Author = {Satish Balay and William D. Gropp and Lois Curfman McInnes and Barry F. Smith},\n  Title = {Efficient Management of Parallelism in Object Oriented Numerical Software Libraries},\n  Booktitle = {Modern Software Tools in Scientific Computing},\n  Editor = {E. Arge and A. M. Bruaset and H. P. Langtangen},\n  Pages = {163--202},\n  Publisher = {Birkh{\\\"{a}}user Press},\n  Year = {1997}\n}\n",NULL));
  PetscFunctionReturn(0);
}

static char programname[PETSC_MAX_PATH_LEN] = ""; /* HP includes entire path in name */

PetscErrorCode  PetscSetProgramName(const char name[])
{
  PetscFunctionBegin;
  CHKERRQ(PetscStrncpy(programname,name,sizeof(programname)));
  PetscFunctionReturn(0);
}

/*@C
    PetscGetProgramName - Gets the name of the running program.

    Not Collective

    Input Parameter:
.   len - length of the string name

    Output Parameter:
.   name - the name of the running program

   Level: advanced

    Notes:
    The name of the program is copied into the user-provided character
    array of length len.  On some machines the program name includes
    its entire path, so one should generally set len >= PETSC_MAX_PATH_LEN.
@*/
PetscErrorCode  PetscGetProgramName(char name[],size_t len)
{
  PetscFunctionBegin;
  CHKERRQ(PetscStrncpy(name,programname,len));
  PetscFunctionReturn(0);
}

/*@C
   PetscGetArgs - Allows you to access the raw command line arguments anywhere
     after PetscInitialize() is called but before PetscFinalize().

   Not Collective

   Output Parameters:
+  argc - count of number of command line arguments
-  args - the command line arguments

   Level: intermediate

   Notes:
      This is usually used to pass the command line arguments into other libraries
   that are called internally deep in PETSc or the application.

      The first argument contains the program name as is normal for C arguments.

.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArguments()

@*/
PetscErrorCode  PetscGetArgs(int *argc,char ***args)
{
  PetscFunctionBegin;
  PetscCheckFalse(!PetscInitializeCalled && PetscFinalizeCalled,PETSC_COMM_SELF,PETSC_ERR_ORDER,"You must call after PetscInitialize() but before PetscFinalize()");
  *argc = PetscGlobalArgc;
  *args = PetscGlobalArgs;
  PetscFunctionReturn(0);
}

/*@C
   PetscGetArguments - Allows you to access the  command line arguments anywhere
     after PetscInitialize() is called but before PetscFinalize().

   Not Collective

   Output Parameters:
.  args - the command line arguments

   Level: intermediate

   Notes:
      This does NOT start with the program name and IS null terminated (final arg is void)

.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscFreeArguments()

@*/
PetscErrorCode  PetscGetArguments(char ***args)
{
  PetscInt       i,argc = PetscGlobalArgc;

  PetscFunctionBegin;
  PetscCheckFalse(!PetscInitializeCalled && PetscFinalizeCalled,PETSC_COMM_SELF,PETSC_ERR_ORDER,"You must call after PetscInitialize() but before PetscFinalize()");
  if (!argc) {*args = NULL; PetscFunctionReturn(0);}
  CHKERRQ(PetscMalloc1(argc,args));
  for (i=0; i<argc-1; i++) {
    CHKERRQ(PetscStrallocpy(PetscGlobalArgs[i+1],&(*args)[i]));
  }
  (*args)[argc-1] = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscFreeArguments - Frees the memory obtained with PetscGetArguments()

   Not Collective

   Output Parameters:
.  args - the command line arguments

   Level: intermediate

.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscGetArguments()

@*/
PetscErrorCode  PetscFreeArguments(char **args)
{
  PetscInt       i = 0;

  PetscFunctionBegin;
  if (!args) PetscFunctionReturn(0);
  while (args[i]) {
    CHKERRQ(PetscFree(args[i]));
    i++;
  }
  CHKERRQ(PetscFree(args));
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_SAWS)
#include <petscconfiginfo.h>

PETSC_INTERN PetscErrorCode PetscInitializeSAWs(const char help[])
{
  PetscFunctionBegin;
  if (!PetscGlobalRank) {
    char           cert[PETSC_MAX_PATH_LEN],root[PETSC_MAX_PATH_LEN],*intro,programname[64],*appline,*options,version[64];
    int            port;
    PetscBool      flg,rootlocal = PETSC_FALSE,flg2,selectport = PETSC_FALSE;
    size_t         applinelen,introlen;
    PetscErrorCode ierr;
    char           sawsurl[256];

    CHKERRQ(PetscOptionsHasName(NULL,NULL,"-saws_log",&flg));
    if (flg) {
      char  sawslog[PETSC_MAX_PATH_LEN];

      CHKERRQ(PetscOptionsGetString(NULL,NULL,"-saws_log",sawslog,sizeof(sawslog),NULL));
      if (sawslog[0]) {
        PetscStackCallSAWs(SAWs_Set_Use_Logfile,(sawslog));
      } else {
        PetscStackCallSAWs(SAWs_Set_Use_Logfile,(NULL));
      }
    }
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-saws_https",cert,sizeof(cert),&flg));
    if (flg) {
      PetscStackCallSAWs(SAWs_Set_Use_HTTPS,(cert));
    }
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-saws_port_auto_select",&selectport,NULL));
    if (selectport) {
        PetscStackCallSAWs(SAWs_Get_Available_Port,(&port));
        PetscStackCallSAWs(SAWs_Set_Port,(port));
    } else {
      CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-saws_port",&port,&flg));
      if (flg) {
        PetscStackCallSAWs(SAWs_Set_Port,(port));
      }
    }
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-saws_root",root,sizeof(root),&flg));
    if (flg) {
      PetscStackCallSAWs(SAWs_Set_Document_Root,(root));
      CHKERRQ(PetscStrcmp(root,".",&rootlocal));
    } else {
      CHKERRQ(PetscOptionsHasName(NULL,NULL,"-saws_options",&flg));
      if (flg) {
        CHKERRQ(PetscStrreplace(PETSC_COMM_WORLD,"${PETSC_DIR}/share/petsc/saws",root,sizeof(root)));
        PetscStackCallSAWs(SAWs_Set_Document_Root,(root));
      }
    }
    CHKERRQ(PetscOptionsHasName(NULL,NULL,"-saws_local",&flg2));
    if (flg2) {
      char jsdir[PETSC_MAX_PATH_LEN];
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"-saws_local option requires -saws_root option");
      CHKERRQ(PetscSNPrintf(jsdir,sizeof(jsdir),"%s/js",root));
      CHKERRQ(PetscTestDirectory(jsdir,'r',&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"-saws_local option requires js directory in root directory");
      PetscStackCallSAWs(SAWs_Push_Local_Header,());
    }
    CHKERRQ(PetscGetProgramName(programname,sizeof(programname)));
    CHKERRQ(PetscStrlen(help,&applinelen));
    introlen   = 4096 + applinelen;
    applinelen += 1024;
    CHKERRQ(PetscMalloc(applinelen,&appline));
    CHKERRQ(PetscMalloc(introlen,&intro));

    if (rootlocal) {
      CHKERRQ(PetscSNPrintf(appline,applinelen,"%s.c.html",programname));
      CHKERRQ(PetscTestFile(appline,'r',&rootlocal));
    }
    CHKERRQ(PetscOptionsGetAll(NULL,&options));
    if (rootlocal && help) {
      CHKERRQ(PetscSNPrintf(appline,applinelen,"<center> Running <a href=\"%s.c.html\">%s</a> %s</center><br><center><pre>%s</pre></center><br>\n",programname,programname,options,help));
    } else if (help) {
      CHKERRQ(PetscSNPrintf(appline,applinelen,"<center>Running %s %s</center><br><center><pre>%s</pre></center><br>",programname,options,help));
    } else {
      CHKERRQ(PetscSNPrintf(appline,applinelen,"<center> Running %s %s</center><br>\n",programname,options));
    }
    CHKERRQ(PetscFree(options));
    CHKERRQ(PetscGetVersion(version,sizeof(version)));
    CHKERRQ(PetscSNPrintf(intro,introlen,"<body>\n"
                          "<center><h2> <a href=\"https://petsc.org/\">PETSc</a> Application Web server powered by <a href=\"https://bitbucket.org/saws/saws\">SAWs</a> </h2></center>\n"
                          "<center>This is the default PETSc application dashboard, from it you can access any published PETSc objects or logging data</center><br><center>%s configured with %s</center><br>\n"
                          "%s",version,petscconfigureoptions,appline));
    PetscStackCallSAWs(SAWs_Push_Body,("index.html",0,intro));
    CHKERRQ(PetscFree(intro));
    CHKERRQ(PetscFree(appline));
    if (selectport) {
      PetscBool silent;

      ierr = SAWs_Initialize();
      /* another process may have grabbed the port so keep trying */
      while (ierr) {
        PetscStackCallSAWs(SAWs_Get_Available_Port,(&port));
        PetscStackCallSAWs(SAWs_Set_Port,(port));
        ierr = SAWs_Initialize();
      }

      CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-saws_port_auto_select_silent",&silent,NULL));
      if (!silent) {
        PetscStackCallSAWs(SAWs_Get_FullURL,(sizeof(sawsurl),sawsurl));
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Point your browser to %s for SAWs\n",sawsurl));
      }
    } else {
      PetscStackCallSAWs(SAWs_Initialize,());
    }
    CHKERRQ(PetscCitationsRegister("@TechReport{ saws,\n"
                                   "  Author = {Matt Otten and Jed Brown and Barry Smith},\n"
                                   "  Title  = {Scientific Application Web Server (SAWs) Users Manual},\n"
                                   "  Institution = {Argonne National Laboratory},\n"
                                   "  Year   = 2013\n}\n",NULL));
  }
  PetscFunctionReturn(0);
}
#endif

/* Things must be done before MPI_Init() when MPI is not yet initialized, and can be shared between C init and Fortran init */
PETSC_INTERN PetscErrorCode PetscPreMPIInit_Private(void)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_HWLOC_SOLARIS_BUG)
    /* see MPI.py for details on this bug */
    (void) setenv("HWLOC_COMPONENTS","-x86",1);
#endif
  PetscFunctionReturn(0);
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

#if PetscDefined(HAVE_DEVICE)
#include <petsc/private/deviceimpl.h>
#  if PetscDefined(HAVE_CUDA)
// REMOVE ME
cudaStream_t PetscDefaultCudaStream = NULL;
#  endif
#  if PetscDefined(HAVE_HIP)
// REMOVE ME
hipStream_t PetscDefaultHipStream = NULL;
#  endif
#endif

#if PetscDefined(HAVE_DLFCN_H)
#include <dlfcn.h>
#endif
#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#endif
#if PetscDefined(HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode PetscViennaCLInit();
PetscBool PetscViennaCLSynchronize = PETSC_FALSE;
#endif

/*
  PetscInitialize_Common  - shared code between C and Fortran initialization

  prog:     program name
  file:     optional PETSc database file name. Might be in Fortran string format when 'ftn' is true
  help:     program help message
  ftn:      is it called from Fortran initilization (petscinitializef_)?
  readarguments,len: used when fortran is true
*/
PETSC_INTERN PetscErrorCode PetscInitialize_Common(const char* prog,const char* file,const char *help,PetscBool ftn,PetscBool readarguments,PetscInt len)
{
  PetscMPIInt size;
  PetscBool   flg = PETSC_TRUE;
  char        hostname[256];

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(0);
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
    char           mpilibraryversion[MPI_MAX_LIBRARY_VERSION_STRING];
    PetscMPIInt    mpilibraryversionlength;
    PetscErrorCode ierr = MPI_Get_library_version(mpilibraryversion,&mpilibraryversionlength);
    if (ierr) PetscFunctionReturn(ierr);
    /* check for MPICH versions before MPI ABI initiative */
#if defined(MPICH_VERSION)
#if MPICH_NUMVERSION < 30100000
    {
      char      *ver,*lf;
      PetscBool flg = PETSC_FALSE;
      ierr = PetscStrstr(mpilibraryversion,"MPICH Version:",&ver);
      if (ierr) PetscFunctionReturn(ierr);
      else if (ver) {
        ierr = PetscStrchr(ver,'\n',&lf);
        if (ierr) PetscFunctionReturn(ierr);
        else if (lf) {
          *lf = 0;
          ierr = PetscStrendswith(ver,MPICH_VERSION,&flg);if (ierr) PetscFunctionReturn(ierr);
        }
      }
      if (!flg) {
        PetscInfo(NULL,"PETSc warning --- MPICH library version \n%s does not match what PETSc was compiled with %s.\n",mpilibraryversion,MPICH_VESION);
        flg = PETSC_TRUE;
      }
    }
#endif
    /* check for OpenMPI version, it is not part of the MPI ABI initiative (is it part of another initiative that needs to be handled?) */
#elif defined(OMPI_MAJOR_VERSION)
    {
      char *ver,bs[MPI_MAX_LIBRARY_VERSION_STRING],*bsf;
      PetscBool flg = PETSC_FALSE;
#define PSTRSZ 2
      char ompistr1[PSTRSZ][MPI_MAX_LIBRARY_VERSION_STRING] = {"Open MPI","FUJITSU MPI"};
      char ompistr2[PSTRSZ][MPI_MAX_LIBRARY_VERSION_STRING] = {"v","Library "};
      int i;
      for (i=0; i<PSTRSZ; i++) {
        ierr = PetscStrstr(mpilibraryversion,ompistr1[i],&ver);
        if (ierr) PetscFunctionReturn(ierr);
        else if (ver) {
          PetscSNPrintf(bs,MPI_MAX_LIBRARY_VERSION_STRING,"%s%d.%d",ompistr2[i],OMPI_MAJOR_VERSION,OMPI_MINOR_VERSION);
          ierr = PetscStrstr(ver,bs,&bsf);
          if (ierr) PetscFunctionReturn(ierr);
          else if (bsf) flg = PETSC_TRUE;
          break;
        }
      }
      if (!flg) {
        PetscInfo(NULL,"PETSc warning --- Open MPI library version \n%s does not match what PETSc was compiled with %d.%d.\n",mpilibraryversion,OMPI_MAJOR_VERSION,OMPI_MINOR_VERSION);
        flg = PETSC_TRUE;
      }
    }
#endif
  }
#endif

#if defined(PETSC_HAVE_DLSYM)
  /* These symbols are currently in the OpenMPI and MPICH libraries; they may not always be, in that case the test will simply not detect the problem */
  if (PetscUnlikely(dlsym(RTLD_DEFAULT,"ompi_mpi_init") && dlsym(RTLD_DEFAULT,"MPID_Abort"))) {
    fprintf(stderr,"PETSc Error --- Application was linked against both OpenMPI and MPICH based MPI libraries and will not run correctly\n");
    CHKERRQ(PetscStackView(stderr));
    PetscFunctionReturn(PETSC_ERR_MPI_LIB_INCOMP);
  }
#endif

  /* these must be initialized in a routine, not as a constant declaration*/
  PETSC_STDOUT = stdout;
  PETSC_STDERR = stderr;

  /*CHKERRQ can be used from now */
  PetscErrorHandlingInitialized = PETSC_TRUE;

  /* on Windows - set printf to default to printing 2 digit exponents */
#if defined(PETSC_HAVE__SET_OUTPUT_FORMAT)
  _set_output_format(_TWO_DIGIT_EXPONENT);
#endif

  CHKERRQ(PetscOptionsCreateDefault());

  PetscFinalizeCalled = PETSC_FALSE;

  CHKERRQ(PetscSetProgramName(prog));
  CHKERRQ(PetscSpinlockCreate(&PetscViewerASCIISpinLockOpen));
  CHKERRQ(PetscSpinlockCreate(&PetscViewerASCIISpinLockStdout));
  CHKERRQ(PetscSpinlockCreate(&PetscViewerASCIISpinLockStderr));
  CHKERRQ(PetscSpinlockCreate(&PetscCommSpinLock));

  if (PETSC_COMM_WORLD == MPI_COMM_NULL) PETSC_COMM_WORLD = MPI_COMM_WORLD;
  CHKERRMPI(MPI_Comm_set_errhandler(PETSC_COMM_WORLD,MPI_ERRORS_RETURN));

  if (PETSC_MPI_ERROR_CLASS == MPI_ERR_LASTCODE) {
    CHKERRMPI(MPI_Add_error_class(&PETSC_MPI_ERROR_CLASS));
    CHKERRMPI(MPI_Add_error_code(PETSC_MPI_ERROR_CLASS,&PETSC_MPI_ERROR_CODE));
  }

  /* Done after init due to a bug in MPICH-GM? */
  CHKERRQ(PetscErrorPrintfInitialize());

  CHKERRMPI(MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank));
  CHKERRMPI(MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize));

  MPIU_BOOL = MPI_INT;
  MPIU_ENUM = MPI_INT;
  MPIU_FORTRANADDR = (sizeof(void*) == sizeof(int)) ? MPI_INT : MPIU_INT64;
  if (sizeof(size_t) == sizeof(unsigned)) MPIU_SIZE_T = MPI_UNSIGNED;
  else if (sizeof(size_t) == sizeof(unsigned long)) MPIU_SIZE_T = MPI_UNSIGNED_LONG;
#if defined(PETSC_SIZEOF_LONG_LONG)
  else if (sizeof(size_t) == sizeof(unsigned long long)) MPIU_SIZE_T = MPI_UNSIGNED_LONG_LONG;
#endif
  else {
    (*PetscErrorPrintf)("PetscInitialize_Common: Could not find MPI type for size_t\n");
    PetscFunctionReturn(PETSC_ERR_SUP_SYS);
  }

  /*
     Initialized the global complex variable; this is because with
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
#if defined(PETSC_HAVE_COMPLEX)
  {
#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_REAL___FLOAT128)
    PetscComplex ic(0.0,1.0);
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
  CHKERRMPI(MPI_Op_create(MPIU_MaxSum_Local,1,&MPIU_MAXSUM_OP));

#if defined(PETSC_USE_REAL___FLOAT128)
  CHKERRMPI(MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU___FLOAT128));
  CHKERRMPI(MPI_Type_commit(&MPIU___FLOAT128));
#if defined(PETSC_HAVE_COMPLEX)
  CHKERRMPI(MPI_Type_contiguous(4,MPI_DOUBLE,&MPIU___COMPLEX128));
  CHKERRMPI(MPI_Type_commit(&MPIU___COMPLEX128));
#endif
  CHKERRMPI(MPI_Op_create(PetscMax_Local,1,&MPIU_MAX));
  CHKERRMPI(MPI_Op_create(PetscMin_Local,1,&MPIU_MIN));
#elif defined(PETSC_USE_REAL___FP16)
  CHKERRMPI(MPI_Type_contiguous(2,MPI_CHAR,&MPIU___FP16));
  CHKERRMPI(MPI_Type_commit(&MPIU___FP16));
  CHKERRMPI(MPI_Op_create(PetscMax_Local,1,&MPIU_MAX));
  CHKERRMPI(MPI_Op_create(PetscMin_Local,1,&MPIU_MIN));
#endif

#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
  CHKERRMPI(MPI_Op_create(PetscSum_Local,1,&MPIU_SUM));
#endif

  CHKERRMPI(MPI_Type_contiguous(2,MPIU_SCALAR,&MPIU_2SCALAR));
  CHKERRMPI(MPI_Type_commit(&MPIU_2SCALAR));

  /* create datatypes used by MPIU_MAXLOC, MPIU_MINLOC and PetscSplitReduction_Op */
#if !defined(PETSC_HAVE_MPIUNI)
  {
    struct PetscRealInt { PetscReal v; PetscInt i; };
    PetscMPIInt  blockSizes[2] = {1,1};
    MPI_Aint     blockOffsets[2] = {offsetof(struct PetscRealInt,v),offsetof(struct PetscRealInt,i)};
    MPI_Datatype blockTypes[2] = {MPIU_REAL,MPIU_INT}, tmpStruct;

    CHKERRMPI(MPI_Type_create_struct(2,blockSizes,blockOffsets,blockTypes,&tmpStruct));
    CHKERRMPI(MPI_Type_create_resized(tmpStruct,0,sizeof(struct PetscRealInt),&MPIU_REAL_INT));
    CHKERRMPI(MPI_Type_free(&tmpStruct));
    CHKERRMPI(MPI_Type_commit(&MPIU_REAL_INT));
  }
  {
    struct PetscScalarInt { PetscScalar v; PetscInt i; };
    PetscMPIInt  blockSizes[2] = {1,1};
    MPI_Aint     blockOffsets[2] = {offsetof(struct PetscScalarInt,v),offsetof(struct PetscScalarInt,i)};
    MPI_Datatype blockTypes[2] = {MPIU_SCALAR,MPIU_INT}, tmpStruct;

    CHKERRMPI(MPI_Type_create_struct(2,blockSizes,blockOffsets,blockTypes,&tmpStruct));
    CHKERRMPI(MPI_Type_create_resized(tmpStruct,0,sizeof(struct PetscScalarInt),&MPIU_SCALAR_INT));
    CHKERRMPI(MPI_Type_free(&tmpStruct));
    CHKERRMPI(MPI_Type_commit(&MPIU_SCALAR_INT));
  }
#endif

#if defined(PETSC_USE_64BIT_INDICES)
  CHKERRMPI(MPI_Type_contiguous(2,MPIU_INT,&MPIU_2INT));
  CHKERRMPI(MPI_Type_commit(&MPIU_2INT));
#endif
  CHKERRMPI(MPI_Type_contiguous(4,MPI_INT,&MPI_4INT));
  CHKERRMPI(MPI_Type_commit(&MPI_4INT));
  CHKERRMPI(MPI_Type_contiguous(4,MPIU_INT,&MPIU_4INT));
  CHKERRMPI(MPI_Type_commit(&MPIU_4INT));

  /*
     Attributes to be set on PETSc communicators
  */
  CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_Counter_Attr_Delete_Fn,&Petsc_Counter_keyval,(void*)0));
  CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_InnerComm_Attr_Delete_Fn,&Petsc_InnerComm_keyval,(void*)0));
  CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_OuterComm_Attr_Delete_Fn,&Petsc_OuterComm_keyval,(void*)0));
  CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_ShmComm_Attr_Delete_Fn,&Petsc_ShmComm_keyval,(void*)0));

#if defined(PETSC_HAVE_FORTRAN)
  if (ftn) CHKERRQ(PetscInitFortran_Private(readarguments,file,len));
  else
#endif
  CHKERRQ(PetscOptionsInsert(NULL,&PetscGlobalArgc,&PetscGlobalArgs,file));

  /* call a second time so it can look in the options database */
  CHKERRQ(PetscErrorPrintfInitialize());

  /*
     Check system options and print help
  */
  CHKERRQ(PetscOptionsCheckInitial_Private(help));

  /*
   Initialize PetscDevice and PetscDeviceContext

   Note to any future devs thinking of moving this, proper initialization requires:
   1. MPI initialized
   2. Options DB initialized
   3. Petsc error handling initialized, specifically signal handlers. This expects to set up its own SIGSEV handler via
      the push/pop interface.
  */
#if (PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP) || PetscDefined(HAVE_SYCL))
  CHKERRQ(PetscDeviceInitializeFromOptions_Internal(PETSC_COMM_WORLD));
#endif

#if PetscDefined(HAVE_VIENNACL)
  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-log_summary",&flg));
  if (!flg) CHKERRQ(PetscOptionsHasName(NULL,NULL,"-log_view",&flg));
  if (!flg) CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-viennacl_synchronize",&flg,NULL));
  PetscViennaCLSynchronize = flg;
  CHKERRQ(PetscViennaCLInit());
#endif

  /*
     Creates the logging data structures; this is enabled even if logging is not turned on
     This is the last thing we do before returning to the user code to prevent having the
     logging numbers contaminated by any startup time associated with MPI
  */
#if defined(PETSC_USE_LOG)
  CHKERRQ(PetscLogInitialize());
#endif

  CHKERRQ(PetscCitationsInitialize());

#if defined(PETSC_HAVE_SAWS)
  CHKERRQ(PetscInitializeSAWs(ftn ? NULL : help));
  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-stack_view",&flg));
  if (flg) CHKERRQ(PetscStackViewSAWs());
#endif

  /*
     Load the dynamic libraries (on machines that support them), this registers all
     the solvers etc. (On non-dynamic machines this initializes the PetscDraw and PetscViewer classes)
  */
  CHKERRQ(PetscInitialize_DynamicLibraries());

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscInfo(NULL,"PETSc successfully started: number of processors = %d\n",size));
  CHKERRQ(PetscGetHostName(hostname,256));
  CHKERRQ(PetscInfo(NULL,"Running on machine: %s\n",hostname));
#if defined(PETSC_HAVE_OPENMP)
  {
    PetscBool       omp_view_flag;
    char           *threads = getenv("OMP_NUM_THREADS");
    PetscErrorCode  ierr;

    if (threads) {
      CHKERRQ(PetscInfo(NULL,"Number of OpenMP threads %s (as given by OMP_NUM_THREADS)\n",threads));
      (void) sscanf(threads, "%" PetscInt_FMT,&PetscNumOMPThreads);
    } else {
      PetscNumOMPThreads = (PetscInt) omp_get_max_threads();
      CHKERRQ(PetscInfo(NULL,"Number of OpenMP threads %" PetscInt_FMT " (as given by omp_get_max_threads())\n",PetscNumOMPThreads));
    }
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"OpenMP options","Sys");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsInt("-omp_num_threads","Number of OpenMP threads to use (can also use environmental variable OMP_NUM_THREADS","None",PetscNumOMPThreads,&PetscNumOMPThreads,&flg));
    CHKERRQ(PetscOptionsName("-omp_view","Display OpenMP number of threads",NULL,&omp_view_flag));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      CHKERRQ(PetscInfo(NULL,"Number of OpenMP theads %" PetscInt_FMT " (given by -omp_num_threads)\n",PetscNumOMPThreads));
      omp_set_num_threads((int)PetscNumOMPThreads);
    }
    if (omp_view_flag) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"OpenMP: number of threads %" PetscInt_FMT "\n",PetscNumOMPThreads));
    }
  }
#endif

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
  /*
      Tell MPI about our own data representation converter, this would/should be used if extern32 is not supported by the MPI

      Currently not used because it is not supported by MPICH.
  */
  if (!PetscBinaryBigEndian()) CHKERRMPI(MPI_Register_datarep((char*)"petsc",PetscDataRep_read_conv_fn,PetscDataRep_write_conv_fn,PetscDataRep_extent_fn,NULL));
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
  CHKERRQ(PetscFPTCreate(10000));
#endif

#if defined(PETSC_HAVE_HWLOC)
  {
    PetscViewer viewer;
    CHKERRQ(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-process_view",&viewer,NULL,&flg));
    if (flg) {
      CHKERRQ(PetscProcessPlacementView(viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }
  }
#endif

  flg  = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-viewfromoptions",&flg,NULL));
  if (!flg) CHKERRQ(PetscOptionsPushGetViewerOff(PETSC_TRUE));

#if defined(PETSC_HAVE_ADIOS)
  CHKERRQ(adios_init_noxml(PETSC_COMM_WORLD));
  CHKERRQ(adios_declare_group(&Petsc_adios_group,"PETSc","",adios_stat_default));
  CHKERRQ(adios_select_method(Petsc_adios_group,"MPI","",""));
  CHKERRQ(adios_read_init_method(ADIOS_READ_METHOD_BP,PETSC_COMM_WORLD,""));
#endif

#if defined(__VALGRIND_H)
  PETSC_RUNNING_ON_VALGRIND = RUNNING_ON_VALGRIND? PETSC_TRUE: PETSC_FALSE;
#if defined(PETSC_USING_DARWIN) && defined(PETSC_BLASLAPACK_SDOT_RETURNS_DOUBLE)
  if (PETSC_RUNNING_ON_VALGRIND) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"WARNING: Running valgrind with the MacOS native BLAS and LAPACK can fail. If it fails suggest configuring with --download-fblaslapack or --download-f2cblaslapack"));
#endif
#endif
  /*
      Set flag that we are completely initialized
  */
  PetscInitializeCalled = PETSC_TRUE;

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-python",&flg));
  if (flg) CHKERRQ(PetscPythonInitialize(NULL,NULL));
  PetscFunctionReturn(0);
}

/*@C
   PetscInitialize - Initializes the PETSc database and MPI.
   PetscInitialize() calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of
   your program -- usually the very first line!

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, append ":yaml" to filename to specify YAML options format.
          Use NULL or empty string to not check for code specific file.
          Also checks ~/.petscrc, .petscrc and petscrc.
          Use -skip_petscrc in the code specific file (or command line) to skip ~/.petscrc, .petscrc and petscrc files.
-  help - [optional] Help message to print, use NULL for no message

   If you wish PETSc code to run ONLY on a subcommunicator of MPI_COMM_WORLD, create that
   communicator first and assign it to PETSC_COMM_WORLD BEFORE calling PetscInitialize(). Thus if you are running a
   four process job and two processes will run PETSc and have PetscInitialize() and PetscFinalize() and two process will not,
   then do this. If ALL processes in the job are using PetscInitialize() and PetscFinalize() then you don't need to do this, even
   if different subcommunicators of the job are doing different things with PETSc.

   Options Database Keys:
+  -help [intro] - prints help method for each option; if intro is given the program stops after printing the introductory help message
.  -start_in_debugger [noxterm,dbx,xdb,gdb,...] - Starts program in debugger
.  -on_error_attach_debugger [noxterm,dbx,xdb,gdb,...] - Starts debugger when error detected
.  -on_error_emacs <machinename> - causes emacsclient to jump to error file
.  -on_error_abort - calls abort() when error detected (no traceback)
.  -on_error_mpiabort - calls MPI_abort() when error detected
.  -error_output_stderr - prints error messages to stderr instead of the default stdout
.  -error_output_none - does not print the error messages (but handles errors in the same way as if this was not called)
.  -debugger_ranks [rank1,rank2,...] - Indicates ranks to start in debugger
.  -debugger_pause [sleeptime] (in seconds) - Pauses debugger
.  -stop_for_debugger - Print message on how to attach debugger manually to
                        process and wait (-debugger_pause) seconds for attachment
.  -malloc - Indicates use of PETSc error-checking malloc (on by default for debug version of libraries) (deprecated, use -malloc_debug)
.  -malloc no - Indicates not to use error-checking malloc (deprecated, use -malloc_debug no)
.  -malloc_debug - check for memory corruption at EVERY malloc or free, see PetscMallocSetDebug()
.  -malloc_dump - prints a list of all unfreed memory at the end of the run
.  -malloc_test - like -malloc_dump -malloc_debug, but only active for debugging builds, ignored in optimized build. May want to set in PETSC_OPTIONS environmental variable
.  -malloc_view - show a list of all allocated memory during PetscFinalize()
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
-  -options_monitor_cancel - cancel options monitoring hard-wired using PetscOptionsMonitorSet()

   Options -options_monitor_{all,cancel} are
   position-independent and apply to all options set since the PETSc start.
   They can be used also in option files.

   See PetscOptionsMonitorSet() to do monitoring programmatically.

   Options Database Keys for Profiling:
   See Users-Manual: ch_profiling for details.
+  -info [filename][:[~]<list,of,classnames>[:[~]self]] - Prints verbose information. See PetscInfo().
.  -log_sync - Enable barrier synchronization for all events. This option is useful to debug imbalance within each event,
        however it slows things down and gives a distorted view of the overall runtime.
.  -log_trace [filename] - Print traces of all PETSc calls to the screen (useful to determine where a program
        hangs without running in the debugger).  See PetscLogTraceBegin().
.  -log_view [:filename:format] - Prints summary of flop and timing information to screen or file, see PetscLogView().
.  -log_view_memory - Includes in the summary from -log_view the memory used in each method, see PetscLogView().
.  -log_summary [filename] - (Deprecated, use -log_view) Prints summary of flop and timing information to screen. If the filename is specified the
        summary is written to the file.  See PetscLogView().
.  -log_exclude: <vec,mat,pc,ksp,snes> - excludes subset of object classes from logging
.  -log_all [filename] - Logs extensive profiling information  See PetscLogDump().
.  -log [filename] - Logs basic profiline information  See PetscLogDump().
.  -log_mpe [filename] - Creates a logfile viewable by the utility Jumpshot (in MPICH distribution)
.  -viewfromoptions on,off - Enable or disable XXXSetFromOptions() calls, for applications with many small solves turn this off
-  -check_pointer_intensity 0,1,2 - if pointers are checked for validity (debug version only), using 0 will result in faster code

    Only one of -log_trace, -log_view, -log_view, -log_all, -log, or -log_mpe may be used at a time

   Options Database Keys for SAWs:
+  -saws_port <portnumber> - port number to publish SAWs data, default is 8080
.  -saws_port_auto_select - have SAWs select a new unique port number where it publishes the data, the URL is printed to the screen
                            this is useful when you are running many jobs that utilize SAWs at the same time
.  -saws_log <filename> - save a log of all SAWs communication
.  -saws_https <certificate file> - have SAWs use HTTPS instead of HTTP
-  -saws_root <directory> - allow SAWs to have access to the given directory to search for requested resources and files

   Environmental Variables:
+   PETSC_TMP - alternative tmp directory
.   PETSC_SHARED_TMP - tmp is shared by all processes
.   PETSC_NOT_SHARED_TMP - each process has its own private tmp
.   PETSC_OPTIONS - a string containing additional options for petsc in the form of command line "-key value" pairs
.   PETSC_OPTIONS_YAML - (requires configuring PETSc to use libyaml) a string containing additional options for petsc in the form of a YAML document
.   PETSC_VIEWER_SOCKET_PORT - socket number to use for socket viewer
-   PETSC_VIEWER_SOCKET_MACHINE - machine to use for socket viewer to connect to

   Level: beginner

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   Fortran Version:
   In Fortran this routine has the format
$       call PetscInitialize(file,ierr)

+  ierr - error return code
-  file - [optional] PETSc database file, also checks ~/.petscrc, .petscrc and petscrc.
          Use PETSC_NULL_CHARACTER to not check for code specific file.
          Use -skip_petscrc in the code specific file (or command line) to skip ~/.petscrc, .petscrc and petscrc files.

   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use NULL as
   in the C version. See Users-Manual: ch_fortran for details.

   If your main program is C but you call Fortran code that also uses PETSc you need to call PetscInitializeFortran() soon after
   calling PetscInitialize().

.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscInitializeNoArguments()

@*/
PetscErrorCode  PetscInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscMPIInt    flag;
  const char     *prog = "Unknown Name";

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Initialized(&flag));
  if (!flag) {
    PetscCheckFalse(PETSC_COMM_WORLD != MPI_COMM_NULL,PETSC_COMM_SELF,PETSC_ERR_SUP,"You cannot set PETSC_COMM_WORLD if you have not initialized MPI first");
    CHKERRQ(PetscPreMPIInit_Private());
#if defined(PETSC_HAVE_MPI_INIT_THREAD)
    {
      PetscMPIInt provided;
      CHKERRMPI(MPI_Init_thread(argc,args,PETSC_MPI_THREAD_REQUIRED,&provided));
    }
#else
    CHKERRMPI(MPI_Init(argc,args));
#endif
    PetscBeganMPI = PETSC_TRUE;
  }

  if (argc && *argc) prog = **args;
  if (argc && args) {
    PetscGlobalArgc = *argc;
    PetscGlobalArgs = *args;
  }
  CHKERRQ(PetscInitialize_Common(prog,file,help,PETSC_FALSE/*C*/,PETSC_FALSE,0));
  PetscFunctionReturn(0);
}

#if PetscDefined(USE_LOG)
PETSC_INTERN PetscObject *PetscObjects;
PETSC_INTERN PetscInt    PetscObjectsCounts;
PETSC_INTERN PetscInt    PetscObjectsMaxCounts;
PETSC_INTERN PetscBool   PetscObjectsLog;
#endif

/*
    Frees all the MPI types and operations that PETSc may have created
*/
PetscErrorCode  PetscFreeMPIResources(void)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_REAL___FLOAT128)
  CHKERRMPI(MPI_Type_free(&MPIU___FLOAT128));
#if defined(PETSC_HAVE_COMPLEX)
  CHKERRMPI(MPI_Type_free(&MPIU___COMPLEX128));
#endif
  CHKERRMPI(MPI_Op_free(&MPIU_MAX));
  CHKERRMPI(MPI_Op_free(&MPIU_MIN));
#elif defined(PETSC_USE_REAL___FP16)
  CHKERRMPI(MPI_Type_free(&MPIU___FP16));
  CHKERRMPI(MPI_Op_free(&MPIU_MAX));
  CHKERRMPI(MPI_Op_free(&MPIU_MIN));
#endif

#if defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
  CHKERRMPI(MPI_Op_free(&MPIU_SUM));
#endif

  CHKERRMPI(MPI_Type_free(&MPIU_2SCALAR));
  CHKERRMPI(MPI_Type_free(&MPIU_REAL_INT));
  CHKERRMPI(MPI_Type_free(&MPIU_SCALAR_INT));
#if defined(PETSC_USE_64BIT_INDICES)
  CHKERRMPI(MPI_Type_free(&MPIU_2INT));
#endif
  CHKERRMPI(MPI_Type_free(&MPI_4INT));
  CHKERRMPI(MPI_Type_free(&MPIU_4INT));
  CHKERRMPI(MPI_Op_free(&MPIU_MAXSUM_OP));
  PetscFunctionReturn(0);
}

#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogFinalize(void);
#endif

/*@C
   PetscFinalize - Checks for options to be called at the conclusion
   of the program. MPI_Finalize() is called only if the user had not
   called MPI_Init() before calling PetscInitialize().

   Collective on PETSC_COMM_WORLD

   Options Database Keys:
+  -options_view - Calls PetscOptionsView()
.  -options_left - Prints unused options that remain in the database
.  -objects_dump [all] - Prints list of objects allocated by the user that have not been freed, the option all cause all outstanding objects to be listed
.  -mpidump - Calls PetscMPIDump()
.  -malloc_dump <optional filename> - Calls PetscMallocDump(), displays all memory allocated that has not been freed
.  -malloc_info - Prints total memory usage
-  -malloc_view <optional filename> - Prints list of all memory allocated and where

   Level: beginner

   Note:
   See PetscInitialize() for more general runtime options.

.seealso: PetscInitialize(), PetscOptionsView(), PetscMallocDump(), PetscMPIDump(), PetscEnd()
@*/
PetscErrorCode  PetscFinalize(void)
{
  PetscMPIInt    rank;
  PetscInt       nopt;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE;
  PetscBool      flg;
#if defined(PETSC_USE_LOG)
  char           mname[PETSC_MAX_PATH_LEN];
#endif

  PetscFunctionBegin;
  if (PetscUnlikely(!PetscInitializeCalled)) {
    fprintf(PETSC_STDOUT,"PetscInitialize() must be called before PetscFinalize()\n");
    CHKERRQ(PetscStackView(PETSC_STDOUT));
    PetscStackClearTop;
    return PETSC_ERR_ARG_WRONGSTATE;
  }
  CHKERRQ(PetscInfo(NULL,"PetscFinalize() called\n"));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
#if defined(PETSC_HAVE_ADIOS)
  CHKERRQ(adios_read_finalize_method(ADIOS_READ_METHOD_BP_AGGREGATE));
  CHKERRQ(adios_finalize(rank));
#endif
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-citations",&flg));
  if (flg) {
    char  *cits, filename[PETSC_MAX_PATH_LEN];
    FILE  *fd = PETSC_STDOUT;

    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-citations",filename,sizeof(filename),NULL));
    if (filename[0]) {
      CHKERRQ(PetscFOpen(PETSC_COMM_WORLD,filename,"w",&fd));
    }
    CHKERRQ(PetscSegBufferGet(PetscCitationsList,1,&cits));
    cits[0] = 0;
    CHKERRQ(PetscSegBufferExtractAlloc(PetscCitationsList,&cits));
    CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,fd,"If you publish results based on this computation please cite the following:\n"));
    CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,fd,"===========================================================================\n"));
    CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,fd,"%s",cits));
    CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,fd,"===========================================================================\n"));
    CHKERRQ(PetscFClose(PETSC_COMM_WORLD,fd));
    CHKERRQ(PetscFree(cits));
  }
  CHKERRQ(PetscSegBufferDestroy(&PetscCitationsList));

#if defined(PETSC_HAVE_SSL) && defined(PETSC_USE_SOCKET_VIEWER)
  /* TextBelt is run for testing purposes only, please do not use this feature often */
  {
    PetscInt nmax = 2;
    char     **buffs;
    CHKERRQ(PetscMalloc1(2,&buffs));
    CHKERRQ(PetscOptionsGetStringArray(NULL,NULL,"-textbelt",buffs,&nmax,&flg1));
    if (flg1) {
      PetscCheck(nmax,PETSC_COMM_WORLD,PETSC_ERR_USER,"-textbelt requires either the phone number or number,\"message\"");
      if (nmax == 1) {
        CHKERRQ(PetscMalloc1(128,&buffs[1]));
        CHKERRQ(PetscGetProgramName(buffs[1],32));
        CHKERRQ(PetscStrcat(buffs[1]," has completed"));
      }
      CHKERRQ(PetscTextBelt(PETSC_COMM_WORLD,buffs[0],buffs[1],NULL));
      CHKERRQ(PetscFree(buffs[0]));
      CHKERRQ(PetscFree(buffs[1]));
    }
    CHKERRQ(PetscFree(buffs));
  }
  {
    PetscInt nmax = 2;
    char     **buffs;
    CHKERRQ(PetscMalloc1(2,&buffs));
    CHKERRQ(PetscOptionsGetStringArray(NULL,NULL,"-tellmycell",buffs,&nmax,&flg1));
    if (flg1) {
      PetscCheck(nmax,PETSC_COMM_WORLD,PETSC_ERR_USER,"-tellmycell requires either the phone number or number,\"message\"");
      if (nmax == 1) {
        CHKERRQ(PetscMalloc1(128,&buffs[1]));
        CHKERRQ(PetscGetProgramName(buffs[1],32));
        CHKERRQ(PetscStrcat(buffs[1]," has completed"));
      }
      CHKERRQ(PetscTellMyCell(PETSC_COMM_WORLD,buffs[0],buffs[1],NULL));
      CHKERRQ(PetscFree(buffs[0]));
      CHKERRQ(PetscFree(buffs[1]));
    }
    CHKERRQ(PetscFree(buffs));
  }
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
  CHKERRQ(PetscFPTDestroy());
#endif

#if defined(PETSC_HAVE_SAWS)
  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-saw_options",&flg,NULL));
  if (flg) {
    CHKERRQ(PetscOptionsSAWsDestroy());
  }
#endif

#if defined(PETSC_HAVE_X)
  flg1 = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-x_virtual",&flg1,NULL));
  if (flg1) {
    /*  this is a crude hack, but better than nothing */
    CHKERRQ(PetscPOpen(PETSC_COMM_WORLD,NULL,"pkill -9 Xvfb","r",NULL));
  }
#endif

#if !defined(PETSC_HAVE_THREADSAFETY)
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-malloc_info",&flg2,NULL));
  if (!flg2) {
    flg2 = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-memory_view",&flg2,NULL));
  }
  if (flg2) {
    CHKERRQ(PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD,"Summary of Memory Usage in PETSc\n"));
  }
#endif

#if defined(PETSC_USE_LOG)
  flg1 = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-get_total_flops",&flg1,NULL));
  if (flg1) {
    PetscLogDouble flops = 0;
    CHKERRMPI(MPI_Reduce(&petsc_TotalFlops,&flops,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Total flops over all processors %g\n",flops));
  }
#endif

#if defined(PETSC_USE_LOG)
#if defined(PETSC_HAVE_MPE)
  mname[0] = 0;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_mpe",mname,sizeof(mname),&flg1));
  if (flg1) {
    if (mname[0]) CHKERRQ(PetscLogMPEDump(mname));
    else          CHKERRQ(PetscLogMPEDump(0));
  }
#endif
#endif

  /*
     Free all objects registered with PetscObjectRegisterDestroy() such as PETSC_VIEWER_XXX_().
  */
  CHKERRQ(PetscObjectRegisterDestroyAll());

#if defined(PETSC_USE_LOG)
  CHKERRQ(PetscOptionsPushGetViewerOff(PETSC_FALSE));
  CHKERRQ(PetscLogViewFromOptions());
  CHKERRQ(PetscOptionsPopGetViewerOff());

  mname[0] = 0;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_summary",mname,sizeof(mname),&flg1));
  if (flg1) {
    PetscViewer viewer;
    CHKERRQ((*PetscHelpPrintf)(PETSC_COMM_WORLD,"\n\n WARNING:   -log_summary is being deprecated; switch to -log_view\n\n\n"));
    if (mname[0]) {
      CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_WORLD,mname,&viewer));
      CHKERRQ(PetscLogView(viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
    } else {
      viewer = PETSC_VIEWER_STDOUT_WORLD;
      CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_DEFAULT));
      CHKERRQ(PetscLogView(viewer));
      CHKERRQ(PetscViewerPopFormat(viewer));
    }
  }

  /*
     Free any objects created by the last block of code.
  */
  CHKERRQ(PetscObjectRegisterDestroyAll());

  mname[0] = 0;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_all",mname,sizeof(mname),&flg1));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log",mname,sizeof(mname),&flg2));
  if (flg1 || flg2) CHKERRQ(PetscLogDump(mname));
#endif

  flg1 = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-no_signal_handler",&flg1,NULL));
  if (!flg1) CHKERRQ(PetscPopSignalHandler());
  flg1 = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-mpidump",&flg1,NULL));
  if (flg1) {
    CHKERRQ(PetscMPIDump(stdout));
  }
  flg1 = PETSC_FALSE;
  flg2 = PETSC_FALSE;
  /* preemptive call to avoid listing this option in options table as unused */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-malloc_dump",&flg1));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-objects_dump",&flg1));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-options_view",&flg2,NULL));

  if (flg2) {
    PetscViewer viewer;
    CHKERRQ(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
    CHKERRQ(PetscViewerSetType(viewer,PETSCVIEWERASCII));
    CHKERRQ(PetscOptionsView(NULL,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  /* to prevent PETSc -options_left from warning */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-nox",&flg1));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-nox_warning",&flg1));

  flg3 = PETSC_FALSE; /* default value is required */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-options_left",&flg3,&flg1));
  if (PetscUnlikelyDebug(!flg1)) flg3 = PETSC_TRUE;
  if (flg3) {
    if (!flg2 && flg1) { /* have not yet printed the options */
      PetscViewer viewer;
      CHKERRQ(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
      CHKERRQ(PetscViewerSetType(viewer,PETSCVIEWERASCII));
      CHKERRQ(PetscOptionsView(NULL,viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }
    CHKERRQ(PetscOptionsAllUsed(NULL,&nopt));
    if (nopt) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"WARNING! There are options you set that were not used!\n"));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"WARNING! could be spelling mistake, etc!\n"));
      if (nopt == 1) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"There is one unused database option. It is:\n"));
      } else {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"There are %" PetscInt_FMT " unused database options. They are:\n",nopt));
      }
    } else if (flg3 && flg1) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"There are no unused options.\n"));
    }
    CHKERRQ(PetscOptionsLeft(NULL));
  }

#if defined(PETSC_HAVE_SAWS)
  if (!PetscGlobalRank) {
    CHKERRQ(PetscStackSAWsViewOff());
    PetscStackCallSAWs(SAWs_Finalize,());
  }
#endif

#if defined(PETSC_USE_LOG)
  /*
       List all objects the user may have forgot to free
  */
  if (PetscObjectsLog) {
    CHKERRQ(PetscOptionsHasName(NULL,NULL,"-objects_dump",&flg1));
    if (flg1) {
      MPI_Comm local_comm;
      char     string[64];

      CHKERRQ(PetscOptionsGetString(NULL,NULL,"-objects_dump",string,sizeof(string),NULL));
      CHKERRMPI(MPI_Comm_dup(MPI_COMM_WORLD,&local_comm));
      CHKERRQ(PetscSequentialPhaseBegin_Private(local_comm,1));
      CHKERRQ(PetscObjectsDump(stdout,(string[0] == 'a') ? PETSC_TRUE : PETSC_FALSE));
      CHKERRQ(PetscSequentialPhaseEnd_Private(local_comm,1));
      CHKERRMPI(MPI_Comm_free(&local_comm));
    }
  }
#endif

#if defined(PETSC_USE_LOG)
  PetscObjectsCounts    = 0;
  PetscObjectsMaxCounts = 0;
  CHKERRQ(PetscFree(PetscObjects));
#endif

  /*
     Destroy any packages that registered a finalize
  */
  CHKERRQ(PetscRegisterFinalizeAll());

#if defined(PETSC_USE_LOG)
  CHKERRQ(PetscLogFinalize());
#endif

  /*
     Print PetscFunctionLists that have not been properly freed

  CHKERRQ(PetscFunctionListPrintAll());
  */

  if (petsc_history) {
    CHKERRQ(PetscCloseHistoryFile(&petsc_history));
    petsc_history = NULL;
  }
  CHKERRQ(PetscOptionsHelpPrintedDestroy(&PetscOptionsHelpPrintedSingleton));
  CHKERRQ(PetscInfoDestroy());

#if !defined(PETSC_HAVE_THREADSAFETY)
  if (!(PETSC_RUNNING_ON_VALGRIND)) {
    char fname[PETSC_MAX_PATH_LEN];
    char sname[PETSC_MAX_PATH_LEN];
    FILE *fd;
    int  err;

    flg2 = PETSC_FALSE;
    flg3 = PETSC_FALSE;
    if (PetscDefined(USE_DEBUG)) CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-malloc_test",&flg2,NULL));
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-malloc_debug",&flg3,NULL));
    fname[0] = 0;
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-malloc_dump",fname,sizeof(fname),&flg1));
    if (flg1 && fname[0]) {

      PetscSNPrintf(sname,sizeof(sname),"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); PetscCheck(fd,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open log file: %s",sname);
      CHKERRQ(PetscMallocDump(fd));
      err  = fclose(fd);
      PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
    } else if (flg1 || flg2 || flg3) {
      MPI_Comm local_comm;

      CHKERRMPI(MPI_Comm_dup(MPI_COMM_WORLD,&local_comm));
      CHKERRQ(PetscSequentialPhaseBegin_Private(local_comm,1));
      CHKERRQ(PetscMallocDump(stdout));
      CHKERRQ(PetscSequentialPhaseEnd_Private(local_comm,1));
      CHKERRMPI(MPI_Comm_free(&local_comm));
    }
    fname[0] = 0;
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-malloc_view",fname,sizeof(fname),&flg1));
    if (flg1 && fname[0]) {

      PetscSNPrintf(sname,sizeof(sname),"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); PetscCheck(fd,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open log file: %s",sname);
      CHKERRQ(PetscMallocView(fd));
      err  = fclose(fd);
      PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
    } else if (flg1) {
      MPI_Comm local_comm;

      CHKERRMPI(MPI_Comm_dup(MPI_COMM_WORLD,&local_comm));
      CHKERRQ(PetscSequentialPhaseBegin_Private(local_comm,1));
      CHKERRQ(PetscMallocView(stdout));
      CHKERRQ(PetscSequentialPhaseEnd_Private(local_comm,1));
      CHKERRMPI(MPI_Comm_free(&local_comm));
    }
  }
#endif

  /*
     Close any open dynamic libraries
  */
  CHKERRQ(PetscFinalize_DynamicLibraries());

  /* Can be destroyed only after all the options are used */
  CHKERRQ(PetscOptionsDestroyDefault());

  PetscGlobalArgc = 0;
  PetscGlobalArgs = NULL;

#if defined(PETSC_HAVE_KOKKOS)
  if (PetscBeganKokkos) {
    CHKERRQ(PetscKokkosFinalize_Private());
    PetscBeganKokkos = PETSC_FALSE;
    PetscKokkosInitialized = PETSC_FALSE;
  }
#endif

#if defined(PETSC_HAVE_NVSHMEM)
  if (PetscBeganNvshmem) {
    CHKERRQ(PetscNvshmemFinalize());
    PetscBeganNvshmem = PETSC_FALSE;
  }
#endif

  CHKERRQ(PetscFreeMPIResources());

  /*
     Destroy any known inner MPI_Comm's and attributes pointing to them
     Note this will not destroy any new communicators the user has created.

     If all PETSc objects were not destroyed those left over objects will have hanging references to
     the MPI_Comms that were freed; but that is ok because those PETSc objects will never be used again
 */
  {
    PetscCommCounter *counter;
    PetscMPIInt      flg;
    MPI_Comm         icomm;
    union {MPI_Comm comm; void *ptr;} ucomm;
    CHKERRMPI(MPI_Comm_get_attr(PETSC_COMM_SELF,Petsc_InnerComm_keyval,&ucomm,&flg));
    if (flg) {
      icomm = ucomm.comm;
      CHKERRMPI(MPI_Comm_get_attr(icomm,Petsc_Counter_keyval,&counter,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");

      CHKERRMPI(MPI_Comm_delete_attr(PETSC_COMM_SELF,Petsc_InnerComm_keyval));
      CHKERRMPI(MPI_Comm_delete_attr(icomm,Petsc_Counter_keyval));
      CHKERRMPI(MPI_Comm_free(&icomm));
    }
    CHKERRMPI(MPI_Comm_get_attr(PETSC_COMM_WORLD,Petsc_InnerComm_keyval,&ucomm,&flg));
    if (flg) {
      icomm = ucomm.comm;
      CHKERRMPI(MPI_Comm_get_attr(icomm,Petsc_Counter_keyval,&counter,&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");

      CHKERRMPI(MPI_Comm_delete_attr(PETSC_COMM_WORLD,Petsc_InnerComm_keyval));
      CHKERRMPI(MPI_Comm_delete_attr(icomm,Petsc_Counter_keyval));
      CHKERRMPI(MPI_Comm_free(&icomm));
    }
  }

  CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Counter_keyval));
  CHKERRMPI(MPI_Comm_free_keyval(&Petsc_InnerComm_keyval));
  CHKERRMPI(MPI_Comm_free_keyval(&Petsc_OuterComm_keyval));
  CHKERRMPI(MPI_Comm_free_keyval(&Petsc_ShmComm_keyval));

  CHKERRQ(PetscSpinlockDestroy(&PetscViewerASCIISpinLockOpen));
  CHKERRQ(PetscSpinlockDestroy(&PetscViewerASCIISpinLockStdout));
  CHKERRQ(PetscSpinlockDestroy(&PetscViewerASCIISpinLockStderr));
  CHKERRQ(PetscSpinlockDestroy(&PetscCommSpinLock));

  if (PetscBeganMPI) {
    PetscMPIInt flag;
    CHKERRMPI(MPI_Finalized(&flag));
    PetscCheckFalse(flag,PETSC_COMM_SELF,PETSC_ERR_LIB,"MPI_Finalize() has already been called, even though MPI_Init() was called by PetscInitialize()");
    CHKERRMPI(MPI_Finalize());
  }
/*

     Note: In certain cases PETSC_COMM_WORLD is never MPI_Comm_free()ed because
   the communicator has some outstanding requests on it. Specifically if the
   flag PETSC_HAVE_BROKEN_REQUEST_FREE is set (for IBM MPI implementation). See
   src/vec/utils/vpscat.c. Due to this the memory allocated in PetscCommDuplicate()
   is never freed as it should be. Thus one may obtain messages of the form
   [ 1] 8 bytes PetscCommDuplicate() line 645 in src/sys/mpiu.c indicating the
   memory was not freed.

*/
  CHKERRQ(PetscMallocClear());
  CHKERRQ(PetscStackReset());

  PetscErrorHandlingInitialized = PETSC_FALSE;
  PetscInitializeCalled = PETSC_FALSE;
  PetscFinalizeCalled   = PETSC_TRUE;
#if defined(PETSC_USE_GCOV)
  /*
     flush gcov, otherwise during CI the flushing continues into the next pipeline resulting in git not being able to delete directories since the
     gcov files are still being added to the directories as git tries to remove the directories.
   */
  __gcov_flush();
#endif
  /* To match PetscFunctionBegin() at the beginning of this function */
  PetscStackClearTop;
  return 0;
}

#if defined(PETSC_MISSING_LAPACK_lsame_)
PETSC_EXTERN int lsame_(char *a,char *b)
{
  if (*a == *b) return 1;
  if (*a + 32 == *b) return 1;
  if (*a - 32 == *b) return 1;
  return 0;
}
#endif

#if defined(PETSC_MISSING_LAPACK_lsame)
PETSC_EXTERN int lsame(char *a,char *b)
{
  if (*a == *b) return 1;
  if (*a + 32 == *b) return 1;
  if (*a - 32 == *b) return 1;
  return 0;
}
#endif
