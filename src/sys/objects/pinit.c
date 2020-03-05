
/*
   This file defines the initialization of PETSc, including PetscInitialize()
*/
#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/
#include <petscvalgrind.h>
#include <petscviewer.h>

#if defined(PETSC_USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogFinalize(void);
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
PETSC_INTERN PetscFPT PetscFPTData;
PetscFPT PetscFPTData = 0;
#endif

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
#endif

#if defined(PETSC_HAVE_OPENMPI_MAJOR)
#include "mpi-ext.h" /* Needed for CUDA-aware check */
#endif
/* -----------------------------------------------------------------------------------------*/

PETSC_INTERN FILE *petsc_history;

PETSC_INTERN PetscErrorCode PetscInitialize_DynamicLibraries(void);
PETSC_INTERN PetscErrorCode PetscFinalize_DynamicLibraries(void);
PETSC_INTERN PetscErrorCode PetscFunctionListPrintAll(void);
PETSC_INTERN PetscErrorCode PetscSequentialPhaseBegin_Private(MPI_Comm,int);
PETSC_INTERN PetscErrorCode PetscSequentialPhaseEnd_Private(MPI_Comm,int);
PETSC_INTERN PetscErrorCode PetscCloseHistoryFile(FILE**);

/* user may set this BEFORE calling PetscInitialize() */
MPI_Comm PETSC_COMM_WORLD = MPI_COMM_NULL;

PetscMPIInt Petsc_Counter_keyval   = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_InnerComm_keyval = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_OuterComm_keyval = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_ShmComm_keyval   = MPI_KEYVAL_INVALID;

/* Do not need put this in a guard like PETSC_HAVE_CUDA. Without configuring PETSc --with-cuda, users can still use option -use_gpu_aware_mpi */
PetscBool use_gpu_aware_mpi = PETSC_FALSE;
#if defined(PETSC_HAVE_CUDA)
PetscBool sf_use_default_cuda_stream = PETSC_FALSE;
#endif

/*
     Declare and set all the string names of the PETSc enums
*/
const char *const PetscBools[]     = {"FALSE","TRUE","PetscBool","PETSC_",NULL};
const char *const PetscCopyModes[] = {"COPY_VALUES","OWN_POINTER","USE_POINTER","PetscCopyMode","PETSC_",NULL};

PetscBool PetscPreLoadingUsed = PETSC_FALSE;
PetscBool PetscPreLoadingOn   = PETSC_FALSE;

PetscInt PetscHotRegionDepth;

#if defined(PETSC_HAVE_THREADSAFETY)
PetscSpinlock PetscViewerASCIISpinLockOpen;
PetscSpinlock PetscViewerASCIISpinLockStdout;
PetscSpinlock PetscViewerASCIISpinLockStderr;
PetscSpinlock PetscCommSpinLock;
#endif

/*
       Checks the options database for initializations related to the
    PETSc components
*/
PETSC_INTERN PetscErrorCode  PetscOptionsCheckInitial_Components(void)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasHelp(NULL,&flg);CHKERRQ(ierr);
  if (flg) {
#if defined(PETSC_USE_LOG)
    MPI_Comm comm = PETSC_COMM_WORLD;
    ierr = (*PetscHelpPrintf)(comm,"------Additional PETSc component options--------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_exclude: <vec,mat,pc,ksp,snes,tao,ts>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -info_exclude: <null,vec,mat,pc,ksp,snes,tao,ts>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"-----------------------------------------------\n");CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

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
  ierr = PetscInitialize(&myargc,&myargs,filename,help);if (ierr) return ierr;
  ierr = PetscPopSignalHandler();CHKERRQ(ierr);
  PetscBeganMPI = PETSC_FALSE;
  PetscFunctionReturn(ierr);
}

/*
      Used by Julia interface to get communicator
*/
PetscErrorCode  PetscGetPETSC_COMM_SELF(MPI_Comm *comm)
{
  PetscFunctionBegin;
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
PetscErrorCode PetscInitialized(PetscBool  *isInitialized)
{
  *isInitialized = PetscInitializeCalled;
  return 0;
}

/*@
      PetscFinalized - Determine whether PetscFinalize() has been called yet

   Level: developer

.seealso: PetscInitialize(), PetscInitializeNoArguments(), PetscInitializeFortran()
@*/
PetscErrorCode  PetscFinalized(PetscBool  *isFinalized)
{
  *isFinalized = PetscFinalizeCalled;
  return 0;
}

PETSC_INTERN PetscErrorCode PetscOptionsCheckInitial_Private(void);

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_REDUCE_SCATTER_BLOCK)
  {
    struct {PetscInt max,sum;} work;
    ierr = MPI_Reduce_scatter_block((void*)sizes,&work,1,MPIU_2INT,MPIU_MAXSUM_OP,comm);CHKERRQ(ierr);
    *max = work.max;
    *sum = work.sum;
  }
#else
  {
    PetscMPIInt    size,rank;
    struct {PetscInt max,sum;} *work;
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = PetscMalloc1(size,&work);CHKERRQ(ierr);
    ierr = MPIU_Allreduce((void*)sizes,work,size,MPIU_2INT,MPIU_MAXSUM_OP,comm);CHKERRQ(ierr);
    *max = work[rank].max;
    *sum = work[rank].sum;
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/

#if (defined(PETSC_HAVE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
MPI_Op MPIU_SUM = 0;

PETSC_EXTERN void PetscSum_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
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

PETSC_EXTERN void PetscMax_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
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

PETSC_EXTERN void PetscMin_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
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
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelCounter(MPI_Comm comm,PetscMPIInt keyval,void *count_val,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(NULL,"Deleting counter data in an MPI_Comm %ld\n",(long)comm);CHKERRMPI(ierr);
  ierr = PetscFree(count_val);CHKERRMPI(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

/*
  This is invoked on the outer comm as a result of either PetscCommDestroy() (via MPI_Comm_delete_attr) or when the user
  calls MPI_Comm_free().

  This is the only entry point for breaking the links between inner and outer comms.

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelComm_Outer(MPI_Comm comm,PetscMPIInt keyval,void *attr_val,void *extra_state)
{
  PetscErrorCode                    ierr;
  PetscMPIInt                       flg;
  union {MPI_Comm comm; void *ptr;} icomm,ocomm;

  PetscFunctionBegin;
  if (keyval != Petsc_InnerComm_keyval) SETERRMPI(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unexpected keyval");
  icomm.ptr = attr_val;

  ierr = MPI_Comm_get_attr(icomm.comm,Petsc_OuterComm_keyval,&ocomm,&flg);CHKERRMPI(ierr);
  if (!flg) SETERRMPI(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected reference to outer comm");
  if (ocomm.comm != comm) SETERRMPI(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm has reference to non-matching outer comm");
  ierr = MPI_Comm_delete_attr(icomm.comm,Petsc_OuterComm_keyval);CHKERRMPI(ierr);
  ierr = PetscInfo1(NULL,"User MPI_Comm %ld is being freed after removing reference from inner PETSc comm to this outer comm\n",(long)comm);CHKERRMPI(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

/*
 * This is invoked on the inner comm when Petsc_DelComm_Outer calls MPI_Comm_delete_attr.  It should not be reached any other way.
 */
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelComm_Inner(MPI_Comm comm,PetscMPIInt keyval,void *attr_val,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(NULL,"Removing reference to PETSc communicator embedded in a user MPI_Comm %ld\n",(long)comm);CHKERRMPI(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelComm_Shm(MPI_Comm,PetscMPIInt,void *,void *);

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSegBufferCreate(1,10000,&PetscCitationsList);CHKERRQ(ierr);
  ierr = PetscCitationsRegister("@TechReport{petsc-user-ref,\n  Author = {Satish Balay and Shrirang Abhyankar and Mark F. Adams and Jed Brown \n            and Peter Brune and Kris Buschelman and Lisandro Dalcin and\n            Victor Eijkhout and William D. Gropp and Dmitry Karpeyev and\n            Dinesh Kaushik and Matthew G. Knepley and Dave A. May and Lois Curfman McInnes\n            and Richard Tran Mills and Todd Munson and Karl Rupp and Patrick Sanan\n            and Barry F. Smith and Stefano Zampini and Hong Zhang and Hong Zhang},\n  Title = {{PETS}c Users Manual},\n  Number = {ANL-95/11 - Revision 3.11},\n  Institution = {Argonne National Laboratory},\n  Year = {2019}\n}\n",NULL);CHKERRQ(ierr);
  ierr = PetscCitationsRegister("@InProceedings{petsc-efficient,\n  Author = {Satish Balay and William D. Gropp and Lois Curfman McInnes and Barry F. Smith},\n  Title = {Efficient Management of Parallelism in Object Oriented Numerical Software Libraries},\n  Booktitle = {Modern Software Tools in Scientific Computing},\n  Editor = {E. Arge and A. M. Bruaset and H. P. Langtangen},\n  Pages = {163--202},\n  Publisher = {Birkh{\\\"{a}}user Press},\n  Year = {1997}\n}\n",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static char programname[PETSC_MAX_PATH_LEN] = ""; /* HP includes entire path in name */

PetscErrorCode  PetscSetProgramName(const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr  = PetscStrncpy(programname,name,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
   ierr = PetscStrncpy(name,programname,len);CHKERRQ(ierr);
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
  if (!PetscInitializeCalled && PetscFinalizeCalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"You must call after PetscInitialize() but before PetscFinalize()");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!PetscInitializeCalled && PetscFinalizeCalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"You must call after PetscInitialize() but before PetscFinalize()");
  if (!argc) {*args = NULL; PetscFunctionReturn(0);}
  ierr = PetscMalloc1(argc,args);CHKERRQ(ierr);
  for (i=0; i<argc-1; i++) {
    ierr = PetscStrallocpy(PetscGlobalArgs[i+1],&(*args)[i]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!args) PetscFunctionReturn(0);
  while (args[i]) {
    ierr = PetscFree(args[i]);CHKERRQ(ierr);
    i++;
  }
  ierr = PetscFree(args);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SAWS)
#include <petscconfiginfo.h>

PETSC_INTERN PetscErrorCode PetscInitializeSAWs(const char help[])
{
  if (!PetscGlobalRank) {
    char           cert[PETSC_MAX_PATH_LEN],root[PETSC_MAX_PATH_LEN],*intro,programname[64],*appline,*options,version[64];
    int            port;
    PetscBool      flg,rootlocal = PETSC_FALSE,flg2,selectport = PETSC_FALSE;
    size_t         applinelen,introlen;
    PetscErrorCode ierr;
    char           sawsurl[256];

    ierr = PetscOptionsHasName(NULL,NULL,"-saws_log",&flg);CHKERRQ(ierr);
    if (flg) {
      char  sawslog[PETSC_MAX_PATH_LEN];

      ierr = PetscOptionsGetString(NULL,NULL,"-saws_log",sawslog,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
      if (sawslog[0]) {
        PetscStackCallSAWs(SAWs_Set_Use_Logfile,(sawslog));
      } else {
        PetscStackCallSAWs(SAWs_Set_Use_Logfile,(NULL));
      }
    }
    ierr = PetscOptionsGetString(NULL,NULL,"-saws_https",cert,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      PetscStackCallSAWs(SAWs_Set_Use_HTTPS,(cert));
    }
    ierr = PetscOptionsGetBool(NULL,NULL,"-saws_port_auto_select",&selectport,NULL);CHKERRQ(ierr);
    if (selectport) {
        PetscStackCallSAWs(SAWs_Get_Available_Port,(&port));
        PetscStackCallSAWs(SAWs_Set_Port,(port));
    } else {
      ierr = PetscOptionsGetInt(NULL,NULL,"-saws_port",&port,&flg);CHKERRQ(ierr);
      if (flg) {
        PetscStackCallSAWs(SAWs_Set_Port,(port));
      }
    }
    ierr = PetscOptionsGetString(NULL,NULL,"-saws_root",root,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      PetscStackCallSAWs(SAWs_Set_Document_Root,(root));CHKERRQ(ierr);
      ierr = PetscStrcmp(root,".",&rootlocal);CHKERRQ(ierr);
    } else {
      ierr = PetscOptionsHasName(NULL,NULL,"-saws_options",&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscStrreplace(PETSC_COMM_WORLD,"${PETSC_DIR}/share/petsc/saws",root,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
        PetscStackCallSAWs(SAWs_Set_Document_Root,(root));CHKERRQ(ierr);
      }
    }
    ierr = PetscOptionsHasName(NULL,NULL,"-saws_local",&flg2);CHKERRQ(ierr);
    if (flg2) {
      char jsdir[PETSC_MAX_PATH_LEN];
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"-saws_local option requires -saws_root option");
      ierr = PetscSNPrintf(jsdir,PETSC_MAX_PATH_LEN,"%s/js",root);CHKERRQ(ierr);
      ierr = PetscTestDirectory(jsdir,'r',&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"-saws_local option requires js directory in root directory");
      PetscStackCallSAWs(SAWs_Push_Local_Header,());CHKERRQ(ierr);
    }
    ierr = PetscGetProgramName(programname,64);CHKERRQ(ierr);
    ierr = PetscStrlen(help,&applinelen);CHKERRQ(ierr);
    introlen   = 4096 + applinelen;
    applinelen += 1024;
    ierr = PetscMalloc(applinelen,&appline);CHKERRQ(ierr);
    ierr = PetscMalloc(introlen,&intro);CHKERRQ(ierr);

    if (rootlocal) {
      ierr = PetscSNPrintf(appline,applinelen,"%s.c.html",programname);CHKERRQ(ierr);
      ierr = PetscTestFile(appline,'r',&rootlocal);CHKERRQ(ierr);
    }
    ierr = PetscOptionsGetAll(NULL,&options);CHKERRQ(ierr);
    if (rootlocal && help) {
      ierr = PetscSNPrintf(appline,applinelen,"<center> Running <a href=\"%s.c.html\">%s</a> %s</center><br><center><pre>%s</pre></center><br>\n",programname,programname,options,help);CHKERRQ(ierr);
    } else if (help) {
      ierr = PetscSNPrintf(appline,applinelen,"<center>Running %s %s</center><br><center><pre>%s</pre></center><br>",programname,options,help);CHKERRQ(ierr);
    } else {
      ierr = PetscSNPrintf(appline,applinelen,"<center> Running %s %s</center><br>\n",programname,options);CHKERRQ(ierr);
    }
    ierr = PetscFree(options);CHKERRQ(ierr);
    ierr = PetscGetVersion(version,sizeof(version));CHKERRQ(ierr);
    ierr = PetscSNPrintf(intro,introlen,"<body>\n"
                                    "<center><h2> <a href=\"https://www.mcs.anl.gov/petsc\">PETSc</a> Application Web server powered by <a href=\"https://bitbucket.org/saws/saws\">SAWs</a> </h2></center>\n"
                                    "<center>This is the default PETSc application dashboard, from it you can access any published PETSc objects or logging data</center><br><center>%s configured with %s</center><br>\n"
                                    "%s",version,petscconfigureoptions,appline);CHKERRQ(ierr);
    PetscStackCallSAWs(SAWs_Push_Body,("index.html",0,intro));
    ierr = PetscFree(intro);CHKERRQ(ierr);
    ierr = PetscFree(appline);CHKERRQ(ierr);
    if (selectport) {
      PetscBool silent;

      ierr = SAWs_Initialize();
      /* another process may have grabbed the port so keep trying */
      while (ierr) {
        PetscStackCallSAWs(SAWs_Get_Available_Port,(&port));
        PetscStackCallSAWs(SAWs_Set_Port,(port));
        ierr = SAWs_Initialize();
      }

      ierr = PetscOptionsGetBool(NULL,NULL,"-saws_port_auto_select_silent",&silent,NULL);CHKERRQ(ierr);
      if (!silent) {
        PetscStackCallSAWs(SAWs_Get_FullURL,(sizeof(sawsurl),sawsurl));
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Point your browser to %s for SAWs\n",sawsurl);CHKERRQ(ierr);
      }
    } else {
      PetscStackCallSAWs(SAWs_Initialize,());
    }
    ierr = PetscCitationsRegister("@TechReport{ saws,\n"
                                  "  Author = {Matt Otten and Jed Brown and Barry Smith},\n"
                                  "  Title  = {Scientific Application Web Server (SAWs) Users Manual},\n"
                                  "  Institution = {Argonne National Laboratory},\n"
                                  "  Year   = 2013\n}\n",NULL);CHKERRQ(ierr);
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

#if defined(PETSC_HAVE_ADIOS)
#include <adios.h>
#include <adios_read.h>
int64_t Petsc_adios_group;
#endif
#if defined(PETSC_HAVE_ADIOS2)
#include <adios2_c.h>
#endif
#if defined(PETSC_HAVE_OPENMP)
#include <omp.h>
PetscInt PetscNumOMPThreads;
#endif

/*@C
   PetscInitialize - Initializes the PETSc database and MPI.
   PetscInitialize() calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of
   your program -- usually the very first line!

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, also checks ~username/.petscrc and .petscrc use NULL to not check for
          code specific file. Use -skip_petscrc in the code specific file to skip the .petscrc files
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
.  -debugger_nodes [node1,node2,...] - Indicates nodes to start in debugger
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
.  -fp_trap - Stops on floating point exceptions
.  -no_signal_handler - Indicates not to trap error signals
.  -shared_tmp - indicates /tmp directory is shared by all processors
.  -not_shared_tmp - each processor has own /tmp
.  -tmp - alternative name of /tmp directory
.  -get_total_flops - returns total flops done by all processors
-  -memory_view - Print memory usage at end of run

   Options Database Keys for Profiling:
   See Users-Manual: ch_profiling for details.
+  -info <optional filename> - Prints verbose information to the screen
.  -info_exclude <null,vec,mat,pc,ksp,snes,ts> - Excludes some of the verbose messages
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
.   PETSC_VIEWER_SOCKET_PORT - socket number to use for socket viewer
-   PETSC_VIEWER_SOCKET_MACHINE - machine to use for socket viewer to connect to


   Level: beginner

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   Fortran Version:
   In Fortran this routine has the format
$       call PetscInitialize(file,ierr)

+   ierr - error return code
-  file - [optional] PETSc database file, also checks ~username/.petscrc and .petscrc use PETSC_NULL_CHARACTER to not check for
          code specific file. Use -skip_petscrc in the code specific file to skip the .petscrc files

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
  PetscErrorCode ierr;
  PetscMPIInt    flag, size;
  PetscBool      flg = PETSC_TRUE;
  char           hostname[256];

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(0);
  /*
      The checking over compatible runtime libraries is complicated by the MPI ABI initiative
      https://wiki.mpich.org/mpich/index.php/ABI_Compatibility_Initiative which started with
        MPICH v3.1 (Released Feburary 2014)
        IBM MPI v2.1 (December 2014)
        Intel® MPI Library v5.0 (2014)
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
    ierr = MPI_Get_library_version(mpilibraryversion,&mpilibraryversionlength);if (ierr) return ierr;
    /* check for MPICH versions before MPI ABI initiative */
#if defined(MPICH_VERSION)
#if MPICH_NUMVERSION < 30100000
    {
      char *ver,*lf;
      flg = PETSC_FALSE;
      ierr = PetscStrstr(mpilibraryversion,"MPICH Version:",&ver);if (ierr) return ierr;
      if (ver) {
        ierr = PetscStrchr(ver,'\n',&lf);if (ierr) return ierr;
        if (lf) {
          *lf = 0;
          ierr = PetscStrendswith(ver,MPICH_VERSION,&flg);if (ierr) return ierr;
        }
      }
      if (!flg) {
        fprintf(stderr,"PETSc Error --- MPICH library version \n%s does not match what PETSc was compiled with %s, aborting\n",mpilibraryversion,MPICH_VERSION);
        return PETSC_ERR_MPI_LIB_INCOMP;
      }
    }
#endif
    /* check for OpenMPI version, it is not part of the MPI ABI initiative (is it part of another initiative that needs to be handled?) */
#elif defined(OMPI_MAJOR_VERSION)
    {
      char *ver,bs[32],*bsf;
      flg = PETSC_FALSE;
      ierr = PetscStrstr(mpilibraryversion,"Open MPI",&ver);if (ierr) return ierr;
      if (ver) {
        PetscSNPrintf(bs,32,"v%d.%d",OMPI_MAJOR_VERSION,OMPI_MINOR_VERSION);
        ierr = PetscStrstr(ver,bs,&bsf);if (ierr) return ierr;
        if (bsf) flg = PETSC_TRUE;
      }
      if (!flg) {
        fprintf(stderr,"PETSc Error --- Open MPI library version \n%s does not match what PETSc was compiled with %d.%d, aborting\n",mpilibraryversion,OMPI_MAJOR_VERSION,OMPI_MINOR_VERSION);
        return PETSC_ERR_MPI_LIB_INCOMP;
      }
    }
#endif
  }
#endif

  /* these must be initialized in a routine, not as a constant declaration*/
  PETSC_STDOUT = stdout;
  PETSC_STDERR = stderr;

  /* on Windows - set printf to default to printing 2 digit exponents */
#if defined(PETSC_HAVE__SET_OUTPUT_FORMAT)
  _set_output_format(_TWO_DIGIT_EXPONENT);
#endif

  ierr = PetscOptionsCreateDefault();CHKERRQ(ierr);

  /*
     We initialize the program name here (before MPI_Init()) because MPICH has a bug in
     it that it sets args[0] on all processors to be args[0] on the first processor.
  */
  if (argc && *argc) {
    ierr = PetscSetProgramName(**args);CHKERRQ(ierr);
  } else {
    ierr = PetscSetProgramName("Unknown Name");CHKERRQ(ierr);
  }

  ierr = MPI_Initialized(&flag);CHKERRQ(ierr);
  if (!flag) {
    if (PETSC_COMM_WORLD != MPI_COMM_NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"You cannot set PETSC_COMM_WORLD if you have not initialized MPI first");
    ierr = PetscPreMPIInit_Private();CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_INIT_THREAD)
    {
      PetscMPIInt provided;
      ierr = MPI_Init_thread(argc,args,MPI_THREAD_FUNNELED,&provided);CHKERRQ(ierr);
    }
#else
    ierr = MPI_Init(argc,args);CHKERRQ(ierr);
#endif
    PetscBeganMPI = PETSC_TRUE;
  }

  if (argc && args) {
    PetscGlobalArgc = *argc;
    PetscGlobalArgs = *args;
  }
  PetscFinalizeCalled = PETSC_FALSE;
  ierr = PetscSpinlockCreate(&PetscViewerASCIISpinLockOpen);CHKERRQ(ierr);
  ierr = PetscSpinlockCreate(&PetscViewerASCIISpinLockStdout);CHKERRQ(ierr);
  ierr = PetscSpinlockCreate(&PetscViewerASCIISpinLockStderr);CHKERRQ(ierr);
  ierr = PetscSpinlockCreate(&PetscCommSpinLock);CHKERRQ(ierr);

  if (PETSC_COMM_WORLD == MPI_COMM_NULL) PETSC_COMM_WORLD = MPI_COMM_WORLD;
  ierr = MPI_Comm_set_errhandler(PETSC_COMM_WORLD,MPI_ERRORS_RETURN);CHKERRQ(ierr);

  if (PETSC_MPI_ERROR_CLASS == MPI_ERR_LASTCODE) {
    ierr = MPI_Add_error_class(&PETSC_MPI_ERROR_CLASS);CHKERRQ(ierr);
    ierr = MPI_Add_error_code(PETSC_MPI_ERROR_CLASS,&PETSC_MPI_ERROR_CODE);CHKERRQ(ierr);
  }

  /* Done after init due to a bug in MPICH-GM? */
  ierr = PetscErrorPrintfInitialize();CHKERRQ(ierr);

  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);CHKERRQ(ierr);

  MPIU_BOOL = MPI_INT;
  MPIU_ENUM = MPI_INT;
  MPIU_FORTRANADDR = (sizeof(void*) == sizeof(int)) ? MPI_INT : MPIU_INT64;
  if (sizeof(size_t) == sizeof(unsigned)) MPIU_SIZE_T = MPI_UNSIGNED;
  else if (sizeof(size_t) == sizeof(unsigned long)) MPIU_SIZE_T = MPI_UNSIGNED_LONG;
#if defined(PETSC_SIZEOF_LONG_LONG)
  else if (sizeof(size_t) == sizeof(unsigned long long)) MPIU_SIZE_T = MPI_UNSIGNED_LONG_LONG;
#endif
  else {(*PetscErrorPrintf)("PetscInitialize: Could not find MPI type for size_t\n"); return PETSC_ERR_SUP_SYS;}

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

#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
  ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_C_DOUBLE_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_C_DOUBLE_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_contiguous(2,MPI_FLOAT,&MPIU_C_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_C_COMPLEX);CHKERRQ(ierr);
#endif
#endif /* PETSC_HAVE_COMPLEX */

  /*
     Create the PETSc MPI reduction operator that sums of the first
     half of the entries and maxes the second half.
  */
  ierr = MPI_Op_create(MPIU_MaxSum_Local,1,&MPIU_MAXSUM_OP);CHKERRQ(ierr);

#if defined(PETSC_USE_REAL___FLOAT128)
  ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU___FLOAT128);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU___FLOAT128);CHKERRQ(ierr);
#if defined(PETSC_HAVE_COMPLEX)
  ierr = MPI_Type_contiguous(4,MPI_DOUBLE,&MPIU___COMPLEX128);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU___COMPLEX128);CHKERRQ(ierr);
#endif
  ierr = MPI_Op_create(PetscMax_Local,1,&MPIU_MAX);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscMin_Local,1,&MPIU_MIN);CHKERRQ(ierr);
#elif defined(PETSC_USE_REAL___FP16)
  ierr = MPI_Type_contiguous(2,MPI_CHAR,&MPIU___FP16);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU___FP16);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscMax_Local,1,&MPIU_MAX);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscMin_Local,1,&MPIU_MIN);CHKERRQ(ierr);
#endif

#if (defined(PETSC_HAVE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
  ierr = MPI_Op_create(PetscSum_Local,1,&MPIU_SUM);CHKERRQ(ierr);
#endif

  ierr = MPI_Type_contiguous(2,MPIU_SCALAR,&MPIU_2SCALAR);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_2SCALAR);CHKERRQ(ierr);

#if defined(PETSC_USE_64BIT_INDICES)
  ierr = MPI_Type_contiguous(2,MPIU_INT,&MPIU_2INT);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_2INT);CHKERRQ(ierr);
#endif


  /*
     Attributes to be set on PETSc communicators
  */
  ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelCounter,&Petsc_Counter_keyval,(void*)0);CHKERRQ(ierr);
  ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelComm_Outer,&Petsc_InnerComm_keyval,(void*)0);CHKERRQ(ierr);
  ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelComm_Inner,&Petsc_OuterComm_keyval,(void*)0);CHKERRQ(ierr);
  ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelComm_Shm,&Petsc_ShmComm_keyval,(void*)0);CHKERRQ(ierr);

  /*
     Build the options database
  */
  ierr = PetscOptionsInsert(NULL,argc,args,file);CHKERRQ(ierr);

  /* call a second time so it can look in the options database */
  ierr = PetscErrorPrintfInitialize();CHKERRQ(ierr);

  /*
     Print main application help message
  */
  ierr = PetscOptionsHasHelp(NULL,&flg);CHKERRQ(ierr);
  if (help && flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsCheckInitial_Private();CHKERRQ(ierr);

  ierr = PetscCitationsInitialize();CHKERRQ(ierr);

#if defined(PETSC_HAVE_SAWS)
  ierr = PetscInitializeSAWs(help);CHKERRQ(ierr);
#endif

  /*
     Load the dynamic libraries (on machines that support them), this registers all
     the solvers etc. (On non-dynamic machines this initializes the PetscDraw and PetscViewer classes)
  */
  ierr = PetscInitialize_DynamicLibraries();CHKERRQ(ierr);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscInfo1(NULL,"PETSc successfully started: number of processors = %d\n",size);CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,256);CHKERRQ(ierr);
  ierr = PetscInfo1(NULL,"Running on machine: %s\n",hostname);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENMP)
  {
    PetscBool omp_view_flag;
    char      *threads = getenv("OMP_NUM_THREADS");

   if (threads) {
     ierr = PetscInfo1(NULL,"Number of OpenMP threads %s (given by OMP_NUM_THREADS)\n",threads);CHKERRQ(ierr);
     (void) sscanf(threads, "%" PetscInt_FMT,&PetscNumOMPThreads);
   } else {
#define NMAX  10000
     int          i;
      PetscScalar *x;
      ierr = PetscMalloc1(NMAX,&x);CHKERRQ(ierr);
#pragma omp parallel for
      for (i=0; i<NMAX; i++) {
        x[i] = 0.0;
        PetscNumOMPThreads  = (PetscInt) omp_get_num_threads();
      }
      ierr = PetscFree(x);CHKERRQ(ierr);
      ierr = PetscInfo1(NULL,"Number of OpenMP threads %D (number not set with OMP_NUM_THREADS, chosen by system)\n",PetscNumOMPThreads);CHKERRQ(ierr);
    }
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"OpenMP options","Sys");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-omp_num_threads","Number of OpenMP threads to use (can also use environmental variable OMP_NUM_THREADS","None",PetscNumOMPThreads,&PetscNumOMPThreads,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsName("-omp_view","Display OpenMP number of threads",NULL,&omp_view_flag);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      ierr = PetscInfo1(NULL,"Number of OpenMP theads %D (given by -omp_num_threads)\n",PetscNumOMPThreads);CHKERRQ(ierr);
      omp_set_num_threads((int)PetscNumOMPThreads);
    }
    if (omp_view_flag) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"OpenMP: number of threads %D\n",PetscNumOMPThreads);CHKERRQ(ierr);
    }
  }
#endif
  ierr = PetscOptionsCheckInitial_Components();CHKERRQ(ierr);
  /* Check the options database for options related to the options database itself */
  ierr = PetscOptionsSetFromOptions(NULL);CHKERRQ(ierr);

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
  /*
      Tell MPI about our own data representation converter, this would/should be used if extern32 is not supported by the MPI

      Currently not used because it is not supported by MPICH.
  */
  if (!PetscBinaryBigEndian()) {
    ierr = MPI_Register_datarep((char*)"petsc",PetscDataRep_read_conv_fn,PetscDataRep_write_conv_fn,PetscDataRep_extent_fn,NULL);CHKERRQ(ierr);
  }
#endif

  /*
      Setup building of stack frames for all function calls
  */
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_THREADSAFETY)
  ierr = PetscStackCreate();CHKERRQ(ierr);
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
  ierr = PetscFPTCreate(10000);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_HWLOC)
  {
    PetscViewer viewer;
    ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-process_view",&viewer,NULL,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscProcessPlacementView(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  }
#endif

  flg = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-viewfromoptions",&flg,NULL);CHKERRQ(ierr);
  if (!flg) {ierr = PetscOptionsPushGetViewerOff(PETSC_TRUE); CHKERRQ(ierr);}

#if defined(PETSC_HAVE_ADIOS)
  ierr = adios_init_noxml(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = adios_declare_group(&Petsc_adios_group,"PETSc","",adios_stat_default);CHKERRQ(ierr);
  ierr = adios_select_method(Petsc_adios_group,"MPI","","");CHKERRQ(ierr);
  ierr = adios_read_init_method(ADIOS_READ_METHOD_BP,PETSC_COMM_WORLD,"");CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ADIOS2)
#endif

  /*
      Once we are completedly initialized then we can set this variables
  */
  PetscInitializeCalled = PETSC_TRUE;

  ierr = PetscOptionsHasName(NULL,NULL,"-python",&flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscPythonInitialize(NULL,NULL);CHKERRQ(ierr);}

#if defined(PETSC_HAVE_CUDA)
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_gpu_aware_mpi",&use_gpu_aware_mpi,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENMPI_MAJOR) && defined(MPIX_CUDA_AWARE_SUPPORT)
  /* OpenMPI supports compile time and runtime cuda support checking */
  if (use_gpu_aware_mpi && 1 != MPIX_Query_cuda_support()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"OpenMPI you used does not support CUDA while you requested -use_gpu_aware_mpi");
#endif
  ierr = PetscOptionsGetBool(NULL,NULL,"-sf_use_default_cuda_stream",&sf_use_default_cuda_stream,NULL);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_LOG)
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_REAL___FLOAT128)
  ierr = MPI_Type_free(&MPIU___FLOAT128);CHKERRQ(ierr);
#if defined(PETSC_HAVE_COMPLEX)
  ierr = MPI_Type_free(&MPIU___COMPLEX128);CHKERRQ(ierr);
#endif
  ierr = MPI_Op_free(&MPIU_MAX);CHKERRQ(ierr);
  ierr = MPI_Op_free(&MPIU_MIN);CHKERRQ(ierr);
#elif defined(PETSC_USE_REAL___FP16)
  ierr = MPI_Type_free(&MPIU___FP16);CHKERRQ(ierr);
  ierr = MPI_Op_free(&MPIU_MAX);CHKERRQ(ierr);
  ierr = MPI_Op_free(&MPIU_MIN);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_COMPLEX)
#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
  ierr = MPI_Type_free(&MPIU_C_DOUBLE_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_free(&MPIU_C_COMPLEX);CHKERRQ(ierr);
#endif
#endif

#if (defined(PETSC_HAVE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128) || defined(PETSC_USE_REAL___FP16)
  ierr = MPI_Op_free(&MPIU_SUM);CHKERRQ(ierr);
#endif

  ierr = MPI_Type_free(&MPIU_2SCALAR);CHKERRQ(ierr);
#if defined(PETSC_USE_64BIT_INDICES)
  ierr = MPI_Type_free(&MPIU_2INT);CHKERRQ(ierr);
#endif
  ierr = MPI_Op_free(&MPIU_MAXSUM_OP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       nopt;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE;
  PetscBool      flg;
#if defined(PETSC_USE_LOG)
  char           mname[PETSC_MAX_PATH_LEN];
#endif

  if (!PetscInitializeCalled) {
    printf("PetscInitialize() must be called before PetscFinalize()\n");
    return(PETSC_ERR_ARG_WRONGSTATE);
  }
  PetscFunctionBegin;
  ierr = PetscInfo(NULL,"PetscFinalize() called\n");CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ADIOS)
  ierr = adios_read_finalize_method(ADIOS_READ_METHOD_BP_AGGREGATE);CHKERRQ(ierr);
  ierr = adios_finalize(rank);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ADIOS2)
#endif
  ierr = PetscOptionsHasName(NULL,NULL,"-citations",&flg);CHKERRQ(ierr);
  if (flg) {
    char  *cits, filename[PETSC_MAX_PATH_LEN];
    FILE  *fd = PETSC_STDOUT;

    ierr = PetscOptionsGetString(NULL,NULL,"-citations",filename,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
    if (filename[0]) {
      ierr = PetscFOpen(PETSC_COMM_WORLD,filename,"w",&fd);CHKERRQ(ierr);
    }
    ierr = PetscSegBufferGet(PetscCitationsList,1,&cits);CHKERRQ(ierr);
    cits[0] = 0;
    ierr = PetscSegBufferExtractAlloc(PetscCitationsList,&cits);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"If you publish results based on this computation please cite the following:\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"===========================================================================\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"%s",cits);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"===========================================================================\n");CHKERRQ(ierr);
    ierr = PetscFClose(PETSC_COMM_WORLD,fd);CHKERRQ(ierr);
    ierr = PetscFree(cits);CHKERRQ(ierr);
  }
  ierr = PetscSegBufferDestroy(&PetscCitationsList);CHKERRQ(ierr);

#if defined(PETSC_HAVE_SSL) && defined(PETSC_USE_SOCKET_VIEWER)
  /* TextBelt is run for testing purposes only, please do not use this feature often */
  {
    PetscInt nmax = 2;
    char     **buffs;
    ierr = PetscMalloc1(2,&buffs);CHKERRQ(ierr);
    ierr = PetscOptionsGetStringArray(NULL,NULL,"-textbelt",buffs,&nmax,&flg1);CHKERRQ(ierr);
    if (flg1) {
      if (!nmax) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-textbelt requires either the phone number or number,\"message\"");
      if (nmax == 1) {
        ierr = PetscMalloc1(128,&buffs[1]);CHKERRQ(ierr);
        ierr = PetscGetProgramName(buffs[1],32);CHKERRQ(ierr);
        ierr = PetscStrcat(buffs[1]," has completed");CHKERRQ(ierr);
      }
      ierr = PetscTextBelt(PETSC_COMM_WORLD,buffs[0],buffs[1],NULL);CHKERRQ(ierr);
      ierr = PetscFree(buffs[0]);CHKERRQ(ierr);
      ierr = PetscFree(buffs[1]);CHKERRQ(ierr);
    }
    ierr = PetscFree(buffs);CHKERRQ(ierr);
  }
  {
    PetscInt nmax = 2;
    char     **buffs;
    ierr = PetscMalloc1(2,&buffs);CHKERRQ(ierr);
    ierr = PetscOptionsGetStringArray(NULL,NULL,"-tellmycell",buffs,&nmax,&flg1);CHKERRQ(ierr);
    if (flg1) {
      if (!nmax) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"-tellmycell requires either the phone number or number,\"message\"");
      if (nmax == 1) {
        ierr = PetscMalloc1(128,&buffs[1]);CHKERRQ(ierr);
        ierr = PetscGetProgramName(buffs[1],32);CHKERRQ(ierr);
        ierr = PetscStrcat(buffs[1]," has completed");CHKERRQ(ierr);
      }
      ierr = PetscTellMyCell(PETSC_COMM_WORLD,buffs[0],buffs[1],NULL);CHKERRQ(ierr);
      ierr = PetscFree(buffs[0]);CHKERRQ(ierr);
      ierr = PetscFree(buffs[1]);CHKERRQ(ierr);
    }
    ierr = PetscFree(buffs);CHKERRQ(ierr);
  }
#endif
  /*
    It should be safe to cancel the options monitors, since we don't expect to be setting options
    here (at least that are worth monitoring).  Monitors ought to be released so that they release
    whatever memory was allocated there before -malloc_dump reports unfreed memory.
  */
  ierr = PetscOptionsMonitorCancel();CHKERRQ(ierr);

#if defined(PETSC_SERIALIZE_FUNCTIONS)
  ierr = PetscFPTDestroy();CHKERRQ(ierr);
#endif


#if defined(PETSC_HAVE_SAWS)
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-saw_options",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsSAWsDestroy();CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_X)
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-x_virtual",&flg1,NULL);CHKERRQ(ierr);
  if (flg1) {
    /*  this is a crude hack, but better than nothing */
    ierr = PetscPOpen(PETSC_COMM_WORLD,NULL,"pkill -9 Xvfb","r",NULL);CHKERRQ(ierr);
  }
#endif

#if !defined(PETSC_HAVE_THREADSAFETY)
  ierr = PetscOptionsGetBool(NULL,NULL,"-malloc_info",&flg2,NULL);CHKERRQ(ierr);
  if (!flg2) {
    flg2 = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-memory_view",&flg2,NULL);CHKERRQ(ierr);
  }
  if (flg2) {
    ierr = PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD,"Summary of Memory Usage in PETSc\n");CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_USE_LOG)
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-get_total_flops",&flg1,NULL);CHKERRQ(ierr);
  if (flg1) {
    PetscLogDouble flops = 0;
    ierr = MPI_Reduce(&petsc_TotalFlops,&flops,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Total flops over all processors %g\n",flops);CHKERRQ(ierr);
  }
#endif


#if defined(PETSC_USE_LOG)
#if defined(PETSC_HAVE_MPE)
  mname[0] = 0;
  ierr = PetscOptionsGetString(NULL,NULL,"-log_mpe",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  if (flg1) {
    if (mname[0]) {ierr = PetscLogMPEDump(mname);CHKERRQ(ierr);}
    else          {ierr = PetscLogMPEDump(0);CHKERRQ(ierr);}
  }
#endif
#endif

  /*
     Free all objects registered with PetscObjectRegisterDestroy() such as PETSC_VIEWER_XXX_().
  */
  ierr = PetscObjectRegisterDestroyAll();CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscOptionsPushGetViewerOff(PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogViewFromOptions();CHKERRQ(ierr);
  ierr = PetscOptionsPopGetViewerOff();CHKERRQ(ierr);

  mname[0] = 0;
  ierr = PetscOptionsGetString(NULL,NULL,"-log_summary",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscViewer viewer;
    ierr = (*PetscHelpPrintf)(PETSC_COMM_WORLD,"\n\n WARNING:   -log_summary is being deprecated; switch to -log_view\n\n\n");CHKERRQ(ierr);
    if (mname[0]) {
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,mname,&viewer);CHKERRQ(ierr);
      ierr = PetscLogView(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      viewer = PETSC_VIEWER_STDOUT_WORLD;
      ierr   = PetscViewerPushFormat(viewer,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
      ierr   = PetscLogView(viewer);CHKERRQ(ierr);
      ierr   = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
  }

  /*
     Free any objects created by the last block of code.
  */
  ierr = PetscObjectRegisterDestroyAll();CHKERRQ(ierr);

  mname[0] = 0;
  ierr = PetscOptionsGetString(NULL,NULL,"-log_all",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-log",mname,PETSC_MAX_PATH_LEN,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {ierr = PetscLogDump(mname);CHKERRQ(ierr);}
#endif

  ierr = PetscStackDestroy();CHKERRQ(ierr);

  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_signal_handler",&flg1,NULL);CHKERRQ(ierr);
  if (!flg1) { ierr = PetscPopSignalHandler();CHKERRQ(ierr);}
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-mpidump",&flg1,NULL);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscMPIDump(stdout);CHKERRQ(ierr);
  }
  flg1 = PETSC_FALSE;
  flg2 = PETSC_FALSE;
  /* preemptive call to avoid listing this option in options table as unused */
  ierr = PetscOptionsHasName(NULL,NULL,"-malloc_dump",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-objects_dump",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-options_view",&flg2,NULL);CHKERRQ(ierr);

  if (flg2) {
    PetscViewer viewer;
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscOptionsView(NULL,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* to prevent PETSc -options_left from warning */
  ierr = PetscOptionsHasName(NULL,NULL,"-nox",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-nox_warning",&flg1);CHKERRQ(ierr);

  flg3 = PETSC_FALSE; /* default value is required */
  ierr = PetscOptionsGetBool(NULL,NULL,"-options_left",&flg3,&flg1);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (!flg1) flg3 = PETSC_TRUE;
#endif
  if (flg3) {
    if (!flg2 && flg1) { /* have not yet printed the options */
      PetscViewer viewer;
      ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscOptionsView(NULL,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = PetscOptionsAllUsed(NULL,&nopt);CHKERRQ(ierr);
    if (nopt) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING! There are options you set that were not used!\n");CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING! could be spelling mistake, etc!\n");CHKERRQ(ierr);
      if (nopt == 1) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"There is one unused database option. It is:\n");CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"There are %D unused database options. They are:\n",nopt);CHKERRQ(ierr);
      }
    } else if (flg3 && flg1) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"There are no unused options.\n");CHKERRQ(ierr);
    }
    ierr = PetscOptionsLeft(NULL);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_SAWS)
  if (!PetscGlobalRank) {
    ierr = PetscStackSAWsViewOff();CHKERRQ(ierr);
    PetscStackCallSAWs(SAWs_Finalize,());
  }
#endif

#if defined(PETSC_USE_LOG)
  /*
       List all objects the user may have forgot to free
  */
  if (PetscObjectsLog) {
    ierr = PetscOptionsHasName(NULL,NULL,"-objects_dump",&flg1);CHKERRQ(ierr);
    if (flg1) {
      MPI_Comm local_comm;
      char     string[64];

      ierr = PetscOptionsGetString(NULL,NULL,"-objects_dump",string,64,NULL);CHKERRQ(ierr);
      ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseBegin_Private(local_comm,1);CHKERRQ(ierr);
      ierr = PetscObjectsDump(stdout,(string[0] == 'a') ? PETSC_TRUE : PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseEnd_Private(local_comm,1);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
    }
  }
#endif

#if defined(PETSC_USE_LOG)
  PetscObjectsCounts    = 0;
  PetscObjectsMaxCounts = 0;
  ierr = PetscFree(PetscObjects);CHKERRQ(ierr);
#endif

  /*
     Destroy any packages that registered a finalize
  */
  ierr = PetscRegisterFinalizeAll();CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscLogFinalize();CHKERRQ(ierr);
#endif

  /*
     Print PetscFunctionLists that have not been properly freed

  ierr = PetscFunctionListPrintAll();CHKERRQ(ierr);
  */

  if (petsc_history) {
    ierr = PetscCloseHistoryFile(&petsc_history);CHKERRQ(ierr);
    petsc_history = NULL;
  }
  ierr = PetscOptionsHelpPrintedDestroy(&PetscOptionsHelpPrintedSingleton);CHKERRQ(ierr);

  ierr = PetscInfoAllow(PETSC_FALSE,NULL);CHKERRQ(ierr);

#if !defined(PETSC_HAVE_THREADSAFETY)
  if (!(PETSC_RUNNING_ON_VALGRIND)) {
    char fname[PETSC_MAX_PATH_LEN];
    char sname[PETSC_MAX_PATH_LEN];
    FILE *fd;
    int  err;

    flg2 = PETSC_FALSE;
    flg3 = PETSC_FALSE;
#if defined(PETSC_USE_DEBUG)
    ierr = PetscOptionsGetBool(NULL,NULL,"-malloc_test",&flg2,NULL);CHKERRQ(ierr);
#endif
    ierr = PetscOptionsGetBool(NULL,NULL,"-malloc_debug",&flg3,NULL);CHKERRQ(ierr);
    fname[0] = 0;
    ierr = PetscOptionsGetString(NULL,NULL,"-malloc_dump",fname,250,&flg1);CHKERRQ(ierr);
    if (flg1 && fname[0]) {

      PetscSNPrintf(sname,PETSC_MAX_PATH_LEN,"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open log file: %s",sname);
      ierr = PetscMallocDump(fd);CHKERRQ(ierr);
      err  = fclose(fd);
      if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
    } else if (flg1 || flg2 || flg3) {
      MPI_Comm local_comm;

      ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseBegin_Private(local_comm,1);CHKERRQ(ierr);
      ierr = PetscMallocDump(stdout);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseEnd_Private(local_comm,1);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
    }
    fname[0] = 0;
    ierr = PetscOptionsGetString(NULL,NULL,"-malloc_view",fname,250,&flg1);CHKERRQ(ierr);
    if (flg1 && fname[0]) {

      PetscSNPrintf(sname,PETSC_MAX_PATH_LEN,"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open log file: %s",sname);
      ierr = PetscMallocView(fd);CHKERRQ(ierr);
      err  = fclose(fd);
      if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
    } else if (flg1) {
      MPI_Comm local_comm;

      ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseBegin_Private(local_comm,1);CHKERRQ(ierr);
      ierr = PetscMallocView(stdout);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseEnd_Private(local_comm,1);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
    }
  }
#endif

  /*
     Close any open dynamic libraries
  */
  ierr = PetscFinalize_DynamicLibraries();CHKERRQ(ierr);

  /* Can be destroyed only after all the options are used */
  ierr = PetscOptionsDestroyDefault();CHKERRQ(ierr);

  PetscGlobalArgc = 0;
  PetscGlobalArgs = NULL;

  ierr = PetscFreeMPIResources();CHKERRQ(ierr);

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
    ierr = MPI_Comm_get_attr(PETSC_COMM_SELF,Petsc_InnerComm_keyval,&ucomm,&flg);CHKERRQ(ierr);
    if (flg) {
      icomm = ucomm.comm;
      ierr = MPI_Comm_get_attr(icomm,Petsc_Counter_keyval,&counter,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");

      ierr = MPI_Comm_delete_attr(PETSC_COMM_SELF,Petsc_InnerComm_keyval);CHKERRQ(ierr);
      ierr = MPI_Comm_delete_attr(icomm,Petsc_Counter_keyval);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&icomm);CHKERRQ(ierr);
    }
    ierr = MPI_Comm_get_attr(PETSC_COMM_WORLD,Petsc_InnerComm_keyval,&ucomm,&flg);CHKERRQ(ierr);
    if (flg) {
      icomm = ucomm.comm;
      ierr = MPI_Comm_get_attr(icomm,Petsc_Counter_keyval,&counter,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");

      ierr = MPI_Comm_delete_attr(PETSC_COMM_WORLD,Petsc_InnerComm_keyval);CHKERRQ(ierr);
      ierr = MPI_Comm_delete_attr(icomm,Petsc_Counter_keyval);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&icomm);CHKERRQ(ierr);
    }
  }

  ierr = MPI_Comm_free_keyval(&Petsc_Counter_keyval);CHKERRQ(ierr);
  ierr = MPI_Comm_free_keyval(&Petsc_InnerComm_keyval);CHKERRQ(ierr);
  ierr = MPI_Comm_free_keyval(&Petsc_OuterComm_keyval);CHKERRQ(ierr);
  ierr = MPI_Comm_free_keyval(&Petsc_ShmComm_keyval);CHKERRQ(ierr);

  ierr = PetscSpinlockDestroy(&PetscViewerASCIISpinLockOpen);CHKERRQ(ierr);
  ierr = PetscSpinlockDestroy(&PetscViewerASCIISpinLockStdout);CHKERRQ(ierr);
  ierr = PetscSpinlockDestroy(&PetscViewerASCIISpinLockStderr);CHKERRQ(ierr);
  ierr = PetscSpinlockDestroy(&PetscCommSpinLock);CHKERRQ(ierr);

  if (PetscBeganMPI) {
#if defined(PETSC_HAVE_MPI_FINALIZED)
    PetscMPIInt flag;
    ierr = MPI_Finalized(&flag);CHKERRQ(ierr);
    if (flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"MPI_Finalize() has already been called, even though MPI_Init() was called by PetscInitialize()");
#endif
    ierr = MPI_Finalize();CHKERRQ(ierr);
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
  ierr = PetscMallocClear();CHKERRQ(ierr);

  PetscInitializeCalled = PETSC_FALSE;
  PetscFinalizeCalled   = PETSC_TRUE;
  PetscFunctionReturn(0);
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
