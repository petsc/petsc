#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: options.c,v 1.177 1998/04/01 20:54:55 bsmith Exp balay $";
#endif
/*
   These routines simplify the use of command line, file options, etc.,
   and are used to manipulate the options database.

  This file uses regular malloc and free because it cannot know 
  what malloc is being used until it has already processed the input.
*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include <math.h>
#include "sys.h"
#include "src/sys/nreg.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/pviewer.h"
#include "src/sys/src/files.h"
#include "pinclude/petscfix.h"

/* 
    For simplicity, we begin with a static size database
*/
#define MAXOPTIONS 256
#define MAXALIASES 25

typedef struct {
  int  N,argc,Naliases;
  char **args,*names[MAXOPTIONS],*values[MAXOPTIONS];
  char *aliases1[MAXALIASES],*aliases2[MAXALIASES];
  int  used[MAXOPTIONS];
  int  namegiven;
  char programname[256]; /* HP includes entire path in name */
} OptionsTable;

static OptionsTable *options = 0;
       int          PetscBeganMPI = 0;

int        OptionsCheckInitial_Private(),
           OptionsCreate_Private(int*,char***,char*),
           OptionsSetAlias_Private(char *,char *);
static int OptionsDestroy_Private();
extern int PLogEventRegisterDestroy_Private();

#if defined(USE_PETSC_COMPLEX)
MPI_Datatype  MPIU_COMPLEX;
Scalar PETSC_i; 
#else
Scalar PETSC_i = 0.0; 
#endif

/*
     These are needed by src/inline/bitarray.H
*/
char _BT_mask, _BT_c;
int  _BT_idx;

/* 
   Optional file where all PETSc output from various prints is saved
*/
FILE *petsc_history = 0;

#undef __FUNC__  
#define __FUNC__ "PLogOpenHistoryFile"
int PLogOpenHistoryFile(char *filename,FILE **fd)
{
  int  ierr,rank,size;
  char pfile[256];

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 
  if (!rank) {
    char arch[10];
    PetscGetArchType(arch,10);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    if (!filename) {
      ierr = PetscGetHomeDirectory(240,pfile); CHKERRQ(ierr);
      PetscStrcat(pfile,"/.petschistory");
      filename = pfile;
    }
    ierr = PetscFixFilename(filename);CHKERRQ(ierr);
    *fd = fopen(filename,"a"); if (!fd) SETERRQ(PETSC_ERR_FILE_OPEN,0,"");
    fprintf(*fd,"---------------------------------------------------------\n");
    fprintf(*fd,"%s %s ",PETSC_VERSION_NUMBER,PetscGetDate());
    fprintf(*fd,"%s on a %s, %d proc. with options:\n",
            options->programname,arch,size);
    OptionsPrint(*fd);
    fprintf(*fd,"---------------------------------------------------------\n");
    fflush(*fd);
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "PLogCloseHistoryFile"
static int PLogCloseHistoryFile(FILE **fd)
{
  int  rank;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 
  if (rank) PetscFunctionReturn(0);
  fprintf(*fd,"---------------------------------------------------------\n");
  fprintf(*fd,"Finished at %s",PetscGetDate());
  fprintf(*fd,"---------------------------------------------------------\n");
  fflush(*fd);
  fclose(*fd);
  PetscFunctionReturn(0); 
}

/*
       Function that is called to display all error messages
*/
extern int  PetscErrorPrintfDefault(char *,...);
extern int  PetscHelpPrintfDefault(MPI_Comm,char *,...);
int (*PetscErrorPrintf)(char *,...) = PetscErrorPrintfDefault;
int (*PetscHelpPrintf)(MPI_Comm,char *,...)  = PetscHelpPrintfDefault;

int      PetscInitializedCalled = 0;
int      PetscGlobalRank = -1, PetscGlobalSize = -1;
MPI_Comm PETSC_COMM_WORLD = 0;
MPI_Comm PETSC_COMM_SELF  = 0;

/* ---------------------------------------------------------------------------*/
int    PetscCompare          = 0;
double PetscCompareTolerance = 1.e-10;

#undef __FUNC__  
#define __FUNC__ "PetscCompareInt"
/*@C
   PetscCompareInt - Compares integers while running with PETSc's incremental
   debugger.

   Input Parameter:
.  d - integer to compare

   Options Database Key:
.  -compare

   Note:
   This routine is activated with the -compare option.

.seealso: PetscCompareDouble(), PetscCompareScalar()
@*/
int PetscCompareInt(int d)
{
  int work = d,ierr;

  PetscFunctionBegin;
  ierr = MPI_Bcast(&work,1,MPI_INT,0,MPI_COMM_WORLD);CHKERRQ(ierr);
  if (d != work) {
    SETERRQ(PETSC_ERR_PLIB,0,"Inconsistent integer");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscCompareDouble"
/*@C
   PetscCompareDouble - Compares doubles while running with PETSc's incremental
   debugger.

   Input Parameter:
.  d - double precision number to compare

   Options Database Key:
.  -compare

   Note:
   This routine is activated with the -compare option.

.seealso: PetscCompareInt(), PetscComparseScalar()
@*/
int PetscCompareDouble(double d)
{
  double work = d;
  int    ierr;

  PetscFunctionBegin;
  ierr = MPI_Bcast(&work,1,MPI_DOUBLE,0,MPI_COMM_WORLD);CHKERRQ(ierr);
  if (!d && !work) PetscFunctionReturn(0);
  if (PetscAbsDouble(work - d)/PetscMax(PetscAbsDouble(d),PetscAbsDouble(work)) 
      > PetscCompareTolerance) {
    SETERRQ(PETSC_ERR_PLIB,0,"Inconsistent double");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscCompareScalar"
/*@C
   PetscCompareScalar - Compares scalars while running with PETSc's incremental
   debugger.

   Input Parameter:
.  d - scalar to compare


   Options Database Key:
.  -compare

   Note:
   This routine is activated with the -compare option.

.seealso: PetscCompareInt(), PetscComparseDouble()
@*/
int PetscCompareScalar(Scalar d)
{
  Scalar work = d;
  int    ierr;

  PetscFunctionBegin;
  ierr = MPI_Bcast(&work,2,MPI_DOUBLE,0,MPI_COMM_WORLD);CHKERRQ(ierr);
  if (!PetscAbsScalar(d) && !PetscAbsScalar(work)) PetscFunctionReturn(0);
  if (PetscAbsScalar(work - d)/PetscMax(PetscAbsScalar(d),PetscAbsScalar(work)) 
      >= PetscCompareTolerance) {
    SETERRQ(PETSC_ERR_PLIB,0,"Inconsistent scalar");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscCompareInitialize"
/*
    PetscCompareInitialize - If there is a command line option -compare then
    this routine calls MPI_Init() and sets up two PETSC_COMM_WORLD, one for 
    each program being compared.

    Note: Only works with C programs.
*/
int PetscCompareInitialize(double tol)
{
  int       ierr,i, len,rank,work,*gflag,size,mysize;
  char      *programname = options->programname, basename[256];
  MPI_Group group_all,group_sub;

  PetscFunctionBegin;
  PetscCompareTolerance = tol;

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if (!rank) {
    PetscStrcpy(basename,programname);
    len = PetscStrlen(basename);
  }

  /* broadcase name from first processor to all processors */
  ierr = MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(basename,len+1,MPI_CHAR,0,MPI_COMM_WORLD);CHKERRQ(ierr);

  /* determine what processors belong to my group */
  if (!PetscStrcmp(programname,basename)) work = 1;
  else                                    work = 0;
  gflag = (int *) malloc( size*sizeof(int) ); CHKPTRQ(gflag);
  ierr = MPI_Allgather(&work,1,MPI_INT,gflag,1 ,MPI_INT,MPI_COMM_WORLD); CHKERRQ(ierr);
  mysize = 0;
  for ( i=0; i<size; i++ ) {
    if (work == gflag[i]) gflag[mysize++] = i;
  }
  /*   printf("[%d] my name %s basename %s mysize %d\n",rank,programname,basename,mysize); */

  if (mysize == 0 || mysize == size) {
    SETERRQ(PETSC_ERR_ARG_IDN,0,"Need two different programs to compare");
  }

  /* create a new communicator for each program */
  ierr = MPI_Comm_group(MPI_COMM_WORLD,&group_all);CHKERRQ(ierr);
  ierr = MPI_Group_incl(group_all,mysize,gflag,&group_sub);CHKERRQ(ierr);
  ierr = MPI_Comm_create(MPI_COMM_WORLD,group_sub,&PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Group_free(&group_all);CHKERRQ(ierr);
  ierr = MPI_Group_free(&group_sub);CHKERRQ(ierr);
  free(gflag);

  PetscCompare = 1;
  PLogInfo(0,"PetscCompareInitialize:Configured to compare two programs\n",rank);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PetscInitializeOptions"
int PetscInitializeOptions(void)
{
  PetscFunctionBegin;
  options = (OptionsTable*) malloc(sizeof(OptionsTable)); CHKPTRQ(options);
  PetscMemzero(options->used,MAXOPTIONS*sizeof(int));
  options->namegiven = 0;
  PetscFunctionReturn(0);
}

extern int PetscInitialize_DynamicLibraries();
extern int PetscFinalize_DynamicLibraries();

#undef __FUNC__  
#define __FUNC__ "OptionsSetProgramName"
int OptionsSetProgramName(char *name)
{ 
  PetscFunctionBegin;
  options->namegiven = 1;
  PetscStrncpy(options->programname,name,256);
  PetscFunctionReturn(0);
}

/*
       This may be called before PetscInitialize() so 
    should not use PETSc specific calls.

       Initializes PETSc without direct access to the 
    command line options.
*/
#undef __FUNC__  
#define __FUNC__ "PetscInitializeNoArguments"
int PetscInitializeNoArguments(void)
{
  int  argc = 0, ierr;
  char **args = 0;

  ierr = PetscInitialize(&argc,&args,PETSC_NULL,PETSC_NULL);

  return ierr;
}


#undef __FUNC__  
#define __FUNC__ "PetscInitialize"
/*@C
   PetscInitialize - Initializes the PETSc database and MPI. 
   PetscInitialize calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of 
   your program -- usually the very first line! 

   Input Parameters:
.  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use PETSC_NULL for default)
.  help - [optional] Help message to print, use PETSC_NULL for no message

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   Fortran Version:
   In Fortran this routine has the format
$       call PetscInitialize(file,ierr)

.   ierr - error return code
.   file - [optional] PETSc database file name, defaults to 
           ~username/.petscrc (use PETSC_NULL_CHARACTER for default)
           
   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use PETSC_NULL as 
   in the C version.  See the users manual for details.

   Options Database Keys:
$  -start_in_debugger [noxterm,dbx,xdb,...]
$  -debugger_nodes [node1,node2,...] : Nodes to start in debugger
$  -debugger_pause [sleeptime] (in seconds) : Pause debugger
$  -trmalloc : Use PETSc error-checking malloc
$  -trmalloc_off : Don't use error-checking malloc
$  -no_signal_handler : Do not trap error signals
$  -fp_trap : Stop on floating point exceptions
$      Note: On the IBM RS6000 this slows code by
$            at least a factor of 10.

   Options Database Keys for Profiling:
   See the 'Profiling' chapter of the users manual for
   details.
$  -log_trace [filename] : Print traces of all PETSc calls
$      to the screen (useful to determine where a program
$      hangs without running in the debugger).  See
$      PLogTraceBegin().
$  -log_info : Print verbose information to the screen.

.keywords: initialize, options, database, startup

.seealso: PetscFinalize()
@*/
int PetscInitialize(int *argc,char ***args,char *file,char *help)
{
  int        ierr,flag,flg,dummy_tag,PETSC_COMM_WORLD_FromUser = 1;

  PetscFunctionBegin;
  if (PetscInitializedCalled) PetscFunctionReturn(0);

  ierr = PetscInitializeOptions(); CHKERRQ(ierr);

  /*
     We initialize the program name here because MPICH has a bug in 
     it that it sets args[0] on all processors to be args[0]
     on the first processor.
  */
  if (argc && *argc) {
    OptionsSetProgramName(**args);
  }

  MPI_Initialized(&flag);
  if (!flag) {
    ierr = MPI_Init(argc,args); CHKERRQ(ierr);
    PetscBeganMPI    = 1;
  }
  PetscInitializedCalled = 1;

  if (!PETSC_COMM_WORLD) {
    PETSC_COMM_WORLD_FromUser = 0;
    PETSC_COMM_WORLD          = MPI_COMM_WORLD;
  }

  MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);
  MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);
#if defined(USE_PETSC_COMPLEX)
  /* 
     Initialized the global variable; this is because with 
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
    Scalar ic(0.0,1.0);
    PETSC_i = ic; 
  }
  ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_COMPLEX);CHKERRQ(ierr);
#endif
  ierr = OptionsCreate_Private(argc,args,file); CHKERRQ(ierr);
  ierr = OptionsCheckInitial_Private(); CHKERRQ(ierr); 

  /*
       Initialize PETSC_COMM_SELF and WORLD as a MPI_Comm with the PETSc 
     attribute.
    
       We delay until here to do it, since PetscMalloc() may not have been
     setup yet.
  */
  ierr = PetscCommDup_Private(MPI_COMM_SELF,&PETSC_COMM_SELF,&dummy_tag);CHKERRQ(ierr);
  if (!PETSC_COMM_WORLD_FromUser) {
    ierr = PetscCommDup_Private(MPI_COMM_WORLD,&PETSC_COMM_WORLD,&dummy_tag); CHKERRQ(ierr);
  }

  ierr = ViewerInitialize_Private(); CHKERRQ(ierr);
  if (PetscBeganMPI) {
    int size;

    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    PLogInfo(0,"PetscInitialize:PETSc successfully started: number of processors = %d\n",size);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (help && flg) {
    PetscPrintf(PETSC_COMM_WORLD,help);
  }
  
  /*
      Initialize the default dynamic libraries
  */
  ierr = PetscInitialize_DynamicLibraries(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern int PetscSequentialPhaseBegin_Private(MPI_Comm,int);
extern int PetscSequentialPhaseEnd_Private(MPI_Comm,int);
extern int DLRegisterDestroyAll();

#undef __FUNC__  
#define __FUNC__ "PetscFinalize"
/*@C 
   PetscFinalize - Checks for options to be called at the conclusion
   of the program and calls MPI_Finalize().

   Options Database Keys:
$  -optionstable : Calls OptionsPrint()
$  -optionsleft : Prints unused options that remain in 
$     the database
$  -mpidump : Calls PetscMPIDump()
$  -trdump : Calls PetscTrDump()
$  -trinfo : Prints total memory usage
$  -trmalloc_log: Prints summary of memory usage
$  -trdebug : Calls malloc_debug(2) to activate memory
$      allocation diagnostics (used by PETSC_ARCH=sun4, 
$      BOPT=[g,g_c++,g_complex] only!)

   Options Database Keys for Profiling:
   See the 'Profiling' chapter of the users manual for
   details.
$  -log_summary [filename] : Prints summary of flop and timing
$      information to screen. If the filename is specified the
$      summary is written to the file. (for code compiled with 
$      USE_PETSC_LOG).  See PLogPrintSummary().
$  -log_all [filename]: Logs extensive profiling information
$      (for code compiled with USE_PETSC_LOG). See PLogDump(). 
$  -log [filename]: Logs basic profiline information (for
$      code compiled with USE_PETSC_LOG).  See PLogDump().
$  -log_sync: Log the synchronization in scatters, inner products
$             and norms
$  -log_mpe [filename]: Creates a logfile viewable by the 
$      utility Upshot/Nupshot (in MPICH distribution)

   Note:
   See PetscInitialize() for more general runtime options.

.keywords: finalize, exit, end

.seealso: PetscInitialize(), OptionsPrint(), PetscTrDump(), PetscMPIDump()
@*/
int PetscFinalize(void)
{
  int        ierr,i,rank = 0,flg1,flg2,flg3,nopt;
  PLogDouble rss;

  PetscFunctionBegin;
  if (!PetscInitializedCalled) {
    (*PetscErrorPrintf)("PETSc ERROR: PetscInitialize() must be called before PetscFinalize()\n");
    PetscFunctionReturn(0);
  }

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ViewerDestroy_Private();
  ViewerDestroyDrawX_Private();
  ViewerDestroyMatlab_Private();
  PetscFinalize_DynamicLibraries();

  ierr = OptionsHasName(PETSC_NULL,"-get_resident_set_size",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscGetResidentSetSize(&rss); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_SELF,"[%d] Size of entire process memory %d\n",rank,(int)rss);
  }

#if defined(USE_PETSC_STACK)
  if (PetscStackActive) {
    ierr = PetscStackDestroy(); CHKERRQ(ierr);
  }
#endif

#if defined(USE_PETSC_LOG)
  {
    char mname[64];
#if defined (HAVE_MPE)
    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_mpe",mname,64,&flg1); CHKERRQ(ierr);
    if (flg1){
      if (mname[0]) PLogMPEDump(mname); 
      else          PLogMPEDump(0);
    }
#endif
    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_summary",mname,64,&flg1); CHKERRQ(ierr);
    if (flg1) { 
      if (mname[0])  PLogPrintSummary(PETSC_COMM_WORLD,mname); 
      else           PLogPrintSummary(PETSC_COMM_WORLD,0); 
    }

    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_all",mname,64,&flg1); CHKERRQ(ierr);
    ierr = OptionsGetString(PETSC_NULL,"-log",mname,64,&flg2); CHKERRQ(ierr);
    if (flg1 || flg2){
      if (mname[0]) PLogDump(mname); 
      else          PLogDump(0);
    }
    ierr = PLogDestroy(); CHKERRQ(ierr);
  }
#endif
  ierr = OptionsHasName(PETSC_NULL,"-no_signal_handler",&flg1);CHKERRQ(ierr);
  if (!flg1) { PetscPopSignalHandler(); }
  ierr = OptionsHasName(PETSC_NULL,"-mpidump",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscMPIDump(stdout); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-trdump",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-optionstable",&flg1); CHKERRQ(ierr);
  if (flg1) {
    if (!rank) OptionsPrint(stdout);
  }
  nopt = OptionsAllUsed();
  ierr = OptionsHasName(PETSC_NULL,"-optionsleft",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = OptionsPrint(stdout);CHKERRQ(ierr);
  }
  if (flg1) {
    if (nopt == 0) { 
      PetscPrintf(PETSC_COMM_WORLD,"There are no unused options.\n");
    } else if (nopt == 1) {
      PetscPrintf(PETSC_COMM_WORLD,"There is one unused database option. It is:\n");
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"There are %d unused database options. They are:\n",nopt);
    }
  }

#if (USE_PETSC_BOPT_g)
  if (nopt && !flg1) {
    PetscPrintf(PETSC_COMM_WORLD,"WARNING! There are options you set that were not used!\n");
    PetscPrintf(PETSC_COMM_WORLD,"WARNING! could be spelling mistake, etc!\n");
  }
  if (nopt || flg1) {
#else 
  if (flg1) {
#endif
    for ( i=0; i<options->N; i++ ) {
      if (!options->used[i]) {
        if (options->values[i]) {
          PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s value: %s\n",options->names[i],
                                                           options->values[i]);
        } else {
          PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s no value \n",options->names[i]);
        }
      }
    } 
  }
  ierr = OptionsHasName(PETSC_NULL,"-log_history",&flg1); CHKERRQ(ierr);
  if (flg1) {
    PLogCloseHistoryFile(&petsc_history);
    petsc_history = 0;
  }

  /*
     Destroy all the function registration lists created
  */
  NRDestroyAll();
  ierr = DLRegisterDestroyAll(); CHKERRQ(ierr); 

  /*
       Destroy PETSC_COMM_SELF as a MPI_Comm with the PETSc 
     attribute.
  */
  ierr = PetscCommFree_Private(&PETSC_COMM_SELF);CHKERRQ(ierr);
  ierr = PetscCommFree_Private(&PETSC_COMM_WORLD);CHKERRQ(ierr);

#if defined(USE_PETSC_LOG)
  PLogEventRegisterDestroy_Private();
#endif

  ierr = OptionsHasName(PETSC_NULL,"-trdump",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trinfo",&flg2); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trmalloc_log",&flg3); CHKERRQ(ierr);
  if (flg1) {
    MPI_Comm local_comm;

    ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
    PetscSequentialPhaseBegin_Private(local_comm,1);
      ierr = PetscTrDump(stderr); CHKERRQ(ierr);
    PetscSequentialPhaseEnd_Private(local_comm,1);
    ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
  } else if (flg2) {
    MPI_Comm local_comm;
    double maxm;

    ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
    ierr = PetscTrSpace(PETSC_NULL,PETSC_NULL,&maxm); CHKERRQ(ierr);
    PetscSequentialPhaseBegin_Private(local_comm,1);
      printf("[%d] Maximum memory used %g\n",rank,maxm);
    PetscSequentialPhaseEnd_Private(local_comm,1);
    ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
  }
  /* Can be dumped only after all the Objects are destroyed */
  if (flg3) {
    ierr = PetscTrLogDump(stdout);CHKERRQ(ierr); 
  }
  /* Can be destroyed only after all the options are used */
  OptionsDestroy_Private();


  if (PetscBeganMPI) {
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    PLogInfo(0,"PetscFinalize:PETSc successfully ended!\n");
    ierr = MPI_Finalize(); CHKERRQ(ierr);
  }

/*

     Note: In certain cases PETSC_COMM_WORLD is never MPI_Comm_free()ed because 
   the communicator has some outstanding requests on it. Specifically if the 
   flag HAVE_BROKEN_REQUEST_FREE is set (for IBM MPI implementation). See 
   src/vec/utils/vpscat.c. Due to this the memory allocated in PetscCommDup_Private()
   is never freed as it should be. Thus one may obtain messages of the form
   [ 1] 8 bytes PetscCommDup_Private() line 645 in src/sys/src/mpiu.c indicating the
   memory was not freed.

*/
  PetscClearMalloc();
  PetscInitializedCalled = 0;
  PetscFunctionReturn(0);
}
 
/* 
   This is ugly and probably belongs somewhere else, but I want to 
  be able to put a true MPI abort error handler with command line args.

    This is so MPI errors in the debugger will leave all the stack 
  frames. The default abort cleans up and exits.
*/

#undef __FUNC__  
#define __FUNC__ "Petsc_MPI_Abort_Function"
void Petsc_MPI_Abort_Function(MPI_Comm *comm,int *flag) 
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("MPI error %d\n",*flag);
  abort();
}

#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
  extern int malloc_debug(int);
}
#elif defined(PARCH_sun4)
  extern int malloc_debug(int);
#endif

extern int PLogInfoAllow(PetscTruth);
extern int PetscSetUseTrMalloc_Private(int);

#include "snes.h" /* so that cookies are defined */

#undef __FUNC__  
#define __FUNC__ "OptionsCheckInitial_Private"
int OptionsCheckInitial_Private(void)
{
  char     string[64];
  MPI_Comm comm = PETSC_COMM_WORLD;
  int      flg1,flg2,flg3,flg4,ierr,*nodes,flag,i,rank;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ierr = OptionsHasName(PETSC_NULL,"-trmalloc_nan",&flg4);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trmalloc_log",&flg3); CHKERRQ(ierr);
#if defined(USE_PETSC_BOPT_g)
  ierr = OptionsHasName(PETSC_NULL,"-trmalloc_off", &flg1); CHKERRQ(ierr);
  if (!flg1) { ierr = PetscSetUseTrMalloc_Private(flg4); CHKERRQ(ierr); }
#else
  ierr = OptionsHasName(PETSC_NULL,"-trdump",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trmalloc",&flg2); CHKERRQ(ierr);
  if (flg1 || flg2 || flg3 || flg4) {ierr = PetscSetUseTrMalloc_Private(flg4);CHKERRQ(ierr);}
#endif
  if (flg3) {
    ierr = PetscTrLog();CHKERRQ(ierr); 
  }
  ierr = OptionsHasName(PETSC_NULL,"-trdebug",&flg1); CHKERRQ(ierr);
  if (flg1) { 
    ierr = PetscTrDebugLevel(1);CHKERRQ(ierr);
#if defined(PARCH_sun4) && defined(USE_PETSC_BOPT_g)
    malloc_debug(2);
#endif
  }
  ierr = PetscSetDisplay(); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-v",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-version",&flg2); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg3); CHKERRQ(ierr);
  if (flg1 || flg2 || flg3 ){
    (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n");
    (*PetscHelpPrintf)(comm,"\t   %s\n",PETSC_VERSION_NUMBER);
    (*PetscHelpPrintf)(comm,"Satish Balay, Bill Gropp, Lois Curfman McInnes, Barry Smith.\n");
    (*PetscHelpPrintf)(comm,"Bug reports, questions: petsc-maint@mcs.anl.gov\n");
    (*PetscHelpPrintf)(comm,"Web page: http://www.mcs.anl.gov/petsc/petsc.html\n");
    (*PetscHelpPrintf)(comm,"See docs/copyright.html for copyright information\n");
    (*PetscHelpPrintf)(comm,"See docs/changes.html for recent updates.\n");
    (*PetscHelpPrintf)(comm,"See docs/troubleshooting.html hints for problems.\n");
    (*PetscHelpPrintf)(comm,"See docs/manualpages/manualpages.html or \n");
    (*PetscHelpPrintf)(comm,"   bin/petscman for help.\n");
    (*PetscHelpPrintf)(comm,"Libraries linked from %s\n",PETSC_LDIR);
    (*PetscHelpPrintf)(comm,"--------------------------------------------\
---------------------------\n");
  }
  ierr = OptionsHasName(PETSC_NULL,"-fp_trap",&flg1); CHKERRQ(ierr);
  if (flg1) { ierr = PetscSetFPTrap(PETSC_FP_TRAP_ON); CHKERRQ(ierr); }
  ierr = OptionsHasName(PETSC_NULL,"-on_error_abort",&flg1); CHKERRQ(ierr);
  if (flg1) { PetscPushErrorHandler(PetscAbortErrorHandler,0); } 
  ierr = OptionsHasName(PETSC_NULL,"-on_error_stop",&flg1); CHKERRQ(ierr);
  if (flg1) { PetscPushErrorHandler(PetscStopErrorHandler,0); }
  /*
     Set default debugger on solaris and rs6000 to dbx and hpux to xdb
     because gdb doesn't work with the native compilers.
  */
#if defined(PARCH_solaris) || defined(PARCH_rs6000) || defined(PARCH_IRIX64) || defined(PARCH_IRIX)
  ierr = PetscSetDebugger("dbx",1,0); CHKERRQ(ierr);
#elif defined(PARCH_hpux) 
  ierr = PetscSetDebugger("xdb",1,0); CHKERRQ(ierr);
#endif

  ierr = OptionsGetString(PETSC_NULL,"-on_error_attach_debugger",string,64,&flg1);CHKERRQ(ierr);
  if (flg1) {
    char *debugger = 0, *display = 0;
    int  xterm     = 1;
    if (PetscStrstr(string,"noxterm")) xterm = 0;
#if defined(PARCH_hpux)
    if (PetscStrstr(string,"xdb"))     debugger = "xdb";
#else
    if (PetscStrstr(string,"dbx"))     debugger = "dbx";
#endif
#if defined(PARCH_rs6000)
    if (PetscStrstr(string,"xldb"))    debugger = "xldb";
#endif
    if (PetscStrstr(string,"gdb"))     debugger = "gdb";
    if (PetscStrstr(string,"xxgdb"))   debugger = "xxgdb";
    if (PetscStrstr(string,"ups"))     debugger = "ups";
    display = (char *) malloc(128*sizeof(char)); CHKPTRQ(display);
    PetscGetDisplay(display,128);
    PetscSetDebugger(debugger,xterm,display);
    PetscPushErrorHandler(PetscAttachDebuggerErrorHandler,0);
  }
  ierr = OptionsGetString(PETSC_NULL,"-start_in_debugger",string,64,&flg1);CHKERRQ(ierr);
  if (flg1) {
    char           *debugger = 0, *display = 0;
    int            xterm     = 1,size;
    MPI_Errhandler abort_handler;
    /*
       we have to make sure that all processors have opened 
       connections to all other processors, otherwise once the 
       debugger has stated it is likely to receive a SIGUSR1
       and kill the program. 
    */
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    if (size > 2) {
      int        dummy;
      MPI_Status status;
      for ( i=0; i<size; i++) {
        ierr = MPI_Send(&dummy,1,MPI_INT,i,109,PETSC_COMM_WORLD);CHKERRQ(ierr);
      }
      for ( i=0; i<size; i++) {
        ierr = MPI_Recv(&dummy,1,MPI_INT,i,109,PETSC_COMM_WORLD,&status);CHKERRQ(ierr);
      }
    }
    /* check if this processor node should be in debugger */
    nodes = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(nodes);
    ierr = OptionsGetIntArray(PETSC_NULL,"-debugger_nodes",nodes,&size,&flag);CHKERRQ(ierr);
    if (flag) {
      for (i=0; i<size; i++) {
        if (nodes[i] == rank) { flag = 0; break; }
      }
    }
    if (!flag) {        
      if (PetscStrstr(string,"noxterm")) xterm = 0;
#if defined(PARCH_hpux)
      if (PetscStrstr(string,"xdb"))     debugger = "xdb";
#else
      if (PetscStrstr(string,"dbx"))     debugger = "dbx";
#endif
#if defined(PARCH_rs6000)
      if (PetscStrstr(string,"xldb"))    debugger = "xldb";
#endif
      if (PetscStrstr(string,"gdb"))     debugger = "gdb";
      if (PetscStrstr(string,"xxgdb"))   debugger = "xxgdb";
      if (PetscStrstr(string,"ups"))     debugger = "ups";
      display = (char *) malloc( 128*sizeof(char) ); CHKPTRQ(display);
      PetscGetDisplay(display,128);
      PetscSetDebugger(debugger,xterm,display);
      PetscPushErrorHandler(PetscAbortErrorHandler,0);
      PetscAttachDebugger();
      ierr = MPI_Errhandler_create((MPI_Handler_function*)Petsc_MPI_Abort_Function,&abort_handler);CHKERRQ(ierr);
      ierr = MPI_Errhandler_set(comm,abort_handler);CHKERRQ(ierr);
    }
    PetscFree(nodes);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mpi_return_on_error", &flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = MPI_Errhandler_set(comm,MPI_ERRORS_RETURN);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-no_signal_handler", &flg1); CHKERRQ(ierr);
  if (!flg1) { PetscPushSignalHandler(PetscDefaultSignalHandler,(void*)0); }
#if defined(USE_PETSC_LOG)
  {
    char mname[256];
    {
      mname[0] = 0;
      ierr = OptionsGetString(PETSC_NULL,"-log_history",mname,256, &flg1);CHKERRQ(ierr);
      if(flg1) {
        if (mname[0]) {
          ierr = PLogOpenHistoryFile(mname,&petsc_history); CHKERRQ(ierr);
        } else {
          ierr = PLogOpenHistoryFile(0,&petsc_history); CHKERRQ(ierr);
        }
      }
    }
    ierr = OptionsHasName(PETSC_NULL,"-log_info", &flg1); CHKERRQ(ierr);
    if (flg1) { 
      PLogInfoAllow(PETSC_TRUE); 
      ierr = OptionsGetString(PETSC_NULL,"-log_info_exclude",mname,256, &flg1);CHKERRQ(ierr);
      if (flg1) {
        if (PetscStrstr(mname,"vec")) {
          PLogInfoDeactivateClass(VEC_COOKIE);
        }
        if (PetscStrstr(mname,"mat")) {
          PLogInfoDeactivateClass(MAT_COOKIE);
        }
        if (PetscStrstr(mname,"sles")) {
          PLogInfoDeactivateClass(SLES_COOKIE);
        }
        if (PetscStrstr(mname,"snes")) {
          PLogInfoDeactivateClass(SNES_COOKIE);
        }
      }
    }
#if defined (HAVE_MPE)
    ierr = OptionsHasName(PETSC_NULL,"-log_mpe", &flg1); CHKERRQ(ierr);
    if (flg1) PLogMPEBegin();
#endif
    ierr = OptionsHasName(PETSC_NULL,"-log_all", &flg1); CHKERRQ(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-log", &flg2); CHKERRQ(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-log_summary", &flg3); CHKERRQ(ierr);
    if (flg1)              {  ierr = PLogAllBegin();  CHKERRQ(ierr); }
    else if (flg2 || flg3) {  ierr = PLogBegin();  CHKERRQ(ierr);}
    if (flg3) {
      ierr = OptionsGetString(PETSC_NULL,"-log_summary_exclude",mname,256, &flg1);CHKERRQ(ierr);
      if (flg1) {
        if (PetscStrstr(mname,"vec")) {
          PLogEventDeactivateClass(VEC_COOKIE);
        }
        if (PetscStrstr(mname,"mat")) {
          PLogEventDeactivateClass(MAT_COOKIE);
        }
        if (PetscStrstr(mname,"sles")) {
          PLogEventDeactivateClass(SLES_COOKIE);
        }
        if (PetscStrstr(mname,"snes")) {
          PLogEventDeactivateClass(SNES_COOKIE);
        }
      }
    }
    
    ierr = OptionsHasName(PETSC_NULL,"-log_sync",&flg1);CHKERRQ(ierr);
    if (flg1) {
      PLogEventActivate(VEC_ScatterBarrier);
      PLogEventActivate(VEC_NormBarrier);
      PLogEventActivate(VEC_NormComm);
      PLogEventActivate(VEC_DotBarrier);
      PLogEventActivate(VEC_DotComm);
      PLogEventActivate(VEC_MDotBarrier);
      PLogEventActivate(VEC_MDotComm);
    }
    ierr = OptionsGetString(PETSC_NULL,"-log_trace",mname,250,&flg1); CHKERRQ(ierr);
    if (flg1) { 
      char fname[256];
      FILE *file;
      if (mname[0]) {
        sprintf(fname,"%s.%d",mname,rank);
        ierr = PetscFixFilename(fname);CHKERRQ(ierr);
        file = fopen(fname,"w"); 
        if (!file) {
          SETERRQ(PETSC_ERR_FILE_OPEN,0,"Unable to open trace file");
        }
      } else {
        file = stdout;
      }
      ierr = PLogTraceBegin(file); CHKERRQ(ierr);
    }
  }
#endif
#if defined(USE_PETSC_STACK)
#if defined(USE_PETSC_BOPT_g)
  ierr = PetscStackCreate(256); CHKERRQ(ierr);
#else
  ierr = OptionsHasName(PETSC_NULL,"-log_stack", &flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscStackCreate(256); CHKERRQ(ierr);
  }
#endif
#endif


  ierr = OptionsHasName(PETSC_NULL,"-help", &flg1); CHKERRQ(ierr);
  if (flg1) {
    (*PetscHelpPrintf)(comm,"Options for all PETSc programs:\n");
    (*PetscHelpPrintf)(comm," -on_error_abort: cause an abort when an error is");
    (*PetscHelpPrintf)(comm," detected. Useful \n       only when run in the debugger\n");
    (*PetscHelpPrintf)(comm," -on_error_attach_debugger [dbx,xxgdb,ups,noxterm]\n"); 
    (*PetscHelpPrintf)(comm,"       start the debugger (gdb by default) in new xterm\n");
    (*PetscHelpPrintf)(comm,"       unless noxterm is given\n");
    (*PetscHelpPrintf)(comm," -start_in_debugger [dbx,xxgdb,ups,noxterm]\n");
    (*PetscHelpPrintf)(comm,"       start all processes in the debugger\n");
    (*PetscHelpPrintf)(comm," -debugger_nodes [n1,n2,..] Nodes to start in debugger\n");
    (*PetscHelpPrintf)(comm," -debugger_pause [m] : delay (in seconds) to attach debugger\n");
    (*PetscHelpPrintf)(comm," -display display: Location where graphics and debuggers are displayed\n");
    (*PetscHelpPrintf)(comm," -no_signal_handler: do not trap error signals\n");
    (*PetscHelpPrintf)(comm," -mpi_return_on_error: MPI returns error code, rather than abort on internal error\n");
    (*PetscHelpPrintf)(comm," -fp_trap: stop on floating point exceptions\n");
    (*PetscHelpPrintf)(comm,"           note on IBM RS6000 this slows run greatly\n");
    (*PetscHelpPrintf)(comm," -trdump: dump list of unfreed memory at conclusion\n");
    (*PetscHelpPrintf)(comm," -trmalloc: use our error checking malloc\n");
    (*PetscHelpPrintf)(comm," -trmalloc_off: don't use error checking malloc\n");
    (*PetscHelpPrintf)(comm," -trmalloc_nan: initialize memory locations with NaNs\n");
    (*PetscHelpPrintf)(comm," -trinfo: prints total memory usage\n");
    (*PetscHelpPrintf)(comm," -trdebug: enables extended checking for memory corruption\n");
    (*PetscHelpPrintf)(comm," -optionstable: dump list of options inputted\n");
    (*PetscHelpPrintf)(comm," -optionsleft: dump list of unused options\n");
#if defined (USE_PETSC_LOG)
    (*PetscHelpPrintf)(comm," -log[_all _summary]: logging objects and events\n");
    (*PetscHelpPrintf)(comm," -log_summary_exclude: <vec,mat,sles,snes>\n");
    (*PetscHelpPrintf)(comm," -log_trace [filename]: prints trace of all PETSc calls\n");
#if defined (HAVE_MPE)
    (*PetscHelpPrintf)(comm," -log_mpe: Also create logfile viewable through upshot\n");
#endif
    (*PetscHelpPrintf)(comm," -log_info: print informative messages about the calculations\n");
#endif
    (*PetscHelpPrintf)(comm," -v: prints PETSc version number and release date\n");
    (*PetscHelpPrintf)(comm," -options_file <file>: reads options from file\n");
    (*PetscHelpPrintf)(comm,"-----------------------------------------------\n");
  }
  ierr = OptionsHasName(PETSC_NULL,"-compare",&flg1); CHKERRQ(ierr);
  if (flg1) {
     double tol = 1.e-12;
     ierr = OptionsGetDouble(PETSC_NULL,"-compare",&tol,&flg1);CHKERRQ(ierr); 
     ierr = PetscCompareInitialize(tol); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscGetProgramName"
/*@C
    PetscGetProgramName - Gets the name of the running program. 

    Input Parameter:
.   len - length of the string name

    Output Parameter:
.   name - the name of the running program

    Notes:
    The name of the program is copied into the user-provided character
    array of length len.  On some machines the program name includes 
    its entire path, so one should generally set len >= 256.
@*/
int PetscGetProgramName(char *name,int len)
{
  PetscFunctionBegin;
  if (!options) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,1,"Must call PetscInitialize() first");
  if (!options->namegiven) SETERRQ(PETSC_ERR_PLIB,1,"Unable to determine program name");
  PetscStrncpy(name,options->programname,len);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsInsertFile_Private"
/*
    Reads options from a file and adds to options database
*/
static int OptionsInsertFile_Private(char *file)
{
  char  string[128],*first,*second,*third,*final;
  int   len,ierr,i;
  FILE  *fd;

  PetscFunctionBegin;
  PetscFixFilename(file);
  fd  = fopen(file,"r"); 
  if (fd) {
    while (fgets(string,128,fd)) {
      /* Comments are indicated by #, ! or % in the first column */
      if (string[0] == '#') continue;
      if (string[0] == '!') continue;
      if (string[0] == '%') continue;
      /* replace tabs with " " */
      len = PetscStrlen(string);
      for ( i=0; i<len; i++ ) {
        if (string[i] == '\t') {
          string[i] = ' ';
        }
      }
      first = PetscStrtok(string," ");
      second = PetscStrtok(0," ");
      if (first && first[0] == '-') {
        if (second) {final = second;} else {final = first;}
        len = PetscStrlen(final);
        while (len > 0 && (final[len-1] == ' ' || final[len-1] == '\n')) {
          len--; final[len] = 0;
        }
        OptionsSetValue(first,second);
      }
      else if (first && !PetscStrcmp(first,"alias")) {
        third = PetscStrtok(0," ");
        if (!third) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Error in options file:alias");
        len = PetscStrlen(third); 
        if (third[len-1] == '\n') third[len-1] = 0;
        ierr = OptionsSetAlias_Private(second,third); CHKERRQ(ierr);
      }
    }
    fclose(fd);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsCreate_Private"
/*
   OptionsCreate_Private - Creates a database of options.

   Input Parameters:
.  argc - count of number of command line arguments
.  args - the command line arguments
.  file - optional filename, defaults to ~username/.petscrc

   Note:
   Since OptionsCreate_Private() is automatically called by PetscInitialize(),
   the user does not typically need to call this routine. OptionsCreate_Private()
   can be called several times, adding additional entries into the database.

.keywords: options, database, create

.seealso: OptionsDestroy_Private(), OptionsPrint()
*/
int OptionsCreate_Private(int *argc,char ***args,char* file)
{
  int  ierr,rank;
  char pfile[256];

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (!file) {
    ierr = PetscGetHomeDirectory(240,pfile); CHKERRQ(ierr);
    PetscStrcat(pfile,"/.petscrc");
    file = pfile;
  }

  options->N        = 0;
  options->Naliases = 0;
  options->argc     = (argc) ? *argc : 0;
  options->args     = (args) ? *args : 0;

  ierr = OptionsInsertFile_Private(file); CHKERRQ(ierr);

  /* insert environmental options */
  {
    char *eoptions = 0, *second, *first;
    int  len;
    if (!rank) {
      eoptions = (char *) getenv("PETSC_OPTIONS");
      len      = PetscStrlen(eoptions);
      ierr     = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    } else {
      ierr     = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      if (len) {
        eoptions = (char *) PetscMalloc((len+1)*sizeof(char *));CHKPTRQ(eoptions);
      }
    }
    if (len) {
      ierr          = MPI_Bcast(eoptions,len,MPI_CHAR,0,PETSC_COMM_WORLD); CHKERRQ(ierr);
      eoptions[len] = 0;
      first         = PetscStrtok(eoptions," ");
      while (first) {
        if (first[0] != '-') {first = PetscStrtok(0," "); continue;}
        second = PetscStrtok(0," ");
        if ((!second) || ((second[0] == '-') && (second[1] > '9'))) {
          OptionsSetValue(first,(char *)0);
          first = second;
        } else {
          OptionsSetValue(first,second);
          first = PetscStrtok(0," ");
        }
      }
      if (rank) PetscFree(eoptions);
    }
  }

  /* insert command line options */
  if (argc && args && *argc) {
    int   left = *argc - 1;
    char  **eargs = *args + 1;
    while (left) {
      if (eargs[0][0] != '-') {
        eargs++; left--;
      } else if (!PetscStrcmp(eargs[0],"-options_file")) {
        ierr = OptionsInsertFile_Private(eargs[1]); CHKERRQ(ierr);
        eargs += 2; left -= 2;

      /*
         These are "bad" options that MPICH, etc put on the command line
         we strip them out here.
      */
      } else if (!PetscStrcmp(eargs[0],"-p4pg")) {
        eargs += 2; left -= 2;

      
      } else if ((left < 2) || ((eargs[1][0] == '-') && 
               ((eargs[1][1] > '9') || (eargs[1][1] < '0')))) {
        OptionsSetValue(eargs[0],(char *)0);
        eargs++; left--;
      } else {
        OptionsSetValue(eargs[0],eargs[1]);
        eargs += 2; left -= 2;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsPrint"
/*@C
   OptionsPrint - Prints the options that have been loaded. This is
        useful for debugging purposes.

   Input Parameter:
.  FILE fd - location to print options (usually stdout or stderr)

   Options Database Key:
$  -optionstable : checked within PetscFinalize()

.keywords: options, database, print, table

.seealso: OptionsAllUsed()
@*/
int OptionsPrint(FILE *fd)
{
  int i;

  PetscFunctionBegin;
  if (!fd) fd = stdout;
  if (!options) OptionsCreate_Private(0,0,0);
  for ( i=0; i<options->N; i++ ) {
    if (options->values[i]) {
      PetscFPrintf(PETSC_COMM_WORLD,fd,"OptionTable: -%s %s\n",options->names[i],options->values[i]);
    } else {
      PetscFPrintf(PETSC_COMM_WORLD,fd,"OptionTable: -%s\n",options->names[i]);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsDestroy_Private"
/*
    OptionsDestroy_Private - Destroys the option database. 

    Note:
    Since OptionsDestroy_Private() is called by PetscFinalize(), the user 
    typically does not need to call this routine.

.keywords: options, database, destroy

.seealso: OptionsCreate_Private()
*/
static int OptionsDestroy_Private(void)
{
  int i;

  PetscFunctionBegin;
  if (!options) PetscFunctionReturn(0);
  for ( i=0; i<options->N; i++ ) {
    if (options->names[i]) free(options->names[i]);
    if (options->values[i]) free(options->values[i]);
  }
  for ( i=0; i<options->Naliases; i++ ) {
    free(options->aliases1[i]);
    free(options->aliases2[i]);
  }
  free(options);
  options = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsSetValue"
/*@C
   OptionsSetValue - Sets an option name-value pair in the options 
   database, overriding whatever is already present.

   Input Parameters:
.  name - name of option, this SHOULD have the - prepended
.  value - the option value (not used for all options)

   Note:
   Only some options have values associated with them, such as
   -ksp_rtol tol.  Other options stand alone, such as -ksp_monitor.

.keywords: options, database, set, value

.seealso: OptionsCreate_Private()
@*/
int OptionsSetValue(char *name,char *value)
{
  int  len, N, n, i;
  char **names;

  PetscFunctionBegin;
  if (!options) OptionsCreate_Private(0,0,0);

  /* this is so that -h and -help are equivalent (p4 don't like -help)*/
  if (!PetscStrcmp(name,"-h")) name = "-help";

  name++;
  /* first check against aliases */
  N = options->Naliases; 
  for ( i=0; i<N; i++ ) {
    if (!PetscStrcmp(options->aliases1[i],name)) {
      name = options->aliases2[i];
      break;
    }
  }

  N = options->N; n = N;
  names = options->names; 
 
  for ( i=0; i<N; i++ ) {
    if (PetscStrcmp(names[i],name) == 0) {
      if (options->values[i]) free(options->values[i]);
      len = PetscStrlen(value);
      if (len) {
        options->values[i] = (char *) malloc( len );CHKPTRQ(options->values[i]);
        PetscStrcpy(options->values[i],value);
      } else { options->values[i] = 0;}
      PetscFunctionReturn(0);
    } else if (PetscStrcmp(names[i],name) > 0) {
      n = i;
      break;
    }
  }
  if (N >= MAXOPTIONS) {
    (*PetscErrorPrintf)("No more room in option table, limit %d\n",MAXOPTIONS);
    (*PetscErrorPrintf)("recompile options/src/options.c with larger ");
    (*PetscErrorPrintf)("value for MAXOPTIONS\n");
    PetscFunctionReturn(0);
  }
  /* shift remaining values down 1 */
  for ( i=N; i>n; i-- ) {
    names[i]           = names[i-1];
    options->values[i] = options->values[i-1];
    options->used[i]   = options->used[i-1];
  }
  /* insert new name and value */
  len = (PetscStrlen(name)+1)*sizeof(char);
  names[n] = (char *) malloc( len ); CHKPTRQ(names[n]);
  PetscStrcpy(names[n],name);
  if (value) {
    len = (PetscStrlen(value)+1)*sizeof(char);
    options->values[n] = (char *) malloc( len ); CHKPTRQ(options->values[n]);
    PetscStrcpy(options->values[n],value);
  } else {options->values[n] = 0;}
  options->used[n] = 0;
  options->N++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsClearValue"
/*@C
   OptionsClearValue - Clears an option name-value pair in the options 
   database, overriding whatever is already present.

   Input Parameters:
.  name - name of option, this SHOULD have the - prepended


.keywords: options, database, set, value, clear

.seealso: OptionsCreate_Private()
@*/
int OptionsClearValue(char *name)
{
  int  N, n, i;
  char **names;

  PetscFunctionBegin;
  if (!options) OptionsCreate_Private(0,0,0);

  name++;

  N     = options->N; n = 0;
  names = options->names; 
 
  for ( i=0; i<N; i++ ) {
    if (PetscStrcmp(names[i],name) == 0) {
      if (options->values[i]) free(options->values[i]);
      break;
    } else if (PetscStrcmp(names[i],name) > 0) {
      PetscFunctionReturn(0); /* it was not listed */
    }
    n++;
  }
  /* shift remaining values down 1 */
  for ( i=n; i<N-1; i++ ) {
    names[i]           = names[i+1];
    options->values[i] = options->values[i+1];
    options->used[i]   = options->used[i+1];
  }
  options->N--;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsSetAlias_Private"
int OptionsSetAlias_Private(char *newname,char *oldname)
{
  int len,n = options->Naliases;

  PetscFunctionBegin;
  if (newname[0] != '-') SETERRQ(PETSC_ERR_ARG_WRONG,0,"aliased must have -");
  if (oldname[0] != '-') SETERRQ(PETSC_ERR_ARG_WRONG,0,"aliasee must have -");
  if (n >= MAXALIASES) {SETERRQ(PETSC_ERR_MEM,0,"You have defined to many PETSc options aliases");}

  newname++; oldname++;
  len = (PetscStrlen(newname)+1)*sizeof(char);
  options->aliases1[n] = (char *) malloc( len ); CHKPTRQ(options->aliases1[n]);
  PetscStrcpy(options->aliases1[n],newname);
  len = (PetscStrlen(oldname)+1)*sizeof(char);
  options->aliases2[n] = (char *) malloc( len );CHKPTRQ(options->aliases2[n]);
  PetscStrcpy(options->aliases2[n],oldname);
  options->Naliases++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsFindPair_Private"
static int OptionsFindPair_Private( char *pre,char *name,char **value,int *flg)
{
  int  i, N,ierr,len;
  char **names,tmp[128];

  PetscFunctionBegin;
  if (!options) {ierr = OptionsCreate_Private(0,0,0); CHKERRQ(ierr);}
  N = options->N;
  names = options->names;

  if (name[0] != '-') SETERRQ(PETSC_ERR_ARG_WRONG,0,"Name must begin with -");

  /* append prefix to name */
  if (pre) {
    PetscStrncpy(tmp,pre,128); 
    len = PetscStrlen(tmp);
    PetscStrncat(tmp,name+1,128-len-1);
  }
  else PetscStrncpy(tmp,name+1,128);

  /* slow search */
  *flg = 0;
  for ( i=0; i<N; i++ ) {
    if (!PetscStrcmp(names[i],tmp)) {
       *value = options->values[i];
       options->used[i]++;
       *flg = 1;
       break;
     }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsReject" 
/*@C
   OptionsReject - Generates an error if a certain option is given.

   Input Parameters:
.  name - the option one is seeking 
.  mess - error message 


.keywords: options, database, has, name

.seealso: OptionsGetInt(), OptionsGetDouble(),OptionsHasName(),
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsReject(char* name,char *mess)
{
  int ierr,flag;

  PetscFunctionBegin;
  ierr = OptionsHasName(PETSC_NULL,name,&flag); CHKERRQ(ierr);
  if (flag) {
    (*PetscErrorPrintf)("Cannot run program with option %s\n",name);
    (*PetscErrorPrintf)("  %s",mess);
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Program has disabled option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsHasName"
/*@C
   OptionsHasName - Determines whether a certain option is given in the database.

   Input Parameters:
.  name - the option one is seeking 
.  pre - string to prepend to the name or PETSC_NULL

   Output Parameters:
.   flg - 1 if found else 0.


.keywords: options, database, has, name

.seealso: OptionsGetInt(), OptionsGetDouble(),
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsHasName(char* pre,char *name,int *flg)
{
  char *value;
  int  ierr;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsGetInt"
/*@C
   OptionsGetInt - Gets the integer value for a particular option in the 
                   database.

   Input Parameters:
.  name - the option one is seeking
.  pre - the string to prepend to the name or PETSC_NULL

   Output Parameter:
.  ivalue - the integer value to return
.  flg - 1 if found, else 0

.keywords: options, database, get, int

.seealso: OptionsGetDouble(), OptionsHasName(), OptionsGetString(),
          OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetInt(char*pre,char *name,int *ivalue,int *flg)
{
  char *value;
  int  flag,ierr;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag); CHKERRQ(ierr);
  if (flag) {
    if (!value) {if (flg) *flg = 0; *ivalue = 0;}
    else        {if (flg) *flg = 1; *ivalue = atoi(value);}
  } else {
    if (flg) *flg = 0;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ "OptionsGetDouble"
/*@C
   OptionsGetDouble - Gets the double precision value for a particular 
   option in the database.

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL

   Output Parameter:
.  dvalue - the double value to return
.  flg - 1 if found, 0 if not found

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetDouble(char* pre,char *name,double *dvalue,int *flg)
{
  char *value;
  int  flag,ierr;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag); CHKERRQ(ierr);
  if (flag) {
    if (!value) {if (flg) *flg = 0; *dvalue = 0.0;}
    else        {if (flg) *flg = 1; *dvalue = atof(value);}
  } else {
    if (flg) *flg = 0;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ "OptionsGetScalar"
/*@C
   OptionsGetScalar - Gets the scalar value for a particular 
   option in the database. At the moment can get only a Scalar with 
   0 imaginary part.

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL

   Output Parameter:
.  dvalue - the double value to return
.  flg - 1 if found, else 0

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetScalar(char* pre,char *name,Scalar *dvalue,int *flg)
{
  char *value;
  int  flag,ierr;
  
  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag); CHKERRQ(ierr);
  if (flag) {
    if (!value) {if (flg) *flg = 0; *dvalue = 0.0;}
    else        {if (flg) *flg = 1; *dvalue = atof(value);}
  } else {
    if (flg) *flg = 0;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ "OptionsGetDoubleArray"
/*@C
   OptionsGetDoubleArray - Gets an array of double precision values for a 
   particular option in the database.  The values must be separated with 
   commas with no intervening spaces.

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL
.  nmax - maximum number of values to retrieve

   Output Parameters:
.  dvalue - the double value to return
.  nmax - actual number of values retreived
.  flg - 1 if found, else 0

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray()
@*/
int OptionsGetDoubleArray(char* pre,char *name,double *dvalue, int *nmax,int *flg)
{
  char *value,*cpy;
  int  flag,n = 0,ierr,len;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag); CHKERRQ(ierr);
  if (!flag)  {if (flg) *flg = 0; *nmax = 0; PetscFunctionReturn(0);}
  if (!value) {if (flg) *flg = 1; *nmax = 0; PetscFunctionReturn(0);}

  if (flg) *flg = 1;
  /* make a copy of the values, otherwise we destroy the old values */
  len = PetscStrlen(value) + 1; 
  cpy = (char *) PetscMalloc(len*sizeof(char));
  PetscStrcpy(cpy,value);
  value = cpy;

  value = PetscStrtok(value,",");
  while (n < *nmax) {
    if (!value) break;
    *dvalue++ = atof(value);
    value = PetscStrtok(0,",");
    n++;
  }
  *nmax = n;
  PetscFree(cpy);
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ "OptionsGetIntArray"
/*@C
   OptionsGetIntArray - Gets an array of integer values for a particular 
   option in the database.  The values must be separated with commas with 
   no intervening spaces. 

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL
.  nmax - maximum number of values to retrieve

   Output Parameter:
.  dvalue - the integer values to return
.  nmax - actual number of values retreived
.  flg - 1 if found, else 0

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetDoubleArray()
@*/
int OptionsGetIntArray(char* pre,char *name,int *dvalue,int *nmax,int *flg)
{
  char *value,*cpy;
  int  flag,n = 0,ierr,len;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag); CHKERRQ(ierr);
  if (!flag)  {if (flg) *flg = 0; *nmax = 0; PetscFunctionReturn(0);}
  if (!value) {if (flg) *flg = 1; *nmax = 0; PetscFunctionReturn(0);}

  if (flg) *flg = 1;
  /* make a copy of the values, otherwise we destroy the old values */
  len = PetscStrlen(value) + 1; 
  cpy = (char *) PetscMalloc(len*sizeof(char));
  PetscStrcpy(cpy,value);
  value = cpy;

  value = PetscStrtok(value,",");
  while (n < *nmax) {
    if (!value) break;
    *dvalue++ = atoi(value);
    value = PetscStrtok(0,",");
    n++;
  }
  *nmax = n;
  PetscFree(cpy);
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ "OptionsGetString"
/*@C
   OptionsGetString - Gets the string value for a particular option in
   the database.

   Input Parameters:
.  name - the option one is seeking
.  len - maximum string length
.  pre - string to prepend to name or PETSC_NULL

   Output Parameter:
.  string - location to copy string
.  flg - 1 if found, else 0

   Fortran Note:
   The Fortran interface is slightly different from the C/C++
   interface (len is not used).  Sample usage in Fortran --

$      character *20 string
$      integer   flg, ierr
$      call OptionsGetString(PETSC_NULL_CHARACTER,'-s',string,flg,ierr)

.keywords: options, database, get, string

.seealso: OptionsGetInt(), OptionsGetDouble(),  
           OptionsHasName(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetString(char *pre,char *name,char *string,int len, int *flg)
{
  char *value;
  int  ierr;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,flg); CHKERRQ(ierr); 
  if (!*flg) {PetscFunctionReturn(0);}
  if (value) PetscStrncpy(string,value,len);
  else PetscMemzero(string,len);
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "OptionsGetStringArray"
/*@C
   OptionsGetStringArray - Gets an array of string values for a particular
   option in the database. The values must be separated with commas with 
   no intervening spaces. 

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to name or PETSC_NULL
.  nmax - maximum number of strings

   Output Parameter:
.  strings - location to copy strings
.  flg - 1 if found, else 0

   Notes: 
   The user is responsible for deallocating the strings that are
   returned. The Fortran interface for this routine is not supported.

   Contributed by Matthew Knepley.

.keywords: options, database, get, string

.seealso: OptionsGetInt(), OptionsGetDouble(),  
           OptionsHasName(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetStringArray(char *pre, char *name, char **strings, int *nmax, int *flg)
{
  char *value;
  char *cpy;
  int   len;
  int   n;
  int   ierr;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,flg); CHKERRQ(ierr); 
  if (!*flg)  {*nmax = 0; PetscFunctionReturn(0);}
  if (!value) {*nmax = 0; PetscFunctionReturn(0);}
  if (*nmax == 0) PetscFunctionReturn(0);

  /* make a copy of the values, otherwise we destroy the old values */
  len = PetscStrlen(value) + 1;
  cpy = (char *) PetscMalloc(len * sizeof(char)); CHKPTRQ(cpy);
  PetscStrcpy(cpy, value);
  value = cpy;

  value = PetscStrtok(value, ",");
  n = 0;
  while (n < *nmax) {
    if (!value) break;
    len        = PetscStrlen(value) + 1;
    strings[n] = (char *) PetscMalloc(len * sizeof(char)); CHKPTRQ(strings[n]);
    PetscStrcpy(strings[n], value);
    value = PetscStrtok(0, ",");
    n++;
  }
  *nmax = n;
  PetscFree(cpy);
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "OptionsAllUsed"
/*@C
   OptionsAllUsed - Returns a count of the number of options in the 
   database that have never been selected.

   Options Database Key:
$  -optionsleft : checked within PetscFinalize()

.keywords: options, database, missed, unused, all, used

.seealso: OptionsPrint()
@*/
int OptionsAllUsed(void)
{
  int  i,n = 0;

  PetscFunctionBegin;
  for ( i=0; i<options->N; i++ ) {
    if (!options->used[i]) { n++; }
  }
  PetscFunctionReturn(n);
}

