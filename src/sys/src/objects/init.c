#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: init.c,v 1.1 1998/04/30 23:15:16 bsmith Exp bsmith $";
#endif
/*

   This file defines the initialization of PETSc, including PetscInitialize()

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
#include "petsc.h"
#include "sys.h"
#include "pinclude/ptime.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_nt)
#include <sys/param.h>
#include <sys/utsname.h>
#endif
#if defined(PARCH_nt)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_nt_gnu)
#include <windows.h>
#endif
#include <fcntl.h>
#include <time.h>  
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "pinclude/petscfix.h"

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#include "pinclude/petscfix.h"

/*
     Global variable indicates if PETSc started up MPI, or it was 
   already started before PETSc was initialized.
*/
int          PetscBeganMPI = 0;

extern int OptionsCheckInitial_Private();
extern int OptionsCreate_Private(int*,char***,char*);
extern int OptionsSetAlias_Private(char *,char *);
extern int OptionsDestroy_Private();
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

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  d - integer to compare

   Options Database Key:
.  -compare - Activates PetscCompareDouble(), PetscCompareInt(), and PetscCompareScalar()

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

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  d - double precision number to compare

   Options Database Key:
.  -compare - Activates PetscCompareDouble(), PetscCompareInt(), and PetscCompareScalar()

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

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  d - scalar to compare

   Options Database Key:
.  -compare - Activates PetscCompareDouble(), PetscCompareInt(), and PetscCompareScalar()

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

    Note: 
    Only works with C programs.
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
  /* printf("[%d] my name %s basename %s mysize %d\n",rank,programname,basename,mysize); */

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

extern int PetscInitialize_DynamicLibraries();
extern int PetscFinalize_DynamicLibraries();

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
   PetscInitialize() calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of 
   your program -- usually the very first line! 

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use PETSC_NULL for default)
-  help - [optional] Help message to print, use PETSC_NULL for no message

   Options Database Keys:
+  -start_in_debugger [noxterm,dbx,xdb,gdb,...] - Starts program in debugger
+  -on_error_attach_debugger [noxterm,dbx,xdb,gdb,...] - Starts debugger when error detected
.  -debugger_nodes [node1,node2,...] - Indicates nodes to start in debugger
.  -debugger_pause [sleeptime] (in seconds) - Pauses debugger
.  -trmalloc - Indicates use of PETSc error-checking malloc
.  -trmalloc_off - Indicates not to use error-checking malloc
.  -fp_trap - Stops on floating point exceptions (Note that on the
              IBM RS6000 this slows code by at least a factor of 10.)
-  -no_signal_handler - Indicates not to trap error signals

   Options Database Keys for Profiling:
   See the 'Profiling' chapter of the users manual for details.
+  -log_trace [filename] - Print traces of all PETSc calls
        to the screen (useful to determine where a program
        hangs without running in the debugger).  See PLogTraceBegin().
.  -log_info - Prints verbose information to the screen
-  -log_info_exclude <null,vec,mat,sles,snes,ts> - Excludes some of the verbose messages

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   Fortran Version:
   In Fortran this routine has the format
$       call PetscInitialize(file,ierr)

+   ierr - error return code
-   file - [optional] PETSc database file name, defaults to 
           ~username/.petscrc (use PETSC_NULL_CHARACTER for default)
           
   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use PETSC_NULL as 
   in the C version.  See the users manual for details.


.keywords: initialize, options, database, startup

.seealso: PetscFinalize(), PetscInitializeFortran()
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

   Collective on PETSC_COMM_WORLD

   Options Database Keys:
+  -optionstable - Calls OptionsPrint()
.  -optionsleft - Prints unused options that remain in the database
.  -mpidump - Calls PetscMPIDump()
.  -trdump - Calls PetscTrDump()
.  -trinfo - Prints total memory usage
.  -trdebug - Calls malloc_debug(2) to activate memory
        allocation diagnostics (used by PETSC_ARCH=sun4, 
        BOPT=[g,g_c++,g_complex] only!)
-  -trmalloc_log - Prints summary of memory usage

   Options Database Keys for Profiling:
   See the 'Profiling' chapter of the users manual for details.
+  -log_summary [filename] - Prints summary of flop and timing
        information to screen. If the filename is specified the
        summary is written to the file. (for code compiled with 
        USE_PETSC_LOG).  See PLogPrintSummary().
.  -log_all [filename] - Logs extensive profiling information
        (for code compiled with USE_PETSC_LOG). See PLogDump(). 
.  -log [filename] - Logs basic profiline information (for
        code compiled with USE_PETSC_LOG).  See PLogDump().
.  -log_sync - Log the synchronization in scatters, inner products
        and norms
-  -log_mpe [filename] - Creates a logfile viewable by the 
      utility Upshot/Nupshot (in MPICH distribution)

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

#if defined(HAVE_MALLOC_VERIFY) && defined(__cplusplus)
extern "C" {
  extern int malloc_debug(int);
}
#elif defined(HAVE_MALLOC_VERIFY)
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
#if defined(HAVE_MALLOC_VERIFY) && defined(USE_PETSC_BOPT_g)
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
    (*PetscHelpPrintf)(comm,"%s",PETSC_AUTHOR_INFO);
    (*PetscHelpPrintf)(comm,"See docs/copyright.html for copyright information\n");
    (*PetscHelpPrintf)(comm,"See docs/changes.html for recent updates.\n");
    (*PetscHelpPrintf)(comm,"See docs/troubleshooting.html hints for problems.\n");
    (*PetscHelpPrintf)(comm,"See docs/manualpages/manualpages.html for help. \n");
#if !defined(PARCH_nt)
    (*PetscHelpPrintf)(comm,"Libraries linked from %s\n",PETSC_LDIR);
#endif
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
#if defined(USE_DBX_DEBUGGER)
  ierr = PetscSetDebugger("dbx",1,0); CHKERRQ(ierr);
#elif defined(USE_XDB_DEBUGGER) 
  ierr = PetscSetDebugger("xdb",1,0); CHKERRQ(ierr);
#endif

  ierr = OptionsGetString(PETSC_NULL,"-on_error_attach_debugger",string,64,&flg1);CHKERRQ(ierr);
  if (flg1) {
    char *debugger = 0, *display = 0;
    int  xterm     = 1;
    if (PetscStrstr(string,"noxterm")) xterm = 0;

    if (PetscStrstr(string,"xdb"))     debugger = "xdb";
    if (PetscStrstr(string,"dbx"))     debugger = "dbx";
    if (PetscStrstr(string,"xldb"))    debugger = "xldb";
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

      if (PetscStrstr(string,"xdb"))     debugger = "xdb";
      if (PetscStrstr(string,"dbx"))     debugger = "dbx";
      if (PetscStrstr(string,"xldb"))    debugger = "xldb";
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
        if (PetscStrstr(mname,"null")) {
          PLogInfoDeactivateClass(PETSC_NULL;
        }
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
        if (PetscStrstr(mname,"ts")) {
          PLogInfoDeactivateClass(TS_COOKIE);
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
    (*PetscHelpPrintf)(comm," -on_error_attach_debugger [gdb,dbx,xxgdb,ups,noxterm]\n"); 
    (*PetscHelpPrintf)(comm,"       start the debugger in new xterm\n");
    (*PetscHelpPrintf)(comm,"       unless noxterm is given\n");
    (*PetscHelpPrintf)(comm," -start_in_debugger [gdb,dbx,xxgdb,ups,noxterm]\n");
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
    (*PetscHelpPrintf)(comm," -log_info_exclude: <null,vec,mat,sles,snes,ts>\n");
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

    Not Collective

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


