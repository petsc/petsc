/*$Id: pinit.c,v 1.36 2000/05/13 04:36:48 bsmith Exp bsmith $*/
/*
   This file defines the initialization of PETSc, including PetscInitialize()
*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "petscsys.h"

/* -----------------------------------------------------------------------------------------*/

extern FILE *petsc_history;

EXTERN int PetscInitialize_DynamicLibraries(void);
EXTERN int PetscFinalize_DynamicLibraries(void);
EXTERN int FListDestroyAll(void);
EXTERN int PLogEventRegisterDestroy_Private(void);
EXTERN int PLogStageDestroy_Private(void);
EXTERN int PetscSequentialPhaseBegin_Private(MPI_Comm,int);
EXTERN int PetscSequentialPhaseEnd_Private(MPI_Comm,int);
EXTERN int PLogCloseHistoryFile(FILE **);

#include "petscsnes.h" /* so that cookies are defined */

/* this is used by the _, __, and ___ macros (see include/petscerror.h) */
int __gierr = 0;

/*
       Checks the options database for initializations related to the 
    PETSc components
*/
#undef __FUNC__  
#define __FUNC__ /*<a name="OptionsCheckInitial_Components"></a>*/"OptionsCheckInitial_Components"
int OptionsCheckInitial_Components(void)
{
  MPI_Comm   comm = PETSC_COMM_WORLD;
  PetscTruth flg1;
  int        ierr;
  char       *f,mname[256];

  PetscFunctionBegin;
  /*
     Publishing to the AMS
  */
#if defined(PETSC_HAVE_AMS)
  ierr = OptionsHasName(PETSC_NULL,"-ams_publish_objects",&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscAMSPublishAll = PETSC_TRUE;
  }
  ierr = OptionsHasName(PETSC_NULL,"-ams_publish_stack",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscStackPublish();CHKERRQ(ierr);
  }
#endif

  ierr = OptionsGetString(PETSC_NULL,"-log_info_exclude",mname,256,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscStrstr(mname,"vec",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogInfoDeactivateClass(VEC_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(mname,"mat",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogInfoDeactivateClass(MAT_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(mname,"sles",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogInfoDeactivateClass(SLES_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(mname,"snes",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogInfoDeactivateClass(SNES_COOKIE);CHKERRQ(ierr);
    }
  }
  ierr = OptionsGetString(PETSC_NULL,"-log_summary_exclude",mname,256,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscStrstr(mname,"vec",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogEventDeactivateClass(VEC_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(mname,"mat",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogEventDeactivateClass(MAT_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(mname,"sles",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogEventDeactivateClass(SLES_COOKIE);CHKERRQ(ierr);
    }
    ierr = PetscStrstr(mname,"snes",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogEventDeactivateClass(SNES_COOKIE);CHKERRQ(ierr);
    }
  }
    
  ierr = OptionsHasName(PETSC_NULL,"-log_sync",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PLogEventActivate(VEC_ScatterBarrier);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_NormBarrier);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_DotBarrier);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_MDotBarrier);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_ReduceBarrier);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {
#if defined (PETSC_USE_LOG)
    ierr = (*PetscHelpPrintf)(comm,"------Additional PETSc component options--------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_summary_exclude: <vec,mat,sles,snes>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_info_exclude: <null,vec,mat,sles,snes,ts>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"-----------------------------------------------\n");CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscInitializeNoArguments"></a>*/"PetscInitializeNoArguments"
/*@C
      PetscInitializeNoArguments - Calls PetscInitialize() from C/C++ without
        the command line arguments.

   Collective
  
   Level: advanced

.seealso: PetscInitialize(), PetscInitializeFortran()
@*/
int PetscInitializeNoArguments(void)
{
  int ierr,argc = 0;
  char **args = 0;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&args,PETSC_NULL,PETSC_NULL);
  PetscFunctionReturn(ierr);
}

EXTERN int        OptionsCheckInitial(void);
extern PetscTruth PetscBeganMPI;

/*
       This function is the MPI reduction operation used to compute the sum of the 
   first half of the entries and the max of the second half.
*/
MPI_Op PetscMaxSum_Op = 0;

EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ /*<a name="PetscMaxSum_Local"></a>*/"PetscMaxSum_Local"
void PetscMaxSum_Local(void *in,void *out,int *cnt,MPI_Datatype *datatype)
{
  int *xin = (int *)in,*xout = (int*)out,i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype != MPI_INT) {
    (*PetscErrorPrintf)("Can only handle MPI_INT data types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  if (count % 2) {
    (*PetscErrorPrintf)("Count must be divisible by 2");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  count = count/2; 
  for (i=0; i<count; i++) {
    xout[i] = PetscMax(xout[i],xin[i]); 
  }
  for (i=count; i<2*count; i++) {
    xout[i] += xin[i]; 
  }

  PetscStackPop;
  return;
}
EXTERN_C_END

#if defined(PETSC_USE_COMPLEX)
MPI_Op PetscSum_Op = 0;

EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ /*<a name="PetscSum_Local"></a>*/"PetscSum_Local"
void PetscSum_Local(void *in,void *out,int *cnt,MPI_Datatype *datatype)
{
  Scalar *xin = (Scalar *)in,*xout = (Scalar*)out;
  int    i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_SCALAR) {
    (*PetscErrorPrintf)("Can only handle MPIU_SCALAR data (i.e. double or complex) types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  for (i=0; i<count; i++) {
    xout[i] += xin[i]; 
  }

  PetscStackPop;
  return;
}
EXTERN_C_END
#endif

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscInitialize"></a>*/"PetscInitialize"
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
.  -on_error_attach_debugger [noxterm,dbx,xdb,gdb,...] - Starts debugger when error detected
.  -on_error_emacs <machinename> causes emacsclient to jump to error file
.  -debugger_nodes [node1,node2,...] - Indicates nodes to start in debugger
.  -debugger_pause [sleeptime] (in seconds) - Pauses debugger
.  -stop_for_debugger - Print message on how to attach debugger manually to 
                        process and wait (-debugger_pause) seconds for attachment
.  -trmalloc - Indicates use of PETSc error-checking malloc
.  -trmalloc_off - Indicates not to use error-checking malloc
.  -fp_trap - Stops on floating point exceptions (Note that on the
              IBM RS6000 this slows code by at least a factor of 10.)
.  -no_signal_handler - Indicates not to trap error signals
.  -shared_tmp - indicates /tmp directory is shared by all processors
.  -not_shared_tmp - each processor has own /tmp
.  -tmp - alternative name of /tmp directory
.  -get_total_flops - returns total flops done by all processors
-  -get_resident_set_size - Print memory usage at end of run

   Options Database Keys for Profiling:
   See the Profiling chapter of the users manual for details.
+  -log_trace [filename] - Print traces of all PETSc calls
        to the screen (useful to determine where a program
        hangs without running in the debugger).  See PLogTraceBegin().
.  -log_info <optional filename> - Prints verbose information to the screen
-  -log_info_exclude <null,vec,mat,sles,snes,ts> - Excludes some of the verbose messages

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
-   file - [optional] PETSc database file name, defaults to 
           ~username/.petscrc (use PETSC_NULL_CHARACTER for default)
           
   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use PETSC_NULL as 
   in the C version.  See the users manual for details.


.keywords: initialize, options, database, startup

.seealso: PetscFinalize(), PetscInitializeFortran()
@*/
int PetscInitialize(int *argc,char ***args,char file[],const char help[])
{
  int        ierr,flag,dummy_tag,size;
  PetscTruth flg;
  char       hostname[16];

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(0);

  ierr = OptionsCreate();CHKERRQ(ierr);

  /*
     We initialize the program name here (before MPI_Init()) because MPICH has a bug in 
     it that it sets args[0] on all processors to be args[0] on the first processor.
  */
  if (argc && *argc) {
    ierr = PetscSetProgramName(**args);CHKERRQ(ierr);
  } else {
    ierr = PetscSetProgramName("Unknown Name");CHKERRQ(ierr);
  }

  /* Also initialize the initial datestamp */
  ierr = PetscSetInitialDate();CHKERRQ(ierr);

  ierr = MPI_Initialized(&flag);CHKERRQ(ierr);
  if (!flag) {
    ierr          = MPI_Init(argc,args);CHKERRQ(ierr);
    PetscBeganMPI = PETSC_TRUE;
  }
  PetscInitializeCalled = PETSC_TRUE;

  if (!PETSC_COMM_WORLD) {
    PETSC_COMM_WORLD = MPI_COMM_WORLD;
  }

  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  /* 
     Initialized the global complex variable; this is because with 
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
    Scalar ic(0.0,1.0);
    PETSC_i = ic; 
  }
  ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscSum_Local,1,&PetscSum_Op);CHKERRQ(ierr);
#endif

  /*
       Create the PETSc MPI reduction operator that sums of the first
     half of the entries and maxes the second half.
  */
  ierr = MPI_Op_create(PetscMaxSum_Local,1,&PetscMaxSum_Op);CHKERRQ(ierr);

  /*
        Build the options database and check for user setup requests
  */
  ierr = OptionsInsert(argc,args,file);CHKERRQ(ierr);
  /*
      Print main application help message
  */
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (help && flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,help);CHKERRQ(ierr);
  }
  ierr = OptionsCheckInitial();CHKERRQ(ierr); 

  /*
       Initialize PETSC_COMM_SELF and WORLD as a MPI_Comm with the PETSc attribute.
    
       We delay until here to do it, since PetscMalloc() may not have been
     setup before this.
  */
  ierr = PetscCommDuplicate_Private(MPI_COMM_SELF,&PETSC_COMM_SELF,&dummy_tag);CHKERRQ(ierr);
  ierr = PetscCommDuplicate_Private(PETSC_COMM_WORLD,&PETSC_COMM_WORLD,&dummy_tag);CHKERRQ(ierr);

  /*
      Load the dynamic libraries (on machines that support them), this registers all
    the solvers etc. (On non-dynamic machines this initializes the Draw and Viewer classes)
  */
  ierr = PetscInitialize_DynamicLibraries();CHKERRQ(ierr);

  /*
      Initialize all the default viewers
  */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  PLogInfo(0,"PetscInitialize:PETSc successfully started: number of processors = %d\n",size);
  ierr = PetscGetHostName(hostname,16);CHKERRQ(ierr);
  PLogInfo(0,"PetscInitialize:Running on machine: %s\n",hostname);

  ierr = OptionsCheckInitial_Components();CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscFinalize"></a>*/"PetscFinalize"
/*@C 
   PetscFinalize - Checks for options to be called at the conclusion
   of the program and calls MPI_Finalize().

   Collective on PETSC_COMM_WORLD

   Options Database Keys:
+  -options_table - Calls OptionsPrint()
.  -options_left - Prints unused options that remain in the database
.  -options_left_off - Does not print unused options that remain in the database
.  -mpidump - Calls PetscMPIDump()
.  -trdump - Calls PetscTrDump()
.  -trinfo - Prints total memory usage
.  -trdebug - Calls malloc_debug(2) to activate memory
        allocation diagnostics (used by PETSC_ARCH=sun4, 
        BOPT=[g,g_c++,g_complex] only!)
-  -trmalloc_log - Prints summary of memory usage

   Options Database Keys for Profiling:
   See the Profiling chapter of the users manual for details.
+  -log_summary [filename] - Prints summary of flop and timing
        information to screen. If the filename is specified the
        summary is written to the file. (for code compiled with 
        PETSC_USE_LOG).  See PLogPrintSummary().
.  -log_all [filename] - Logs extensive profiling information
        (for code compiled with PETSC_USE_LOG). See PLogDump(). 
.  -log [filename] - Logs basic profiline information (for
        code compiled with PETSC_USE_LOG).  See PLogDump().
.  -log_sync - Log the synchronization in scatters, inner products
        and norms
-  -log_mpe [filename] - Creates a logfile viewable by the 
      utility Upshot/Nupshot (in MPICH distribution)

   Level: beginner

   Note:
   See PetscInitialize() for more general runtime options.

.keywords: finalize, exit, end

.seealso: PetscInitialize(), OptionsPrint(), PetscTrDump(), PetscMPIDump()
@*/
int PetscFinalize(void)
{
  int        ierr,rank = 0,nopt;
  PLogDouble rss;
  PetscTruth flg1,flg2,flg3;
  
  PetscFunctionBegin;

  if (!PetscInitializeCalled) {
    (*PetscErrorPrintf)("PETSc ERROR: PetscInitialize() must be called before PetscFinalize()\n");
    PetscFunctionReturn(0);
  }
  /*
     Destroy all the function registration lists created
  */
  ierr = PetscFinalize_DynamicLibraries();CHKERRQ(ierr);


  ierr = OptionsHasName(PETSC_NULL,"-get_resident_set_size",&flg1);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscGetResidentSetSize(&rss);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Size of entire process memory %d\n",rank,(int)rss);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-get_total_flops",&flg1);CHKERRQ(ierr);
  if (flg1) {
    PLogDouble flops = 0;
    ierr = MPI_Reduce(&_TotalFlops,&flops,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Total flops over all processors %g\n",flops);CHKERRQ(ierr);
  }

  /*
     Free all objects registered with PetscObjectRegisterDestroy() such ast
    VIEWER_XXX_().
  */
  ierr = PetscObjectRegisterDestroyAll();CHKERRQ(ierr);  

#if defined(PETSC_USE_STACK)
  if (PetscStackActive) {
    ierr = PetscStackDestroy();CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_USE_LOG)
  {
    char mname[64];
#if defined(PETSC_HAVE_MPE)
    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_mpe",mname,64,&flg1);CHKERRQ(ierr);
    if (flg1){
      if (mname[0]) {ierr = PLogMPEDump(mname);CHKERRQ(ierr);}
      else          {ierr = PLogMPEDump(0);CHKERRQ(ierr);}
    }
#endif
    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_summary",mname,64,&flg1);CHKERRQ(ierr);
    if (flg1) { 
      if (mname[0])  {ierr = PLogPrintSummary(PETSC_COMM_WORLD,mname);CHKERRQ(ierr);}
      else           {ierr = PLogPrintSummary(PETSC_COMM_WORLD,0);CHKERRQ(ierr);}
    }

    mname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-log_all",mname,64,&flg1);CHKERRQ(ierr);
    ierr = OptionsGetString(PETSC_NULL,"-log",mname,64,&flg2);CHKERRQ(ierr);
    if (flg1 || flg2){
      if (mname[0]) PLogDump(mname); 
      else          PLogDump(0);
    }
    ierr = PLogDestroy();CHKERRQ(ierr);
  }
#endif
  ierr = OptionsHasName(PETSC_NULL,"-no_signal_handler",&flg1);CHKERRQ(ierr);
  if (!flg1) { ierr = PetscPopSignalHandler();CHKERRQ(ierr);}
  ierr = OptionsHasName(PETSC_NULL,"-mpidump",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscMPIDump(stdout);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-trdump",&flg1);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-optionstable",&flg1);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-options_table",&flg2);CHKERRQ(ierr);
  if (flg1 && flg2) {
    if (!rank) {ierr = OptionsPrint(stdout);CHKERRQ(ierr);}
  }
  /* to prevent PETSc -options_left from warning */
  ierr = OptionsHasName(PETSC_NULL,"-nox_warning",&flg1);CHKERRQ(ierr)

  ierr = OptionsHasName(PETSC_NULL,"-optionsleft",&flg1);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-options_left",&flg2);CHKERRQ(ierr);
  ierr = OptionsAllUsed(&nopt);CHKERRQ(ierr);
  if (flg1 || flg2) {
    ierr = OptionsPrint(stdout);CHKERRQ(ierr);
  }
  if (flg1) {
    if (!nopt) { 
      ierr = PetscPrintf(PETSC_COMM_WORLD,"There are no unused options.\n");CHKERRQ(ierr);
    } else if (nopt == 1) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"There is one unused database option. It is:\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"There are %d unused database options. They are:\n",nopt);CHKERRQ(ierr);
    }
  }

#if defined(PETSC_USE_BOPT_g)
  ierr = OptionsHasName(PETSC_NULL,"-options_left_off",&flg2);CHKERRQ(ierr);
  if (nopt && !flg1 && !flg2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING! There are options you set that were not used!\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING! could be spelling mistake, etc!\n");CHKERRQ(ierr);
  }
  if ((nopt || flg1) && !flg2) {
#else 
  if (flg1) {
#endif
    ierr = OptionsLeft();CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-log_history",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PLogCloseHistoryFile(&petsc_history);CHKERRQ(ierr);
    petsc_history = 0;
  }


  /*
       Destroy PETSC_COMM_SELF/WORLD as a MPI_Comm with the PETSc 
     attribute.
  */
  ierr = PetscCommDestroy_Private(&PETSC_COMM_SELF);CHKERRQ(ierr);
  ierr = PetscCommDestroy_Private(&PETSC_COMM_WORLD);CHKERRQ(ierr);

  /*
       Free all the registered create functions, such as KSPList, VecList, SNESList, etc
  */
  ierr = FListDestroyAll();CHKERRQ(ierr); 

#if defined(PETSC_USE_LOG)
  ierr = PLogEventRegisterDestroy_Private();CHKERRQ(ierr);
  ierr = PLogStageDestroy_Private();CHKERRQ(ierr);
#endif

  ierr = OptionsHasName(PETSC_NULL,"-trdump",&flg1);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trinfo",&flg2);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-trmalloc_log",&flg3);CHKERRQ(ierr);
  if (flg1) {
    char fname[256];
    FILE *fd;
    
    fname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-trdump",fname,250,&flg1);CHKERRQ(ierr);
    if (flg1 && fname[0]) {
      char sname[256];

      sprintf(sname,"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); if (!fd) SETERRQ1(1,1,"Cannot open log file: %s",sname);
      ierr = PetscTrDump(fd);CHKERRQ(ierr);
      fclose(fd);
    } else {
      MPI_Comm local_comm;

      ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseBegin_Private(local_comm,1);CHKERRQ(ierr);
        ierr = PetscTrDump(stdout);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseEnd_Private(local_comm,1);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
    }
  } else if (flg2) {
    MPI_Comm   local_comm;
    PLogDouble maxm;

    ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
    ierr = PetscTrSpace(PETSC_NULL,PETSC_NULL,&maxm);CHKERRQ(ierr);
    ierr = PetscSequentialPhaseBegin_Private(local_comm,1);CHKERRQ(ierr);
      printf("[%d] Maximum memory used %g\n",rank,maxm);
    ierr = PetscSequentialPhaseEnd_Private(local_comm,1);CHKERRQ(ierr);
    ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
  }
  if (flg3) {
    char fname[256];
    FILE *fd;
    
    fname[0] = 0;
    ierr = OptionsGetString(PETSC_NULL,"-trmalloc_log",fname,250,&flg1);CHKERRQ(ierr);
    if (flg1 && fname[0]) {
      char sname[256];

      sprintf(sname,"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); if (!fd) SETERRQ1(1,1,"Cannot open log file: %s",sname);
      ierr = PetscTrLogDump(fd);CHKERRQ(ierr); 
      fclose(fd);
    } else {
      ierr = PetscTrLogDump(stdout);CHKERRQ(ierr); 
    }
  }
  /* Can be destroyed only after all the options are used */
  ierr = OptionsDestroy();CHKERRQ(ierr);

  PLogInfo(0,"PetscFinalize:PETSc successfully ended!\n");
  if (PetscBeganMPI) {
    ierr = MPI_Finalize();CHKERRQ(ierr);
  }

/*

     Note: In certain cases PETSC_COMM_WORLD is never MPI_Comm_free()ed because 
   the communicator has some outstanding requests on it. Specifically if the 
   flag PETSC_HAVE_BROKEN_REQUEST_FREE is set (for IBM MPI implementation). See 
   src/vec/utils/vpscat.c. Due to this the memory allocated in PetscCommDuplicate_Private()
   is never freed as it should be. Thus one may obtain messages of the form
   [ 1] 8 bytes PetscCommDuplicate_Private() line 645 in src/sys/src/mpiu.c indicating the
   memory was not freed.

*/
  ierr = PetscClearMalloc();CHKERRQ(ierr);
  PetscInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(ierr);
}

