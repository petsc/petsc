#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pinit.c,v 1.17 1999/05/04 20:29:12 balay Exp bsmith $";
#endif
/*

   This file defines the initialization of PETSc, including PetscInitialize()

*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "sys.h"

/* -----------------------------------------------------------------------------------------*/


extern int PetscInitialize_DynamicLibraries(void);
extern int PetscFinalize_DynamicLibraries(void);
extern int FListDestroyAll(void);

#include "snes.h" /* so that cookies are defined */

/*
       Checks the options database for initializations related to the 
    PETSc components
*/
#undef __FUNC__  
#define __FUNC__ "OptionsCheckInitial_Components"
int OptionsCheckInitial_Components(void)
{
  MPI_Comm comm = PETSC_COMM_WORLD;
  int      flg1,ierr;
  char     *f,mname[256];

  PetscFunctionBegin;
  /*
     Publishing to the AMS
  */
#if defined(HAVE_AMS)
  ierr = OptionsHasName(PETSC_NULL,"-ams_publish_objects",&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscAMSPublishAll = PETSC_TRUE;
  }
  ierr = OptionsHasName(0, "-ams_publish_stack", &flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscStackPublish();CHKERRQ(ierr);
  }
#endif


  ierr = OptionsGetString(PETSC_NULL,"-log_info_exclude",mname,256, &flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscStrstr(mname,"null",&f);CHKERRQ(ierr);
    if (f) {
      ierr = PLogInfoDeactivateClass(PETSC_NULL);CHKERRQ(ierr);
    }
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
  ierr = OptionsGetString(PETSC_NULL,"-log_summary_exclude",mname,256, &flg1);CHKERRQ(ierr);
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
    ierr = PLogEventActivate(VEC_NormComm);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_DotBarrier);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_DotComm);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_MDotBarrier);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_MDotComm);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_ReduceBarrier);CHKERRQ(ierr);
    ierr = PLogEventActivate(VEC_ReduceCommOnly);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help", &flg1);CHKERRQ(ierr);
  if (flg1) {
#if defined (USE_PETSC_LOG)
    ierr = (*PetscHelpPrintf)(comm,"------Additional PETSc component options--------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_summary_exclude: <vec,mat,sles,snes>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_info_exclude: <null,vec,mat,sles,snes,ts>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"-----------------------------------------------\n");CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscInitializeNoArguments"
/*@C
      PetscInitializeNoArguments - Calls PetscInitialize() from C/C++ without
        the command line arguments.

   Collective
  
   Level: advanced

.seealso: PetscInitialize(), PetscInitializeFortran()
@*/
int PetscInitializeNoArguments(void)
{
  int ierr;

  PetscFunctionBegin;
  ierr = AliceInitializeNoArguments();
  PetscFunctionReturn(ierr);
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
.  -on_error_attach_debugger [noxterm,dbx,xdb,gdb,...] - Starts debugger when error detected
.  -debugger_nodes [node1,node2,...] - Indicates nodes to start in debugger
.  -debugger_pause [sleeptime] (in seconds) - Pauses debugger
.  -stop_for_debugger - Print message on how to attach debugger manually to 
                        process and wait (-debugger_pause) seconds for attachment
.  -trmalloc - Indicates use of PETSc error-checking malloc
.  -trmalloc_off - Indicates not to use error-checking malloc
.  -fp_trap - Stops on floating point exceptions (Note that on the
              IBM RS6000 this slows code by at least a factor of 10.)
.  -no_signal_handler - Indicates not to trap error signals
-  -get_resident_set_size - Print memory usage at end of run

   Options Database Keys for Profiling:
   See the 'Profiling' chapter of the users manual for details.
+  -log_trace [filename] - Print traces of all PETSc calls
        to the screen (useful to determine where a program
        hangs without running in the debugger).  See PLogTraceBegin().
.  -log_info <optional filename> - Prints verbose information to the screen
-  -log_info_exclude <null,vec,mat,sles,snes,ts> - Excludes some of the verbose messages

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
  int        ierr;

  PetscFunctionBegin;
  ierr = AliceInitialize(argc,args,file,help);CHKERRQ(ierr);
  ierr = OptionsCheckInitial_Components();CHKERRQ(ierr);

  /*
      Initialize the default dynamic libraries
  */
  PetscFunctionReturn(ierr);
}

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

   Level: beginner

   Note:
   See PetscInitialize() for more general runtime options.

.keywords: finalize, exit, end

.seealso: PetscInitialize(), OptionsPrint(), PetscTrDump(), PetscMPIDump()
@*/
int PetscFinalize(void)
{
  int ierr;
  
  PetscFunctionBegin;
  /*
     Destroy all the function registration lists created
  */
  ierr = PetscFinalize_DynamicLibraries();CHKERRQ(ierr);
  ierr = AliceFinalize();
  PetscFunctionReturn(ierr);
}
