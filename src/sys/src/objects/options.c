
#ifndef lint
static char vcid[] = "$Id: options.c,v 1.31 1995/08/04 01:51:30 bsmith Exp bsmith $";
#endif
/*
    Routines to simplify the use of command line, file options etc.

    This file uses regular malloc and free because it cannot know 
  what malloc is being used until it has already processed the input.

*/
#include "petsc.h"
#include <stdio.h>
#include <math.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "sys.h"
#include "sysio.h"
#include "sys/nreg.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pviewer.h"
#include "petscfix.h"

/* 
    For simplicity, we begin with a static size database
*/
#define MAXOPTIONS 256
#define MAXALIASES 10

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

int OptionsCheckInitial_Private(),
    OptionsCreate_Private(int*,char***,char*,char*),
    OptionsSetAlias_Private(char *,char *);
static int OptionsDestroy_Private();

#if defined(PETSC_COMPLEX)
MPI_Datatype  MPIU_COMPLEX;
#endif

/*@C
   PetscInitialize - Initializes the PETSc database and MPI. 
   PetscInitialize calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of 
   your program -- usually the very first line! 

   Input Parameters:
.  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
.  env  - [optional] PETSc database environmental variable, defaults to 
          PETSC_OPTIONS

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   In FORTRAN this takes one integer argument!

.keywords: initialize, options, database, startup

.seealso: PetscFinalize()
@*/
int PetscInitialize(int *argc,char ***args,char *file,char *env)
{
  int ierr,flag;

  MPI_Initialized(&flag);
  if (!flag) {
    ierr = MPI_Init(argc,args); CHKERRQ(ierr);
    PetscBeganMPI = 1;
  }
#if defined(PETSC_COMPLEX)
  MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_COMPLEX);
  MPI_Type_commit(&MPIU_COMPLEX);
#endif
  ierr = ViewerInitialize_Private(); CHKERRQ(ierr);
  ierr = OptionsCreate_Private(argc,args,file,env); CHKERRQ(ierr);
  ierr = OptionsCheckInitial_Private(); CHKERRQ(ierr);
  return 0;
}

/*@ 
   PetscFinalize - Checks for options to be called at the conclusion
   of the program and calls MPI_Finalize().

   Options Database Keys:
$  -optionstable : Calls OptionsPrint()
$  -optionsused : Calls OptionsAllUsed()
$  -optionsleft : Prints unused options that remain in 
$     the database
$  -no_signal_handler : Turns off the signal handler
$  -trdump : Calls TrDump()
$  -log_all : Prints extensive log information (for
$      code compiled with PETSC_LOG)
$  -log : Prints basic log information (for code 
$      compiled with PETSC_LOG)
$  -log_summary : Prints summary of flop and timing
$      information to screen (for code compiled with 
$      PETSC_LOG)
$  -fp_trap : stops on floating point exceptions, note on the IBM RS6000
$             this slows your code by at least a factor of 10.

.keywords: finalize, exit, end

.seealso: PetscInitialize(), OptionsPrint(), TrDump()
@*/
int PetscFinalize()
{
  int  ierr,i,mytid = 0;

#if defined(PETSC_LOG)
  if (OptionsHasName(0,"-log_summary")) {
    PLogPrint(MPI_COMM_WORLD,stdout);
  }
  {
    char monitorname[64];
    if (OptionsGetString(0,"-log_all",monitorname,64) || 
        OptionsGetString(0,"-log",monitorname,64)) {
      if (monitorname[0]) PLogDump(monitorname); 
      else PLogDump(0);
    }
  }
#endif
  if (!OptionsHasName(0,"-no_signal_handler")) {
    PetscPopSignalHandler();
  }
  OptionsHasName(0,"-trdump");
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  if (OptionsHasName(0,"-optionstable")) {
    if (!mytid) OptionsPrint(stdout);
  }
  if (OptionsHasName(0,"-optionsleft")) {
    if (!mytid) {
      int nopt = OptionsAllUsed();
      if (nopt == 0) 
        fprintf(stdout,"There are no unused options.\n");
      else if (nopt == 1) 
        fprintf(stdout,"There is one unused database option. It is:\n");
      else
        fprintf(stdout,"There are %d unused database options. They are:\n",nopt);
      for ( i=0; i<options->N; i++ ) {
        if (!options->used[i]) {
          fprintf(stdout,"Option left: name:%s value: %s\n",options->names[i],
                                                           options->values[i]);
        }
      }
    } 
  }
  if (OptionsHasName(0,"-trdump")) {
    OptionsDestroy_Private();
    NRDestroyAll();
    MPIU_Seq_begin(MPI_COMM_WORLD,1);
      ierr = TrDump(stderr); CHKERRQ(ierr);
    MPIU_Seq_end(MPI_COMM_WORLD,1);
  }
  else {
    OptionsDestroy_Private();
    NRDestroyAll(); 
  }
  if (PetscBeganMPI) {
      ierr = MPI_Finalize(); CHKERRQ(ierr);
  }
  return 0;
}
 
/* 
   This is ugly and probably belongs somewhere else, but I want to 
  be able to put a true MPI abort error handler with command line args.

    This is so MPI errors in the debugger will leave all the stack 
  frames. The default abort cleans up and exits.
*/

void abort_function(MPI_Comm *comm,int *flag) 
{
  fprintf(stderr,"MPI error %d\n",*flag);
  abort();
}

#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
  extern int malloc_debug(int);
};
#elif defined(PARCH_sun4)
  extern int malloc_debug(int);
#endif

extern int PLogAllowInfo(PetscTruth);
extern int PetscSetUseTrMalloc_Private();

int OptionsCheckInitial_Private()
{
  char     string[64];
  MPI_Comm comm = MPI_COMM_WORLD;

#if defined(PETSC_BOPT_g)
  if (!OptionsHasName(0,"-notrmalloc")) {
    PetscSetUseTrMalloc_Private();
  }
#else
  if (OptionsHasName(0,"-trdump") || OptionsHasName(0,"-trmalloc")) {
    PetscSetUseTrMalloc_Private();
  }
#endif
#if defined(PARCH_sun4) && defined(PETSC_BOPT_g)
  if (OptionsHasName(0,"-malloc_debug")) {
    malloc_debug(2);
  }
#endif
  if (OptionsHasName(0,"-v") || OptionsHasName(0,"-version") ||
      OptionsHasName(0,"-help")) {
    MPIU_printf(comm,"--------------------------------------------\
------------------------------\n");
    MPIU_printf(comm,"\t   %s\n",PETSC_VERSION_NUMBER);
    MPIU_printf(comm,"Bill Gropp,Lois Curfman McInnes,Barry Smith.\
 Bugs: petsc-maint@mcs.anl.gov\n");
    MPIU_printf(comm,"See petsc/COPYRIGHT for copyright information,\
 petsc/Changes for recent updates.\n");
    MPIU_printf(comm,"--------------------------------------------\
---------------------------\n");
  }
  if (OptionsHasName(0,"-fp_trap")) {
    PetscSetFPTrap(FP_TRAP_ALWAYS);
  }
  if (OptionsHasName(0,"-on_error_abort")) {
    PetscPushErrorHandler(PetscAbortErrorHandler,0);
  }
  if (OptionsGetString(0,"-on_error_attach_debugger",string,64)) {
    char *debugger = 0, *display = 0;
    int  xterm     = 1, sfree = 0;
    if (strstr(string,"noxterm")) xterm = 0;
#if defined(PARCH_hpux)
    if (strstr(string,"xdb"))     debugger = "xdb";
#else
    if (strstr(string,"dbx"))     debugger = "dbx";
#endif
    if (strstr(string,"xxgdb"))   debugger = "xxgdb";
    if (OptionsGetString(0,"-display",string,64)){
      display = string;
    }
    if (!display) {
      display = (char *) malloc( 128*sizeof(char)); CHKPTRQ(display);
      MPIU_Set_display(comm,display,128); sfree = 1;
    } 
    PetscSetDebugger(debugger,xterm,display);
    if (sfree) free(display);
    PetscPushErrorHandler(PetscAttachDebuggerErrorHandler,0);
  }
  if (OptionsGetString(0,"-start_in_debugger",string,64)) {
    char *debugger = 0, *display = 0;
    int  xterm     = 1, sfree = 0,numtid = 1;
    MPI_Errhandler abort_handler;
    /*
       we have to make sure that all processors have opened 
       connections to all other processors, otherwise once the 
       debugger has stated it is likely to receive a SIGUSR1
       and kill the program. 
    */
    MPI_Comm_size(MPI_COMM_WORLD,&numtid);
    if (numtid > 2) {
      int        i,dummy;
      MPI_Status status;
      for ( i=0; i<numtid; i++) {
        MPI_Send(&dummy,1,MPI_INT,i,109,MPI_COMM_WORLD);
      }
      for ( i=0; i<numtid; i++) {
        MPI_Recv(&dummy,1,MPI_INT,i,109,MPI_COMM_WORLD,&status);
      }
    }
    if (strstr(string,"noxterm")) xterm = 0;
#if defined(PARCH_hpux)
    if (strstr(string,"xdb"))     debugger = "xdb";
#else
    if (strstr(string,"dbx"))     debugger = "dbx";
#endif
    if (strstr(string,"xxgdb"))   debugger = "xxgdb";
    if (OptionsGetString(0,"-display",string,64)){
      display = string;
    }
    if (!display) {
      display = (char *) malloc( 128*sizeof(char) ); CHKPTRQ(display);
      MPIU_Set_display(comm,display,128);
      sfree = 1;
    } 
    PetscSetDebugger(debugger,xterm,display);
    if (sfree) free(display);
    PetscPushErrorHandler(PetscAbortErrorHandler,0);
    PetscAttachDebugger();
    MPI_Errhandler_create((MPI_Handler_function*)abort_function,
                                                       &abort_handler);
    MPI_Errhandler_set(comm,abort_handler);
  }
  if (!OptionsHasName(0,"-no_signal_handler")) {
    PetscPushSignalHandler(PetscDefaultSignalHandler,(void*)0);
  }
#if defined(PETSC_LOG)
  if (OptionsHasName(0,"-info")) {
    PLogAllowInfo(PETSC_TRUE);
  }
  if (OptionsHasName(0,"-log_all")) {
    PLogAllBegin();
  }
  else if (OptionsHasName(0,"-log") || OptionsHasName(0,"-log_summary")) {
    PLogBegin();
  }
#endif
  if (OptionsHasName(0,"-help")) {
    MPIU_printf(comm,"Options for all PETSc programs:\n");
    MPIU_printf(comm," -on_error_abort: cause an abort when an error is");
    MPIU_printf(comm," detected. Useful \n       only when run in the debugger\n");
    MPIU_printf(comm," -on_error_attach_debugger [dbx,xxgdb,noxterm]"); 
    MPIU_printf(comm," [-display display]:\n");
    MPIU_printf(comm,"       start the debugger (gdb by default) in new xterm\n");
    MPIU_printf(comm,"       unless noxterm is given\n");
    MPIU_printf(comm," -start_in_debugger [dbx,xxgdb,noxterm]");
    MPIU_printf(comm," [-display display]:\n");
    MPIU_printf(comm,"       start all processes in the debugger\n");
    MPIU_printf(comm," -no_signal_handler: do not trap error signals\n");
    MPIU_printf(comm," -fp_trap: stop on floating point exceptions\n");
    MPIU_printf(comm,"           note on IBM RS6000 this slows run greatly\n");
    MPIU_printf(comm," -trdump: dump list of unfreed memory at conclusion\n");
    MPIU_printf(comm," -trmalloc: use our error checking malloc\n");
    MPIU_printf(comm," -notrmalloc: don't use error checking malloc\n");
    MPIU_printf(comm," -optionstable: dump list of options inputted\n");
    MPIU_printf(comm," -optionsleft: dump list of unused options\n");
    MPIU_printf(comm," -log[_all _summary]: logging objects and events\n");
    MPIU_printf(comm," -v: prints PETSc version number and release date\n");
    MPIU_printf(comm,"-----------------------------------------------\n");
  }
  return 0;
}

char *OptionsGetProgramName()
{
  if (!options) return (char *) 0;
  if (!options->namegiven) return (char *) 0;
  return options->programname;
}

/*
   OptionsCreate_Private - Creates a database of options.

   Input Parameters:
.  argc - count of number of command line arguments
.  args - the command line arguments
.  file - optional filename, defaults to ~username/.petscrc
.  env  - environmental variable, defaults to PETSC_OPTIONS

   Note:
   Since OptionsCreate_Private() is automatically called by PetscInitialize(),
   the user does not typically need to call this routine. OptionsCreate_Private()
   can be called several times, adding additional entries into the database.

.keywords: options, database, create

.seealso: OptionsDestroy_Private(), OptionsPrint()
*/
int OptionsCreate_Private(int *argc,char ***args,char* file,char* env)
{
  int  ierr;
  char pfile[128];
  if (!options) {
    options = (OptionsTable*) malloc(sizeof(OptionsTable)); CHKPTRQ(options);
    PETSCMEMSET(options->used,0,MAXOPTIONS*sizeof(int));
  }
  if (!env) env = "PETSC_OPTIONS";
  if (!file) {
    if ((ierr = SYGetHomeDirectory(120,pfile))) return ierr;
    strcat(pfile,"/.petscrc");
    file = pfile;
  }

  if (*argc) {options->namegiven = 1; strncpy(options->programname,**args,64);}
  else {options->namegiven = 0;}
  options->N = 0;
  options->Naliases = 0;
  options->argc = *argc;
  options->args = *args;

  /* insert file options */
  {
    char string[128],*first,*second,*third,*final;
    int   len;
    FILE *fd = fopen(file,"r"); 
    if (fd) {
      while (fgets(string,128,fd)) {
        first = strtok(string," ");
        second = strtok(0," ");
        if (first && first[0] == '-') {
          if (second) {final = second;} else {final = first;}
          len = strlen(final);
          while (len > 0 && (final[len-1] == ' ' || final[len-1] == '\n')) {
            len--; final[len] = 0;
          }
          OptionsSetValue(first,second);
        }
        else if (first && !strcmp(first,"alias")) {
          third = strtok(0," ");
          if (!third)
            SETERRQ(1,"OptionsCreate_Private: Error in options file on alias");
          len = strlen(third); third[len-1] = 0;
          ierr = OptionsSetAlias_Private(second,third); CHKERRQ(ierr);
        }
      }
      fclose(fd);
    }
  }
  /* insert environmental options */
  {
    char *eoptions = (char *) getenv(env);
    char *second, *first = strtok(eoptions," ");
    while (first) {
      if (first[0] != '-') {first = strtok(0," "); continue;}
      second = strtok(0," ");
      if ((!second) || ((second[0] == '-') && (second[1] > '9'))) {
        OptionsSetValue(first,(char *)0);
        first = second;
      }
      else {
        OptionsSetValue(first,second);
        first = strtok(0," ");
      }
    }
  }

  /* insert command line options */
  if (*argc) {
    int   left = *argc - 1;
    char  **eargs = *args + 1;
    while (left) {
      if (eargs[0][0] != '-') {
        eargs++; left--;
      }
      else if ((left < 2) || ((eargs[1][0] == '-') && 
          ((eargs[1][1] > '9') || (eargs[1][1] < '0')))) {
        OptionsSetValue(eargs[0],(char *)0);
        eargs++; left--;
      }
      else {
        OptionsSetValue(eargs[0],eargs[1]);
        eargs += 2; left -= 2;
      }
    }
  }
  return 0;
}

/*@C
   OptionsPrint - Prints the options that have been loaded. This is
   mainly useful for debugging purposes.

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
  if (!fd) fd = stdout;
  if (!options) OptionsCreate_Private(0,0,0,0);
  for ( i=0; i<options->N; i++ ) {
    fprintf(fd,"OptionTable: %s %s\n",options->names[i],options->values[i]);
  }
  return 0;
}

/*
    OptionsDestroy_Private - Destroys the option database. 

    Note:
    Since OptionsDestroy_Private() is called by PetscFinalize(), the user 
    typically does not need to call this routine.

.keywords: options, database, destroy

.seealso: OptionsCreate_Private()
*/
static int OptionsDestroy_Private()
{
  int i;
  if (!options) return 0;
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
  return 0;
}
/*@
   OptionsSetValue - Sets an option name-value pair in the options 
   database, overriding whatever is already present.

   Input Parameters:
.  name - name of option
.  value - the option value (not used for all options)

   Note:
   Only some options have values associated with them, such as
   -ksp_rtol tol.  Other options stand alone, such as -ksp_monitor.

.keywords: options, database, set, value

.seealso: OptionsCreate_Private()
@*/
int OptionsSetValue(char *name,char *value)
{
  int len, N, n, i;
  char **names;
  if (!options) OptionsCreate_Private(0,0,0,0);

  /* this is so that -h and -help are equivalent (p4 don't like -help)*/
  if (!strcmp(name,"-h")) name = "-help";

  /* first check against aliases */
  N = options->Naliases; 
  for ( i=0; i<N; i++ ) {
    if (!strcmp(options->aliases1[i],name)) {
      name = options->aliases2[i];
      break;
    }
  }

  N = options->N; n = N;
  names = options->names; 
 
  for ( i=0; i<N; i++ ) {
    if (strcmp(names[i],name) == 0) {
      if (options->values[i]) free(options->values[i]);
      len = strlen(value);
      if (len) {
        options->values[i] = (char *) malloc( len ); 
        CHKPTRQ(options->values[i]);
        strcpy(options->values[i],value);
      }
      else { options->values[i] = 0;}
      return 0;
    }
    else if (strcmp(names[i],name) > 0) {
      n = i;
      break;
    }
  }
  if (N >= MAXOPTIONS) {
    fprintf(stderr,"No more room in option table, limit %d\n",MAXOPTIONS);
    fprintf(stderr,"recompile options/src/options.c with larger ");
    fprintf(stderr,"value for MAXOPTIONS\n");
    return 0;
  }
  /* shift remaining values down 1 */
  for ( i=N; i>n; i-- ) {
    names[i] = names[i-1];
    options->values[i] = options->values[i-1];
  }
  /* insert new name and value */
  len = (strlen(name)+1)*sizeof(char);
  names[n] = (char *) malloc( len ); CHKPTRQ(names[n]);
  strcpy(names[n],name);
  if (value) {
    len = (strlen(value)+1)*sizeof(char);
    options->values[n] = (char *) malloc( len ); CHKPTRQ(options->values[n]);
    strcpy(options->values[n],value);
  }
  else {options->values[n] = 0;}
  options->N++;
  return 0;
}

int OptionsSetAlias_Private(char *newname,char *oldname)
{
  int len,n = options->Naliases;

  if (n >= MAXALIASES) {
    SETERRQ(1,"OptionsSetAlias_Private: Too many option aliases defined");
  }
  len = (strlen(newname)+1)*sizeof(char);
  options->aliases1[n] = (char *) malloc( len ); CHKPTRQ(options->aliases1[n]);
  strcpy(options->aliases1[n],newname);
  len = (strlen(oldname)+1)*sizeof(char);
  options->aliases2[n] = (char *) malloc( len );CHKPTRQ(options->aliases2[n]);
  strcpy(options->aliases2[n],oldname);
  options->Naliases++;
  return 0;
}
static int OptionsFindPair_Private( char *pre,char *name,char **value)
{
  int  i, N;
  char **names,tmp[128];
  if (!options) OptionsCreate_Private(0,0,0,0);
  N = options->N;
  names = options->names;

  /* append prefix to name */
  if (pre) { strcpy(tmp,pre); strcat(tmp,name+1);}
  else strcpy(tmp,name);

  /* slow search */
  for ( i=0; i<N; i++ ) {
    if (!strcmp(names[i],tmp)) {
       *value = options->values[i];
       options->used[i]++;
       return 1;
     }
  }
  return 0;
}

/*@
   OptionsHasName - Determines if a certain option is given in the database.

   Input Parameters:
.  name - the option one is seeking 
.  pre - string to prepend to the name

   Returns:
$   1 if the option is found;
$   0 if the option is not found;
$  -1 if an error is detected.

.keywords: options, database, has, name

.seealso: OptionsGetInt(), OptionsGetDouble(),
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsHasName(char* pre,char *name)
{
  char *value;
  if (!OptionsFindPair_Private(pre,name,&value)) {return 0;}
  return 1;
}

/*@
   OptionsGetInt - Gets the integer value for a particular option in the 
                    database.

   Input Parameters:
.  name - the option one is seeking
.  pre - the string to preappend to the name

   Output Parameter:
.  ivalue - the integer value to return

.keywords: options, database, get, int

.seealso: OptionsGetDouble(), OptionsHasName(), OptionsGetString(),
          OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetInt(char*pre,char *name,int *ivalue)
{
  char *value;
  if (!OptionsFindPair_Private(pre,name,&value)) {return 0;}
  if (!value) SETERRQ(1,"OptionsGetInt: Missing value for option");
  *ivalue = atoi(value);
  return 1; 
} 

/*@
   OptionsGetDouble - Gets the double precision value for a particular 
   option in the database.

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name

   Output Parameter:
.  dvalue - the double value to return

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetDouble(char* pre,char *name,double *dvalue)
{
  char *value;
  if (!OptionsFindPair_Private(pre,name,&value)) {return 0;}
  *dvalue = atof(value);
  return 1; 
} 
/*@
   OptionsGetDoubleArray - Gets an array of double precision values for a 
   particular option in the database.  The values must be separated with 
   commas with no intervening spaces.

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name
.  nmax - maximum number of values to retrieve

   Output Parameters:
.  dvalue - the double value to return
.  nmax - actual number of values retreived


.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray()
@*/
int OptionsGetDoubleArray(char* pre,char *name,
                          double *dvalue, int *nmax)
{
  char *value;
  int  n = 0;
  if (!OptionsFindPair_Private(pre,name,&value)) {*nmax = 0; return 0;}
  value = strtok(value,",");
  while (n < *nmax) {
    if (!value) break;
    *dvalue++ = atof(value);
    value = strtok(0,",");
    n++;
  }
  *nmax = n;
  return 1; 
} 

/*@
   OptionsGetIntArray - Gets an array of integer values for a particular 
   option in the database.  The values must be separated with commas with 
   no intervening spaces. 

   Input Parameters:
.  name - the option one is seeking
.  pre - string to prepend to each name
.  nmax - maximum number of values to retrieve

   Output Parameter:
.  dvalue - the integer values to return
.  nmax - actual number of values retreived


.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetDoubleArray()
@*/
int OptionsGetIntArray(char* pre,char *name,int *dvalue,int *nmax)
{
  char *value;
  int  n = 0;
  if (!OptionsFindPair_Private(pre,name,&value)) {*nmax = 0; return 0;}
  value = strtok(value,",");
  while (n < *nmax) {
    if (!value) break;
    *dvalue++ = atoi(value);
    value = strtok(0,",");
    n++;
  }
  *nmax = n;
  return 1; 
} 


/*@
   OptionsGetString - Gets the string value for a particular option in
   the database.

   Input Parameters:
.  name - the option one is seeking
.  len - maximum string length
.  pre - string to prepend to name

   Output Parameter:
.  string - location to copy string


.keywords: options, database, get, string

.seealso: OptionsGetInt(), OptionsGetDouble(),  
           OptionsHasName(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetString(char *pre,char *name,char *string,int len)
{
  char *value;
  if (!OptionsFindPair_Private(pre,name,&value)) {return 0;}
  if (value) strncpy(string,value,len);
  else PETSCMEMSET(string,0,len);
  return 1; 
}

/*@
   OptionsAllUsed - Returns a count of the number of options in the 
   database that have never been selected.

   Options Database Key:
$  -optionsleft : checked within PetscFinalize()

.keywords: options, database, missed, unused, all, used

.seealso: OptionsPrint()
@*/
int OptionsAllUsed()
{
  int  i,n = 0;
  for ( i=0; i<options->N; i++ ) {
    if (!options->used[i]) { n++; }
  }
  return n;
}
