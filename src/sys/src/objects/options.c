#ifndef lint
static char vcid[] = "$Id: options.c,v 1.4 1995/05/15 20:27:21 curfman Exp curfman $";
#endif
/*
    Routines to simplify the use of command line, file options etc.
*/
#include <stdio.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "ptscimpl.h"
#include "sys.h"
#include "sysio.h"
#include "options.h"
#include "sys/nreg.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "petscfix.h"

/* 
    For simplicity, we begin with a static size database
*/
#define MAXOPTIONS 256

typedef struct {
  int  N,argc;
  char **args,*names[MAXOPTIONS],*values[MAXOPTIONS];
  int  used[MAXOPTIONS];
  int  namegiven;
  char programname[64];
} OptionsTable;

static OptionsTable *options = 0;
static int PetscBeganMPI = 0;

extern int ViewerInitialize_Private(),
           OptionsCreate_Private(int*,char***,char*,char*),
           OptionsDestroy_Private();
/*@
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

.keywords: initialize, options, database, startup

.seealso: PetscFinalize()
@*/
int PetscInitialize(int *argc,char ***args,char *file,char *env)
{
  int ierr,flag;

  MPI_Initialized(&flag);
  if (!flag) {
    ierr = MPI_Init(argc,args); CHKERR(ierr);
    PetscBeganMPI = 1;
  }
  ViewerInitialize_Private();
  return OptionsCreate_Private(argc,args,file,env);
}

/*@ 
   PetscFinalize - Checks for options to be called at the conclusion
   of the program and calls MPI_Finalize().

   Options Database Keys:
$  -optionstable : Calls OptionsPrint()
$  -optionused : Calls OptionsAllUsed()
$  -optionsleft : Prints unused options that remain in 
$     the database
$  -no_signal_handler : Turns off the signal handler
$  -trdump : Calls Trdump()
$  -logall : Prints log information (for code compiled
$      with PETSC_LOG)
$  -log : Prints log information (for code compiled 
$      with PETSC_LOG)

.keywords: finalize, exit, end

.seealso: PetscInitialize(), OptionsPrint(), Trdump()
@*/
int PetscFinalize()
{
  int  ierr,i,mytid = 0,MPI_Used;

  MPI_Initialized(&MPI_Used);
  if (!OptionsHasName(0,0,"-no_signal_handler")) {
    PetscPopSignalHandler();
  }
  OptionsHasName(0,0,"-trdump");
  if (MPI_Used) {MPI_Comm_rank(MPI_COMM_WORLD,&mytid);}
  if (OptionsHasName(0,0,"-optionstable")) {
    if (!mytid) OptionsPrint(stderr);
  }
  if (OptionsHasName(0,0,"-optionsused")) {
    if (!mytid) {
      int nopt = OptionsAllUsed();
      if (nopt == 1) 
        fprintf(stderr,"There is %d unused database option.\n",nopt);
      else
        fprintf(stderr,"There are %d unused database options.\n",nopt);
    }
  }
  if (OptionsHasName(0,0,"-optionsleft")) {
    if (!mytid) {
      for ( i=0; i<options->N; i++ ) {
        if (!options->used[i]) {
          fprintf(stderr,"Option left: name:%s value: %s\n",options->names[i],
                                                           options->values[i]);
        }
      }
    } 
  }
#if defined(PETSC_LOG)
  {
    char monitorname[64];
    if (OptionsGetString(0,0,"-logall",monitorname,64) || 
        OptionsGetString(0,0,"-log",monitorname,64)) {
      if (monitorname[0]) PLogDump(monitorname); 
      else PLogDump(0);
    }
  }
#endif
#if defined(PETSC_MALLOC)
  if (OptionsHasName(0,0,"-trdump")) {
    OptionsDestroy_Private();
    NRDestroyAll();
    if (MPI_Used) {MPE_Seq_begin(MPI_COMM_WORLD,1);}
      ierr = Trdump(stderr); CHKERR(ierr);
    if (MPI_Used) {MPE_Seq_end(MPI_COMM_WORLD,1);}
  }
  else {
    OptionsDestroy_Private();
    NRDestroyAll(); 
  }
#else
  OptionsDestroy_Private();
  NRDestroyAll();
#endif
  if (MPI_Used) { 
    if (PetscBeganMPI) {
      ierr = MPI_Finalize(); CHKERR(ierr);
    }
  }
  return 0;
}
 
/* 
   This is ugly and probably belongs somewhere else, but I want to 
  be able to put a true MPI abort error handler with commandline args.

    This is so MPI errors in the debugger will leave all the stack 
  frames. The default abort cleans up and exits.
*/

void abort_function(MPI_Comm *comm,int *flag) 
{
  fprintf(stderr,"MPI error %d\n",*flag);
  abort();
}

#if defined(PARCH_sun4) && defined(__cplusplus) && defined(PETSC_MALLOC)
extern "C" {
  extern int malloc_debug(int);
};
#endif

extern int PLogAllowInfo(PetscTruth);
/* 
   This is called by OptionsCreate_Private(). It checks for any initialization
  options the user may like. At the moment is only support for type
  of run-time error handling.
*/
int OptionsCheckInitial()
{
  char     string[64];
  MPI_Comm comm = MPI_COMM_WORLD;

#if defined(PARCH_sun4) && defined(PETSC_DEBUG) && defined(PETSC_MALLOC)
  if (OptionsHasName(0,0,"-malloc_debug")) {
    malloc_debug(2);
  }
#endif
  if (OptionsHasName(0,0,"-v") || OptionsHasName(0,0,"-version") ||
      OptionsHasName(0,0,"-help")) {
    MPE_printf(comm,"--------------------------------------------\
------------------------------\n");
    MPE_printf(comm,"\t   %s\n",PETSC_VERSION_NUMBER);
    MPE_printf(comm,"Bill Gropp,Lois Curfman McInnes,Barry Smith.\
 Bugs: petsc-maint@mcs.anl.gov\n");
    MPE_printf(comm,"See petsc/COPYRIGHT for copyright information,\
 petsc/Changes for recent updates.\n");
    MPE_printf(comm,"--------------------------------------------\
---------------------------\n");
  }
  if (OptionsHasName(0,0,"-fp_trap")) {
    PetscSetFPTrap(1);
  }
  if (OptionsHasName(0,0,"-on_error_abort")) {
    PetscPushErrorHandler(PetscAbortErrorHandler,0);
  }
  if (OptionsGetString(0,0,"-on_error_attach_debugger",string,64)) {
    char *debugger = 0, *display = 0;
    int  xterm     = 1, sfree = 0;
    if (strstr(string,"noxterm")) xterm = 0;
    if (strstr(string,"dbx"))     debugger = "dbx";
    if (strstr(string,"xxgdb"))   debugger = "xxgdb";
    if (OptionsGetString(0,0,"-display",string,64)){
      display = string;
    }
    if (!display) {MPE_Set_display(comm,&display); sfree = 1;}; 
    PetscSetDebugger(debugger,xterm,display);
    if (sfree) FREE(display);
    PetscPushErrorHandler(PetscAttachDebuggerErrorHandler,0);
  }
  if (OptionsGetString(0,0,"-start_in_debugger",string,64)) {
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
    if (strstr(string,"dbx"))     debugger = "dbx";
    if (strstr(string,"xxgdb"))   debugger = "xxgdb";
    if (OptionsGetString(0,0,"-display",string,64)){
      display = string;
    }
    if (!display) {MPE_Set_display(comm,&display);sfree = 1;} 
    PetscSetDebugger(debugger,xterm,display);
    if (sfree) FREE(display);
    PetscPushErrorHandler(PetscAbortErrorHandler,0);
    PetscAttachDebugger();
    MPI_Errhandler_create((MPI_Handler_function*)abort_function,
                                                       &abort_handler);
    MPI_Errhandler_set(comm,abort_handler);
  }
  if (!OptionsHasName(0,0,"-no_signal_handler")) {
    PetscPushSignalHandler(PetscDefaultSignalHandler,(void*)0);
  }
#if defined(PETSC_LOG)
  if (OptionsHasName(0,0,"-log")) {
    PLogBegin();
  }
  if (OptionsHasName(0,0,"-logall")) {
    PLogAllBegin();
  }
  if (OptionsHasName(0,0,"-info")) {
    PLogAllowInfo(PETSC_TRUE);
  }
#endif
  if (OptionsHasName(0,0,"-help")) {
    fprintf(stderr,"Options for all PETSc programs:\n");
    fprintf(stderr," -on_error_abort: cause an abort when an error is");
    fprintf(stderr," detected. Useful \n       only when run in the debugger\n");
    fprintf(stderr," -on_error_attach_debugger [dbx,xxgdb,noxterm]"); 
    fprintf(stderr," [-display display]:\n");
    fprintf(stderr,"       start the debugger (gdb by default) in new xterm\n");
    fprintf(stderr,"       unless noxterm is given\n");
    fprintf(stderr," -start_in_debugger [dbx,xxgdb,noxterm]");
    fprintf(stderr," [-display display]:\n");
    fprintf(stderr,"       start all processes in the debugger\n");
    fprintf(stderr," -no_signal_handler: do not trap error signals\n");
    fprintf(stderr," -fp_trap: stop on floating point exceptions\n");
    fprintf(stderr," -trdump: dump list of unfreed memory at conclusion\n");
    fprintf(stderr," -optionstable: dump list of options inputted\n");
    fprintf(stderr," -optionsleft: dump list of unused options\n");
    fprintf(stderr," -optionsused: print number of unused options\n");
    fprintf(stderr," -monitor: logging objects and events\n");
    fprintf(stderr," -v: prints PETSc version number and release date\n");
    fprintf(stderr,"-----------------------------------------------\n");
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
    options = (OptionsTable*) MALLOC(sizeof(OptionsTable)); CHKPTR(options);
    MEMSET(options->used,0,MAXOPTIONS*sizeof(int));
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
  options->argc = *argc;
  options->args = *args;

  /* insert file options */
  {
    char string[128],*first,*second;
    int   len;
    FILE *fd = fopen(file,"r"); 
    if (fd) {
      while (fgets(string,128,fd)) {
        first = strtok(string," ");
        second = strtok(0," ");
        if (second) {len = strlen(second); second[len-1] = 0;}
        if (first && first[0] == '-') OptionsSetValue(first,second);
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
  OptionsCheckInitial();
  return 0;
}

/*@
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
int OptionsDestroy_Private()
{
  int i;
  if (!options) return 0;
  for ( i=0; i<options->N; i++ ) {
    if (options->names[i]) FREE(options->names[i]);
    if (options->values[i]) FREE(options->values[i]);
  }
  FREE(options);
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

  N = options->N; n = N;
  names = options->names; 
 
  for ( i=0; i<N; i++ ) {
    if (strcmp(names[i],name) == 0) {
      if (options->values[i]) FREE(options->values[i]);
      len = strlen(value);
      if (len) {
        options->values[i] = (char *) MALLOC( len ); 
        CHKPTR(options->values[i]);
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
  names[n] = (char *) MALLOC( len ); CHKPTR(names[n]);
  strcpy(names[n],name);
  if (value) {
    len = (strlen(value)+1)*sizeof(char);
    options->values[n] = (char *) MALLOC( len ); CHKPTR(options->values[n]);
    strcpy(options->values[n],value);
  }
  else {options->values[n] = 0;}
  options->N++;
  return 0;
}

static int OptionsFindPair(int keep, char *pre,char *name,char **value)
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
.  keep - flag to detemine if option is kept in database
.  name - the option one is seeking 
.  pre - string to prepend to the name

   Returns:
$   1 if the option is found;
$   0 if the option is not found;
$  -1 if an error is detected.

   Notes:
   If keep=0, the entry is removed from the database, and a further request
   for it will return a 0.  If keep is nonzero, then the argument will
   remain in the database.  Keep is not yet implemented.

.keywords: options, database, has, name

.seealso: OptionsGetInt(), OptionsGetDouble(), OptionsGetScalar(),
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsHasName(int keep,char* pre,char *name)
{
  char *value;
  if (!OptionsFindPair(keep,pre,name,&value)) {return 0;}
  return 1;
}

/*@
   OptionsGetInt - Gets the integer value for a particular option in the 
                    database.

   Input Parameters:
.  keep - flag to detemine if option is kept in database
.  name - the option one is seeking
.  pre - the string to preappend to the name

   Output Parameter:
.  ivalue - the integer value to return

   Note:
   Keep is not yet implemented.

.keywords: options, database, get, int

.seealso: OptionsGetDouble(), OptionsHasName(), OptionsGetString(),
           OptionsGetScalar(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetInt(int keep,char*pre,char *name,int *ivalue)
{
  char *value;
  if (!OptionsFindPair(keep,pre,name,&value)) {return 0;}
  if (!value) SETERR(1,"Missing value for option");
  *ivalue = atoi(value);
  return 1; 
} 

/*@
   OptionsGetDouble - Gets the double precision value for a particular 
   option in the database.

   Input Parameters:
.  keep - flag to detemine if option is kept in database
.  name - the option one is seeking
.  pre - string to prepend to each name

   Output Parameter:
.  dvalue - the double value to return

   Note:
   The keep flag is not yet implemented.

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsGetScalar(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetDouble(int keep,char* pre,char *name,double *dvalue)
{
  char *value;
  if (!OptionsFindPair(keep,pre,name,&value)) {return 0;}
  *dvalue = atof(value);
  return 1; 
} 
/*@
   OptionsGetDoubleArray - Gets an array of double precision values for a 
   particular option in the database.  The values must be separated with 
   commas with no intervening spaces.

   Input Parameters:
.  keep - flag to detemine if option is kept in database
.  name - the option one is seeking
.  pre - string to prepend to each name
.  nmax - maximum number of values to retrieve

   Output Parameters:
.  dvalue - the double value to return
.  nmax - actual number of values retreived

   Note:
   The keep flag is not yet implemented.

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsGetScalar(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray()
@*/
int OptionsGetDoubleArray(int keep,char* pre,char *name,
                          double *dvalue, int *nmax)
{
  char *value;
  int  n = 0;
  if (!OptionsFindPair(keep,pre,name,&value)) {*nmax = 0; return 0;}
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
.  keep - flag to detemine if option is kept in database
.  name - the option one is seeking
.  pre - string to prepend to each name
.  nmax - maximum number of values to retrieve

   Output Parameter:
.  dvalue - the integer values to return
.  nmax - actual number of values retreived

   Note:
   The keep flag is not yet implemented.

.keywords: options, database, get, double

.seealso: OptionsGetInt(), OptionsGetScalar(), OptionsHasName(), 
           OptionsGetString(), OptionsGetDoubleArray()
@*/
int OptionsGetIntArray(int keep,char* pre,char *name,int *dvalue,int *nmax)
{
  char *value;
  int  n = 0;
  if (!OptionsFindPair(keep,pre,name,&value)) {*nmax = 0; return 0;}
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
   OptionsGetScalar - Gets the double or complex value for a
   particular option in the database.

   Input Parameters:
.  keep - flag to detemine if option is kept in database
.  name - the option one is seeking
.  pre - string to prepend to the name (usually 0)

   Output Parameter:
.  dvalue - the value to return

   Note:
   The keep flag is not yet implemented.

.keywords: options, database, get, scalar

.seealso: OptionsGetInt(), OptionsGetDouble(), OptionsGetString(), 
           OptionsHasName(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetScalar(int keep,char* pre,char *name,Scalar *dvalue)
{
  char *value;
  if (!OptionsFindPair(keep,pre,name,&value)) {return 0;}
  if (!value) SETERR(1,"Missing value for option");
  *dvalue = atof(value);
  return 1; 
} 

/*@
   OptionsGetString - Gets the string value for a particular option in
   the database.

   Input Parameters:
.  keep - flag to detemine if option is kept in database
.  name - the option one is seeking
.  len - maximum string length
.  pre - string to prepend to name

   Output Parameter:
.  string - location to copy string

   Note:
   The keep flag is not yet implemented.

.keywords: options, database, get, string

.seealso: OptionsGetInt(), OptionsGetDouble(), OptionsGetScalar(), 
           OptionsHasName(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetString(int keep,char *pre,char *name,char *string,int len)
{
  char *value;
  if (!OptionsFindPair(keep,pre,name,&value)) {return 0;}
  if (value) strncpy(string,value,len);
  else MEMSET(string,0,len);
  return 1; 
}

/*@
   OptionsAllUsed - Returns a count of the number of options in the 
   database that have never been selected.

   Options Database Key:
$  -optionsused : checked within PetscFinalize()

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
