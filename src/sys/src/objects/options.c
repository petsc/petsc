#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: options.c,v 1.189 1998/04/30 17:35:23 bsmith Exp bsmith $";
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

int        OptionsCheckInitial_Private(),
           OptionsCreate_Private(int*,char***,char*),
           OptionsSetAlias_Private(char *,char *);


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

#undef __FUNC__  
#define __FUNC__ "PetscSetProgramName"
int PetscSetProgramName(char *name)
{ 
  PetscFunctionBegin;
  options->namegiven = 1;
  PetscStrncpy(options->programname,name,256);
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

       } else if (!PetscStrcmp(eargs[0],"-np")) {
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

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  FILE fd - location to print options (usually stdout or stderr)

   Options Database Key:
.  -optionstable - Activates OptionsPrint() within PetscFinalize()

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
#define __FUNC__ "OptionsDestroy"
/*
    OptionsDestroy - Destroys the option database. 

    Note:
    Since OptionsDestroy() is called by PetscFinalize(), the user 
    typically does not need to call this routine.

.keywords: options, database, destroy

.seealso: OptionsCreate_Private()
*/
int OptionsDestroy(void)
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

   Not collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  name - name of option, this SHOULD have the - prepended
-  value - the option value (not used for all options)

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

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameter:
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

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  name - the option one is seeking 
-  mess - error message 

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

   Not Collective

   Input Parameters:
+  name - the option one is seeking 
-  pre - string to prepend to the name or PETSC_NULL

   Output Parameters:
.  flg - 1 if found else 0.

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
   OptionsGetInt - Gets the integer value for a particular option in the database.

   Not Collective

   Input Parameters:
+  name - the option one is seeking
-  pre - the string to prepend to the name or PETSC_NULL

   Output Parameter:
+  ivalue - the integer value to return
-  flg - 1 if found, else 0

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

   Not Collective

   Input Parameters:
+  name - the option one is seeking
-  pre - string to prepend to each name or PETSC_NULL

   Output Parameter:
+  dvalue - the double value to return
-  flg - 1 if found, 0 if not found

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

   Not Collective

   Input Parameters:
+  name - the option one is seeking
-  pre - string to prepend to each name or PETSC_NULL

   Output Parameter:
+  dvalue - the double value to return
-  flg - 1 if found, else 0

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

   Not Collective

   Input Parameters:
+  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL
-  nmax - maximum number of values to retrieve

   Output Parameters:
+  dvalue - the double value to return
.  nmax - actual number of values retreived
-  flg - 1 if found, else 0

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

   Not Collective

   Input Parameters:
+  name - the option one is seeking
.  pre - string to prepend to each name or PETSC_NULL
-  nmax - maximum number of values to retrieve

   Output Parameter:
+  dvalue - the integer values to return
.  nmax - actual number of values retreived
-  flg - 1 if found, else 0

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

   Not Collective

   Input Parameters:
+  name - the option one is seeking
.  len - maximum string length
-  pre - string to prepend to name or PETSC_NULL

   Output Parameters:
+  string - location to copy string
-  flg - 1 if found, else 0

   Fortran Note:
   The Fortran interface is slightly different from the C/C++
   interface (len is not used).  Sample usage in Fortran follows
.vb
      character *20 string
      integer   flg, ierr
      call OptionsGetString(PETSC_NULL_CHARACTER,'-s',string,flg,ierr)
.ve

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

   Not Collective

   Input Parameters:
+  name - the option one is seeking
.  pre - string to prepend to name or PETSC_NULL
-  nmax - maximum number of strings

   Output Parameter:
+  strings - location to copy strings
-  flg - 1 if found, else 0

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

   Not Collective

   Options Database Key:
.  -optionsleft - Activates OptionsAllUsed() within PetscFinalize()

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

/*@
    OptionsLeft - Prints to screen any options that were set and never used.

.seealso: OptionsAllUsed()
@*/
int OptionsLeft(void)
{
  int i;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "OptionsCreate"
int OptionsCreate(void)
{
  PetscFunctionBegin;
  options = (OptionsTable*) malloc(sizeof(OptionsTable)); CHKPTRQ(options);
  PetscMemzero(options->used,MAXOPTIONS*sizeof(int));
  options->namegiven = 0;
  PetscFunctionReturn(0);
}
