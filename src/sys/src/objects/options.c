/*$Id: options.c,v 1.239 2000/09/28 21:09:12 bsmith Exp bsmith $*/
/*
   These routines simplify the use of command line, file options, etc.,
   and are used to manipulate the options database.

  This file uses regular malloc and free because it cannot know 
  what malloc is being used until it has already processed the input.
*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "petscfix.h"

/* 
    For simplicity, we use a static size database
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

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsAtoi"
int OptionsAtoi(const char name[],int *a)
{
  int        i,ierr,len;
  PetscTruth decide,tdefault,mouse;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) SETERRQ(1,"charactor string of length zero has no numerical value");

  ierr = PetscStrcasecmp(name,"PETSC_DEFAULT",&tdefault);CHKERRQ(ierr);
  if (!tdefault) {
    ierr = PetscStrcasecmp(name,"DEFAULT",&tdefault);CHKERRQ(ierr);
  }
  ierr = PetscStrcasecmp(name,"PETSC_DECIDE",&decide);CHKERRQ(ierr);
  if (!decide) {
    ierr = PetscStrcasecmp(name,"DECIDE",&decide);CHKERRQ(ierr);
  }
  ierr = PetscStrcasecmp(name,"mouse",&mouse);CHKERRQ(ierr);

  if (tdefault) {
    *a = PETSC_DEFAULT;
  } else if (decide) {
    *a = PETSC_DECIDE;
  } else if (mouse) {
    *a = -1;
  } else {
    if (name[0] != '+' && name[0] != '-' && name[0] < '0' && name[0] > '9') {
      SETERRQ1(1,"Input string %s has no integer value (do not include . in it)",name);
    }
    for (i=1; i<len; i++) {
      if (name[i] < '0' || name[i] > '9') {
        SETERRQ1(1,"Input string %s has no integer value (do not include . in it)",name);
      }
    }
    *a  = atoi(name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsAtod"
int OptionsAtod(const char name[],PetscReal *a)
{
  int        ierr,len;
  PetscTruth decide,tdefault;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) SETERRQ(1,"charactor string of length zero has no numerical value");

  ierr = PetscStrcasecmp(name,"PETSC_DEFAULT",&tdefault);CHKERRQ(ierr);
  if (!tdefault) {
    ierr = PetscStrcasecmp(name,"DEFAULT",&tdefault);CHKERRQ(ierr);
  }
  ierr = PetscStrcasecmp(name,"PETSC_DECIDE",&decide);CHKERRQ(ierr);
  if (!decide) {
    ierr = PetscStrcasecmp(name,"DECIDE",&decide);CHKERRQ(ierr);
  }

  if (tdefault) {
    *a = PETSC_DEFAULT;
  } else if (decide) {
    *a = PETSC_DECIDE;
  } else {
    if (name[0] != '+' && name[0] != '-' && name[0] != '.' && name[0] < '0' && name[0] > '9') {
      SETERRQ1(1,"Input string %s has no numeric value ",name);
    }
    *a  = atof(name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscGetProgramName"
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
    its entire path, so one should generally set len >= 256.
@*/
int PetscGetProgramName(char name[],int len)
{
  int ierr;

  PetscFunctionBegin;
  if (!options) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call PetscInitialize() first");
  if (!options->namegiven) SETERRQ(PETSC_ERR_PLIB,"Unable to determine program name");
  ierr = PetscStrncpy(name,options->programname,len);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSetProgramName"
int PetscSetProgramName(const char name[])
{ 
  char *sname = 0;
  int  ierr;

  PetscFunctionBegin;
  options->namegiven = 1;
  /* Now strip away the path, if absulute path is specified */
  ierr = PetscStrrchr(name,'/',&sname);CHKERRQ(ierr);
  ierr  = PetscStrncpy(options->programname,sname,256);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsInsertFile"
/*@C
     OptionsInsertFile - Inserts options into the database from a file.

     Not collective: but only processes that call this routine will set the options
                     included in the file

  Input Parameter:
.   file - name of file


  Level: intermediate

.seealso: OptionsSetValue(), OptionsPrint(), OptionsHasName(), OptionsGetInt(),
          OptionsGetDouble(), OptionsGetString(), OptionsGetIntArray()

@*/
int OptionsInsertFile(const char file[])
{
  char  string[128],fname[256],*first,*second,*third,*final;
  int   len,ierr,i;
  FILE  *fd;

  PetscFunctionBegin;
  ierr = PetscFixFilename(file,fname);CHKERRQ(ierr);
  fd  = fopen(fname,"r"); 
  if (fd) {
    while (fgets(string,128,fd)) {
      /* Comments are indicated by #, ! or % in the first column */
      if (string[0] == '#') continue;
      if (string[0] == '!') continue;
      if (string[0] == '%') continue;
      /* replace tabs with " " */
      ierr = PetscStrlen(string,&len);CHKERRQ(ierr);
      for (i=0; i<len; i++) {
        if (string[i] == '\t') {
          string[i] = ' ';
        }
      }
      ierr = PetscStrtok(string," ",&first);CHKERRQ(ierr);
      ierr = PetscStrtok(0," ",&second);CHKERRQ(ierr);
      if (first && first[0] == '-') {
        if (second) {final = second;} else {final = first;}
        ierr = PetscStrlen(final,&len);CHKERRQ(ierr);
        while (len > 0 && (final[len-1] == ' ' || final[len-1] == '\n')) {
          len--; final[len] = 0;
        }
        ierr = OptionsSetValue(first,second);CHKERRQ(ierr);
      } else if (first) {
        PetscTruth match;

        ierr = PetscStrcmp(first,"alias",&match);CHKERRQ(ierr);
        if (match) {
          ierr = PetscStrtok(0," ",&third);CHKERRQ(ierr);
          if (!third) SETERRQ1(PETSC_ERR_ARG_WRONG,"Error in options file:alias missing (%s)",second);
          ierr = PetscStrlen(third,&len);CHKERRQ(ierr);
          if (third[len-1] == '\n') third[len-1] = 0;
          ierr = OptionsSetAlias(second,third);CHKERRQ(ierr);
        }
      }
    }
    fclose(fd);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsInsert"
/*
   OptionsInsert - Inserts into the options database from the command line,
                   the environmental variable and a file.

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
-  file - optional filename, defaults to ~username/.petscrc

   Note:
   Since OptionsInsert() is automatically called by PetscInitialize(),
   the user does not typically need to call this routine. OptionsInsert()
   can be called several times, adding additional entries into the database.

   Concepts: options database^adding

.seealso: OptionsDestroy_Private(), OptionsPrint()
*/
int OptionsInsert(int *argc,char ***args,const char file[])
{
  int  ierr,rank;
  char pfile[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  options->N        = 0;
  options->Naliases = 0;
  options->argc     = (argc) ? *argc : 0;
  options->args     = (args) ? *args : 0;

  if (file) {
    ierr = OptionsInsertFile(file);CHKERRQ(ierr);
  } else {
    ierr = PetscGetHomeDirectory(pfile,240);CHKERRQ(ierr);
    ierr = PetscStrcat(pfile,"/.petscrc");CHKERRQ(ierr);
    ierr = OptionsInsertFile(pfile);CHKERRQ(ierr);
  }

  /* insert environmental options */
  {
    char *eoptions = 0,*second,*first;
    int  len;
    if (!rank) {
      eoptions = (char*)getenv("PETSC_OPTIONS");
      ierr     = PetscStrlen(eoptions,&len);CHKERRQ(ierr);
      ierr     = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    } else {
      ierr     = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      if (len) {
        eoptions = (char*)PetscMalloc((len+1)*sizeof(char *));CHKPTRQ(eoptions);
      }
    }
    if (len) {
      ierr          = MPI_Bcast(eoptions,len,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      eoptions[len] = 0;
      ierr          =  PetscStrtok(eoptions," ",&first);CHKERRQ(ierr);
      while (first) {
        if (first[0] != '-') {ierr = PetscStrtok(0," ",&first);CHKERRQ(ierr); continue;}
        ierr = PetscStrtok(0," ",&second);CHKERRQ(ierr);
        if ((!second) || ((second[0] == '-') && (second[1] > '9'))) {
          ierr = OptionsSetValue(first,(char *)0);CHKERRQ(ierr);
          first = second;
        } else {
          ierr = OptionsSetValue(first,second);CHKERRQ(ierr);
          ierr = PetscStrtok(0," ",&first);CHKERRQ(ierr);
        }
      }
      if (rank) {ierr = PetscFree(eoptions);CHKERRQ(ierr);}
    }
  }

  /* insert command line options */
  if (argc && args && *argc) {
    int        left    = *argc - 1;
    char       **eargs = *args + 1;
    PetscTruth isoptions_file,isp4,tisp4;

    while (left) {
      ierr = PetscStrcmp(eargs[0],"-options_file",&isoptions_file);CHKERRQ(ierr);
      ierr = PetscStrcmp(eargs[0],"-p4pg",&isp4);CHKERRQ(ierr);
      ierr = PetscStrcmp(eargs[0],"-p4wd",&tisp4);CHKERRQ(ierr);
      isp4 = (PetscTruth) (isp4 || tisp4);
      ierr = PetscStrcmp(eargs[0],"-np",&tisp4);CHKERRQ(ierr);
      isp4 = (PetscTruth) (isp4 || tisp4);
      ierr = PetscStrcmp(eargs[0],"-p4amslave",&tisp4);CHKERRQ(ierr);

      if (eargs[0][0] != '-') {
        eargs++; left--;
      } else if (isoptions_file) {
        ierr = OptionsInsertFile(eargs[1]);CHKERRQ(ierr);
        eargs += 2; left -= 2;

      /*
         These are "bad" options that MPICH, etc put on the command line
         we strip them out here.
      */
      } else if (tisp4) {
        eargs += 1; left -= 1;        
      } else if (isp4) {
        eargs += 2; left -= 2;
      } else if ((left < 2) || ((eargs[1][0] == '-') && 
               ((eargs[1][1] > '9') || (eargs[1][1] < '0')))) {
        ierr = OptionsSetValue(eargs[0],PETSC_NULL);CHKERRQ(ierr);
        eargs++; left--;
      } else {
        ierr = OptionsSetValue(eargs[0],eargs[1]);CHKERRQ(ierr);
        eargs += 2; left -= 2;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsPrint"
/*@C
   OptionsPrint - Prints the options that have been loaded. This is
   useful for debugging purposes.

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  FILE fd - location to print options (usually stdout or stderr)

   Options Database Key:
.  -optionstable - Activates OptionsPrint() within PetscFinalize()

   Level: advanced

   Concepts: options database^printing

.seealso: OptionsAllUsed()
@*/
int OptionsPrint(FILE *fd)
{
  int i,ierr;

  PetscFunctionBegin;
  if (!fd) fd = stdout;
  if (!options) {ierr = OptionsInsert(0,0,0);CHKERRQ(ierr);}
  for (i=0; i<options->N; i++) {
    if (options->values[i]) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"OptionTable: -%s %s\n",options->names[i],options->values[i]);CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"OptionTable: -%s\n",options->names[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetAll"
/*@C
   OptionsGetAll - Lists all the options the program was run with in a single string.

   Not Collective

   Output Parameter:
.  copts - pointer where string pointer is stored

   Level: advanced

   Concepts: options database^listing

.seealso: OptionsAllUsed(), OptionsPrintf()
@*/
int OptionsGetAll(char *copts[])
{
  int  i,ierr,len = 1,lent;
  char *coptions;

  PetscFunctionBegin;
  if (!options) {ierr = OptionsInsert(0,0,0);CHKERRQ(ierr);}

  /* count the length of the required string */
  for (i=0; i<options->N; i++) {
    ierr = PetscStrlen(options->names[i],&lent);CHKERRQ(ierr);
    len += 1 + lent;
    if (options->values[i]) {
      ierr = PetscStrlen(options->values[i],&lent);CHKERRQ(ierr);
      len += 1 + lent;
    } 
  }
  coptions    = (char*)PetscMalloc(len*sizeof(char));CHKPTRQ(coptions);
  coptions[0] = 0;
  for (i=0; i<options->N; i++) {
    ierr = PetscStrcat(coptions,"-");CHKERRQ(ierr);
    ierr = PetscStrcat(coptions,options->names[i]);CHKERRQ(ierr);
    ierr = PetscStrcat(coptions," ");CHKERRQ(ierr);
    if (options->values[i]) {
      ierr = PetscStrcat(coptions,options->values[i]);CHKERRQ(ierr);
      ierr = PetscStrcat(coptions," ");CHKERRQ(ierr);
    } 
  }
  *copts = coptions;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsDestroy"
/*
    OptionsDestroy - Destroys the option database. 

    Note:
    Since OptionsDestroy() is called by PetscFinalize(), the user 
    typically does not need to call this routine.

.seealso: OptionsInsert()
*/
int OptionsDestroy(void)
{
  int i;

  PetscFunctionBegin;
  if (!options) PetscFunctionReturn(0);
  for (i=0; i<options->N; i++) {
    if (options->names[i]) free(options->names[i]);
    if (options->values[i]) free(options->values[i]);
  }
  for (i=0; i<options->Naliases; i++) {
    free(options->aliases1[i]);
    free(options->aliases2[i]);
  }
  free(options);
  options = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsSetValue"
/*@C
   OptionsSetValue - Sets an option name-value pair in the options 
   database, overriding whatever is already present.

   Not collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  name - name of option, this SHOULD have the - prepended
-  value - the option value (not used for all options)

   Level: intermediate

   Note:
   Only some options have values associated with them, such as
   -ksp_rtol tol.  Other options stand alone, such as -ksp_monitor.

  Concepts: options database^adding option

.seealso: OptionsInsert()
@*/
int OptionsSetValue(const char iname[],const char value[])
{
  int        len,N,n,i,ierr;
  char       **names,*name = (char*)iname;
  PetscTruth gt,match;

  PetscFunctionBegin;
  if (!options) {ierr = OptionsInsert(0,0,0);CHKERRQ(ierr);}

  /* this is so that -h and -help are equivalent (p4 does not like -help)*/
  ierr = PetscStrcmp(name,"-h",&match);CHKERRQ(ierr);
  if (match) name = "-help";

  name++;
  /* first check against aliases */
  N = options->Naliases; 
  for (i=0; i<N; i++) {
    ierr = PetscStrcmp(options->aliases1[i],name,&match);CHKERRQ(ierr);
    if (match) {
      name = options->aliases2[i];
      break;
    }
  }

  N     = options->N;
  n     = N;
  names = options->names; 
 
  for (i=0; i<N; i++) {
    ierr = PetscStrcmp(names[i],name,&match);CHKERRQ(ierr);
    ierr  = PetscStrgrt(names[i],name,&gt);CHKERRQ(ierr);
    if (match) {
      if (options->values[i]) free(options->values[i]);
      ierr = PetscStrlen(value,&len);CHKERRQ(ierr);
      if (len) {
        options->values[i] = (char*)malloc((len+1)*sizeof(char));CHKPTRQ(options->values[i]);
        ierr = PetscStrcpy(options->values[i],value);CHKERRQ(ierr);
      } else { options->values[i] = 0;}
      PetscFunctionReturn(0);
    } else if (gt) {
      n = i;
      break;
    }
  }
  if (N >= MAXOPTIONS) {
    SETERRQ1(1,"No more room in option table, limit %d recompile \n src/sys/src/objects/options.c with larger value for MAXOPTIONS\n",MAXOPTIONS);
  }
  /* shift remaining values down 1 */
  for (i=N; i>n; i--) {
    names[i]           = names[i-1];
    options->values[i] = options->values[i-1];
    options->used[i]   = options->used[i-1];
  }
  /* insert new name and value */
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  names[n] = (char*)malloc((len+1)*sizeof(char));CHKPTRQ(names[n]);
  ierr = PetscStrcpy(names[n],name);CHKERRQ(ierr);
  if (value) {
    ierr = PetscStrlen(value,&len);CHKERRQ(ierr);
    options->values[n] = (char*)malloc((len+1)*sizeof(char));CHKPTRQ(options->values[n]);
    ierr = PetscStrcpy(options->values[n],value);CHKERRQ(ierr);
  } else {options->values[n] = 0;}
  options->used[n] = 0;
  options->N++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsClearValue"
/*@C
   OptionsClearValue - Clears an option name-value pair in the options 
   database, overriding whatever is already present.

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameter:
.  name - name of option, this SHOULD have the - prepended

   Level: intermediate

   Concepts: options database^removing option
.seealso: OptionsInsert()
@*/
int OptionsClearValue(const char iname[])
{
  int        N,n,i,ierr;
  char       **names,*name=(char*)iname;
  PetscTruth gt,match;

  PetscFunctionBegin;
  if (!options) {ierr = OptionsInsert(0,0,0);CHKERRQ(ierr);}

  name++;

  N     = options->N; n = 0;
  names = options->names; 
 
  for (i=0; i<N; i++) {
    ierr = PetscStrcmp(names[i],name,&match);CHKERRQ(ierr);
    ierr  = PetscStrgrt(names[i],name,&gt);CHKERRQ(ierr);
    if (match) {
      if (options->values[i]) free(options->values[i]);
      break;
    } else if (gt) {
      PetscFunctionReturn(0); /* it was not listed */
    }
    n++;
  }
  /* shift remaining values down 1 */
  for (i=n; i<N-1; i++) {
    names[i]           = names[i+1];
    options->values[i] = options->values[i+1];
    options->used[i]   = options->used[i+1];
  }
  options->N--;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsSetAlias"
int OptionsSetAlias(const char inewname[],const char ioldname[])
{
  int  ierr,len,n = options->Naliases;
  char *newname = (char *)inewname,*oldname = (char*)ioldname;

  PetscFunctionBegin;
  if (newname[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"aliased must have -: Instead %s",newname);
  if (oldname[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"aliasee must have -: Instead %s",oldname);
  if (n >= MAXALIASES) {
    SETERRQ1(PETSC_ERR_MEM,"You have defined to many PETSc options aliases, limit %d recompile \n  src/sys/src/objects/options.c with larger value for MAXALIASES",MAXALIASES);
  }

  newname++; oldname++;
  ierr = PetscStrlen(newname,&len);CHKERRQ(ierr);
  options->aliases1[n] = (char*)malloc((len+1)*sizeof(char));CHKPTRQ(options->aliases1[n]);
  ierr = PetscStrcpy(options->aliases1[n],newname);CHKERRQ(ierr);
  ierr = PetscStrlen(oldname,&len);CHKERRQ(ierr);
  options->aliases2[n] = (char*)malloc((len+1)*sizeof(char));CHKPTRQ(options->aliases2[n]);
  ierr = PetscStrcpy(options->aliases2[n],oldname);CHKERRQ(ierr);
  options->Naliases++;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsFindPair_Private"
static int OptionsFindPair_Private(const char pre[],const char name[],char *value[],PetscTruth *flg)
{
  int        i,N,ierr,len;
  char       **names,tmp[256];
  PetscTruth match;

  PetscFunctionBegin;
  if (!options) {ierr = OptionsInsert(0,0,0);CHKERRQ(ierr);}
  N = options->N;
  names = options->names;

  if (name[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"Name must begin with -: Instead %s",name);

  /* append prefix to name */
  if (pre) {
    ierr = PetscStrncpy(tmp,pre,256);CHKERRQ(ierr);
    ierr = PetscStrlen(tmp,&len);CHKERRQ(ierr);
    ierr = PetscStrncat(tmp,name+1,256-len-1);CHKERRQ(ierr);
  } else {
    ierr = PetscStrncpy(tmp,name+1,256);CHKERRQ(ierr);
  }

  /* slow search */
  *flg = PETSC_FALSE;
  for (i=0; i<N; i++) {
    ierr = PetscStrcmp(names[i],tmp,&match);CHKERRQ(ierr);
    if (match) {
       *value = options->values[i];
       options->used[i]++;
       *flg = PETSC_TRUE;
       break;
     }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsReject" 
/*@C
   OptionsReject - Generates an error if a certain option is given.

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  name - the option one is seeking 
-  mess - error message (may be PETSC_NULL)

   Level: advanced

   Concepts: options database^rejecting option

.seealso: OptionsGetInt(), OptionsGetDouble(),OptionsHasName(),
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsReject(const char name[],const char mess[])
{
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = OptionsHasName(PETSC_NULL,name,&flag);CHKERRQ(ierr);
  if (flag) {
    if (mess) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Program has disabled option: %s with %s",name,mess);
    } else {
      SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Program has disabled option: %s",name);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsHasName"
/*@C
   OptionsHasName - Determines whether a certain option is given in the database.

   Not Collective

   Input Parameters:
+  name - the option one is seeking 
-  pre - string to prepend to the name or PETSC_NULL

   Output Parameters:
.  flg - PETSC_TRUE if found else PETSC_FALSE.

   Level: beginner

   Concepts: options database^has option name

.seealso: OptionsGetInt(), OptionsGetDouble(),
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsHasName(const char pre[],const char name[],PetscTruth *flg)
{
  char       *value;
  int        ierr;
  PetscTruth isfalse,flag;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);

  /* remove if turned off */
  if (flag) {    
    ierr = PetscStrcmp(value,"FALSE",&isfalse);CHKERRQ(ierr);
    if (isfalse) flag = PETSC_FALSE;
    ierr = PetscStrcmp(value,"NO",&isfalse);CHKERRQ(ierr);
    if (isfalse) flag = PETSC_FALSE;
    ierr = PetscStrcmp(value,"0",&isfalse);CHKERRQ(ierr);
    if (isfalse) flag = PETSC_FALSE;
    ierr = PetscStrcmp(value,"false",&isfalse);CHKERRQ(ierr);
    if (isfalse) flag = PETSC_FALSE;
    ierr = PetscStrcmp(value,"no",&isfalse);CHKERRQ(ierr);
  }
  if (flg) *flg = flag;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetInt"
/*@C
   OptionsGetInt - Gets the integer value for a particular option in the database.

   Not Collective

   Input Parameters:
+  name - the option one is seeking
-  pre - the string to prepend to the name or PETSC_NULL

   Output Parameter:
+  ivalue - the integer value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

.seealso: OptionsGetDouble(), OptionsHasName(), OptionsGetString(),
          OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetInt(const char pre[],const char name[],int *ivalue,PetscTruth *flg)
{
  char       *value;
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {if (flg) *flg = PETSC_FALSE; *ivalue = 0;}
    else {
      if (flg) *flg = PETSC_TRUE; 
      ierr = OptionsAtoi(value,ivalue);CHKERRQ(ierr);
    }
  } else {
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetLogical"
/*@C
   OptionsGetLogical - Gets the Logical (true or false) value for a particular 
            option in the database.

   Not Collective

   Input Parameters:
+  name - the option one is seeking
-  pre - the string to prepend to the name or PETSC_NULL

   Output Parameter:
+  ivalue - the logical value to return
-  flg - PETSC_TRUE  if found, else PETSC_FALSE

   Level: beginner

   Notes:
       TRUE, true, YES, yes, nostring, and 1 all translate to PETSC_TRUE
       FALSE, false, NO, no, and 0 all translate to PETSC_FALSE

   Concepts: options database^has logical

.seealso: OptionsGetDouble(), OptionsHasName(), OptionsGetString(),
          OptionsGetIntArray(), OptionsGetDoubleArray(), OptionsGetInt()
@*/
int OptionsGetLogical(const char pre[],const char name[],PetscTruth *ivalue,PetscTruth *flg)
{
  char       *value;
  PetscTruth flag,istrue,isfalse;
  int        ierr;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (flg) *flg = PETSC_TRUE;
    if (!value) {
      *ivalue = PETSC_TRUE;
    } else {
      *ivalue = PETSC_TRUE;
      ierr = PetscStrcmp(value,"TRUE",&istrue);CHKERRQ(ierr);
      if (istrue) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"YES",&istrue);CHKERRQ(ierr);
      if (istrue) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"YES",&istrue);CHKERRQ(ierr);
      if (istrue) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"1",&istrue);CHKERRQ(ierr);
      if (istrue) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"true",&istrue);CHKERRQ(ierr);
      if (istrue) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"yes",&istrue);CHKERRQ(ierr);
      if (istrue) PetscFunctionReturn(0);

      *ivalue = PETSC_FALSE;
      ierr = PetscStrcmp(value,"FALSE",&isfalse);CHKERRQ(ierr);
      if (isfalse) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"NO",&isfalse);CHKERRQ(ierr);
      if (isfalse) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"0",&isfalse);CHKERRQ(ierr);
      if (isfalse) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"false",&isfalse);CHKERRQ(ierr);
      if (isfalse) PetscFunctionReturn(0);
      ierr = PetscStrcmp(value,"no",&isfalse);CHKERRQ(ierr);
      if (isfalse) PetscFunctionReturn(0);

      SETERRQ1(1,"Unknown logical value: %s",value);
    }
  } else {
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetDouble"
/*@C
   OptionsGetDouble - Gets the double precision value for a particular 
   option in the database.

   Not Collective

   Input Parameters:
+  name - the option one is seeking
-  pre - string to prepend to each name or PETSC_NULL

   Output Parameter:
+  dvalue - the double value to return
-  flg - PETSC_TRUE if found, PETSC_FALSE if not found

   Level: beginner

   Concepts: options database^has double

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetDouble(const char pre[],const char name[],double *dvalue,PetscTruth *flg)
{
  char       *value;
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {if (flg) *flg = PETSC_FALSE; *dvalue = 0.0;}
    else        {if (flg) *flg = PETSC_TRUE; ierr = OptionsAtod(value,dvalue);CHKERRQ(ierr);}
  } else {
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetScalar"
/*@C
   OptionsGetScalar - Gets the scalar value for a particular 
   option in the database.

   Not Collective

   Input Parameters:
+  name - the option one is seeking
-  pre - string to prepend to each name or PETSC_NULL

   Output Parameter:
+  dvalue - the double value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Usage:
   A complex number 2+3i can be specified as 2,3 at the command line.
   or a number 2.0e-10 - 3.3e-20 i  can be specified as 2.0e-10,3.3e-20

   Concepts: options database^has scalar

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetScalar(const char pre[],const char name[],Scalar *dvalue,PetscTruth *flg)
{
  char       *value;
  PetscTruth flag;
  int        ierr;
  
  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {
      if (flg) *flg = PETSC_FALSE; *dvalue = 0.0;
    } else { 
#if !defined(PETSC_USE_COMPLEX)
      ierr = OptionsAtod(value,dvalue);
#else
      double re=0.0,im=0.0;
      char   *tvalue = 0;

      ierr = PetscStrtok(value,",",&tvalue);CHKERRQ(ierr);
      if (!tvalue) { SETERRQ(1,"unknown string specified\n"); }
      ierr    = OptionsAtod(tvalue,&re);CHKERRQ(ierr);
      ierr    = PetscStrtok(0,",",&tvalue);CHKERRQ(ierr);
      if (!tvalue) { /* Unknown separator used. using only real value */
        *dvalue = re;
      } else {
        ierr    = OptionsAtod(tvalue,&im);CHKERRQ(ierr);
        *dvalue = re + PETSC_i*im;
      } 
#endif
      if (flg) *flg    = PETSC_TRUE;
    } 
  } else { /* flag */
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetDoubleArray"
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
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^array of doubles

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetIntArray()
@*/
int OptionsGetDoubleArray(const char pre[],const char name[],double dvalue[],int *nmax,PetscTruth *flg)
{
  char       *value,*cpy;
  int        n = 0,ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag)  {if (flg) *flg = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (!value) {if (flg) *flg = PETSC_TRUE; *nmax = 0; PetscFunctionReturn(0);}

  if (flg) *flg = PETSC_TRUE;
  /* make a copy of the values, otherwise we destroy the old values */
  ierr  = PetscStrallocpy(value,&cpy);CHKERRQ(ierr);
  value = cpy;

  ierr = PetscStrtok(value,",",&value);CHKERRQ(ierr);
  while (n < *nmax) {
    if (!value) break;
    ierr = OptionsAtod(value,dvalue++);CHKERRQ(ierr);
    ierr = PetscStrtok(0,",",&value);CHKERRQ(ierr);
    n++;
  }
  *nmax = n;
  ierr = PetscFree(cpy);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetIntArray"
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
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^array of ints

.seealso: OptionsGetInt(), OptionsHasName(), 
           OptionsGetString(), OptionsGetDoubleArray()
@*/
int OptionsGetIntArray(const char pre[],const char name[],int dvalue[],int *nmax,PetscTruth *flg)
{
  char       *value,*cpy;
  int        n = 0,ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag)  {if (flg) *flg = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (!value) {if (flg) *flg = PETSC_TRUE; *nmax = 0; PetscFunctionReturn(0);}

  if (flg) *flg = PETSC_TRUE;
  /* make a copy of the values, otherwise we destroy the old values */
  ierr  = PetscStrallocpy(value,&cpy);CHKERRQ(ierr);
  value = cpy;

  ierr = PetscStrtok(value,",",&value);CHKERRQ(ierr);
  while (n < *nmax) {
    if (!value) break;
    ierr      = OptionsAtoi(value,dvalue);CHKERRQ(ierr);
    dvalue++;
    ierr      = PetscStrtok(0,",",&value);CHKERRQ(ierr);
    n++;
  }
  *nmax = n;
  ierr = PetscFree(cpy);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetString"
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
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Fortran Note:
   The Fortran interface is slightly different from the C/C++
   interface (len is not used).  Sample usage in Fortran follows
.vb
      character *20 string
      integer   flg, ierr
      call OptionsGetString(PETSC_NULL_CHARACTER,'-s',string,flg,ierr)
.ve

   Concepts: options database^string

.seealso: OptionsGetInt(), OptionsGetDouble(),  
           OptionsHasName(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetString(const char pre[],const char name[],char string[],int len,PetscTruth *flg)
{
  char       *value;
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr); 
  if (!flag) {
    if (flg) *flg = PETSC_FALSE;
  } else {
    if (flg) *flg = PETSC_TRUE;
    if (value) {
      ierr = PetscStrncpy(string,value,len);CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(string,len);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsGetStringArray"
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
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes: 
   The user should pass in an array of pointers to char, to hold all the
   strings returned by this function.

   The user is responsible for deallocating the strings that are
   returned. The Fortran interface for this routine is not supported.

   Contributed by Matthew Knepley.

   Concepts: options database^array of strings

.seealso: OptionsGetInt(), OptionsGetDouble(),  
           OptionsHasName(), OptionsGetIntArray(), OptionsGetDoubleArray()
@*/
int OptionsGetStringArray(const char pre[],const char name[],char **strings,int *nmax,PetscTruth *flg)
{
  char       *value,*cpy;
  int        len,n,ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = OptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr); 
  if (!flag)  {*nmax = 0; if (flg) *flg = PETSC_FALSE; PetscFunctionReturn(0);}
  if (!value) {*nmax = 0; if (flg) *flg = PETSC_FALSE;PetscFunctionReturn(0);}
  if (!*nmax) {if (flg) *flg = PETSC_FALSE;PetscFunctionReturn(0);}
  if (flg) *flg = PETSC_TRUE;

  /* make a copy of the values, otherwise we destroy the old values */
  ierr  = PetscStrallocpy(value,&cpy);CHKERRQ(ierr);
  value = cpy;

  ierr = PetscStrtok(value,",",&value);CHKERRQ(ierr);
  n = 0;
  while (n < *nmax) {
    if (!value) break;
    ierr        = PetscStrlen(value,&len);CHKERRQ(ierr);
    strings[n] = (char*)PetscMalloc((len+1)*sizeof(char));CHKPTRQ(strings[n]);
    ierr       = PetscStrcpy(strings[n],value);CHKERRQ(ierr);
    ierr       = PetscStrtok(0,",",&value);CHKERRQ(ierr);
    n++;
  }
  *nmax = n;
  ierr = PetscFree(cpy);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsAllUsed"
/*@C
   OptionsAllUsed - Returns a count of the number of options in the 
   database that have never been selected.

   Not Collective

   Output Parameter:
.   N - count of options not used

   Level: advanced

.seealso: OptionsPrint()
@*/
int OptionsAllUsed(int *N)
{
  int  i,n = 0;

  PetscFunctionBegin;
  for (i=0; i<options->N; i++) {
    if (!options->used[i]) { n++; }
  }
  *N = n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"OptionsLeft"
/*@
    OptionsLeft - Prints to screen any options that were set and never used.

  Not collective

   Options Database Key:
.  -optionsleft - Activates OptionsAllUsed() within PetscFinalize()

  Level: advanced

.seealso: OptionsAllUsed()
@*/
int OptionsLeft(void)
{
  int i,ierr;

  PetscFunctionBegin;
  for (i=0; i<options->N; i++) {
    if (!options->used[i]) {
      if (options->values[i]) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s value: %s\n",options->names[i],options->values[i]);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s no value \n",options->names[i]);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
    OptionsCreate - Creates the empty options database.

*/
#undef __FUNC__  
#define __FUNC__ /*<a name="OptionsCreate"></a>*/"OptionsCreate"
int OptionsCreate(void)
{
  int ierr;

  PetscFunctionBegin;
  options = (OptionsTable*) malloc(sizeof(OptionsTable));CHKPTRQ(options);
  ierr = PetscMemzero(options->used,MAXOPTIONS*sizeof(int));CHKERRQ(ierr);
  options->namegiven = 0;
  PetscFunctionReturn(0);
}
