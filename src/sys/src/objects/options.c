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
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#if defined(PETSC_HAVE_SYS_PARAM_H)
#include "sys/param.h"
#endif
#include "petscfix.h"

/* 
    For simplicity, we use a static size database
*/
#define MAXOPTIONS 512
#define MAXALIASES 25

typedef struct {
  int        N,argc,Naliases;
  char       **args,*names[MAXOPTIONS],*values[MAXOPTIONS];
  char       *aliases1[MAXALIASES],*aliases2[MAXALIASES];
  int        used[MAXOPTIONS];
  PetscTruth namegiven;
  char       programname[PETSC_MAX_PATH_LEN]; /* HP includes entire path in name */
} PetscOptionsTable;

static PetscOptionsTable *options = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsAtoi"
PetscErrorCode PetscOptionsAtoi(const char name[],PetscInt *a)
{
  PetscErrorCode ierr;
  size_t         i,len;
  PetscTruth     decide,tdefault,mouse;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_ERR_ARG_WRONG,"character string of length zero has no numerical value");

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
      SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Input string %s has no integer value (do not include . in it)",name);
    }
    for (i=1; i<len; i++) {
      if (name[i] < '0' || name[i] > '9') {
        SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Input string %s has no integer value (do not include . in it)",name);
      }
    }
    *a  = atoi(name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsAtod"
PetscErrorCode PetscOptionsAtod(const char name[],PetscReal *a)
{
  PetscErrorCode ierr;
  size_t     len;
  PetscTruth decide,tdefault;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_ERR_ARG_WRONG,"character string of length zero has no numerical value");

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
      SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Input string %s has no numeric value ",name);
    }
    *a  = atof(name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGetProgramName"
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
PetscErrorCode PetscGetProgramName(char name[],size_t len)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!options) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call PetscInitialize() first");
  if (!options->namegiven) SETERRQ(PETSC_ERR_PLIB,"Unable to determine program name");
  ierr = PetscStrncpy(name,options->programname,len);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSetProgramName"
PetscErrorCode PetscSetProgramName(const char name[])
{ 
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->namegiven = PETSC_TRUE;
  ierr  = PetscStrncpy(options->programname,name,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsInsertString"
/*@C
     PetscOptionsInsertString - Inserts options into the database from a string

     Not collective: but only processes that call this routine will set the options
                     included in the file

  Input Parameter:
.   in_str - string that contains options seperated by blanks


  Level: intermediate

  Contributed by Boyana Norris

.seealso: PetscOptionsSetValue(), PetscOptionsPrint(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsInsertFile()

@*/
PetscErrorCode PetscOptionsInsertString(const char in_str[])
{
  char           *str,*first,*second,*third,*final;
  size_t         len;
  PetscErrorCode ierr;
  PetscToken     *token;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(in_str, &str);CHKERRQ(ierr);
  ierr = PetscTokenCreate(str,' ',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
  if (first && first[0] == '-') {
    if (second) {final = second;} else {final = first;}
    ierr = PetscStrlen(final,&len);CHKERRQ(ierr);
    while (len > 0 && (final[len-1] == ' ' || final[len-1] == 'n')) {
      len--; final[len] = 0;
    }
    ierr = PetscOptionsSetValue(first,second);CHKERRQ(ierr);
  } else if (first) {
    PetscTruth match;
    
    ierr = PetscStrcmp(first,"alias",&match);CHKERRQ(ierr);
    if (match) {
      ierr = PetscTokenFind(token,&third);CHKERRQ(ierr);
      if (!third) SETERRQ1(PETSC_ERR_ARG_WRONG,"Error in options string:alias missing (%s)",second);
      ierr = PetscStrlen(third,&len);CHKERRQ(ierr);
      if (third[len-1] == 'n') third[len-1] = 0;
      ierr = PetscOptionsSetAlias(second,third);CHKERRQ(ierr);
    }
  }
  ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
  ierr = PetscFree(str);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsInsertFile"
/*@C
     PetscOptionsInsertFile - Inserts options into the database from a file.

     Not collective: but only processes that call this routine will set the options
                     included in the file

  Input Parameter:
.   file - name of file


  Level: intermediate

.seealso: PetscOptionsSetValue(), PetscOptionsPrint(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()

@*/
PetscErrorCode PetscOptionsInsertFile(const char file[])
{
  char           string[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN],*first,*second,*third,*final;
  PetscErrorCode ierr;
  size_t         i,len,startIndex;
  FILE           *fd;
  PetscToken     *token;

  PetscFunctionBegin;
  ierr = PetscFixFilename(file,fname);CHKERRQ(ierr);
  fd   = fopen(fname,"r"); 
  if (fd) {
    while (fgets(string,128,fd)) {
      /* Comments are indicated by #, ! or % in the first column */
      if (string[0] == '#') continue;
      if (string[0] == '!') continue;
      if (string[0] == '%') continue;

      ierr = PetscStrlen(string,&len);CHKERRQ(ierr);

      /* replace tabs, ^M with " " */
      for (i=0; i<len; i++) {
        if (string[i] == '\t' || string[i] == '\r') {
          string[i] = ' ';
        }
      }
      for(startIndex = 0; startIndex < len-1; startIndex++) {
        if (string[startIndex] != ' ') break;
      }
      ierr = PetscTokenCreate(&string[startIndex],' ',&token);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
      if (first && first[0] == '-') {
        if (second) {final = second;} else {final = first;}
        ierr = PetscStrlen(final,&len);CHKERRQ(ierr);
        while (len > 0 && (final[len-1] == ' ' || final[len-1] == '\n')) {
          len--; final[len] = 0;
        }
        ierr = PetscOptionsSetValue(first,second);CHKERRQ(ierr);
      } else if (first) {
        PetscTruth match;

        ierr = PetscStrcmp(first,"alias",&match);CHKERRQ(ierr);
        if (match) {
          ierr = PetscTokenFind(token,&third);CHKERRQ(ierr);
          if (!third) SETERRQ1(PETSC_ERR_ARG_WRONG,"Error in options file:alias missing (%s)",second);
          ierr = PetscStrlen(third,&len);CHKERRQ(ierr);
          if (third[len-1] == '\n') third[len-1] = 0;
          ierr = PetscOptionsSetAlias(second,third);CHKERRQ(ierr);
        }
      }
      ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
    }
    fclose(fd);
  } else {
    SETERRQ1(PETSC_ERR_USER,"Unable to open Options File %s",fname);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsInsert"
/*@C
   PetscOptionsInsert - Inserts into the options database from the command line,
                   the environmental variable and a file.

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
-  file - optional filename, defaults to ~username/.petscrc

   Note:
   Since PetscOptionsInsert() is automatically called by PetscInitialize(),
   the user does not typically need to call this routine. PetscOptionsInsert()
   can be called several times, adding additional entries into the database.

   Level: advanced

   Concepts: options database^adding

.seealso: PetscOptionsDestroy_Private(), PetscOptionsPrint()
@*/
PetscErrorCode PetscOptionsInsert(int *argc,char ***args,const char file[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           pfile[PETSC_MAX_PATH_LEN];
  PetscToken     *token;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  options->argc     = (argc) ? *argc : 0;
  options->args     = (args) ? *args : 0;

  if (file) {
    ierr = PetscOptionsInsertFile(file);CHKERRQ(ierr);
  } else {
    ierr = PetscGetHomeDirectory(pfile,PETSC_MAX_PATH_LEN-16);CHKERRQ(ierr);
    if (pfile[0]) {
      PetscTruth flag;
      ierr = PetscStrcat(pfile,"/.petscrc");CHKERRQ(ierr);
      ierr = PetscTestFile(pfile,'r',&flag);CHKERRQ(ierr);
      if (flag) {
	ierr = PetscOptionsInsertFile(pfile);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscLogInfo((0,"PetscOptionsInsert:Unable to determine home directory; skipping loading ~/.petscrc\n"));CHKERRQ(ierr);
    }
  }

  /* insert environmental options */
  {
    char   *eoptions = 0,*second,*first;
    size_t len = 0;
    if (!rank) {
      eoptions = (char*)getenv("PETSC_OPTIONS");
      ierr     = PetscStrlen(eoptions,&len);CHKERRQ(ierr);
      ierr     = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    } else {
      ierr = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      if (len) {
        ierr = PetscMalloc((len+1)*sizeof(char*),&eoptions);CHKERRQ(ierr);
      }
    }
    if (len) {
      ierr          = MPI_Bcast(eoptions,len,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      eoptions[len] = 0;
      ierr          =  PetscTokenCreate(eoptions,' ',&token);CHKERRQ(ierr);
      ierr          =  PetscTokenFind(token,&first);CHKERRQ(ierr);
      while (first) {
        if (first[0] != '-') {ierr = PetscTokenFind(token,&first);CHKERRQ(ierr); continue;}
        ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
        if ((!second) || ((second[0] == '-') && (second[1] > '9'))) {
          ierr = PetscOptionsSetValue(first,(char *)0);CHKERRQ(ierr);
          first = second;
        } else {
          ierr = PetscOptionsSetValue(first,second);CHKERRQ(ierr);
          ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
        }
      }
      ierr =  PetscTokenDestroy(token);CHKERRQ(ierr);
      if (rank) {ierr = PetscFree(eoptions);CHKERRQ(ierr);}
    }
  }

  /* insert command line options */
  if (argc && args && *argc) {
    int        left    = *argc - 1;
    char       **eargs = *args + 1;
    PetscTruth isoptions_file,isp4,tisp4,isp4yourname,isp4rmrank;

    while (left) {
      ierr = PetscStrcmp(eargs[0],"-options_file",&isoptions_file);CHKERRQ(ierr);
      ierr = PetscStrcmp(eargs[0],"-p4pg",&isp4);CHKERRQ(ierr);
      ierr = PetscStrcmp(eargs[0],"-p4yourname",&isp4yourname);CHKERRQ(ierr);
      ierr = PetscStrcmp(eargs[0],"-p4rmrank",&isp4rmrank);CHKERRQ(ierr);
      ierr = PetscStrcmp(eargs[0],"-p4wd",&tisp4);CHKERRQ(ierr);
      isp4 = (PetscTruth) (isp4 || tisp4);
      ierr = PetscStrcmp(eargs[0],"-np",&tisp4);CHKERRQ(ierr);
      isp4 = (PetscTruth) (isp4 || tisp4);
      ierr = PetscStrcmp(eargs[0],"-p4amslave",&tisp4);CHKERRQ(ierr);

      if (eargs[0][0] != '-') {
        eargs++; left--;
      } else if (isoptions_file) {
        if (left <= 1) SETERRQ(PETSC_ERR_USER,"Missing filename for -options_file filename option");
        if (eargs[1][0] == '-') SETERRQ(PETSC_ERR_USER,"Missing filename for -options_file filename option");
        ierr = PetscOptionsInsertFile(eargs[1]);CHKERRQ(ierr);
        eargs += 2; left -= 2;

      /*
         These are "bad" options that MPICH, etc put on the command line
         we strip them out here.
      */
      } else if (tisp4 || isp4rmrank) {
        eargs += 1; left -= 1;        
      } else if (isp4 || isp4yourname) {
        eargs += 2; left -= 2;
      } else if ((left < 2) || ((eargs[1][0] == '-') && 
               ((eargs[1][1] > '9') || (eargs[1][1] < '0')))) {
        ierr = PetscOptionsSetValue(eargs[0],PETSC_NULL);CHKERRQ(ierr);
        eargs++; left--;
      } else {
        ierr = PetscOptionsSetValue(eargs[0],eargs[1]);CHKERRQ(ierr);
        eargs += 2; left -= 2;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsPrint"
/*@C
   PetscOptionsPrint - Prints the options that have been loaded. This is
   useful for debugging purposes.

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  FILE fd - location to print options (usually stdout or stderr)

   Options Database Key:
.  -optionstable - Activates PetscOptionsPrint() within PetscFinalize()

   Level: advanced

   Concepts: options database^printing

.seealso: PetscOptionsAllUsed()
@*/
PetscErrorCode PetscOptionsPrint(FILE *fd)
{
  PetscErrorCode ierr;
  int i;

  PetscFunctionBegin;
  if (!fd) fd = stdout;
  if (!options) {ierr = PetscOptionsInsert(0,0,0);CHKERRQ(ierr);}
  for (i=0; i<options->N; i++) {
    if (options->values[i]) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"OptionTable: -%s %s\n",options->names[i],options->values[i]);CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"OptionTable: -%s\n",options->names[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetAll"
/*@C
   PetscOptionsGetAll - Lists all the options the program was run with in a single string.

   Not Collective

   Output Parameter:
.  copts - pointer where string pointer is stored

   Level: advanced

   Concepts: options database^listing

.seealso: PetscOptionsAllUsed(), PetscOptionsPrint()
@*/
PetscErrorCode PetscOptionsGetAll(char *copts[])
{
  PetscErrorCode ierr;
  int    i;
  size_t len = 1,lent;
  char   *coptions;

  PetscFunctionBegin;
  if (!options) {ierr = PetscOptionsInsert(0,0,0);CHKERRQ(ierr);}

  /* count the length of the required string */
  for (i=0; i<options->N; i++) {
    ierr = PetscStrlen(options->names[i],&lent);CHKERRQ(ierr);
    len += 2 + lent;
    if (options->values[i]) {
      ierr = PetscStrlen(options->values[i],&lent);CHKERRQ(ierr);
      len += 1 + lent;
    } 
  }
  ierr = PetscMalloc(len*sizeof(char),&coptions);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsDestroy"
/*@C
    PetscOptionsDestroy - Destroys the option database. 

    Note:
    Since PetscOptionsDestroy() is called by PetscFinalize(), the user 
    typically does not need to call this routine.

   Level: developer

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode PetscOptionsDestroy(void)
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

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsSetValue"
/*@C
   PetscOptionsSetValue - Sets an option name-value pair in the options 
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

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode PetscOptionsSetValue(const char iname[],const char value[])
{
  size_t     len;
  PetscErrorCode ierr;
  int        N,n,i;
  char       **names;
  const char *name = (char*)iname;
  PetscTruth gt,match;

  PetscFunctionBegin;
  if (!options) {ierr = PetscOptionsInsert(0,0,0);CHKERRQ(ierr);}

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
        options->values[i] = (char*)malloc((len+1)*sizeof(char));
        ierr = PetscStrcpy(options->values[i],value);CHKERRQ(ierr);
      } else { options->values[i] = 0;}
      PetscFunctionReturn(0);
    } else if (gt) {
      n = i;
      break;
    }
  }
  if (N >= MAXOPTIONS) {
    SETERRQ1(PETSC_ERR_PLIB,"No more room in option table, limit %d recompile \n src/sys/src/objects/options.c with larger value for MAXOPTIONS\n",MAXOPTIONS);
  }
  /* shift remaining values down 1 */
  for (i=N; i>n; i--) {
    names[i]           = names[i-1];
    options->values[i] = options->values[i-1];
    options->used[i]   = options->used[i-1];
  }
  /* insert new name and value */
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  names[n] = (char*)malloc((len+1)*sizeof(char));
  ierr = PetscStrcpy(names[n],name);CHKERRQ(ierr);
  if (value) {
    ierr = PetscStrlen(value,&len);CHKERRQ(ierr);
    options->values[n] = (char*)malloc((len+1)*sizeof(char));
    ierr = PetscStrcpy(options->values[n],value);CHKERRQ(ierr);
  } else {options->values[n] = 0;}
  options->used[n] = 0;
  options->N++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsClearValue"
/*@C
   PetscOptionsClearValue - Clears an option name-value pair in the options 
   database, overriding whatever is already present.

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameter:
.  name - name of option, this SHOULD have the - prepended

   Level: intermediate

   Concepts: options database^removing option
.seealso: PetscOptionsInsert()
@*/
PetscErrorCode PetscOptionsClearValue(const char iname[])
{
  PetscErrorCode ierr;
  int        N,n,i;
  char       **names,*name=(char*)iname;
  PetscTruth gt,match;

  PetscFunctionBegin;
  if (name[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"Name must begin with -: Instead %s",name);
  if (!options) {ierr = PetscOptionsInsert(0,0,0);CHKERRQ(ierr);}

  name++;

  N     = options->N; n = 0;
  names = options->names; 
 
  for (i=0; i<N; i++) {
    ierr  = PetscStrcmp(names[i],name,&match);CHKERRQ(ierr);
    ierr  = PetscStrgrt(names[i],name,&gt);CHKERRQ(ierr);
    if (match) {
      if (options->values[i]) free(options->values[i]);
      break;
    } else if (gt) {
      PetscFunctionReturn(0); /* it was not listed */
    }
    n++;
  }
  if (n == N) PetscFunctionReturn(0); /* it was not listed */

  /* shift remaining values down 1 */
  for (i=n; i<N-1; i++) {
    names[i]           = names[i+1];
    options->values[i] = options->values[i+1];
    options->used[i]   = options->used[i+1];
  }
  options->N--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsSetAlias"
/*@C
   PetscOptionsReject - Generates an error if a certain option is given.

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  name - the option one is seeking 
-  mess - error message (may be PETSC_NULL)

   Level: advanced

   Concepts: options database^rejecting option

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),OptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(),PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsSetAlias(const char inewname[],const char ioldname[])
{
  PetscErrorCode ierr;
  int    n = options->Naliases;
  size_t len;
  char   *newname = (char *)inewname,*oldname = (char*)ioldname;

  PetscFunctionBegin;
  if (newname[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"aliased must have -: Instead %s",newname);
  if (oldname[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"aliasee must have -: Instead %s",oldname);
  if (n >= MAXALIASES) {
    SETERRQ1(PETSC_ERR_MEM,"You have defined to many PETSc options aliases, limit %d recompile \n  src/sys/src/objects/options.c with larger value for MAXALIASES",MAXALIASES);
  }

  newname++; oldname++;
  ierr = PetscStrlen(newname,&len);CHKERRQ(ierr);
  options->aliases1[n] = (char*)malloc((len+1)*sizeof(char));
  ierr = PetscStrcpy(options->aliases1[n],newname);CHKERRQ(ierr);
  ierr = PetscStrlen(oldname,&len);CHKERRQ(ierr);
  options->aliases2[n] = (char*)malloc((len+1)*sizeof(char));
  ierr = PetscStrcpy(options->aliases2[n],oldname);CHKERRQ(ierr);
  options->Naliases++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsFindPair_Private"
static PetscErrorCode PetscOptionsFindPair_Private(const char pre[],const char name[],char *value[],PetscTruth *flg)
{
  PetscErrorCode ierr;
  PetscInt       i,N;
  size_t         len;
  char           **names,tmp[256];
  PetscTruth     match;

  PetscFunctionBegin;
  if (!options) {ierr = PetscOptionsInsert(0,0,0);CHKERRQ(ierr);}
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
  if (!*flg) {
    PetscInt j,cnt = 0,locs[16],loce[16];
    size_t   n;
    ierr = PetscStrlen(tmp,&n);CHKERRQ(ierr);
    /* determine the location and number of all _%d_ in the key */
    for (i=0; i< (PetscInt)n; i++) {
      if (tmp[i] == '_') {
        for (j=i+1; j< (PetscInt)n; j++) {
          if (tmp[j] >= '0' && tmp[j] <= '9') continue;
          if (tmp[j] == '_' && j > i+1) { /* found a number */
            locs[cnt]   = i+1;
            loce[cnt++] = j+1;
          }
          break;
        }
      }
    }
    if (cnt) {
      char tmp2[256];
      for (i=0; i<cnt; i++) {
        ierr = PetscStrcpy(tmp2,"-");CHKERRQ(ierr);
        ierr = PetscStrncat(tmp2,tmp,locs[i]);CHKERRQ(ierr);
        ierr = PetscStrcat(tmp2,tmp+loce[i]);CHKERRQ(ierr);
        ierr = PetscOptionsFindPair_Private(PETSC_NULL,tmp2,value,flg);CHKERRQ(ierr);
        if (*flg) break;
      }
    }        
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsReject" 
/*@C
   PetscOptionsReject - Generates an error if a certain option is given.

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  name - the option one is seeking 
-  mess - error message (may be PETSC_NULL)

   Level: advanced

   Concepts: options database^rejecting option

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),OptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsReject(const char name[],const char mess[])
{
  PetscErrorCode ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL,name,&flag);CHKERRQ(ierr);
  if (flag) {
    if (mess) {
      SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Program has disabled option: %s with %s",name,mess);
    } else {
      SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Program has disabled option: %s",name);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsHasName"
/*@C
   PetscOptionsHasName - Determines whether a certain option is given in the database.

   Not Collective

   Input Parameters:
+  name - the option one is seeking 
-  pre - string to prepend to the name or PETSC_NULL

   Output Parameters:
.  flg - PETSC_TRUE if found else PETSC_FALSE.

   Level: beginner

   Concepts: options database^has option name

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsHasName(const char pre[],const char name[],PetscTruth *flg)
{
  char       *value;
  PetscErrorCode ierr;
  PetscTruth isfalse,flag;

  PetscFunctionBegin;
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);

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

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetInt"
/*@C
   PetscOptionsGetInt - Gets the integer value for a particular option in the database.

   Not Collective

   Input Parameters:
+  pre - the string to prepend to the name or PETSC_NULL
-  name - the option one is seeking

   Output Parameter:
+  ivalue - the integer value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetInt(const char pre[],const char name[],PetscInt *ivalue,PetscTruth *flg)
{
  char       *value;
  PetscErrorCode ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(ivalue,3);
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {if (flg) *flg = PETSC_FALSE;}
    else {
      if (flg) *flg = PETSC_TRUE; 
      ierr = PetscOptionsAtoi(value,ivalue);CHKERRQ(ierr);
    }
  } else {
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetLogical"
/*@C
   PetscOptionsGetLogical - Gets the Logical (true or false) value for a particular 
            option in the database.

   Not Collective

   Input Parameters:
+  pre - the string to prepend to the name or PETSC_NULL
-  name - the option one is seeking

   Output Parameter:
+  ivalue - the logical value to return
-  flg - PETSC_TRUE  if found, else PETSC_FALSE

   Level: beginner

   Notes:
       TRUE, true, YES, yes, nostring, and 1 all translate to PETSC_TRUE
       FALSE, false, NO, no, and 0 all translate to PETSC_FALSE

   Concepts: options database^has logical

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetInt(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetLogical(const char pre[],const char name[],PetscTruth *ivalue,PetscTruth *flg)
{
  char       *value;
  PetscTruth flag,istrue,isfalse;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(ivalue,3);
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
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
      ierr = PetscStrcmp(value,"on",&istrue);CHKERRQ(ierr);
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
      ierr = PetscStrcmp(value,"off",&isfalse);CHKERRQ(ierr);
      if (isfalse) PetscFunctionReturn(0);

      SETERRQ1(PETSC_ERR_ARG_WRONG,"Unknown logical value: %s",value);
    }
  } else {
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetReal"
/*@C
   PetscOptionsGetReal - Gets the double precision value for a particular 
   option in the database.

   Not Collective

   Input Parameters:
+  pre - string to prepend to each name or PETSC_NULL
-  name - the option one is seeking

   Output Parameter:
+  dvalue - the double value to return
-  flg - PETSC_TRUE if found, PETSC_FALSE if not found

   Level: beginner

   Concepts: options database^has double

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(), 
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(),PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetReal(const char pre[],const char name[],PetscReal *dvalue,PetscTruth *flg)
{
  char       *value;
  PetscErrorCode ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidDoublePointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {if (flg) *flg = PETSC_FALSE;}
    else        {if (flg) *flg = PETSC_TRUE; ierr = PetscOptionsAtod(value,dvalue);CHKERRQ(ierr);}
  } else {
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetScalar"
/*@C
   PetscOptionsGetScalar - Gets the scalar value for a particular 
   option in the database.

   Not Collective

   Input Parameters:
+  pre - string to prepend to each name or PETSC_NULL
-  name - the option one is seeking

   Output Parameter:
+  dvalue - the double value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Usage:
   A complex number 2+3i can be specified as 2,3 at the command line.
   or a number 2.0e-10 - 3.3e-20 i  can be specified as 2.0e-10,3.3e-20

   Concepts: options database^has scalar

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(), 
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetScalar(const char pre[],const char name[],PetscScalar *dvalue,PetscTruth *flg)
{
  char       *value;
  PetscTruth flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidScalarPointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {
      if (flg) *flg = PETSC_FALSE;
    } else { 
#if !defined(PETSC_USE_COMPLEX)
      ierr = PetscOptionsAtod(value,dvalue);CHKERRQ(ierr);
#else
      PetscReal  re=0.0,im=0.0;
      PetscToken *token;
      char       *tvalue = 0;

      ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&tvalue);CHKERRQ(ierr);
      if (!tvalue) { SETERRQ(PETSC_ERR_ARG_WRONG,"unknown string specified\n"); }
      ierr    = PetscOptionsAtod(tvalue,&re);CHKERRQ(ierr);
      ierr    = PetscTokenFind(token,&tvalue);CHKERRQ(ierr);
      if (!tvalue) { /* Unknown separator used. using only real value */
        *dvalue = re;
      } else {
        ierr    = PetscOptionsAtod(tvalue,&im);CHKERRQ(ierr);
        *dvalue = re + PETSC_i*im;
      } 
      ierr    = PetscTokenDestroy(token);CHKERRQ(ierr);
#endif
      if (flg) *flg    = PETSC_TRUE;
    } 
  } else { /* flag */
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetRealArray"
/*@C
   PetscOptionsGetRealArray - Gets an array of double precision values for a 
   particular option in the database.  The values must be separated with 
   commas with no intervening spaces.

   Not Collective

   Input Parameters:
+  pre - string to prepend to each name or PETSC_NULL
.  name - the option one is seeking
-  nmax - maximum number of values to retrieve

   Output Parameters:
+  dvalue - the double value to return
.  nmax - actual number of values retreived
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^array of doubles

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(), 
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetRealArray(const char pre[],const char name[],PetscReal dvalue[],PetscInt *nmax,PetscTruth *flg)
{
  char       *value;
  PetscErrorCode ierr;
  int        n = 0;
  PetscTruth flag;
  PetscToken *token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidDoublePointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag)  {if (flg) *flg = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (!value) {if (flg) *flg = PETSC_TRUE; *nmax = 0; PetscFunctionReturn(0);}

  if (flg) *flg = PETSC_TRUE;

  ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (n < *nmax) {
    if (!value) break;
    ierr = PetscOptionsAtod(value,dvalue++);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    n++;
  }
  ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetIntArray"
/*@C
   PetscOptionsGetIntArray - Gets an array of integer values for a particular 
   option in the database.  The values must be separated with commas with 
   no intervening spaces. 

   Not Collective

   Input Parameters:
+  pre - string to prepend to each name or PETSC_NULL
.  name - the option one is seeking
-  nmax - maximum number of values to retrieve

   Output Parameter:
+  dvalue - the integer values to return
.  nmax - actual number of values retreived
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^array of ints

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(), 
           PetscOptionsGetString(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetIntArray(const char pre[],const char name[],PetscInt dvalue[],PetscInt *nmax,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscTruth     flag;
  PetscToken     *token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag)  {if (flg) *flg = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (!value) {if (flg) *flg = PETSC_TRUE; *nmax = 0; PetscFunctionReturn(0);}

  if (flg) *flg = PETSC_TRUE;

  ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (n < *nmax) {
    if (!value) break;
    ierr      = PetscOptionsAtoi(value,dvalue);CHKERRQ(ierr);
    dvalue++;
    ierr      = PetscTokenFind(token,&value);CHKERRQ(ierr);
    n++;
  }
  ierr      = PetscTokenDestroy(token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetString"
/*@C
   PetscOptionsGetString - Gets the string value for a particular option in
   the database.

   Not Collective

   Input Parameters:
+  pre - string to prepend to name or PETSC_NULL
.  name - the option one is seeking
-  len - maximum string length

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
      call PetscOptionsGetString(PETSC_NULL_CHARACTER,'-s',string,flg,ierr)
.ve

   Concepts: options database^string

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetString(const char pre[],const char name[],char string[],size_t len,PetscTruth *flg)
{
  char       *value;
  PetscErrorCode ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidCharPointer(string,3);
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr); 
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

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetStringArray"
/*@C
   PetscOptionsGetStringArray - Gets an array of string values for a particular
   option in the database. The values must be separated with commas with 
   no intervening spaces. 

   Not Collective

   Input Parameters:
+  pre - string to prepend to name or PETSC_NULL
.  name - the option one is seeking
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

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsLogical(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsLogicalGroupBegin(), PetscOptionsLogicalGroup(), PetscOptionsLogicalGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetStringArray(const char pre[],const char name[],char *strings[],PetscInt *nmax,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n;
  PetscTruth     flag;
  PetscToken     *token;
 
  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidPointer(strings,3);
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr); 
  if (!flag)  {*nmax = 0; if (flg) *flg = PETSC_FALSE; PetscFunctionReturn(0);}
  if (!value) {*nmax = 0; if (flg) *flg = PETSC_FALSE;PetscFunctionReturn(0);}
  if (!*nmax) {if (flg) *flg = PETSC_FALSE;PetscFunctionReturn(0);}
  if (flg) *flg = PETSC_TRUE;

  ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  n = 0;
  while (n < *nmax) {
    if (!value) break;
    ierr = PetscStrallocpy(value,&strings[n]);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    n++;
  }
  ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsAllUsed"
/*@C
   PetscOptionsAllUsed - Returns a count of the number of options in the 
   database that have never been selected.

   Not Collective

   Output Parameter:
.   N - count of options not used

   Level: advanced

.seealso: PetscOptionsPrint()
@*/
PetscErrorCode PetscOptionsAllUsed(int *N)
{
  int  i,n = 0;

  PetscFunctionBegin;
  for (i=0; i<options->N; i++) {
    if (!options->used[i]) { n++; }
  }
  *N = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsLeft"
/*@
    PetscOptionsLeft - Prints to screen any options that were set and never used.

  Not collective

   Options Database Key:
.  -options_left - Activates OptionsAllUsed() within PetscFinalize()

  Level: advanced

.seealso: PetscOptionsAllUsed()
@*/
PetscErrorCode PetscOptionsLeft(void)
{
  PetscErrorCode ierr;
  int        i;

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
    PetscOptionsCreate - Creates the empty options database.

*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsCreate"
PetscErrorCode PetscOptionsCreate(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options = (PetscOptionsTable*)malloc(sizeof(PetscOptionsTable));
  ierr    = PetscMemzero(options->used,MAXOPTIONS*sizeof(int));CHKERRQ(ierr);
  options->namegiven = PETSC_FALSE;
  options->N         = 0;
  options->Naliases  = 0;
  PetscFunctionReturn(0);
}
