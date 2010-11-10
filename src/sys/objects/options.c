#define PETSC_DLL

/* Feature test macros to make sure atoll is available (SVr4, POSIX.1-2001, 4.3BSD, C99), not in (C89 and POSIX.1-1996) */
#define _XOPEN_SOURCE 600
#define _BSD_SOURCE
/*
   These routines simplify the use of command line, file options, etc., and are used to manipulate the options database.
   This provides the low-level interface, the high level interface is in aoptions.c

   Some routines use regular malloc and free because it cannot know  what malloc is requested with the 
   options database until it has already processed the input.
*/

#include "petscsys.h"        /*I  "petscsys.h"   I*/
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#if defined(PETSC_HAVE_SYS_PARAM_H)
#include "sys/param.h"
#endif

/* 
    This table holds all the options set by the user. For simplicity, we use a static size database
*/
#define MAXOPTIONS 512
#define MAXALIASES 25
#define MAXOPTIONSMONITORS 5

typedef struct {
  int            N,argc,Naliases;
  char           **args,*names[MAXOPTIONS],*values[MAXOPTIONS];
  char           *aliases1[MAXALIASES],*aliases2[MAXALIASES];
  PetscTruth     used[MAXOPTIONS];
  PetscTruth     namegiven;
  char           programname[PETSC_MAX_PATH_LEN]; /* HP includes entire path in name */

  /* --------User (or default) routines (most return -1 on error) --------*/
  PetscErrorCode (*monitor[MAXOPTIONSMONITORS])(const char[], const char[], void*); /* returns control to user after */
  PetscErrorCode (*monitordestroy[MAXOPTIONSMONITORS])(void*);         /* */
  void           *monitorcontext[MAXOPTIONSMONITORS];                  /* to pass arbitrary user data into monitor */
  PetscInt       numbermonitors;                                       /* to, for instance, detect options being set */

} PetscOptionsTable;


static PetscOptionsTable      *options = 0;
extern PetscOptionsObjectType PetscOptionsObject;

/*
    Options events monitor
*/
#define PetscOptionsMonitor(name,value)                                     \
        { PetscErrorCode _ierr; PetscInt _i,_im = options->numbermonitors; \
          for (_i=0; _i<_im; _i++) {\
            _ierr = (*options->monitor[_i])(name, value, options->monitorcontext[_i]);CHKERRQ(_ierr); \
	  } \
	}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsAtoi"
/*
   PetscOptionsAtoi - Converts a string to an integer value. Handles special cases such as "default" and "decide"
*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsAtoi(const char name[],PetscInt *a)
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
#if defined(PETSC_USE_64BIT_INDICES)
    *a = atoll(name);
#else
    *a = atoi(name);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsAtod"
/*
   Converts a string to PetscReal value. Handles special cases like "default" and "decide"
*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsAtod(const char name[],PetscReal *a)
{
  PetscErrorCode ierr;
  size_t         len;
  PetscTruth     decide,tdefault;

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
#define __FUNCT__ "PetscOptionsAtol"
/*
   PetscOptionsAtol - Converts string to PetscTruth, handles cases like "yes", "no", "true", "false", "0", "1"
*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsAtol(const char value[], PetscTruth *a)
{
  PetscTruth     istrue, isfalse;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(value, &len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_ERR_ARG_WRONG, "Character string of length zero has no logical value");
  ierr = PetscStrcasecmp(value,"TRUE",&istrue);CHKERRQ(ierr);
  if (istrue) {*a = PETSC_TRUE; PetscFunctionReturn(0);}
  ierr = PetscStrcasecmp(value,"YES",&istrue);CHKERRQ(ierr);
  if (istrue) {*a = PETSC_TRUE; PetscFunctionReturn(0);}
  ierr = PetscStrcasecmp(value,"1",&istrue);CHKERRQ(ierr);
  if (istrue) {*a = PETSC_TRUE; PetscFunctionReturn(0);}
  ierr = PetscStrcasecmp(value,"on",&istrue);CHKERRQ(ierr);
  if (istrue) {*a = PETSC_TRUE; PetscFunctionReturn(0);}
  ierr = PetscStrcasecmp(value,"FALSE",&isfalse);CHKERRQ(ierr);
  if (isfalse) {*a = PETSC_FALSE; PetscFunctionReturn(0);}
  ierr = PetscStrcasecmp(value,"NO",&isfalse);CHKERRQ(ierr);
  if (isfalse) {*a = PETSC_FALSE; PetscFunctionReturn(0);}
  ierr = PetscStrcasecmp(value,"0",&isfalse);CHKERRQ(ierr);
  if (isfalse) {*a = PETSC_FALSE; PetscFunctionReturn(0);}
  ierr = PetscStrcasecmp(value,"off",&isfalse);CHKERRQ(ierr);
  if (isfalse) {*a = PETSC_FALSE; PetscFunctionReturn(0);}
  SETERRQ1(PETSC_ERR_ARG_WRONG, "Unknown logical value: %s", value);
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
PetscErrorCode PETSC_DLLEXPORT PetscGetProgramName(char name[],size_t len)
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
PetscErrorCode PETSC_DLLEXPORT PetscSetProgramName(const char name[])
{ 
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->namegiven = PETSC_TRUE;
  ierr  = PetscStrncpy(options->programname,name,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsValidKey"
/*@
    PetscOptionsValidKey - PETSc Options database keys must begin with a - followed by a letter.

   Input Parameter:
.    in_str - string to check if valid

   Output Parameter:
.    key - PETSC_TRUE if a valid key

  Level: intermediate

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsValidKey(const char in_str[],PetscTruth *key)
{
  PetscFunctionBegin;
  *key = PETSC_FALSE;
  if (!in_str) PetscFunctionReturn(0);
  if (in_str[0] != '-') PetscFunctionReturn(0);
  if ((in_str[1] < 'A') || (in_str[1] > 'z')) PetscFunctionReturn(0);
  *key = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsInsertString"
/*@C
     PetscOptionsInsertString - Inserts options into the database from a string

     Not collective: but only processes that call this routine will set the options
                     included in the file

  Input Parameter:
.   in_str - string that contains options separated by blanks


  Level: intermediate

  Contributed by Boyana Norris

.seealso: PetscOptionsSetValue(), PetscOptionsPrint(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsInsertFile()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsInsertString(const char in_str[])
{
  char           *first,*second;
  PetscErrorCode ierr;
  PetscToken     token;
  PetscTruth     key;

  PetscFunctionBegin;
  ierr = PetscTokenCreate(in_str,' ',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
  while (first) {
    ierr = PetscOptionsValidKey(first,&key);CHKERRQ(ierr);
    if (key) {
      ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
      ierr = PetscOptionsValidKey(second,&key);CHKERRQ(ierr);
      if (!key) {
        ierr = PetscOptionsSetValue(first,second);CHKERRQ(ierr);
        ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);        
      } else {
        ierr  = PetscOptionsSetValue(first,PETSC_NULL);CHKERRQ(ierr);
        first = second;
      }
    } else {
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);        
    }
  }
  ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Returns a line (ended by a \n, \r or null character of any length. Result should be freed with free()
*/
static char *Petscgetline(FILE * f)
{
  size_t size = 0;
  size_t len  = 0;
  size_t last = 0;
  char * buf  = PETSC_NULL;

  if (feof(f)) return 0;
  do {
    size += 1024; /* BUFSIZ is defined as "the optimal read size for this platform" */
    buf = (char*)realloc((void *)buf,size); /* realloc(NULL,n) is the same as malloc(n) */            
    /* Actually do the read. Note that fgets puts a terminal '\0' on the
    end of the string, so we make sure we overwrite this */
    if (!fgets(buf+len,size,f)) buf[len]=0;
    PetscStrlen(buf,&len);
    last = len - 1;
  } while (!feof(f) && buf[last] != '\n' && buf[last] != '\r');
  if (len) return buf;
  free(buf); 
  return 0;
}


#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsInsertFile"
/*@C
     PetscOptionsInsertFile - Inserts options into the database from a file.

     Collective on MPI_Comm

  Input Parameter:
+   comm - the processes that will share the options (usually PETSC_COMM_WORLD)
.   file - name of file
-   require - if PETSC_TRUE will generate an error if the file does not exist


  Level: intermediate

.seealso: PetscOptionsSetValue(), PetscOptionsPrint(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsInsertFile(MPI_Comm comm,const char file[],PetscTruth require)
{
  char           *string,fname[PETSC_MAX_PATH_LEN],*first,*second,*third,*vstring = 0,*astring = 0;
  PetscErrorCode ierr;
  size_t         i,len;
  FILE           *fd;
  PetscToken     token;
  int            err;
  char           cmt[3]={'#','!','%'},*cmatch;
  PetscMPIInt    rank,cnt=0,acnt=0;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    /* Warning: assume a maximum size for all options in a string */
    ierr = PetscMalloc(128000*sizeof(char),&vstring);CHKERRQ(ierr);
    vstring[0] = 0;
    ierr = PetscMalloc(64000*sizeof(char),&astring);CHKERRQ(ierr);
    astring[0] = 0;
    cnt     = 0;
    acnt    = 0;

    ierr = PetscFixFilename(file,fname);CHKERRQ(ierr);
    fd   = fopen(fname,"r"); 
    if (fd) {
      /* the following line will not work when opening initial files (like .petscrc) since info is not yet set */
      ierr = PetscInfo1(0,"Opened options file %s\n",file);CHKERRQ(ierr);
      while ((string = Petscgetline(fd))) {
	/* eliminate comments from each line */
	for (i=0; i<3; i++){
	  ierr = PetscStrchr(string,cmt[i],&cmatch);
	  if (cmatch) *cmatch = 0;
	}
	ierr = PetscStrlen(string,&len);CHKERRQ(ierr);
	/* replace tabs, ^M, \n with " " */
	for (i=0; i<len; i++) {
	  if (string[i] == '\t' || string[i] == '\r' || string[i] == '\n') {
	    string[i] = ' ';
	  }
	}
	ierr = PetscTokenCreate(string,' ',&token);CHKERRQ(ierr);
        free(string);
	ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
	if (!first) {
	  goto destroy;
	} else if (!first[0]) { /* if first token is empty spaces, redo first token */
	  ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
	}
	ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
	if (!first) {
	  goto destroy;
	} else if (first[0] == '-') {
          /* warning: should be making sure we do not overfill vstring */
          ierr = PetscStrcat(vstring,first);CHKERRQ(ierr);
          ierr = PetscStrcat(vstring," ");CHKERRQ(ierr);
          if (second) {
            /* protect second with quotes in case it contains strings */
            ierr = PetscStrcat(vstring,"\"");CHKERRQ(ierr);
	    ierr = PetscStrcat(vstring,second);CHKERRQ(ierr);
            ierr = PetscStrcat(vstring,"\"");CHKERRQ(ierr);
          }
          ierr = PetscStrcat(vstring," ");CHKERRQ(ierr);
	} else {
	  PetscTruth match;

	  ierr = PetscStrcasecmp(first,"alias",&match);CHKERRQ(ierr);
	  if (match) {
	    ierr = PetscTokenFind(token,&third);CHKERRQ(ierr);
	    if (!third) SETERRQ1(PETSC_ERR_ARG_WRONG,"Error in options file:alias missing (%s)",second);
            ierr = PetscStrcat(astring,second);CHKERRQ(ierr);
            ierr = PetscStrcat(astring," ");CHKERRQ(ierr);
            ierr = PetscStrcat(astring,third);CHKERRQ(ierr);
            ierr = PetscStrcat(astring," ");CHKERRQ(ierr);
	  } else {
	    SETERRQ1(PETSC_ERR_ARG_WRONG,"Unknown statement in options file: (%s)",string);
	  }
	}
        destroy:
	ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
      }
      err = fclose(fd);
      if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
      ierr = PetscStrlen(astring,&len);CHKERRQ(ierr);
      acnt = PetscMPIIntCast(len);CHKERRQ(ierr);
      ierr = PetscStrlen(vstring,&len);CHKERRQ(ierr);
      cnt  = PetscMPIIntCast(len);CHKERRQ(ierr);
    } else if (require) {
      SETERRQ1(PETSC_ERR_USER,"Unable to open Options File %s",fname);
    }
  }

  ierr = MPI_Bcast(&acnt,1,MPI_INT,0,comm);CHKERRQ(ierr);
  if (acnt) {
    PetscToken token;
    char       *first,*second;

    if (rank) {
      ierr = PetscMalloc((acnt+1)*sizeof(char),&astring);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(astring,acnt,MPI_CHAR,0,comm);CHKERRQ(ierr);
    astring[acnt] = 0;
    ierr = PetscTokenCreate(astring,' ',&token);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    while (first) {
      ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
      ierr = PetscOptionsSetAlias(first,second);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    }
    ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
  }

  ierr = MPI_Bcast(&cnt,1,MPI_INT,0,comm);CHKERRQ(ierr);
  if (cnt) {
    if (rank) {
      ierr = PetscMalloc((cnt+1)*sizeof(char),&vstring);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(vstring,cnt,MPI_CHAR,0,comm);CHKERRQ(ierr);
    vstring[cnt] = 0;
    ierr = PetscOptionsInsertString(vstring);CHKERRQ(ierr);
  }
  ierr = PetscFree(astring);CHKERRQ(ierr);
  ierr = PetscFree(vstring);CHKERRQ(ierr);
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

   Options Database Keys:
+   -options_monitor <optional filename> - print options names and values as they are set

   Level: advanced

   Concepts: options database^adding

.seealso: PetscOptionsDestroy_Private(), PetscOptionsPrint(), PetscOptionsInsertString(), PetscOptionsInsertFile(),
          PetscInitialize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsInsert(int *argc,char ***args,const char file[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           pfile[PETSC_MAX_PATH_LEN];
  PetscTruth     flag = PETSC_FALSE;

  PetscFunctionBegin;
  if (options == PETSC_NULL) {
    fprintf(stderr, "Options have not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    MPI_Abort(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  options->argc     = (argc) ? *argc : 0;
  options->args     = (args) ? *args : PETSC_NULL;

  if (file) {
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,file,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-skip_petscrc",&flag,PETSC_NULL);CHKERRQ(ierr);
  if (!flag) {
    ierr = PetscGetHomeDirectory(pfile,PETSC_MAX_PATH_LEN-16);CHKERRQ(ierr);
    /* warning: assumes all processes have a home directory or none, but nothing in between */
    if (pfile[0]) {
      ierr = PetscStrcat(pfile,"/.petscrc");CHKERRQ(ierr);
      ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,pfile,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,".petscrc",PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,"petscrc",PETSC_FALSE);CHKERRQ(ierr);
  }

  /* insert environmental options */
  {
    char   *eoptions = 0;
    size_t len = 0;
    if (!rank) {
      eoptions = (char*)getenv("PETSC_OPTIONS");
      ierr     = PetscStrlen(eoptions,&len);CHKERRQ(ierr);
      ierr     = MPI_Bcast(&len,1,MPIU_SIZE_T,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    } else {
      ierr = MPI_Bcast(&len,1,MPIU_SIZE_T,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      if (len) {
        ierr = PetscMalloc((len+1)*sizeof(char*),&eoptions);CHKERRQ(ierr);
      }
    }
    if (len) {
      ierr = MPI_Bcast(eoptions,len,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      if (rank) eoptions[len] = 0;
      ierr = PetscOptionsInsertString(eoptions);CHKERRQ(ierr);
      if (rank) {ierr = PetscFree(eoptions);CHKERRQ(ierr);}
    }
  }

  /* insert command line options */
  if (argc && args && *argc) {
    int        left    = *argc - 1;
    char       **eargs = *args + 1;
    PetscTruth isoptions_file,isp4,tisp4,isp4yourname,isp4rmrank;

    while (left) {
      ierr = PetscStrcasecmp(eargs[0],"-options_file",&isoptions_file);CHKERRQ(ierr);
      ierr = PetscStrcasecmp(eargs[0],"-p4pg",&isp4);CHKERRQ(ierr);
      ierr = PetscStrcasecmp(eargs[0],"-p4yourname",&isp4yourname);CHKERRQ(ierr);
      ierr = PetscStrcasecmp(eargs[0],"-p4rmrank",&isp4rmrank);CHKERRQ(ierr);
      ierr = PetscStrcasecmp(eargs[0],"-p4wd",&tisp4);CHKERRQ(ierr);
      isp4 = (PetscTruth) (isp4 || tisp4);
      ierr = PetscStrcasecmp(eargs[0],"-np",&tisp4);CHKERRQ(ierr);
      isp4 = (PetscTruth) (isp4 || tisp4);
      ierr = PetscStrcasecmp(eargs[0],"-p4amslave",&tisp4);CHKERRQ(ierr);

      if (eargs[0][0] != '-') {
        eargs++; left--;
      } else if (isoptions_file) {
        if (left <= 1) SETERRQ(PETSC_ERR_USER,"Missing filename for -options_file filename option");
        if (eargs[1][0] == '-') SETERRQ(PETSC_ERR_USER,"Missing filename for -options_file filename option");
        ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,eargs[1],PETSC_TRUE);CHKERRQ(ierr);
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
PetscErrorCode PETSC_DLLEXPORT PetscOptionsPrint(FILE *fd)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!fd) fd = PETSC_STDOUT;
  if (!options) {ierr = PetscOptionsInsert(0,0,0);CHKERRQ(ierr);}
  if (options->N) {
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"#PETSc Option Table entries:\n");CHKERRQ(ierr);
  } else {
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"#No PETSc Option Table entries\n");CHKERRQ(ierr);
  }
  for (i=0; i<options->N; i++) {
    if (options->values[i]) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"-%s %s\n",options->names[i],options->values[i]);CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"-%s\n",options->names[i]);CHKERRQ(ierr);
    }
  }
  if (options->N) {
    ierr = PetscFPrintf(PETSC_COMM_WORLD,fd,"#End of PETSc Option Table entries\n");CHKERRQ(ierr);
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

   Notes: the array and each entry in the array should be freed with PetscFree()

   Level: advanced

   Concepts: options database^listing

.seealso: PetscOptionsAllUsed(), PetscOptionsPrint()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetAll(char *copts[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  size_t         len = 1,lent;
  char           *coptions;

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
#define __FUNCT__ "PetscOptionsClear"
/*@C
    PetscOptionsClear - Removes all options form the database leaving it empty.

   Level: developer

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsClear(void)
{
  PetscInt i;

  PetscFunctionBegin;
  if (!options) PetscFunctionReturn(0);
  for (i=0; i<options->N; i++) {
    if (options->names[i])  free(options->names[i]);
    if (options->values[i]) free(options->values[i]);
  }
  for (i=0; i<options->Naliases; i++) {
    free(options->aliases1[i]);
    free(options->aliases2[i]);
  }
  options->N        = 0;
  options->Naliases = 0;
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
PetscErrorCode PETSC_DLLEXPORT PetscOptionsDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!options) PetscFunctionReturn(0);
  ierr = PetscOptionsClear();CHKERRQ(ierr);
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
PetscErrorCode PETSC_DLLEXPORT PetscOptionsSetValue(const char iname[],const char value[])
{
  size_t         len;
  PetscErrorCode ierr;
  PetscInt       N,n,i;
  char           **names;
  const char     *name = (char*)iname;
  PetscTruth     gt,match;

  PetscFunctionBegin;
  if (!options) {ierr = PetscOptionsInsert(0,0,0);CHKERRQ(ierr);}

  /* this is so that -h and -hel\p are equivalent (p4 does not like -help)*/
  ierr = PetscStrcasecmp(name,"-h",&match);CHKERRQ(ierr);
  if (match) name = "-help";

  name++;
  /* first check against aliases */
  N = options->Naliases; 
  for (i=0; i<N; i++) {
    ierr = PetscStrcasecmp(options->aliases1[i],name,&match);CHKERRQ(ierr);
    if (match) {
      name = options->aliases2[i];
      break;
    }
  }

  N     = options->N;
  n     = N;
  names = options->names; 
 
  for (i=0; i<N; i++) {
    ierr = PetscStrcasecmp(names[i],name,&match);CHKERRQ(ierr);
    ierr  = PetscStrgrt(names[i],name,&gt);CHKERRQ(ierr);
    if (match) {
      if (options->values[i]) free(options->values[i]);
      ierr = PetscStrlen(value,&len);CHKERRQ(ierr);
      if (len) {
        options->values[i] = (char*)malloc((len+1)*sizeof(char));
        ierr = PetscStrcpy(options->values[i],value);CHKERRQ(ierr);
      } else { options->values[i] = 0;}
      PetscOptionsMonitor(name,value);
      PetscFunctionReturn(0);
    } else if (gt) {
      n = i;
      break;
    }
  }
  if (N >= MAXOPTIONS) {
    SETERRQ1(PETSC_ERR_PLIB,"No more room in option table, limit %d recompile \n src/sys/objects/options.c with larger value for MAXOPTIONS\n",MAXOPTIONS);
  }
  /* shift remaining values down 1 */
  for (i=N; i>n; i--) {
    options->names[i]  = options->names[i-1];
    options->values[i] = options->values[i-1];
    options->used[i]   = options->used[i-1];
  }
  /* insert new name and value */
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  options->names[n] = (char*)malloc((len+1)*sizeof(char));
  ierr = PetscStrcpy(options->names[n],name);CHKERRQ(ierr);
  ierr = PetscStrlen(value,&len);CHKERRQ(ierr);
  if (len) {
    options->values[n] = (char*)malloc((len+1)*sizeof(char));
    ierr = PetscStrcpy(options->values[n],value);CHKERRQ(ierr);
  } else {options->values[n] = 0;}
  options->used[n] = PETSC_FALSE;
  options->N++;
  PetscOptionsMonitor(name,value);
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
PetscErrorCode PETSC_DLLEXPORT PetscOptionsClearValue(const char iname[])
{
  PetscErrorCode ierr;
  PetscInt       N,n,i;
  char           **names,*name=(char*)iname;
  PetscTruth     gt,match;

  PetscFunctionBegin;
  if (name[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"Name must begin with -: Instead %s",name);
  if (!options) {ierr = PetscOptionsInsert(0,0,0);CHKERRQ(ierr);}

  name++;

  N     = options->N; n = 0;
  names = options->names; 
 
  for (i=0; i<N; i++) {
    ierr  = PetscStrcasecmp(names[i],name,&match);CHKERRQ(ierr);
    ierr  = PetscStrgrt(names[i],name,&gt);CHKERRQ(ierr);
    if (match) {
      if (options->names[i])  free(options->names[i]);
      if (options->values[i]) free(options->values[i]);
      PetscOptionsMonitor(name,"");
      break;
    } else if (gt) {
      PetscFunctionReturn(0); /* it was not listed */
    }
    n++;
  }
  if (n == N) PetscFunctionReturn(0); /* it was not listed */

  /* shift remaining values down 1 */
  for (i=n; i<N-1; i++) {
    options->names[i]  = options->names[i+1];
    options->values[i] = options->values[i+1];
    options->used[i]   = options->used[i+1];
  }
  options->N--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsSetAlias"
/*@C
   PetscOptionsSetAlias - Makes a key and alias for another key

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  inewname - the alias
-  ioldname - the name that alias will refer to 

   Level: advanced

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),OptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(),PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsSetAlias(const char inewname[],const char ioldname[])
{
  PetscErrorCode ierr;
  PetscInt       n = options->Naliases;
  size_t         len;
  char           *newname = (char *)inewname,*oldname = (char*)ioldname;

  PetscFunctionBegin;
  if (newname[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"aliased must have -: Instead %s",newname);
  if (oldname[0] != '-') SETERRQ1(PETSC_ERR_ARG_WRONG,"aliasee must have -: Instead %s",oldname);
  if (n >= MAXALIASES) {
    SETERRQ1(PETSC_ERR_MEM,"You have defined to many PETSc options aliases, limit %d recompile \n  src/sys/objects/options.c with larger value for MAXALIASES",MAXALIASES);
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
    if (pre[0] == '-') SETERRQ(PETSC_ERR_ARG_WRONG,"Prefix should not begin with a -");
    ierr = PetscStrncpy(tmp,pre,256);CHKERRQ(ierr);
    ierr = PetscStrlen(tmp,&len);CHKERRQ(ierr);
    ierr = PetscStrncat(tmp,name+1,256-len-1);CHKERRQ(ierr);
  } else {
    ierr = PetscStrncpy(tmp,name+1,256);CHKERRQ(ierr);
  }

  /* slow search */
  *flg = PETSC_FALSE;
  for (i=0; i<N; i++) {
    ierr = PetscStrcasecmp(names[i],tmp,&match);CHKERRQ(ierr);
    if (match) {
       *value           = options->values[i];
       options->used[i] = PETSC_TRUE;
       *flg             = PETSC_TRUE;
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
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsReject(const char name[],const char mess[])
{
  PetscErrorCode ierr;
  PetscTruth     flag = PETSC_FALSE;

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
   PetscOptionsHasName - Determines whether a certain option is given in the database. This returns true whether the option is a number, string or boolean, even 
                      its value is set to false.

   Not Collective

   Input Parameters:
+  name - the option one is seeking 
-  pre - string to prepend to the name or PETSC_NULL

   Output Parameters:
.  flg - PETSC_TRUE if found else PETSC_FALSE.

   Level: beginner

   Concepts: options database^has option name

   Notes: Name cannot be simply -h

          In many cases you probably want to use PetscOptionsGetTruth() instead of calling this, to allowing toggling values.

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsHasName(const char pre[],const char name[],PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr = PetscOptionsFindPair_Private(pre,name,&value,&flag);CHKERRQ(ierr);
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
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetInt(const char pre[],const char name[],PetscInt *ivalue,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscTruth     flag;

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
#define __FUNCT__ "PetscOptionsGetEList"
/*@C
     PetscOptionsGetEList - Puts a list of option values that a single one may be selected from

   Not Collective

   Input Parameters:
+  pre - the string to prepend to the name or PETSC_NULL
.  opt - option name
.  list - the possible choices
.  ntext - number of choices

   Output Parameter:
+  value - the index of the value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE
   
   Level: intermediate

   See PetscOptionsList() for when the choices are given in a PetscFList()

   Concepts: options database^list

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),  
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetEList(const char pre[],const char opt[],const char *const*list,PetscInt ntext,PetscInt *value,PetscTruth *set)
{
  PetscErrorCode ierr;
  size_t         alen,len = 0;
  char           *svalue;
  PetscTruth     aset,flg = PETSC_FALSE;
  PetscInt       i;

  PetscFunctionBegin;
  for ( i=0; i<ntext; i++) {
    ierr = PetscStrlen(list[i],&alen);CHKERRQ(ierr);
    if (alen > len) len = alen;
  }
  len += 5; /* a little extra space for user mistypes */
  ierr = PetscMalloc(len*sizeof(char),&svalue);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(pre,opt,svalue,len,&aset);CHKERRQ(ierr);
  if (aset) {
    if (set) *set = PETSC_TRUE;
    for (i=0; i<ntext; i++) {
      ierr = PetscStrcasecmp(svalue,list[i],&flg);CHKERRQ(ierr);
      if (flg) {
        *value = i;
        break;
      }
    }
    if (!flg) SETERRQ3(PETSC_ERR_USER,"Unknown option %s for -%s%s",svalue,pre?pre:"",opt+1);
  } else if (set) {
    *set = PETSC_FALSE;
  }
  ierr = PetscFree(svalue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsEnum"
/*@C
   PetscOptionsGetEnum - Gets the enum value for a particular option in the database.

   Not Collective

   Input Parameters:
+  pre - option prefix or PETSC_NULL
.  opt - option name
.  list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
-  defaultv - the default (current) value

   Output Parameter:
+  value - the  value to return
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          list is usually something like PCASMTypes or some other predefined list of enum names

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList(), PetscOptionsGetEList(), PetscOptionsEnum()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetEnum(const char pre[],const char opt[],const char *const*list,PetscEnum *value,PetscTruth *set)
{
  PetscErrorCode ierr;
  PetscInt       ntext = 0,tval;
  PetscTruth     fset;

  PetscFunctionBegin;
  while (list[ntext++]) {
    if (ntext > 50) SETERRQ(PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  }
  if (ntext < 3) SETERRQ(PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  ntext -= 3;
  ierr = PetscOptionsGetEList(pre,opt,list,ntext,&tval,&fset);CHKERRQ(ierr);
  /* with PETSC_USE_64BIT_INDICES sizeof(PetscInt) != sizeof(PetscEnum) */
  if (fset) *value = (PetscEnum)tval;
  if (set) *set = fset;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetTruth"
/*@C
   PetscOptionsGetTruth - Gets the Logical (true or false) value for a particular 
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

       If the user does not supply the option (as either true or false) ivalue is NOT changed. Thus
     you NEED TO ALWAYS initialize the ivalue.

   Concepts: options database^has logical

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetInt(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetTruth(const char pre[],const char name[],PetscTruth *ivalue,PetscTruth *flg)
{
  char           *value;
  PetscTruth     flag;
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
      ierr = PetscOptionsAtol(value, ivalue);CHKERRQ(ierr);
    }
  } else {
    if (flg) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetTruthArray"
/*@C
   PetscOptionsGetTruthArray - Gets an array of Logical (true or false) values for a particular 
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

   Notes:
       TRUE, true, YES, yes, nostring, and 1 all translate to PETSC_TRUE
       FALSE, false, NO, no, and 0 all translate to PETSC_FALSE

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(), 
           PetscOptionsGetString(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetTruthArray(const char pre[],const char name[],PetscTruth dvalue[],PetscInt *nmax,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscTruth     flag;
  PetscToken     token;

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
    ierr = PetscOptionsAtol(value,dvalue);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    dvalue++;
    n++;
  }
  ierr  = PetscTokenDestroy(token);CHKERRQ(ierr);
  *nmax = n;
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
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(),PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetReal(const char pre[],const char name[],PetscReal *dvalue,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscTruth     flag;

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
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetScalar(const char pre[],const char name[],PetscScalar *dvalue,PetscTruth *flg)
{
  char           *value;
  PetscTruth     flag;
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
      PetscToken token;
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
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetRealArray(const char pre[],const char name[],PetscReal dvalue[],PetscInt *nmax,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscTruth     flag;
  PetscToken     token;

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
-  nmax - maximum number of values to retrieve, can include d-D to indicate d,d+1,..,D-1

   Output Parameter:
+  dvalue - the integer values to return
.  nmax - actual number of values retreived
-  flg - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^array of ints

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(), 
           PetscOptionsGetString(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetIntArray(const char pre[],const char name[],PetscInt dvalue[],PetscInt *nmax,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0,i,start,end;
  size_t         len;
  PetscTruth     flag,foundrange;
  PetscToken     token;

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
    
    /* look for form  d-D where d and D are integers */
    foundrange = PETSC_FALSE;
    ierr      = PetscStrlen(value,&len);CHKERRQ(ierr); 
    if (value[0] == '-') i=2;
    else i=1;
    for (;i<(int)len; i++) {
      if (value[i] == '-') {
        if (i == (int)len-1) SETERRQ2(PETSC_ERR_USER,"Error in %D-th array entry %s\n",n,value);
        value[i] = 0;
        ierr     = PetscOptionsAtoi(value,&start);CHKERRQ(ierr);        
        ierr     = PetscOptionsAtoi(value+i+1,&end);CHKERRQ(ierr);        
        if (end <= start) SETERRQ3(PETSC_ERR_USER,"Error in %D-th array entry, %s-%s cannot have decreasing list",n,value,value+i+1);
        if (n + end - start - 1 >= *nmax) SETERRQ4(PETSC_ERR_USER,"Error in %D-th array entry, not enough space in left in array (%D) to contain entire range from %D to %D",n,*nmax-n,start,end);
        for (;start<end; start++) {
          *dvalue = start; dvalue++;n++;
        }
        foundrange = PETSC_TRUE;
        break;
      }
    }
    if (!foundrange) {
      ierr      = PetscOptionsAtoi(value,dvalue);CHKERRQ(ierr);
      dvalue++;
      n++;
    }
    ierr      = PetscTokenFind(token,&value);CHKERRQ(ierr);
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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetString(const char pre[],const char name[],char string[],size_t len,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscTruth     flag;

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
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsTruth(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsTruthGroupBegin(), PetscOptionsTruthGroup(), PetscOptionsTruthGroupEnd(),
          PetscOptionsList(), PetscOptionsEList()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetStringArray(const char pre[],const char name[],char *strings[],PetscInt *nmax,PetscTruth *flg)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n;
  PetscTruth     flag;
  PetscToken     token;
 
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
PetscErrorCode PETSC_DLLEXPORT PetscOptionsAllUsed(int *N)
{
  PetscInt i,n = 0;

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
PetscErrorCode PETSC_DLLEXPORT PetscOptionsLeft(void)
{
  PetscErrorCode ierr;
  PetscInt       i;

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


#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsCreate"
/*
    PetscOptionsCreate - Creates the empty options database.

*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsCreate(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options = (PetscOptionsTable*)malloc(sizeof(PetscOptionsTable));
  ierr    = PetscMemzero(options->used,MAXOPTIONS*sizeof(PetscTruth));CHKERRQ(ierr);
  options->namegiven 		= PETSC_FALSE;
  options->N         		= 0;
  options->Naliases  		= 0;
  options->numbermonitors 	= 0;

  PetscOptionsObject.prefix = PETSC_NULL;
  PetscOptionsObject.title  = PETSC_NULL;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsSetFromOptions"
/*@
   PetscOptionsSetFromOptions - Sets various SNES and KSP parameters from user options.

   Collective on PETSC_COMM_WORLD

   Options Database Keys:
+  -options_monitor <optional filename> - prints the names and values of all runtime options as they are set. The monitor functionality is not 
                available for options set through a file, environment variable, or on 
                the command line. Only options set after PetscInitialize completes will 
                be monitored.
.  -options_monitor_cancel - cancel all options database monitors    

   Notes:
   To see all options, run your program with the -help option or consult
   the users manual. 

   Level: intermediate

.keywords: set, options, database
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsSetFromOptions(void)
{
  PetscTruth          flgc,flgm;
  PetscErrorCode      ierr;
  char                monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer         monviewer; 

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options database options","PetscOptions");CHKERRQ(ierr);
    ierr = PetscOptionsString("-options_monitor","Monitor options database","PetscOptionsMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flgm);CHKERRQ(ierr);
    ierr = PetscOptionsName("-options_monitor_cancel","Cancel all options database monitors","PetscOptionsMonitorCancel",&flgc);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flgm) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,monfilename,&monviewer);CHKERRQ(ierr);
    ierr = PetscOptionsMonitorSet(PetscOptionsMonitorDefault,monviewer,(PetscErrorCode (*)(void*))PetscViewerDestroy);CHKERRQ(ierr);
  }
  if (flgc) { ierr = PetscOptionsMonitorCancel();CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsMonitorDefault"
/*@C
   PetscOptionsMonitorDefault - Print all options set value events.

   Collective on PETSC_COMM_WORLD

   Input Parameters:
+  name  - option name string
.  value - option value string
-  dummy - unused monitor context 

   Level: intermediate

.keywords: PetscOptions, default, monitor

.seealso: PetscOptionsMonitorSet()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsMonitorDefault(const char name[], const char value[], void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"Setting option: %s = %s\n",name,value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsMonitorSet"
/*@C
   PetscOptionsMonitorSet - Sets an ADDITIONAL function to be called at every method that
   modified the PETSc options database.
      
   Not collective

   Input Parameters:
+  monitor - pointer to function (if this is PETSC_NULL, it turns off monitoring
.  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be PETSC_NULL)

   Calling Sequence of monitor:
$     monitor (const char name[], const char value[], void *mctx)

+  name - option name string
.  value - option value string
-  mctx  - optional monitoring context, as set by PetscOptionsMonitorSet()

   Options Database Keys:
+    -options_monitor    - sets PetscOptionsMonitorDefault()
-    -options_monitor_cancel - cancels all monitors that have
                          been hardwired into a code by 
                          calls to PetscOptionsMonitorSet(), but
                          does not cancel those set via
                          the options database.

   Notes:  
   The default is to do nothing.  To print the name and value of options 
   being inserted into the database, use PetscOptionsMonitorDefault() as the monitoring routine, 
   with a null monitoring context. 

   Several different monitoring routines may be set by calling
   PetscOptionsMonitorSet() multiple times; all will be called in the 
   order in which they were set.

   Level: beginner

.keywords: PetscOptions, set, monitor

.seealso: PetscOptionsMonitorDefault(), PetscOptionsMonitorCancel()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsMonitorSet(PetscErrorCode (*monitor)(const char name[], const char value[], void*),void *mctx,PetscErrorCode (*monitordestroy)(void*))
{
  PetscFunctionBegin;
  if (options->numbermonitors >= MAXOPTIONSMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscOptions monitors set");
  }
  options->monitor[options->numbermonitors]           = monitor;
  options->monitordestroy[options->numbermonitors]    = monitordestroy;
  options->monitorcontext[options->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsMonitorCancel"
/*@
   PetscOptionsMonitorCancel - Clears all monitors for a PetscOptions object.

   Not collective 

   Options Database Key:
.  -options_monitor_cancel - Cancels all monitors that have
    been hardwired into a code by calls to PetscOptionsMonitorSet(), 
    but does not cancel those set via the options database.

   Level: intermediate

.keywords: PetscOptions, set, monitor

.seealso: PetscOptionsMonitorDefault(), PetscOptionsMonitorSet()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsMonitorCancel(void)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<options->numbermonitors; i++) {
    if (options->monitordestroy[i]) {
      ierr = (*options->monitordestroy[i])(options->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  options->numbermonitors = 0;
  PetscFunctionReturn(0);
}
