
/* Define Feature test macros to make sure atoll is available (SVr4, POSIX.1-2001, 4.3BSD, C99), not in (C89 and POSIX.1-1996) */
#define PETSC_DESIRE_FEATURE_TEST_MACROS

/*
   These routines simplify the use of command line, file options, etc., and are used to manipulate the options database.
   This provides the low-level interface, the high level interface is in aoptions.c

   Some routines use regular malloc and free because it cannot know  what malloc is requested with the
   options database until it has already processed the input.
*/

#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/
#include <petscviewer.h>
#include <ctype.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#if defined(PETSC_HAVE_STRING_H)
#include <string.h>             /* strstr */
#endif
#if defined(PETSC_HAVE_STRINGS_H)
#  include <strings.h>          /* strcasecmp */
#endif
#if defined(PETSC_HAVE_YAML)
#include <yaml.h>
#endif

/*
    This table holds all the options set by the user. For simplicity, we use a static size database
*/
#define MAXOPTIONS 512
#define MAXALIASES 25
#define MAXOPTIONSMONITORS 5
#define MAXPREFIXES 25

struct  _n_PetscOptions {
  int            N,argc,Naliases;
  char           **args,*names[MAXOPTIONS],*values[MAXOPTIONS];
  char           *aliases1[MAXALIASES],*aliases2[MAXALIASES];
  PetscBool      used[MAXOPTIONS];
  PetscBool      namegiven;
  char           programname[PETSC_MAX_PATH_LEN]; /* HP includes entire path in name */

  /* --------User (or default) routines (most return -1 on error) --------*/
  PetscErrorCode (*monitor[MAXOPTIONSMONITORS])(const char[], const char[], void*); /* returns control to user after */
  PetscErrorCode (*monitordestroy[MAXOPTIONSMONITORS])(void**);         /* */
  void           *monitorcontext[MAXOPTIONSMONITORS];                  /* to pass arbitrary user data into monitor */
  PetscInt       numbermonitors;                                       /* to, for instance, detect options being set */

  /* Prefixes */
  PetscInt prefixind,prefixstack[MAXPREFIXES];
  char     prefix[2048];
};


static PetscOptions      defaultoptions = NULL;

/*
    Options events monitor
*/
#define PetscOptionsMonitor(name,value)                              \
  { PetscErrorCode _ierr; PetscInt _i,_im = options->numbermonitors; \
    for (_i=0; _i<_im; _i++) { \
      _ierr = (*options->monitor[_i])(name, value, options->monitorcontext[_i]);CHKERRQ(_ierr); \
    } \
  }

/*
   PetscOptionsStringToInt - Converts a string to an integer value. Handles special cases such as "default" and "decide"
*/
PetscErrorCode  PetscOptionsStringToInt(const char name[],PetscInt *a)
{
  PetscErrorCode ierr;
  size_t         len;
  PetscBool      decide,tdefault,mouse;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"character string of length zero has no numerical value");

  ierr = PetscStrcasecmp(name,"PETSC_DEFAULT",&tdefault);CHKERRQ(ierr);
  if (!tdefault) {
    ierr = PetscStrcasecmp(name,"DEFAULT",&tdefault);CHKERRQ(ierr);
  }
  ierr = PetscStrcasecmp(name,"PETSC_DECIDE",&decide);CHKERRQ(ierr);
  if (!decide) {
    ierr = PetscStrcasecmp(name,"DECIDE",&decide);CHKERRQ(ierr);
  }
  ierr = PetscStrcasecmp(name,"mouse",&mouse);CHKERRQ(ierr);

  if (tdefault)    *a = PETSC_DEFAULT;
  else if (decide) *a = PETSC_DECIDE;
  else if (mouse)  *a = -1;
  else {
    char *endptr;
    long strtolval;

    strtolval = strtol(name,&endptr,10);
    if ((size_t) (endptr - name) != len) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Input string %s has no integer value (do not include . in it)",name);

#if defined(PETSC_USE_64BIT_INDICES) && defined(PETSC_HAVE_ATOLL)
    (void) strtolval;
    *a = atoll(name);
#elif defined(PETSC_USE_64BIT_INDICES) && defined(PETSC_HAVE___INT64)
    (void) strtolval;
    *a = _atoi64(name);
#else
    *a = (PetscInt)strtolval;
#endif
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_REAL___FLOAT128)
#include <quadmath.h>
#endif

static PetscErrorCode PetscStrtod(const char name[],PetscReal *a,char **endptr)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_REAL___FLOAT128)
  *a = strtoflt128(name,endptr);
#else
  *a = (PetscReal)strtod(name,endptr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStrtoz(const char name[],PetscScalar *a,char **endptr,PetscBool *isImaginary)
{
  PetscBool      hasi = PETSC_FALSE;
  char           *ptr;
  PetscReal      strtoval;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrtod(name,&strtoval,&ptr);CHKERRQ(ierr);
  if (ptr == name) {
    strtoval = 1.;
    hasi = PETSC_TRUE;
    if (name[0] == 'i') {
      ptr++;
    } else if (name[0] == '+' && name[1] == 'i') {
      ptr += 2;
    } else if (name[0] == '-' && name[1] == 'i') {
      strtoval = -1.;
      ptr += 2;
    }
  } else if (*ptr == 'i') {
    hasi = PETSC_TRUE;
    ptr++;
  }
  *endptr = ptr;
  *isImaginary = hasi;
  if (hasi) {
#if !defined(PETSC_USE_COMPLEX)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Input string %s contains imaginary but complex not supported ",name);
#else
    *a = PetscCMPLX(0.,strtoval);
#endif
  } else {
    *a = strtoval;
  }
  PetscFunctionReturn(0);
}

/*
   Converts a string to PetscReal value. Handles special cases like "default" and "decide"
*/
PetscErrorCode  PetscOptionsStringToReal(const char name[],PetscReal *a)
{
  PetscErrorCode ierr;
  size_t         len;
  PetscBool      decide,tdefault;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"character string of length zero has no numerical value");

  ierr = PetscStrcasecmp(name,"PETSC_DEFAULT",&tdefault);CHKERRQ(ierr);
  if (!tdefault) {
    ierr = PetscStrcasecmp(name,"DEFAULT",&tdefault);CHKERRQ(ierr);
  }
  ierr = PetscStrcasecmp(name,"PETSC_DECIDE",&decide);CHKERRQ(ierr);
  if (!decide) {
    ierr = PetscStrcasecmp(name,"DECIDE",&decide);CHKERRQ(ierr);
  }

  if (tdefault)    *a = PETSC_DEFAULT;
  else if (decide) *a = PETSC_DECIDE;
  else {
    char   *endptr;

    ierr = PetscStrtod(name,a,&endptr);CHKERRQ(ierr);
    if ((size_t) (endptr - name) != len) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Input string %s has no numeric value ",name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscOptionsStringToScalar(const char name[],PetscScalar *a)
{
  PetscBool      imag1;
  size_t         len;
  PetscScalar    val = 0.;
  char           *ptr = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"character string of length zero has no numerical value");
  ierr = PetscStrtoz(name,&val,&ptr,&imag1);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  if ((size_t) (ptr - name) < len) {
    PetscBool   imag2;
    PetscScalar val2;

    ierr = PetscStrtoz(ptr,&val2,&ptr,&imag2);CHKERRQ(ierr);
    if (imag1 || !imag2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Input string %s: must specify imaginary component second",name);
    val = PetscCMPLX(PetscRealPart(val),PetscImaginaryPart(val2));
  }
#endif
  if ((size_t) (ptr - name) != len) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Input string %s has no numeric value ",name);
  *a = val;
  PetscFunctionReturn(0);
}

/*
   PetscOptionsStringToBool - Converts string to PetscBool , handles cases like "yes", "no", "true", "false", "0", "1"
*/
PetscErrorCode  PetscOptionsStringToBool(const char value[], PetscBool  *a)
{
  PetscBool      istrue, isfalse;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(value, &len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Character string of length zero has no logical value");
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
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Unknown logical value: %s", value);
}

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
PetscErrorCode  PetscGetProgramName(char name[],size_t len)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
   ierr = PetscStrncpy(name,defaultoptions->programname,len);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscSetProgramName(const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr  = PetscStrncpy(defaultoptions->programname,name,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PetscOptionsValidKey - PETSc Options database keys must begin with one or two dashes (-) followed by a letter.

   Input Parameter:
.    in_str - string to check if valid

   Output Parameter:
.    key - PETSC_TRUE if a valid key

  Level: intermediate

@*/
PetscErrorCode  PetscOptionsValidKey(const char in_str[],PetscBool  *key)
{
  char           *ptr;

  PetscFunctionBegin;
  *key = PETSC_FALSE;
  if (!in_str) PetscFunctionReturn(0);
  if (in_str[0] != '-') PetscFunctionReturn(0);
  if (in_str[1] == '-') in_str++;
  if (!isalpha((int)(in_str[1]))) PetscFunctionReturn(0);
  (void) strtod(in_str,&ptr);
  if (ptr != in_str && !(*ptr == '_' || isalnum((int)*ptr))) PetscFunctionReturn(0);
  *key = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
     PetscOptionsInsertString - Inserts options into the database from a string

     Not collective: but only processes that call this routine will set the options
                     included in the string

  Input Parameter:
.   in_str - string that contains options separated by blanks


  Level: intermediate

  Contributed by Boyana Norris

.seealso: PetscOptionsSetValue(), PetscOptionsView(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsInsertFile()

@*/
PetscErrorCode  PetscOptionsInsertString(PetscOptions options,const char in_str[])
{
  char           *first,*second;
  PetscErrorCode ierr;
  PetscToken     token;
  PetscBool      key,ispush,ispop;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  ierr = PetscTokenCreate(in_str,' ',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
  while (first) {
    ierr = PetscStrcasecmp(first,"-prefix_push",&ispush);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(first,"-prefix_pop",&ispop);CHKERRQ(ierr);
    ierr = PetscOptionsValidKey(first,&key);CHKERRQ(ierr);
    if (ispush) {
      ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
      ierr = PetscOptionsPrefixPush(options,second);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    } else if (ispop) {
      ierr = PetscOptionsPrefixPop(options);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    } else if (key) {
      ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
      ierr = PetscOptionsValidKey(second,&key);CHKERRQ(ierr);
      if (!key) {
        ierr = PetscOptionsSetValue(options,first,second);CHKERRQ(ierr);
        ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
      } else {
        ierr  = PetscOptionsSetValue(options,first,NULL);CHKERRQ(ierr);
        first = second;
      }
    } else {
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    }
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Returns a line (ended by a \n, \r or null character of any length. Result should be freed with free()
*/
static char *Petscgetline(FILE * f)
{
  size_t size  = 0;
  size_t len   = 0;
  size_t last  = 0;
  char   *buf  = NULL;

  if (feof(f)) return 0;
  do {
    size += 1024; /* BUFSIZ is defined as "the optimal read size for this platform" */
    buf   = (char*)realloc((void*)buf,size); /* realloc(NULL,n) is the same as malloc(n) */
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


/*@C
     PetscOptionsInsertFile - Inserts options into the database from a file.

     Collective on MPI_Comm

  Input Parameter:
+   comm - the processes that will share the options (usually PETSC_COMM_WORLD)
.   options - options database, use NULL for default global database
.   file - name of file
-   require - if PETSC_TRUE will generate an error if the file does not exist


  Notes: Use  # for lines that are comments and which should be ignored.

   Usually, instead of using this command, one should list the file name in the call to PetscInitialize(), this insures that certain options
   such as -log_view or -malloc_debug are processed properly. This routine only sets options into the options database that will be processed by later
   calls to XXXSetFromOptions() it should not be used for options listed under PetscInitialize().

  Level: developer

.seealso: PetscOptionsSetValue(), PetscOptionsView(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()

@*/
PetscErrorCode  PetscOptionsInsertFile(MPI_Comm comm,PetscOptions options,const char file[],PetscBool require)
{
  char           *string,fname[PETSC_MAX_PATH_LEN],*first,*second,*third,*vstring = 0,*astring = 0,*packed = 0;
  PetscErrorCode ierr;
  size_t         i,len,bytes;
  FILE           *fd;
  PetscToken     token;
  int            err;
  char           cmt[1]={'#'},*cmatch;
  PetscMPIInt    rank,cnt=0,acnt=0,counts[2];
  PetscBool      isdir;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    cnt        = 0;
    acnt       = 0;

    ierr = PetscFixFilename(file,fname);CHKERRQ(ierr);
    fd   = fopen(fname,"r");
    ierr = PetscTestDirectory(fname,'r',&isdir);CHKERRQ(ierr);
    if (isdir && require) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Specified options file %s is a directory",fname);
    if (fd && !isdir) {
      PetscSegBuffer vseg,aseg;
      ierr = PetscSegBufferCreate(1,4000,&vseg);CHKERRQ(ierr);
      ierr = PetscSegBufferCreate(1,2000,&aseg);CHKERRQ(ierr);

      /* the following line will not work when opening initial files (like .petscrc) since info is not yet set */
      ierr = PetscInfo1(0,"Opened options file %s\n",file);CHKERRQ(ierr);

      while ((string = Petscgetline(fd))) {
        /* eliminate comments from each line */
        for (i=0; i<1; i++) {
          ierr = PetscStrchr(string,cmt[i],&cmatch);CHKERRQ(ierr);
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
          ierr = PetscStrlen(first,&len);CHKERRQ(ierr);
          ierr = PetscSegBufferGet(vseg,len+1,&vstring);CHKERRQ(ierr);
          ierr = PetscMemcpy(vstring,first,len);CHKERRQ(ierr);
          vstring[len] = ' ';
          if (second) {
            ierr = PetscStrlen(second,&len);CHKERRQ(ierr);
            ierr = PetscSegBufferGet(vseg,len+3,&vstring);CHKERRQ(ierr);
            vstring[0] = '"';
            ierr = PetscMemcpy(vstring+1,second,len);CHKERRQ(ierr);
            vstring[len+1] = '"';
            vstring[len+2] = ' ';
          }
        } else {
          PetscBool match;

          ierr = PetscStrcasecmp(first,"alias",&match);CHKERRQ(ierr);
          if (match) {
            ierr = PetscTokenFind(token,&third);CHKERRQ(ierr);
            if (!third) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in options file:alias missing (%s)",second);
            ierr = PetscStrlen(second,&len);CHKERRQ(ierr);
            ierr = PetscSegBufferGet(aseg,len+1,&astring);CHKERRQ(ierr);
            ierr = PetscMemcpy(astring,second,len);CHKERRQ(ierr);
            astring[len] = ' ';

            ierr = PetscStrlen(third,&len);CHKERRQ(ierr);
            ierr = PetscSegBufferGet(aseg,len+1,&astring);CHKERRQ(ierr);
            ierr = PetscMemcpy(astring,third,len);CHKERRQ(ierr);
            astring[len] = ' ';
          } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown statement in options file: (%s)",string);
        }
destroy:
        free(string);
        ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
      }
      err = fclose(fd);
      if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
      ierr = PetscSegBufferGetSize(aseg,&bytes);CHKERRQ(ierr); /* size without null termination */
      ierr = PetscMPIIntCast(bytes,&acnt);CHKERRQ(ierr);
      ierr = PetscSegBufferGet(aseg,1,&astring);CHKERRQ(ierr);
      astring[0] = 0;
      ierr = PetscSegBufferGetSize(vseg,&bytes);CHKERRQ(ierr); /* size without null termination */
      ierr = PetscMPIIntCast(bytes,&cnt);CHKERRQ(ierr);
      ierr = PetscSegBufferGet(vseg,1,&vstring);CHKERRQ(ierr);
      vstring[0] = 0;
      ierr = PetscMalloc1(2+acnt+cnt,&packed);CHKERRQ(ierr);
      ierr = PetscSegBufferExtractTo(aseg,packed);CHKERRQ(ierr);
      ierr = PetscSegBufferExtractTo(vseg,packed+acnt+1);CHKERRQ(ierr);
      ierr = PetscSegBufferDestroy(&aseg);CHKERRQ(ierr);
      ierr = PetscSegBufferDestroy(&vseg);CHKERRQ(ierr);
    } else if (require) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unable to open Options File %s",fname);
  }

  counts[0] = acnt;
  counts[1] = cnt;
  ierr = MPI_Bcast(counts,2,MPI_INT,0,comm);CHKERRQ(ierr);
  acnt = counts[0];
  cnt = counts[1];
  if (rank) {
    ierr = PetscMalloc1(2+acnt+cnt,&packed);CHKERRQ(ierr);
  }
  if (acnt || cnt) {
    ierr = MPI_Bcast(packed,2+acnt+cnt,MPI_CHAR,0,comm);CHKERRQ(ierr);
    astring = packed;
    vstring = packed + acnt + 1;
  }

  if (acnt) {
    PetscToken token;
    char       *first,*second;

    ierr = PetscTokenCreate(astring,' ',&token);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    while (first) {
      ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
      ierr = PetscOptionsSetAlias(options,first,second);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    }
    ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  }

  if (cnt) {
    ierr = PetscOptionsInsertString(options,vstring);CHKERRQ(ierr);
  }
  ierr = PetscFree(packed);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscOptionsInsertArgs_Private(PetscOptions options,int argc,char *args[])
{
  PetscErrorCode ierr;
  int            left    = argc - 1;
  char           **eargs = args + 1;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  while (left) {
    PetscBool isoptions_file,isprefixpush,isprefixpop,isp4,tisp4,isp4yourname,isp4rmrank,key;
    ierr = PetscStrcasecmp(eargs[0],"-options_file",&isoptions_file);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(eargs[0],"-prefix_push",&isprefixpush);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(eargs[0],"-prefix_pop",&isprefixpop);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(eargs[0],"-p4pg",&isp4);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(eargs[0],"-p4yourname",&isp4yourname);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(eargs[0],"-p4rmrank",&isp4rmrank);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(eargs[0],"-p4wd",&tisp4);CHKERRQ(ierr);
    isp4 = (PetscBool) (isp4 || tisp4);
    ierr = PetscStrcasecmp(eargs[0],"-np",&tisp4);CHKERRQ(ierr);
    isp4 = (PetscBool) (isp4 || tisp4);
    ierr = PetscStrcasecmp(eargs[0],"-p4amslave",&tisp4);CHKERRQ(ierr);
    ierr = PetscOptionsValidKey(eargs[0],&key);CHKERRQ(ierr);

    if (!key) {
      eargs++; left--;
    } else if (isoptions_file) {
      if (left <= 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing filename for -options_file filename option");
      if (eargs[1][0] == '-') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing filename for -options_file filename option");
      ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,eargs[1],PETSC_TRUE);CHKERRQ(ierr);
      eargs += 2; left -= 2;
    } else if (isprefixpush) {
      if (left <= 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing prefix for -prefix_push option");
      if (eargs[1][0] == '-') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Missing prefix for -prefix_push option (prefixes cannot start with '-')");
      ierr = PetscOptionsPrefixPush(options,eargs[1]);CHKERRQ(ierr);
      eargs += 2; left -= 2;
    } else if (isprefixpop) {
      ierr = PetscOptionsPrefixPop(options);CHKERRQ(ierr);
      eargs++; left--;

      /*
       These are "bad" options that MPICH, etc put on the command line
       we strip them out here.
       */
    } else if (tisp4 || isp4rmrank) {
      eargs += 1; left -= 1;
    } else if (isp4 || isp4yourname) {
      eargs += 2; left -= 2;
    } else {
      PetscBool nextiskey = PETSC_FALSE;
      if (left >= 2) {ierr = PetscOptionsValidKey(eargs[1],&nextiskey);CHKERRQ(ierr);}
      if (left < 2 || nextiskey) {
        ierr = PetscOptionsSetValue(options,eargs[0],NULL);CHKERRQ(ierr);
        eargs++; left--;
      } else {
        ierr = PetscOptionsSetValue(options,eargs[0],eargs[1]);CHKERRQ(ierr);
        eargs += 2; left -= 2;
      }
    }
  }
  PetscFunctionReturn(0);
}


/*@C
   PetscOptionsInsert - Inserts into the options database from the command line,
                   the environmental variable and a file.

   Input Parameters:
+  options - options database or NULL for the default global database
.  argc - count of number of command line arguments
.  args - the command line arguments
-  file - optional filename, defaults to ~username/.petscrc

   Note:
   Since PetscOptionsInsert() is automatically called by PetscInitialize(),
   the user does not typically need to call this routine. PetscOptionsInsert()
   can be called several times, adding additional entries into the database.

   Options Database Keys:
+   -options_monitor <optional filename> - print options names and values as they are set
.   -options_file <filename> - read options from a file

   Level: advanced

   Concepts: options database^adding

.seealso: PetscOptionsDestroy_Private(), PetscOptionsView(), PetscOptionsInsertString(), PetscOptionsInsertFile(),
          PetscInitialize()
@*/
PetscErrorCode  PetscOptionsInsert(PetscOptions options,int *argc,char ***args,const char file[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           pfile[PETSC_MAX_PATH_LEN];
  PetscBool      flag = PETSC_FALSE;

  
  PetscFunctionBegin;
  if (!options) {
    if (!defaultoptions) {
      ierr = PetscOptionsCreateDefault();CHKERRQ(ierr);
    }
    options = defaultoptions;
  }
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  options->argc = (argc) ? *argc : 0;
  options->args = (args) ? *args : NULL;

  if (file && file[0]) {
    char fullpath[PETSC_MAX_PATH_LEN];

    ierr = PetscStrreplace(PETSC_COMM_WORLD,file,fullpath,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,fullpath,PETSC_TRUE);CHKERRQ(ierr);
  }
  /*
     We want to be able to give -skip_petscrc on the command line, but need to parse it first.  Since the command line
     should take precedence, we insert it twice.  It would be sufficient to just scan for -skip_petscrc.
  */
  if (argc && args && *argc) {ierr = PetscOptionsInsertArgs_Private(options,*argc,*args);CHKERRQ(ierr);}
  ierr = PetscOptionsGetBool(NULL,NULL,"-skip_petscrc",&flag,NULL);CHKERRQ(ierr);
  if (!flag) {
    ierr = PetscGetHomeDirectory(pfile,PETSC_MAX_PATH_LEN-16);CHKERRQ(ierr);
    /* PetscOptionsInsertFile() does a fopen() on rank0 only - so only rank0 HomeDir value is relavent */
    if (pfile[0]) { ierr = PetscStrcat(pfile,"/.petscrc");CHKERRQ(ierr); }
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,pfile,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,".petscrc",PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,"petscrc",PETSC_FALSE);CHKERRQ(ierr);
  }

  /* insert environmental options */
  {
    char   *eoptions = 0;
    size_t len       = 0;
    if (!rank) {
      eoptions = (char*)getenv("PETSC_OPTIONS");
      ierr     = PetscStrlen(eoptions,&len);CHKERRQ(ierr);
      ierr     = MPI_Bcast(&len,1,MPIU_SIZE_T,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    } else {
      ierr = MPI_Bcast(&len,1,MPIU_SIZE_T,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      if (len) {
        ierr = PetscMalloc1(len+1,&eoptions);CHKERRQ(ierr);
      }
    }
    if (len) {
      ierr = MPI_Bcast(eoptions,len,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      if (rank) eoptions[len] = 0;
      ierr = PetscOptionsInsertString(options,eoptions);CHKERRQ(ierr);
      if (rank) {ierr = PetscFree(eoptions);CHKERRQ(ierr);}
    }
  }

#if defined(PETSC_HAVE_YAML)
  {
    char      yaml_file[PETSC_MAX_PATH_LEN];
    PetscBool yaml_flg;
    ierr = PetscOptionsGetString(NULL,NULL,"-options_file_yaml",yaml_file,PETSC_MAX_PATH_LEN,&yaml_flg);CHKERRQ(ierr);
    if (yaml_flg) {
      ierr = PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,yaml_file,PETSC_TRUE);CHKERRQ(ierr);
    }
  }
#endif

  /* insert command line options again because they take precedence over arguments in petscrc/environment */
  if (argc && args && *argc) {ierr = PetscOptionsInsertArgs_Private(options,*argc,*args);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsView - Prints the options that have been loaded. This is
   useful for debugging purposes.

   Logically Collective on PetscViewer

   Input Parameter:
-  options - options database, use NULL for default global database
+  viewer - must be an PETSCVIEWERASCII viewer

   Options Database Key:
.  -options_table - Activates PetscOptionsView() within PetscFinalize()

   Level: advanced

   Concepts: options database^printing

.seealso: PetscOptionsAllUsed()
@*/
PetscErrorCode  PetscOptionsView(PetscOptions options,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      isascii;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Only supports ASCII viewer");

  if (options->N) {
    ierr = PetscViewerASCIIPrintf(viewer,"#PETSc Option Table entries:\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"#No PETSc Option Table entries\n");CHKERRQ(ierr);
  }
  for (i=0; i<options->N; i++) {
    if (options->values[i]) {
      ierr = PetscViewerASCIIPrintf(viewer,"-%s %s\n",options->names[i],options->values[i]);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"-%s\n",options->names[i]);CHKERRQ(ierr);
    }
  }
  if (options->N) {
    ierr = PetscViewerASCIIPrintf(viewer,"#End of PETSc Option Table entries\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   Called by error handlers to print options used in run
*/
PetscErrorCode  PetscOptionsViewError(void)
{
  PetscInt       i;
  PetscOptions   options = defaultoptions;

  PetscFunctionBegin;
  if (options->N) {
    (*PetscErrorPrintf)("PETSc Option Table entries:\n");
  } else {
    (*PetscErrorPrintf)("No PETSc Option Table entries\n");
  }
  for (i=0; i<options->N; i++) {
    if (options->values[i]) {
      (*PetscErrorPrintf)("-%s %s\n",options->names[i],options->values[i]);
    } else {
      (*PetscErrorPrintf)("-%s\n",options->names[i]);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetAll - Lists all the options the program was run with in a single string.

   Not Collective

   Input Paramter:
.  options - the options database, use NULL for the default global database

   Output Parameter:
.  copts - pointer where string pointer is stored

   Notes: the array and each entry in the array should be freed with PetscFree()

   Level: advanced

   Concepts: options database^listing

.seealso: PetscOptionsAllUsed(), PetscOptionsView()
@*/
PetscErrorCode  PetscOptionsGetAll(PetscOptions options,char *copts[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  size_t         len       = 1,lent = 0;
  char           *coptions = NULL;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;

  /* count the length of the required string */
  for (i=0; i<options->N; i++) {
    ierr = PetscStrlen(options->names[i],&lent);CHKERRQ(ierr);
    len += 2 + lent;
    if (options->values[i]) {
      ierr = PetscStrlen(options->values[i],&lent);CHKERRQ(ierr);
      len += 1 + lent;
    }
  }
  ierr = PetscMalloc1(len,&coptions);CHKERRQ(ierr);
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

/*@C
   PetscOptionsPrefixPush - Designate a prefix to be used by all options insertions to follow.

   Not Collective, but prefix will only be applied on calling ranks

   Input Parameter:
+  options - options database, or NULL for the default global database
-  prefix - The string to append to the existing prefix

   Options Database Keys:
 +   -prefix_push <some_prefix_> - push the given prefix
 -   -prefix_pop - pop the last prefix

   Notes:
   It is common to use this in conjunction with -options_file as in

 $ -prefix_push system1_ -options_file system1rc -prefix_pop -prefix_push system2_ -options_file system2rc -prefix_pop

   where the files no longer require all options to be prefixed with -system2_.

Level: advanced

.seealso: PetscOptionsPrefixPop()
@*/
PetscErrorCode  PetscOptionsPrefixPush(PetscOptions options,const char prefix[])
{
  PetscErrorCode ierr;
  size_t         n;
  PetscInt       start;
  char           buf[2048];
  PetscBool      key;

  PetscFunctionBegin;
  PetscValidCharPointer(prefix,1);
  options = options ? options : defaultoptions;
  /* Want to check validity of the key using PetscOptionsValidKey(), which requires that the first character is a '-' */
  buf[0] = '-';
  ierr = PetscStrncpy(buf+1,prefix,sizeof(buf) - 1);CHKERRQ(ierr);
  buf[sizeof(buf) - 1] = 0;
  ierr = PetscOptionsValidKey(buf,&key);CHKERRQ(ierr);
  if (!key) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Given prefix \"%s\" not valid (the first character must be a letter, do not include leading '-')",prefix);

  if (!options) {
    ierr = PetscOptionsInsert(NULL,0,0,0);CHKERRQ(ierr);
    options = defaultoptions;
  }
  if (options->prefixind >= MAXPREFIXES) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Maximum depth of prefix stack %d exceeded, recompile \n src/sys/objects/options.c with larger value for MAXPREFIXES",MAXPREFIXES);
  start = options->prefixind ? options->prefixstack[options->prefixind-1] : 0;
  ierr = PetscStrlen(prefix,&n);CHKERRQ(ierr);
  if (n+1 > sizeof(options->prefix)-start) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Maximum prefix length %d exceeded",sizeof(options->prefix));
  ierr = PetscMemcpy(options->prefix+start,prefix,n+1);CHKERRQ(ierr);
  options->prefixstack[options->prefixind++] = start+n;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsPrefixPop - Remove the latest options prefix, see PetscOptionsPrefixPush() for details

   Not  Collective, but prefix will only be popped on calling ranks

  Input Parameters:
.  options - options database, or NULL for the default global database

   Level: advanced

.seealso: PetscOptionsPrefixPush()
@*/
PetscErrorCode  PetscOptionsPrefixPop(PetscOptions options)
{
  PetscInt offset;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  if (options->prefixind < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"More prefixes popped than pushed");
  options->prefixind--;
  offset = options->prefixind ? options->prefixstack[options->prefixind-1] : 0;
  options->prefix[offset] = 0;
  PetscFunctionReturn(0);
}

/*@C
    PetscOptionsClear - Removes all options form the database leaving it empty.

  Input Parameters:
.  options - options database, use NULL for the default global database

   Level: developer

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode  PetscOptionsClear(PetscOptions options)
{
  PetscInt     i;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  for (i=0; i<options->N; i++) {
    if (options->names[i])  free(options->names[i]);
    if (options->values[i]) free(options->values[i]);
  }
  for (i=0; i<options->Naliases; i++) {
    free(options->aliases1[i]);
    free(options->aliases2[i]);
  }
  options->prefix[0] = 0;
  options->prefixind = 0;
  options->N         = 0;
  options->Naliases  = 0;
  PetscFunctionReturn(0);
}

/*@
    PetscObjectSetPrintedOptions - indicate to an object that it should behave as if it has already printed the help for its options

  Input Parameters:
.  obj  - the PetscObject

   Level: developer

   Developer Notes: This is used, for example to prevent sequential objects that are created from a parallel object; such as the KSP created by 
    PCBJACOBI from all printing the same help messages to the screen

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode  PetscObjectSetPrintedOptions(PetscObject obj)
{
  PetscFunctionBegin;
  obj->optionsprinted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
    PetscObjectInheritPrintedOptions - If the child object is not on the rank 0 process of the parent object and the child is sequential then the child gets it set.

  Input Parameters:
+  pobj - the parent object
-  obj  - the PetscObject

   Level: developer

   Developer Notes: This is used, for example to prevent sequential objects that are created from a parallel object; such as the KSP created by 
    PCBJACOBI from all printing the same help messages to the screen

    This will not handle more complicated situations like with GASM where children may live on any subset of the parent's processes and overlap

.seealso: PetscOptionsInsert(), PetscObjectSetPrintedOptions()
@*/
PetscErrorCode  PetscObjectInheritPrintedOptions(PetscObject pobj,PetscObject obj)
{
  PetscErrorCode ierr;
  PetscMPIInt    prank,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(pobj->comm,&prank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(obj->comm,&size);CHKERRQ(ierr);
  if (size == 1 && prank > 0) obj->optionsprinted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
    PetscOptionsDestroy - Destroys an option database.

  Input Parameter:
.  options - the PetscOptions object

   Level: developer

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode  PetscOptionsDestroy(PetscOptions *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsClear(*options);CHKERRQ(ierr);
  free(*options);
  *options = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscOptionsDestroyDefault(void)
{
  PetscErrorCode ierr;

  ierr = PetscOptionsDestroy(&defaultoptions);if (ierr) return ierr;
  return 0;
}


/*@C
   PetscOptionsSetValue - Sets an option name-value pair in the options
   database, overriding whatever is already present.

   Not collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  options - options database, use NULL for the default global database
.  name - name of option, this SHOULD have the - prepended
-  value - the option value (not used for all options)

   Level: intermediate

   Note:
   This function can be called BEFORE PetscInitialize()

   Only some options have values associated with them, such as
   -ksp_rtol tol.  Other options stand alone, such as -ksp_monitor.

  Developers Note: Uses malloc() directly because PETSc may not yet have been fully initialized

  Concepts: options database^adding option

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode  PetscOptionsSetValue(PetscOptions options,const char iname[],const char value[])
{
  size_t         len;
  PetscErrorCode ierr;
  PetscInt       N,n,i;
  char           **names;
  char           fullname[2048];
  const char     *name = iname;
  int            match;

  if (!options) {
    if (!defaultoptions) {
      ierr = PetscOptionsCreateDefault();
      if (ierr) return ierr;
    }
    options = defaultoptions;
  }

  /* this is so that -h and -help are equivalent (p4 does not like -help)*/
  match = strcmp(name,"-h");
  if (!match) name = "-help";

  name++; /* skip starting hyphen */
  if (options->prefixind > 0) {
    strncpy(fullname,options->prefix,sizeof(fullname));
    strncat(fullname,name,sizeof(fullname)-strlen(fullname)-1);
    name = fullname;
  }

  /* check against aliases */
  N = options->Naliases;
  for (i=0; i<N; i++) {
#if defined(PETSC_HAVE_STRCASECMP)
    match = strcasecmp(options->aliases1[i],name);
#elif defined(PETSC_HAVE_STRICMP)
    match = stricmp(options->aliases1[i],name);
#else
    Error
#endif
    if (!match) {
      name = options->aliases2[i];
      break;
    }
  }

  N     = options->N;
  n     = N;
  names = options->names;

  for (i=0; i<N; i++) {
#if defined(PETSC_HAVE_STRCASECMP)
    match = strcasecmp(names[i],name);
#elif defined(PETSC_HAVE_STRICMP)
    match = stricmp(names[i],name);
#else
    Error
#endif
    if (!match) {
      if (options->values[i]) free(options->values[i]);
      len = value ? strlen(value) : 0;
      if (len) {
        options->values[i] = (char*)malloc((len+1)*sizeof(char));
        if (!options->values[i]) return PETSC_ERR_MEM;
        strcpy(options->values[i],value);
      } else options->values[i] = 0;
      return 0;
    } else if (strcmp(names[i],name) > 0) {
      n = i;
      break;
    }
  }
  if (N >= MAXOPTIONS) abort();

  /* shift remaining values down 1 */
  for (i=N; i>n; i--) {
    options->names[i]  = options->names[i-1];
    options->values[i] = options->values[i-1];
    options->used[i]   = options->used[i-1];
  }
  /* insert new name and value */
  len = strlen(name);
  options->names[n] = (char*)malloc((len+1)*sizeof(char));
  if (!options->names[n]) return PETSC_ERR_MEM;
  strcpy(options->names[n],name);
  len = value ? strlen(value) : 0;
  if (len) {
    options->values[n] = (char*)malloc((len+1)*sizeof(char));
    if (!options->values[n]) return PETSC_ERR_MEM;
    strcpy(options->values[n],value);
  } else options->values[n] = NULL;
  options->used[n] = PETSC_FALSE;
  options->N++;
  return 0;
}

/*@C
   PetscOptionsClearValue - Clears an option name-value pair in the options
   database, overriding whatever is already present.

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameter:
+  options - options database, use NULL for the default global database
.  name - name of option, this SHOULD have the - prepended

   Level: intermediate

   Concepts: options database^removing option
.seealso: PetscOptionsInsert()
@*/
PetscErrorCode  PetscOptionsClearValue(PetscOptions options,const char iname[])
{
  PetscErrorCode ierr;
  PetscInt       N,n,i;
  char           **names,*name=(char*)iname;
  PetscBool      gt,match;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  if (name[0] != '-') SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Name must begin with -: Instead %s",name);
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
    } else if (gt) PetscFunctionReturn(0); /* it was not listed */

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

/*@C
   PetscOptionsSetAlias - Makes a key and alias for another key

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  options - options database or NULL for default global database
.  inewname - the alias
-  ioldname - the name that alias will refer to

   Level: advanced

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),OptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(),PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsSetAlias(PetscOptions options,const char inewname[],const char ioldname[])
{
  PetscErrorCode ierr;
  PetscInt       n = options->Naliases;
  size_t         len;
  char           *newname = (char*)inewname,*oldname = (char*)ioldname;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  if (newname[0] != '-') SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"aliased must have -: Instead %s",newname);
  if (oldname[0] != '-') SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"aliasee must have -: Instead %s",oldname);
  if (n >= MAXALIASES) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEM,"You have defined to many PETSc options aliases, limit %d recompile \n  src/sys/objects/options.c with larger value for MAXALIASES",MAXALIASES);

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

PetscErrorCode PetscOptionsFindPair_Private(PetscOptions options,const char pre[],const char name[],char *value[],PetscBool  *flg)
{
  PetscErrorCode ierr;
  PetscInt       i,N;
  size_t         len;
  char           **names,tmp[256];
  PetscBool      match;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  N     = options->N;
  names = options->names;

  if (name[0] != '-') SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Name must begin with -: Instead %s",name);

  /* append prefix to name, if prefix="foo_" and option='--bar", prefixed option is --foo_bar */
  if (pre) {
    char       *ptr   = tmp;
    const char *namep = name;
    if (pre[0] == '-') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Prefix should not begin with a -");
    if (name[1] == '-') {
      *ptr++ = '-';
      namep++;
    }
    ierr = PetscStrncpy(ptr,pre,tmp+sizeof(tmp)-ptr);CHKERRQ(ierr);
    tmp[sizeof(tmp)-1] = 0;
    ierr = PetscStrlen(tmp,&len);CHKERRQ(ierr);
    ierr = PetscStrncat(tmp,namep+1,sizeof(tmp)-len-1);CHKERRQ(ierr);
  } else {
    ierr = PetscStrncpy(tmp,name+1,sizeof(tmp));CHKERRQ(ierr);
    tmp[sizeof(tmp)-1] = 0;
  }
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool valid;
    char      key[sizeof(tmp)+1] = "-";

    ierr = PetscMemcpy(key+1,tmp,sizeof(tmp));CHKERRQ(ierr);
    ierr = PetscOptionsValidKey(key,&valid);CHKERRQ(ierr);
    if (!valid) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid option '%s' obtained from pre='%s' and name='%s'",key,pre?pre:"",name);
  }
#endif

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
        ierr = PetscOptionsFindPair_Private(options,NULL,tmp2,value,flg);CHKERRQ(ierr);
        if (*flg) break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscOptionsFindPairPrefix_Private(PetscOptions options,const char pre[], const char name[], char *value[], PetscBool *flg)
{
  PetscErrorCode ierr;
  PetscInt       i,N;
  size_t         len;
  char           **names,tmp[256];
  PetscBool      match;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  N     = options->N;
  names = options->names;

  if (name[0] != '-') SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Name must begin with -: Instead %s",name);

  /* append prefix to name, if prefix="foo_" and option='--bar", prefixed option is --foo_bar */
  if (pre) {
    char       *ptr   = tmp;
    const char *namep = name;
    if (pre[0] == '-') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Prefix should not begin with a -");
    if (name[1] == '-') {
      *ptr++ = '-';
      namep++;
    }
    ierr = PetscStrncpy(ptr,pre,tmp+sizeof(tmp)-ptr);CHKERRQ(ierr);
    tmp[sizeof(tmp)-1] = 0;
    ierr = PetscStrlen(tmp,&len);CHKERRQ(ierr);
    ierr = PetscStrncat(tmp,namep+1,sizeof(tmp)-len-1);CHKERRQ(ierr);
  } else {
    ierr = PetscStrncpy(tmp,name+1,sizeof(tmp));CHKERRQ(ierr);
    tmp[sizeof(tmp)-1] = 0;
  }
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool valid;
    char      key[sizeof(tmp)+1] = "-";

    ierr = PetscMemcpy(key+1,tmp,sizeof(tmp));CHKERRQ(ierr);
    ierr = PetscOptionsValidKey(key,&valid);CHKERRQ(ierr);
    if (!valid) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid option '%s' obtained from pre='%s' and name='%s'",key,pre?pre:"",name);
  }
#endif

  /* slow search */
  *flg = PETSC_FALSE;
  ierr = PetscStrlen(tmp,&len);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    ierr = PetscStrncmp(names[i], tmp, len, &match);CHKERRQ(ierr);
    if (match) {
      if (value) *value = options->values[i];
      options->used[i]  = PETSC_TRUE;
      if (flg)   *flg   = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsReject - Generates an error if a certain option is given.

   Not Collective, but setting values on certain processors could cause problems
   for parallel objects looking for options.

   Input Parameters:
+  options - options database, use NULL for default global database
.  name - the option one is seeking
-  mess - error message (may be NULL)

   Level: advanced

   Concepts: options database^rejecting option

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),OptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsReject(PetscOptions options,const char name[],const char mess[])
{
  PetscErrorCode ierr;
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  ierr = PetscOptionsHasName(options,NULL,name,&flag);CHKERRQ(ierr);
  if (flag) {
    if (mess) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Program has disabled option: %s with %s",name,mess);
    else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Program has disabled option: %s",name);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsHasName - Determines whether a certain option is given in the database. This returns true whether the option is a number, string or boolean, even
                      its value is set to false.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  name - the option one is seeking
-  pre - string to prepend to the name or NULL

   Output Parameters:
.  set - PETSC_TRUE if found else PETSC_FALSE.

   Level: beginner

   Concepts: options database^has option name

   Notes: Name cannot be simply -h

          In many cases you probably want to use PetscOptionsGetBool() instead of calling this, to allowing toggling values.

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsHasName(PetscOptions options,const char pre[],const char name[],PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (set) *set = flag;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetInt - Gets the integer value for a particular option in the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - the string to prepend to the name or NULL
-  name - the option one is seeking

   Output Parameter:
+  ivalue - the integer value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^has int

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetInt(PetscOptions options,const char pre[],const char name[],PetscInt *ivalue,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(ivalue,3);
  options = options ? options : defaultoptions;
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {
      if (set) *set = PETSC_FALSE;
    } else {
      if (set) *set = PETSC_TRUE;
      ierr = PetscOptionsStringToInt(value,ivalue);CHKERRQ(ierr);
    }
  } else {
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscOptionsGetEList - Puts a list of option values that a single one may be selected from

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - the string to prepend to the name or NULL
.  opt - option name
.  list - the possible choices (one of these must be selected, anything else is invalid)
.  ntext - number of choices

   Output Parameter:
+  value - the index of the value to return (defaults to zero if the option name is given but choice is listed)
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   See PetscOptionsFList() for when the choices are given in a PetscFunctionList()

   Concepts: options database^list

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetEList(PetscOptions options,const char pre[],const char opt[],const char * const *list,PetscInt ntext,PetscInt *value,PetscBool  *set)
{
  PetscErrorCode ierr;
  size_t         alen,len = 0;
  char           *svalue;
  PetscBool      aset,flg = PETSC_FALSE;
  PetscInt       i;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  for (i=0; i<ntext; i++) {
    ierr = PetscStrlen(list[i],&alen);CHKERRQ(ierr);
    if (alen > len) len = alen;
  }
  len += 5; /* a little extra space for user mistypes */
  ierr = PetscMalloc1(len,&svalue);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(options,pre,opt,svalue,len,&aset);CHKERRQ(ierr);
  if (aset) {
    ierr = PetscEListFind(ntext,list,svalue,value,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown option %s for -%s%s",svalue,pre ? pre : "",opt+1);
    if (set) *set = PETSC_TRUE;
  } else if (set) *set = PETSC_FALSE;
  ierr = PetscFree(svalue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetEnum - Gets the enum value for a particular option in the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - option prefix or NULL
.  opt - option name
.  list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
-  defaultv - the default (current) value

   Output Parameter:
+  value - the  value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database

   Notes: Must be between a PetscOptionsBegin() and a PetscOptionsEnd()

          list is usually something like PCASMTypes or some other predefined list of enum names

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsGetEList(), PetscOptionsEnum()
@*/
PetscErrorCode  PetscOptionsGetEnum(PetscOptions options,const char pre[],const char opt[],const char * const *list,PetscEnum *value,PetscBool  *set)
{
  PetscErrorCode ierr;
  PetscInt       ntext = 0,tval;
  PetscBool      fset;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  while (list[ntext++]) {
    if (ntext > 50) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  }
  if (ntext < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  ntext -= 3;
  ierr = PetscOptionsGetEList(options,pre,opt,list,ntext,&tval,&fset);CHKERRQ(ierr);
  /* with PETSC_USE_64BIT_INDICES sizeof(PetscInt) != sizeof(PetscEnum) */
  if (fset) *value = (PetscEnum)tval;
  if (set) *set = fset;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetBool - Gets the Logical (true or false) value for a particular
            option in the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - the string to prepend to the name or NULL
-  name - the option one is seeking

   Output Parameter:
+  ivalue - the logical value to return
-  set - PETSC_TRUE  if found, else PETSC_FALSE

   Level: beginner

   Notes:
       TRUE, true, YES, yes, nostring, and 1 all translate to PETSC_TRUE
       FALSE, false, NO, no, and 0 all translate to PETSC_FALSE

       If the user does not supply the option (as either true or false) ivalue is NOT changed. Thus
     you NEED TO ALWAYS initialize the ivalue.

   Concepts: options database^has logical

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetInt(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetBool(PetscOptions options,const char pre[],const char name[],PetscBool  *ivalue,PetscBool  *set)
{
  char           *value;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  if (ivalue) PetscValidIntPointer(ivalue,3);
  options = options ? options : defaultoptions;
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (set) *set = PETSC_TRUE;
    if (!value) {
      if (ivalue) *ivalue = PETSC_TRUE;
    } else {
      ierr = PetscOptionsStringToBool(value, ivalue);CHKERRQ(ierr);
    }
  } else {
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetBoolArray - Gets an array of Logical (true or false) values for a particular
   option in the database.  The values must be separated with commas with
   no intervening spaces.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to each name or NULL
.  name - the option one is seeking
-  nmax - maximum number of values to retrieve

   Output Parameter:
+  dvalue - the integer values to return
.  nmax - actual number of values retreived
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^array of ints

   Notes:
       TRUE, true, YES, yes, nostring, and 1 all translate to PETSC_TRUE
       FALSE, false, NO, no, and 0 all translate to PETSC_FALSE

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetBoolArray(PetscOptions options,const char pre[],const char name[],PetscBool dvalue[],PetscInt *nmax,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscBool      flag;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag)  {if (set) *set = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (!value) {if (set) *set = PETSC_TRUE;  *nmax = 0; PetscFunctionReturn(0);}

  if (set) *set = PETSC_TRUE;

  ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (n < *nmax) {
    if (!value) break;
    ierr = PetscOptionsStringToBool(value,dvalue);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    dvalue++;
    n++;
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetReal - Gets the double precision value for a particular
   option in the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to each name or NULL
-  name - the option one is seeking

   Output Parameter:
+  dvalue - the double value to return
-  set - PETSC_TRUE if found, PETSC_FALSE if not found

   Note: if the option is given but no value is provided then set is given the value PETSC_FALSE

   Level: beginner

   Concepts: options database^has double

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(),PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetReal(PetscOptions options,const char pre[],const char name[],PetscReal *dvalue,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidRealPointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {
      if (set) *set = PETSC_FALSE;
    } else {
      if (set) *set = PETSC_TRUE;
      ierr = PetscOptionsStringToReal(value,dvalue);CHKERRQ(ierr);
    }
  } else {
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetScalar - Gets the scalar value for a particular
   option in the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to each name or NULL
-  name - the option one is seeking

   Output Parameter:
+  dvalue - the double value to return
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Usage:
   A complex number 2+3i must be specified with NO spaces

   Note: if the option is given but no value is provided then set is given the value PETSC_FALSE

   Concepts: options database^has scalar

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetScalar(PetscOptions options,const char pre[],const char name[],PetscScalar *dvalue,PetscBool  *set)
{
  char           *value;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidScalarPointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (!value) {
      if (set) *set = PETSC_FALSE;
    } else {
#if !defined(PETSC_USE_COMPLEX)
      ierr = PetscOptionsStringToReal(value,dvalue);CHKERRQ(ierr);
#else
      ierr = PetscOptionsStringToScalar(value,dvalue);CHKERRQ(ierr);
#endif
      if (set) *set = PETSC_TRUE;
    }
  } else { /* flag */
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetRealArray - Gets an array of double precision values for a
   particular option in the database.  The values must be separated with
   commas with no intervening spaces.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to each name or NULL
.  name - the option one is seeking
-  nmax - maximum number of values to retrieve

   Output Parameters:
+  dvalue - the double values to return
.  nmax - actual number of values retreived
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^array of doubles

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetRealArray(PetscOptions options,const char pre[],const char name[],PetscReal dvalue[],PetscInt *nmax,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscBool      flag;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidRealPointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }
  if (!value) {
    if (set) *set = PETSC_TRUE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }

  if (set) *set = PETSC_TRUE;

  ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (n < *nmax) {
    if (!value) break;
    ierr = PetscOptionsStringToReal(value,dvalue++);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    n++;
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetScalarArray - Gets an array of scalars for a
   particular option in the database.  The values must be separated with
   commas with no intervening spaces.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to each name or NULL
.  name - the option one is seeking
-  nmax - maximum number of values to retrieve

   Output Parameters:
+  dvalue - the scalar values to return
.  nmax - actual number of values retreived
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database^array of doubles

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetScalarArray(PetscOptions options,const char pre[],const char name[],PetscScalar dvalue[],PetscInt *nmax,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscBool      flag;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidRealPointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }
  if (!value) {
    if (set) *set = PETSC_TRUE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }

  if (set) *set = PETSC_TRUE;

  ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (n < *nmax) {
    if (!value) break;
    ierr = PetscOptionsStringToScalar(value,dvalue++);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    n++;
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetIntArray - Gets an array of integer values for a particular
   option in the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to each name or NULL
.  name - the option one is seeking
-  nmax - maximum number of values to retrieve

   Output Parameter:
+  dvalue - the integer values to return
.  nmax - actual number of values retreived
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The array can be passed as
   a comma separated list:                                 0,1,2,3,4,5,6,7
   a range (start-end+1):                                  0-8
   a range with given increment (start-end+1:inc):         0-7:2
   a combination of values and ranges separated by commas: 0,1-8,8-15:2

   There must be no intervening spaces between the values.

   Concepts: options database^array of ints

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
           PetscOptionsGetString(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetIntArray(PetscOptions options,const char pre[],const char name[],PetscInt dvalue[],PetscInt *nmax,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0,i,j,start,end,inc,nvalues;
  size_t         len;
  PetscBool      flag,foundrange;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(dvalue,3);
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }
  if (!value) {
    if (set) *set = PETSC_TRUE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }

  if (set) *set = PETSC_TRUE;

  ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (n < *nmax) {
    if (!value) break;

    /* look for form  d-D where d and D are integers */
    foundrange = PETSC_FALSE;
    ierr       = PetscStrlen(value,&len);CHKERRQ(ierr);
    if (value[0] == '-') i=2;
    else i=1;
    for (;i<(int)len; i++) {
      if (value[i] == '-') {
        if (i == (int)len-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry %s\n",n,value);
        value[i] = 0;

        ierr = PetscOptionsStringToInt(value,&start);CHKERRQ(ierr);
        inc  = 1;
        j    = i+1;
        for (;j<(int)len; j++) {
          if (value[j] == ':') {
            value[j] = 0;

            ierr = PetscOptionsStringToInt(value+j+1,&inc);CHKERRQ(ierr);
            if (inc <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry,%s cannot have negative increment",n,value+j+1);CHKERRQ(ierr);
            break;
          }
        }
        ierr = PetscOptionsStringToInt(value+i+1,&end);CHKERRQ(ierr);
        if (end <= start) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry, %s-%s cannot have decreasing list",n,value,value+i+1);
        nvalues = (end-start)/inc + (end-start)%inc;
        if (n + nvalues  > *nmax) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry, not enough space left in array (%D) to contain entire range from %D to %D",n,*nmax-n,start,end);
        for (;start<end; start+=inc) {
          *dvalue = start; dvalue++;n++;
        }
        foundrange = PETSC_TRUE;
        break;
      }
    }
    if (!foundrange) {
      ierr = PetscOptionsStringToInt(value,dvalue);CHKERRQ(ierr);
      dvalue++;
      n++;
    }
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetEnumArray - Gets an array of enum values for a particular option in the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - option prefix or NULL
.  name - option name
.  list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
-  nmax - maximum number of values to retrieve

   Output Parameters:
+  dvalue - the  enum values to return
.  nmax - actual number of values retreived
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Concepts: options database

   Notes:
   The array must be passed as a comma separated list.

   There must be no intervening spaces between the values.

   list is usually something like PCASMTypes or some other predefined list of enum names.

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetEnum(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(), PetscOptionsName(),
          PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(), PetscOptionsStringArray(),PetscOptionsRealArray(),
          PetscOptionsScalar(), PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsGetEList(), PetscOptionsEnum()
@*/
PetscErrorCode PetscOptionsGetEnumArray(PetscOptions options,const char pre[],const char name[],const char *const *list,PetscEnum dvalue[],PetscInt *nmax,PetscBool *set)
{
  char           *svalue;
  PetscInt       n = 0;
  PetscEnum      evalue;
  PetscBool      flag;
  PetscToken     token;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidPointer(list,3);
  PetscValidPointer(dvalue,4);
  PetscValidIntPointer(nmax,5);

  ierr = PetscOptionsFindPair_Private(options,pre,name,&svalue,&flag);CHKERRQ(ierr);
  if (!flag) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }
  if (!svalue) {
    if (set) *set = PETSC_TRUE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }
  if (set) *set = PETSC_TRUE;

  ierr = PetscTokenCreate(svalue,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&svalue);CHKERRQ(ierr);
  while (svalue && n < *nmax) {
    ierr = PetscEnumFind(list,svalue,&evalue,&flag);CHKERRQ(ierr);
    if (!flag) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown enum value '%s' for -%s%s",svalue,pre ? pre : "",name+1);
    dvalue[n++] = evalue;
    ierr = PetscTokenFind(token,&svalue);CHKERRQ(ierr);
  }
  *nmax = n;
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetString - Gets the string value for a particular option in
   the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to name or NULL
.  name - the option one is seeking
-  len - maximum length of the string including null termination

   Output Parameters:
+  string - location to copy string
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Fortran Note:
   The Fortran interface is slightly different from the C/C++
   interface (len is not used).  Sample usage in Fortran follows
.vb
      character *20    string
      PetscErrorCode   ierr
      PetscBool        set
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-s',string,set,ierr)
.ve

   Notes: if the option is given but no string is provided then an empty string is returned and set is given the value of PETSC_TRUE

   Concepts: options database^string

    Note:
      Even if the user provided no string (for example -optionname -someotheroption) the flag is set to PETSC_TRUE (and the string is fulled with nulls).

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetString(PetscOptions options,const char pre[],const char name[],char string[],size_t len,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidCharPointer(string,3);
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag) {
    if (set) *set = PETSC_FALSE;
  } else {
    if (set) *set = PETSC_TRUE;
    if (value) {
      ierr = PetscStrncpy(string,value,len);CHKERRQ(ierr);
      string[len-1] = 0;        /* Ensure that the string is NULL terminated */
    } else {
      ierr = PetscMemzero(string,len);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

char *PetscOptionsGetStringMatlab(PetscOptions options,const char pre[],const char name[])
{
  char           *value;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);if (ierr) PetscFunctionReturn(0);
  if (flag) PetscFunctionReturn(value);
  else PetscFunctionReturn(0);
}


/*@C
   PetscOptionsGetStringArray - Gets an array of string values for a particular
   option in the database. The values must be separated with commas with
   no intervening spaces.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to name or NULL
.  name - the option one is seeking
-  nmax - maximum number of strings

   Output Parameter:
+  strings - location to copy strings
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The user should pass in an array of pointers to char, to hold all the
   strings returned by this function.

   The user is responsible for deallocating the strings that are
   returned. The Fortran interface for this routine is not supported.

   Contributed by Matthew Knepley.

   Concepts: options database^array of strings

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
           PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  PetscOptionsGetStringArray(PetscOptions options,const char pre[],const char name[],char *strings[],PetscInt *nmax,PetscBool  *set)
{
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n;
  PetscBool      flag;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidPointer(strings,3);
  ierr = PetscOptionsFindPair_Private(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag) {
    *nmax = 0;
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  if (!value) {
    *nmax = 0;
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  if (!*nmax) {
    if (set) *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  if (set) *set = PETSC_TRUE;

  ierr = PetscTokenCreate(value,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  n    = 0;
  while (n < *nmax) {
    if (!value) break;
    ierr = PetscStrallocpy(value,&strings[n]);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    n++;
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsUsed - Indicates if PETSc has used a particular option set in the database

   Not Collective

   Input Parameter:
+   options - options database, use NULL for default global database
-   option - string name of option

   Output Parameter:
.   used - PETSC_TRUE if the option was used, otherwise false, including if option was not found in options database

   Level: advanced

.seealso: PetscOptionsView(), PetscOptionsLeft(), PetscOptionsAllUsed()
@*/
PetscErrorCode  PetscOptionsUsed(PetscOptions options,const char *option,PetscBool *used)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  *used = PETSC_FALSE;
  for (i=0; i<options->N; i++) {
    ierr = PetscStrcmp(options->names[i],option,used);CHKERRQ(ierr);
    if (*used) {
      *used = options->used[i];
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsAllUsed - Returns a count of the number of options in the
   database that have never been selected.

   Not Collective

   Input Parameter:
.  options - options database, use NULL for default global database

   Output Parameter:
.   N - count of options not used

   Level: advanced

.seealso: PetscOptionsView()
@*/
PetscErrorCode  PetscOptionsAllUsed(PetscOptions options,PetscInt *N)
{
  PetscInt     i,n = 0;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  for (i=0; i<options->N; i++) {
    if (!options->used[i]) n++;
  }
  *N = n;
  PetscFunctionReturn(0);
}

/*@C
    PetscOptionsLeft - Prints to screen any options that were set and never used.

  Not collective

   Input Parameter:
.  options - options database; use NULL for default global database

   Options Database Key:
.  -options_left - Activates OptionsAllUsed() within PetscFinalize()

  Level: advanced

.seealso: PetscOptionsAllUsed()
@*/
PetscErrorCode  PetscOptionsLeft(PetscOptions options)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  for (i=0; i<options->N; i++) {
    if (!options->used[i]) {
      if (options->values[i]) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s value: %s\n",options->names[i],options->values[i]);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s (no value)\n",options->names[i]);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscOptionsLeftGet - Returns all options that were set and never used.

  Not collective

   Input Parameter:
.  options - options database, use NULL for default global database

   Output Parameter:
.   N - count of options not used
.   names - names of options not used
.   values - values of options not used

  Level: advanced

  Notes:
  Users should call PetscOptionsLeftRestore() to free the memory allocated in this routine

.seealso: PetscOptionsAllUsed(), PetscOptionsLeft()
@*/
PetscErrorCode  PetscOptionsLeftGet(PetscOptions options,PetscInt *N,char **names[],char **values[])
{
  PetscErrorCode ierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;

  /* The number of unused PETSc options */
  n = 0;
  for (i=0; i<options->N; i++) {
    if (!options->used[i]) {
      n++;
    }
  }
  if (N) {*N = n;}
  if (names)  { ierr = PetscMalloc1(n,names);CHKERRQ(ierr); }
  if (values) { ierr = PetscMalloc1(n,values);CHKERRQ(ierr); }

  n = 0;
  if (names || values) {
    for (i=0; i<options->N; i++) {
      if (!options->used[i]) {
        if (names)  (*names)[n]  = options->names[i];
        if (values) (*values)[n] = options->values[i];
        n++;
      }
    }
  }
  PetscFunctionReturn(0);
}


/*@C
  PetscOptionsLeftRestore - Free memory for the unused PETSc options obtained using PetscOptionsLeftGet.

  Not collective

   Input Parameter:
.   options - options database, use NULL for default global database
.   names - names of options not used
.   values - values of options not used

  Level: advanced

.seealso: PetscOptionsAllUsed(), PetscOptionsLeft(), PetscOptionsLeftGet
@*/
PetscErrorCode  PetscOptionsLeftRestore(PetscOptions options,PetscInt *N,char **names[],char **values[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(N) *N = 0;
  if (names)  { ierr = PetscFree(*names);CHKERRQ(ierr); }
  if (values) { ierr = PetscFree(*values);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}


/*@
    PetscOptionsCreate - Creates the empty options database.

  Output Parameter:
.   options - Options database object

  Level: advanced

@*/
PetscErrorCode  PetscOptionsCreate(PetscOptions *options)
{
  *options = (PetscOptions)calloc(1,sizeof(struct _n_PetscOptions));
  if (!options) return PETSC_ERR_MEM;
  (*options)->namegiven      = PETSC_FALSE;
  (*options)->N              = 0;
  (*options)->Naliases       = 0;
  (*options)->numbermonitors = 0;
  return 0;
}

/*
    PetscOptionsCreateDefault - Creates the default global options database

*/
PetscErrorCode  PetscOptionsCreateDefault(void)
{
  PetscErrorCode ierr;

  if (!defaultoptions) {
    ierr = PetscOptionsCreate(&defaultoptions);if (ierr) return ierr;
  }
  return 0;
}

/*@C
   PetscOptionsSetFromOptions - Sets options related to the handling of options in PETSc

   Collective on PETSC_COMM_WORLD

   Input Parameter:
.  options - options database, use NULL for default global database

   Options Database Keys:
+  -options_monitor <optional filename> - prints the names and values of all runtime options as they are set. The monitor functionality is not
                available for options set through a file, environment variable, or on
                the command line. Only options set after PetscInitialize() completes will
                be monitored.
.  -options_monitor_cancel - cancel all options database monitors

   Notes:
   To see all options, run your program with the -help option or consult Users-Manual: sec_gettingstarted

   Level: intermediate

.keywords: set, options, database
@*/
PetscErrorCode  PetscOptionsSetFromOptions(PetscOptions options)
{
  PetscBool      flgc = PETSC_FALSE,flgm;
  PetscErrorCode ierr;
  char           monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer    monviewer;

  PetscFunctionBegin;
  /*
     The options argument is currently ignored since we currently maintain only a single options database

     options = options ? options : defaultoptions;
  */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for handling options","PetscOptions");CHKERRQ(ierr);
  ierr = PetscOptionsString("-options_monitor","Monitor options database","PetscOptionsMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flgm);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-options_monitor_cancel","Cancel all options database monitors","PetscOptionsMonitorCancel",flgc,&flgc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flgm) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,monfilename,&monviewer);CHKERRQ(ierr);
    ierr = PetscOptionsMonitorSet(PetscOptionsMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
  }
  if (flgc) { ierr = PetscOptionsMonitorCancel();CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}


/*@C
   PetscOptionsMonitorDefault - Print all options set value events.

   Logically Collective on PETSC_COMM_WORLD

   Input Parameters:
+  name  - option name string
.  value - option value string
-  dummy - an ASCII viewer

   Level: intermediate

.keywords: PetscOptions, default, monitor

.seealso: PetscOptionsMonitorSet()
@*/
PetscErrorCode  PetscOptionsMonitorDefault(const char name[], const char value[], void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"Setting option: %s = %s\n",name,value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsMonitorSet - Sets an ADDITIONAL function to be called at every method that
   modified the PETSc options database.

   Not collective

   Input Parameters:
+  monitor - pointer to function (if this is NULL, it turns off monitoring
.  mctx    - [optional] context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be NULL)

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
PetscErrorCode  PetscOptionsMonitorSet(PetscErrorCode (*monitor)(const char name[], const char value[], void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscOptions options = defaultoptions;

  PetscFunctionBegin;
  if (options->numbermonitors >= MAXOPTIONSMONITORS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscOptions monitors set");
  options->monitor[options->numbermonitors]          = monitor;
  options->monitordestroy[options->numbermonitors]   = monitordestroy;
  options->monitorcontext[options->numbermonitors++] = (void*)mctx;
  PetscFunctionReturn(0);
}

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
PetscErrorCode  PetscOptionsMonitorCancel(void)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscOptions   options = defaultoptions;

  PetscFunctionBegin;
  for (i=0; i<options->numbermonitors; i++) {
    if (options->monitordestroy[i]) {
      ierr = (*options->monitordestroy[i])(&options->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  options->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#define CHKERRQI(incall,ierr) if (ierr) {incall = PETSC_FALSE; CHKERRQ(ierr);}

/*@C
  PetscObjectViewFromOptions - Processes command line options to determine if/how a PetscObject is to be viewed.

  Collective on PetscObject

  Input Parameters:
+ obj   - the object
. bobj  - optional other object that provides prefix (if NULL then the prefix in obj is used)
- optionname - option to activate viewing

  Level: intermediate

@*/
PetscErrorCode PetscObjectViewFromOptions(PetscObject obj,PetscObject bobj,const char optionname[])
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;
  char              *prefix;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  prefix = bobj ? bobj->prefix : obj->prefix;
  ierr   = PetscOptionsGetViewer(PetscObjectComm((PetscObject)obj),prefix,optionname,&viewer,&format,&flg);CHKERRQI(incall,ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQI(incall,ierr);
    ierr = PetscObjectView(obj,viewer);CHKERRQI(incall,ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQI(incall,ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQI(incall,ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}
