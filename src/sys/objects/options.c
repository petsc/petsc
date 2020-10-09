/* Define Feature test macros to make sure atoll is available (SVr4, POSIX.1-2001, 4.3BSD, C99), not in (C89 and POSIX.1-1996) */
#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for atoll() */

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
#if defined(PETSC_HAVE_STRINGS_H)
#  include <strings.h>          /* strcasecmp */
#endif
#if defined(PETSC_HAVE_YAML)
#include <yaml.h>
#endif

#if defined(PETSC_HAVE_STRCASECMP)
#define PetscOptNameCmp(a,b) strcasecmp(a,b)
#elif defined(PETSC_HAVE_STRICMP)
#define PetscOptNameCmp(a,b) stricmp(a,b)
#else
#define PetscOptNameCmp(a,b) Error_strcasecmp_not_found
#endif

#include <petsc/private/hashtable.h>

/* This assumes ASCII encoding and ignores locale settings */
/* Using tolower() is about 2X slower in microbenchmarks   */
PETSC_STATIC_INLINE int PetscToLower(int c)
{
  return ((c >= 'A') & (c <= 'Z')) ? c + 'a' - 'A' : c;
}

/* Bob Jenkins's one at a time hash function (case-insensitive) */
PETSC_STATIC_INLINE unsigned int PetscOptHash(const char key[])
{
  unsigned int hash = 0;
  while (*key) {
    hash += PetscToLower(*key++);
    hash += hash << 10;
    hash ^= hash >>  6;
  }
  hash += hash <<  3;
  hash ^= hash >> 11;
  hash += hash << 15;
  return hash;
}

PETSC_STATIC_INLINE int PetscOptEqual(const char a[],const char b[])
{
  return !PetscOptNameCmp(a,b);
}

KHASH_INIT(HO, kh_cstr_t, int, 1, PetscOptHash, PetscOptEqual)

/*
    This table holds all the options set by the user. For simplicity, we use a static size database
*/
#define MAXOPTNAME 512
#define MAXOPTIONS 512
#define MAXALIASES  25
#define MAXPREFIXES 25
#define MAXOPTIONSMONITORS 5

struct  _n_PetscOptions {
  PetscOptions   previous;
  int            N;                    /* number of options */
  char           *names[MAXOPTIONS];   /* option names */
  char           *values[MAXOPTIONS];  /* option values */
  PetscBool      used[MAXOPTIONS];     /* flag option use */
  PetscBool      precedentProcessed;

  /* Hash table */
  khash_t(HO)    *ht;

  /* Prefixes */
  int            prefixind;
  int            prefixstack[MAXPREFIXES];
  char           prefix[MAXOPTNAME];

  /* Aliases */
  int            Naliases;                   /* number or aliases */
  char           *aliases1[MAXALIASES];      /* aliased */
  char           *aliases2[MAXALIASES];      /* aliasee */

  /* Help */
  PetscBool      help;       /* flag whether "-help" is in the database */
  PetscBool      help_intro; /* flag whether "-help intro" is in the database */

  /* Monitors */
  PetscBool      monitorFromOptions, monitorCancel;
  PetscErrorCode (*monitor[MAXOPTIONSMONITORS])(const char[],const char[],void*); /* returns control to user after */
  PetscErrorCode (*monitordestroy[MAXOPTIONSMONITORS])(void**);         /* */
  void           *monitorcontext[MAXOPTIONSMONITORS];                  /* to pass arbitrary user data into monitor */
  PetscInt       numbermonitors;                                       /* to, for instance, detect options being set */
};

static PetscOptions defaultoptions = NULL;  /* the options database routines query this object for options */

/* list of options which preceed others, i.e., are processed in PetscOptionsProcessPrecedentFlags() */
static const char *precedentOptions[] = {"-options_monitor","-options_monitor_cancel","-help","-skip_petscrc","-options_file_yaml","-options_string_yaml"};
enum PetscPrecedentOption {PO_OPTIONS_MONITOR,PO_OPTIONS_MONITOR_CANCEL,PO_HELP,PO_SKIP_PETSCRC,PO_OPTIONS_FILE_YAML,PO_OPTIONS_STRING_YAML,PO_NUM};

static PetscErrorCode PetscOptionsSetValue_Private(PetscOptions,const char[],const char[],int*);

/*
    Options events monitor
*/
static PetscErrorCode PetscOptionsMonitor(PetscOptions options,const char name[],const char value[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  if (!PetscErrorHandlingInitialized) return 0;
  PetscFunctionBegin;
  if (!value) value = "";
  if (options->monitorFromOptions) {
    ierr = PetscOptionsMonitorDefault(name,value,NULL);CHKERRQ(ierr);
  }
  for (i=0; i<options->numbermonitors; i++) {
    ierr = (*options->monitor[i])(name,value,options->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscOptionsCreate - Creates an empty options database.

   Logically collective

   Output Parameter:
.  options - Options database object

   Level: advanced

   Developer Note: We may want eventually to pass a MPI_Comm to determine the ownership of the object

.seealso: PetscOptionsDestroy(), PetscOptionsPush(), PetscOptionsPop(), PetscOptionsInsert(), PetscOptionsSetValue()
@*/
PetscErrorCode PetscOptionsCreate(PetscOptions *options)
{
  if (!options) return PETSC_ERR_ARG_NULL;
  *options = (PetscOptions)calloc(1,sizeof(**options));
  if (!*options) return PETSC_ERR_MEM;
  return 0;
}

/*@
    PetscOptionsDestroy - Destroys an option database.

    Logically collective on whatever communicator was associated with the call to PetscOptionsCreate()

  Input Parameter:
.  options - the PetscOptions object

   Level: advanced

.seealso: PetscOptionsInsert(), PetscOptionsPush(), PetscOptionsPop(), PetscOptionsInsert(), PetscOptionsSetValue()
@*/
PetscErrorCode PetscOptionsDestroy(PetscOptions *options)
{
  PetscErrorCode ierr;

  if (!*options) return 0;
  if ((*options)->previous) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"You are destroying an option that has been used with PetscOptionsPush() but does not have a corresponding PetscOptionsPop()");
  ierr = PetscOptionsClear(*options);if (ierr) return ierr;
  /* XXX what about monitors ? */
  free(*options);
  *options = NULL;
  PetscFunctionReturn(0);
}

/*
    PetscOptionsCreateDefault - Creates the default global options database
*/
PetscErrorCode PetscOptionsCreateDefault(void)
{
  PetscErrorCode ierr;

  if (!defaultoptions) {
    ierr = PetscOptionsCreate(&defaultoptions);if (ierr) return ierr;
  }
  return 0;
}

/*@
      PetscOptionsPush - Push a new PetscOptions object as the default provider of options
                         Allows using different parts of a code to use different options databases

  Logically Collective

  Input Parameter:
.   opt - the options obtained with PetscOptionsCreate()

  Notes:
  Use PetscOptionsPop() to return to the previous default options database

  The collectivity of this routine is complex; only the MPI processes that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

   Level: advanced

.seealso: PetscOptionsPop(), PetscOptionsCreate(), PetscOptionsInsert(), PetscOptionsSetValue(), PetscOptionsLeft()

@*/
PetscErrorCode PetscOptionsPush(PetscOptions opt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsCreateDefault();CHKERRQ(ierr);
  opt->previous        = defaultoptions;
  defaultoptions       = opt;
  PetscFunctionReturn(0);
}

/*@
      PetscOptionsPop - Pop the most recent PetscOptionsPush() to return to the previous default options

      Logically collective on whatever communicator was associated with the call to PetscOptionsCreate()

  Notes:
  Use PetscOptionsPop() to return to the previous default options database
  Allows using different parts of a code to use different options databases

   Level: advanced

.seealso: PetscOptionsPop(), PetscOptionsCreate(), PetscOptionsInsert(), PetscOptionsSetValue(), PetscOptionsLeft()

@*/
PetscErrorCode PetscOptionsPop(void)
{
  PetscOptions current = defaultoptions;

  PetscFunctionBegin;
  if (!defaultoptions) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing default options");
  if (!defaultoptions->previous) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscOptionsPop() called too many times");
  defaultoptions = defaultoptions->previous;
  current->previous    = NULL;
  PetscFunctionReturn(0);
}

/*
    PetscOptionsDestroyDefault - Destroys the default global options database
*/
PetscErrorCode PetscOptionsDestroyDefault(void)
{
  PetscErrorCode ierr;
  PetscOptions   tmp;

  /* Destroy any options that the user forgot to pop */
  while (defaultoptions->previous) {
    tmp = defaultoptions;
    ierr = PetscOptionsPop();CHKERRQ(ierr);
    ierr = PetscOptionsDestroy(&tmp);CHKERRQ(ierr);
  }
  ierr = PetscOptionsDestroy(&defaultoptions);if (ierr) return ierr;
  return 0;
}

/*@C
   PetscOptionsValidKey - PETSc Options database keys must begin with one or two dashes (-) followed by a letter.

   Not collective

   Input Parameter:
.  key - string to check if valid

   Output Parameter:
.  valid - PETSC_TRUE if a valid key

   Level: intermediate
@*/
PetscErrorCode PetscOptionsValidKey(const char key[],PetscBool *valid)
{
  char           *ptr;

  PetscFunctionBegin;
  if (key) PetscValidCharPointer(key,1);
  PetscValidPointer(valid,2);
  *valid = PETSC_FALSE;
  if (!key) PetscFunctionReturn(0);
  if (key[0] != '-') PetscFunctionReturn(0);
  if (key[1] == '-') key++;
  if (!isalpha((int)key[1])) PetscFunctionReturn(0);
  (void) strtod(key,&ptr);
  if (ptr != key && !(*ptr == '_' || isalnum((int)*ptr))) PetscFunctionReturn(0);
  *valid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsInsertString - Inserts options into the database from a string

   Logically Collective

   Input Parameter:
+  options - options object
-  in_str - string that contains options separated by blanks

   Level: intermediate

  The collectivity of this routine is complex; only the MPI processes that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

   Contributed by Boyana Norris

.seealso: PetscOptionsSetValue(), PetscOptionsView(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsInsertFile()
@*/
PetscErrorCode PetscOptionsInsertString(PetscOptions options,const char in_str[])
{
  char           *first,*second;
  PetscErrorCode ierr;
  PetscToken     token;
  PetscBool      key,ispush,ispop,isopts;

  PetscFunctionBegin;
  ierr = PetscTokenCreate(in_str,' ',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
  while (first) {
    ierr = PetscStrcasecmp(first,"-prefix_push",&ispush);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(first,"-prefix_pop",&ispop);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(first,"-options_file",&isopts);CHKERRQ(ierr);
    ierr = PetscOptionsValidKey(first,&key);CHKERRQ(ierr);
    if (ispush) {
      ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
      ierr = PetscOptionsPrefixPush(options,second);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    } else if (ispop) {
      ierr = PetscOptionsPrefixPop(options);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&first);CHKERRQ(ierr);
    } else if (isopts) {
      ierr = PetscTokenFind(token,&second);CHKERRQ(ierr);
      ierr = PetscOptionsInsertFile(PETSC_COMM_SELF,options,second,PETSC_TRUE);CHKERRQ(ierr);
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

  if (feof(f)) return NULL;
  do {
    size += 1024; /* BUFSIZ is defined as "the optimal read size for this platform" */
    buf   = (char*)realloc((void*)buf,size); /* realloc(NULL,n) is the same as malloc(n) */
    /* Actually do the read. Note that fgets puts a terminal '\0' on the
    end of the string, so we make sure we overwrite this */
    if (!fgets(buf+len,1024,f)) buf[len]=0;
    PetscStrlen(buf,&len);
    last = len - 1;
  } while (!feof(f) && buf[last] != '\n' && buf[last] != '\r');
  if (len) return buf;
  free(buf);
  return NULL;
}

/*@C
     PetscOptionsInsertFile - Inserts options into the database from a file.

     Collective

  Input Parameter:
+   comm - the processes that will share the options (usually PETSC_COMM_WORLD)
.   options - options database, use NULL for default global database
.   file - name of file
-   require - if PETSC_TRUE will generate an error if the file does not exist


  Notes:
    Use  # for lines that are comments and which should be ignored.
    Usually, instead of using this command, one should list the file name in the call to PetscInitialize(), this insures that certain options
   such as -log_view or -malloc_debug are processed properly. This routine only sets options into the options database that will be processed by later
   calls to XXXSetFromOptions() it should not be used for options listed under PetscInitialize().
   The collectivity of this routine is complex; only the MPI processes in comm will
   have the affect of these options. If some processes that create objects call this routine and others do
   not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
   on different ranks.

  Level: developer

.seealso: PetscOptionsSetValue(), PetscOptionsView(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()

@*/
PetscErrorCode PetscOptionsInsertFile(MPI_Comm comm,PetscOptions options,const char file[],PetscBool require)
{
  char           *string,fname[PETSC_MAX_PATH_LEN],*vstring = NULL,*astring = NULL,*packed = NULL;
  char           *tokens[4];
  PetscErrorCode ierr;
  size_t         i,len,bytes;
  FILE           *fd;
  PetscToken     token=NULL;
  int            err;
  char           *cmatch;
  const char     cmt='#';
  PetscInt       line=1;
  PetscMPIInt    rank,cnt=0,acnt=0,counts[2];
  PetscBool      isdir,alias=PETSC_FALSE,valid;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscMemzero(tokens,sizeof(tokens));CHKERRQ(ierr);
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
      ierr = PetscInfo1(NULL,"Opened options file %s\n",file);CHKERRQ(ierr);

      while ((string = Petscgetline(fd))) {
        /* eliminate comments from each line */
        ierr = PetscStrchr(string,cmt,&cmatch);CHKERRQ(ierr);
        if (cmatch) *cmatch = 0;
        ierr = PetscStrlen(string,&len);CHKERRQ(ierr);
        /* replace tabs, ^M, \n with " " */
        for (i=0; i<len; i++) {
          if (string[i] == '\t' || string[i] == '\r' || string[i] == '\n') {
            string[i] = ' ';
          }
        }
        ierr = PetscTokenCreate(string,' ',&token);CHKERRQ(ierr);
        ierr = PetscTokenFind(token,&tokens[0]);CHKERRQ(ierr);
        if (!tokens[0]) {
          goto destroy;
        } else if (!tokens[0][0]) { /* if token 0 is empty (string begins with spaces), redo */
          ierr = PetscTokenFind(token,&tokens[0]);CHKERRQ(ierr);
        }
        for (i=1; i<4; i++) {
          ierr = PetscTokenFind(token,&tokens[i]);CHKERRQ(ierr);
        }
        if (!tokens[0]) {
          goto destroy;
        } else if (tokens[0][0] == '-') {
          ierr = PetscOptionsValidKey(tokens[0],&valid);CHKERRQ(ierr);
          if (!valid) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in options file %s line %D: invalid option %s",fname,line,tokens[0]);
          ierr = PetscStrlen(tokens[0],&len);CHKERRQ(ierr);
          ierr = PetscSegBufferGet(vseg,len+1,&vstring);CHKERRQ(ierr);
          ierr = PetscArraycpy(vstring,tokens[0],len);CHKERRQ(ierr);
          vstring[len] = ' ';
          if (tokens[1]) {
            ierr = PetscOptionsValidKey(tokens[1],&valid);CHKERRQ(ierr);
            if (valid) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in options file %s line %D: cannot specify two options per line (%s %s)",fname,line,tokens[0],tokens[1]);
            ierr = PetscStrlen(tokens[1],&len);CHKERRQ(ierr);
            ierr = PetscSegBufferGet(vseg,len+3,&vstring);CHKERRQ(ierr);
            vstring[0] = '"';
            ierr = PetscArraycpy(vstring+1,tokens[1],len);CHKERRQ(ierr);
            vstring[len+1] = '"';
            vstring[len+2] = ' ';
          }
        } else {
          ierr = PetscStrcasecmp(tokens[0],"alias",&alias);CHKERRQ(ierr);
          if (alias) {
            ierr = PetscOptionsValidKey(tokens[1],&valid);CHKERRQ(ierr);
            if (!valid) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in options file %s line %D: invalid aliased option %s",fname,line,tokens[1]);
            if (!tokens[2]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in options file %s line %D: alias missing for %s",fname,line,tokens[1]);
            ierr = PetscOptionsValidKey(tokens[2],&valid);CHKERRQ(ierr);
            if (!valid) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in options file %s line %D: invalid aliasee option %s",fname,line,tokens[2]);
            ierr = PetscStrlen(tokens[1],&len);CHKERRQ(ierr);
            ierr = PetscSegBufferGet(aseg,len+1,&astring);CHKERRQ(ierr);
            ierr = PetscArraycpy(astring,tokens[1],len);CHKERRQ(ierr);
            astring[len] = ' ';

            ierr = PetscStrlen(tokens[2],&len);CHKERRQ(ierr);
            ierr = PetscSegBufferGet(aseg,len+1,&astring);CHKERRQ(ierr);
            ierr = PetscArraycpy(astring,tokens[2],len);CHKERRQ(ierr);
            astring[len] = ' ';
          } else SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown first token in options file %s line %D: %s",fname,line,tokens[0]);
        }
        {
          const char *extraToken = alias ? tokens[3] : tokens[2];
          if (extraToken) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Error in options file %s line %D: extra token %s",fname,line,extraToken);
        }
destroy:
        free(string);
        ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
        alias = PETSC_FALSE;
        line++;
      }
      err = fclose(fd);
      if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file %s",fname);
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
    } else if (require) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unable to open options file %s",fname);
  }

  counts[0] = acnt;
  counts[1] = cnt;
  err = MPI_Bcast(counts,2,MPI_INT,0,comm);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in first MPI collective call, could be caused by using an incorrect mpiexec or a network problem, it can be caused by having VPN running: see https://www.mcs.anl.gov/petsc/documentation/faq.html");
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
    ierr = PetscTokenCreate(astring,' ',&token);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&tokens[0]);CHKERRQ(ierr);
    while (tokens[0]) {
      ierr = PetscTokenFind(token,&tokens[1]);CHKERRQ(ierr);
      ierr = PetscOptionsSetAlias(options,tokens[0],tokens[1]);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&tokens[0]);CHKERRQ(ierr);
    }
    ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  }

  if (cnt) {
    ierr = PetscOptionsInsertString(options,vstring);CHKERRQ(ierr);
  }
  ierr = PetscFree(packed);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscOptionsInsertArgs(PetscOptions options,int argc,char *args[])
{
  PetscErrorCode ierr;
  int            left    = argc - 1;
  char           **eargs = args + 1;

  PetscFunctionBegin;
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

PETSC_STATIC_INLINE PetscErrorCode PetscOptionsStringToBoolIfSet_Private(enum PetscPrecedentOption opt,const char *val[],PetscBool set[],PetscBool *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (set[opt]) {
    ierr = PetscOptionsStringToBool(val[opt],flg);CHKERRQ(ierr);
  } else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* Process options with absolute precedence */
static PetscErrorCode PetscOptionsProcessPrecedentFlags(PetscOptions options,int argc,char *args[],PetscBool *skip_petscrc,PetscBool *skip_petscrc_set)
{
  const char* const *opt = precedentOptions;
  const size_t      n = PO_NUM;
  size_t            o;
  int               a;
  const char        **val;
  PetscBool         *set;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc2(n,&val,n,&set);CHKERRQ(ierr);

  /* Look for options possibly set using PetscOptionsSetValue beforehand */
  for (o=0; o<n; o++) {
    ierr = PetscOptionsFindPair(options,NULL,opt[o],&val[o],&set[o]);CHKERRQ(ierr);
  }

  /* Loop through all args to collect last occuring value of each option */
  for (a=1; a<argc; a++) {
    PetscBool valid, eq;

    ierr = PetscOptionsValidKey(args[a],&valid);CHKERRQ(ierr);
    if (!valid) continue;
    for (o=0; o<n; o++) {
      ierr = PetscStrcasecmp(args[a],opt[o],&eq);CHKERRQ(ierr);
      if (eq) {
        set[o] = PETSC_TRUE;
        if (a == argc-1 || !args[a+1] || !args[a+1][0] || args[a+1][0] == '-') val[o] = NULL;
        else val[o] = args[a+1];
        break;
      }
    }
  }

  /* Process flags */
  ierr = PetscStrcasecmp(val[PO_HELP], "intro", &options->help_intro);CHKERRQ(ierr);
  if (options->help_intro) options->help = PETSC_TRUE;
  else {ierr = PetscOptionsStringToBoolIfSet_Private(PO_HELP,            val,set,&options->help);CHKERRQ(ierr);}
  ierr = PetscOptionsStringToBoolIfSet_Private(PO_OPTIONS_MONITOR_CANCEL,val,set,&options->monitorCancel);CHKERRQ(ierr);
  ierr = PetscOptionsStringToBoolIfSet_Private(PO_OPTIONS_MONITOR,       val,set,&options->monitorFromOptions);CHKERRQ(ierr);
  ierr = PetscOptionsStringToBoolIfSet_Private(PO_SKIP_PETSCRC,          val,set,skip_petscrc);CHKERRQ(ierr);
  *skip_petscrc_set = set[PO_SKIP_PETSCRC];

  /* Store precedent options in database and mark them as used */
  for (o=0; o<n; o++) {
    if (set[o]) {
      int pos;

      ierr = PetscOptionsSetValue_Private(options,opt[o],val[o],&pos);CHKERRQ(ierr);
      options->used[pos] = PETSC_TRUE;
    }
  }

  ierr = PetscFree2(val,set);CHKERRQ(ierr);
  options->precedentProcessed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscOptionsSkipPrecedent(PetscOptions options,const char name[],PetscBool *flg)
{
  int i;
  PetscErrorCode ierr;

  *flg = PETSC_FALSE;
  if (options->precedentProcessed) {
    for (i=0; i<PO_NUM; i++) {
      if (!PetscOptNameCmp(precedentOptions[i],name)) {
        /* check if precedent option has been set already */
        ierr = PetscOptionsFindPair(options,NULL,name,NULL,flg);CHKERRQ(ierr);
        if (*flg) break;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsInsert - Inserts into the options database from the command line,
                        the environmental variable and a file.

   Collective on PETSC_COMM_WORLD

   Input Parameters:
+  options - options database or NULL for the default global database
.  argc - count of number of command line arguments
.  args - the command line arguments
-  file - [optional] PETSc database file, also checks ~/.petscrc, .petscrc and petscrc.
          Use NULL to not check for code specific file.
          Use -skip_petscrc in the code specific file (or command line) to skip ~/.petscrc, .petscrc and petscrc files.

   Note:
   Since PetscOptionsInsert() is automatically called by PetscInitialize(),
   the user does not typically need to call this routine. PetscOptionsInsert()
   can be called several times, adding additional entries into the database.

   Options Database Keys:
.   -options_file <filename> - read options from a file

   See PetscInitialize() for options related to option database monitoring.

   Level: advanced

.seealso: PetscOptionsDestroy(), PetscOptionsView(), PetscOptionsInsertString(), PetscOptionsInsertFile(),
          PetscInitialize()
@*/
PetscErrorCode PetscOptionsInsert(PetscOptions options,int *argc,char ***args,const char file[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      hasArgs = (argc && *argc) ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      skipPetscrc = PETSC_FALSE, skipPetscrcSet = PETSC_FALSE;


  PetscFunctionBegin;
  if (hasArgs && !(args && *args)) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "*argc > 1 but *args not given");
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (!options) {
    ierr = PetscOptionsCreateDefault();CHKERRQ(ierr);
    options = defaultoptions;
  }
  if (hasArgs) {
    /* process options with absolute precedence */
    ierr = PetscOptionsProcessPrecedentFlags(options,*argc,*args,&skipPetscrc,&skipPetscrcSet);CHKERRQ(ierr);
  }
  if (file && file[0]) {
    ierr = PetscStrreplace(PETSC_COMM_WORLD,file,filename,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,filename,PETSC_TRUE);CHKERRQ(ierr);
    /* if -skip_petscrc has not been set from command line, check whether it has been set in the file */
    if (!skipPetscrcSet) {ierr = PetscOptionsGetBool(options,NULL,"-skip_petscrc",&skipPetscrc,NULL);CHKERRQ(ierr);}
  }
  if (!skipPetscrc) {
    ierr = PetscGetHomeDirectory(filename,PETSC_MAX_PATH_LEN-16);CHKERRQ(ierr);
    /* PetscOptionsInsertFile() does a fopen() on rank0 only - so only rank0 HomeDir value is relavent */
    if (filename[0]) { ierr = PetscStrcat(filename,"/.petscrc");CHKERRQ(ierr); }
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,filename,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,".petscrc",PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,options,"petscrc",PETSC_FALSE);CHKERRQ(ierr);
  }

  /* insert environment options */
  {
    char   *eoptions = NULL;
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
    char   *eoptions = NULL;
    size_t len       = 0;
    if (!rank) {
      eoptions = (char*)getenv("PETSC_OPTIONS_YAML");
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
      ierr = PetscOptionsInsertStringYAML(options,eoptions);CHKERRQ(ierr);
      if (rank) {ierr = PetscFree(eoptions);CHKERRQ(ierr);}
    }
  }
  {
    char      yaml_file[PETSC_MAX_PATH_LEN];
    char      yaml_string[BUFSIZ];
    PetscBool yaml_flg;
    ierr = PetscOptionsGetString(NULL,NULL,"-options_file_yaml",yaml_file,sizeof(yaml_file),&yaml_flg);CHKERRQ(ierr);
    if (yaml_flg) {
      ierr = PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,yaml_file,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsGetString(NULL,NULL,"-options_string_yaml",yaml_string,sizeof(yaml_string),&yaml_flg);CHKERRQ(ierr);
    if (yaml_flg) {
      ierr = PetscOptionsInsertStringYAML(NULL,yaml_string);CHKERRQ(ierr);
    }
  }
#endif

  /* insert command line options here because they take precedence over arguments in petscrc/environment */
  if (hasArgs) {ierr = PetscOptionsInsertArgs(options,*argc,*args);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsView - Prints the options that have been loaded. This is
   useful for debugging purposes.

   Logically Collective on PetscViewer

   Input Parameter:
+  options - options database, use NULL for default global database
-  viewer - must be an PETSCVIEWERASCII viewer

   Options Database Key:
.  -options_view - Activates PetscOptionsView() within PetscFinalize()

   Notes:
   Only the rank zero process of MPI_Comm used to create view prints the option values. Other processes
   may have different values but they are not printed.

   Level: advanced

.seealso: PetscOptionsAllUsed()
@*/
PetscErrorCode PetscOptionsView(PetscOptions options,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      isascii;

  PetscFunctionBegin;
  if (viewer) PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  options = options ? options : defaultoptions;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Only supports ASCII viewer");

  if (!options->N) {
    ierr = PetscViewerASCIIPrintf(viewer,"#No PETSc Option Table entries\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscViewerASCIIPrintf(viewer,"#PETSc Option Table entries:\n");CHKERRQ(ierr);
  for (i=0; i<options->N; i++) {
    if (options->values[i]) {
      ierr = PetscViewerASCIIPrintf(viewer,"-%s %s\n",options->names[i],options->values[i]);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"-%s\n",options->names[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerASCIIPrintf(viewer,"#End of PETSc Option Table entries\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Called by error handlers to print options used in run
*/
PETSC_EXTERN PetscErrorCode PetscOptionsViewError(void)
{
  PetscInt     i;
  PetscOptions options = defaultoptions;

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
   PetscOptionsPrefixPush - Designate a prefix to be used by all options insertions to follow.

   Logically Collective

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

   The collectivity of this routine is complex; only the MPI processes that call this routine will
   have the affect of these options. If some processes that create objects call this routine and others do
   not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
   on different ranks.

Level: advanced

.seealso: PetscOptionsPrefixPop(), PetscOptionsPush(), PetscOptionsPop(), PetscOptionsCreate(), PetscOptionsSetValue()
@*/
PetscErrorCode PetscOptionsPrefixPush(PetscOptions options,const char prefix[])
{
  PetscErrorCode ierr;
  size_t         n;
  PetscInt       start;
  char           key[MAXOPTNAME+1];
  PetscBool      valid;

  PetscFunctionBegin;
  PetscValidCharPointer(prefix,1);
  options = options ? options : defaultoptions;
  if (options->prefixind >= MAXPREFIXES) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Maximum depth of prefix stack %d exceeded, recompile \n src/sys/objects/options.c with larger value for MAXPREFIXES",MAXPREFIXES);
  key[0] = '-'; /* keys must start with '-' */
  ierr = PetscStrncpy(key+1,prefix,sizeof(key)-1);CHKERRQ(ierr);
  ierr = PetscOptionsValidKey(key,&valid);CHKERRQ(ierr);
  if (!valid) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Given prefix \"%s\" not valid (the first character must be a letter, do not include leading '-')",prefix);
  start = options->prefixind ? options->prefixstack[options->prefixind-1] : 0;
  ierr = PetscStrlen(prefix,&n);CHKERRQ(ierr);
  if (n+1 > sizeof(options->prefix)-start) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Maximum prefix length %d exceeded",sizeof(options->prefix));
  ierr = PetscArraycpy(options->prefix+start,prefix,n+1);CHKERRQ(ierr);
  options->prefixstack[options->prefixind++] = start+n;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsPrefixPop - Remove the latest options prefix, see PetscOptionsPrefixPush() for details

   Logically Collective on the MPI_Comm that called PetscOptionsPrefixPush()

  Input Parameters:
.  options - options database, or NULL for the default global database

   Level: advanced

.seealso: PetscOptionsPrefixPush(), PetscOptionsPush(), PetscOptionsPop(), PetscOptionsCreate(), PetscOptionsSetValue()
@*/
PetscErrorCode PetscOptionsPrefixPop(PetscOptions options)
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

    Logically Collective

  Input Parameters:
.  options - options database, use NULL for the default global database

   The collectivity of this routine is complex; only the MPI processes that call this routine will
   have the affect of these options. If some processes that create objects call this routine and others do
   not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
   on different ranks.

   Level: developer

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode PetscOptionsClear(PetscOptions options)
{
  PetscInt i;

  options = options ? options : defaultoptions;
  if (!options) return 0;

  for (i=0; i<options->N; i++) {
    if (options->names[i])  free(options->names[i]);
    if (options->values[i]) free(options->values[i]);
  }
  options->N = 0;

  for (i=0; i<options->Naliases; i++) {
    free(options->aliases1[i]);
    free(options->aliases2[i]);
  }
  options->Naliases = 0;

  /* destroy hash table */
  kh_destroy(HO,options->ht);
  options->ht = NULL;

  options->prefixind = 0;
  options->prefix[0] = 0;
  options->help      = PETSC_FALSE;
  return 0;
}

/*@C
   PetscOptionsSetAlias - Makes a key and alias for another key

   Logically Collective

   Input Parameters:
+  options - options database, or NULL for default global database
.  newname - the alias
-  oldname - the name that alias will refer to

   Level: advanced

   The collectivity of this routine is complex; only the MPI processes that call this routine will
   have the affect of these options. If some processes that create objects call this routine and others do
   not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
   on different ranks.

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),OptionsHasName(),
          PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(),PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsSetAlias(PetscOptions options,const char newname[],const char oldname[])
{
  PetscInt       n;
  size_t         len;
  PetscBool      valid;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(newname,2);
  PetscValidCharPointer(oldname,3);
  options = options ? options : defaultoptions;
  ierr = PetscOptionsValidKey(newname,&valid);CHKERRQ(ierr);
  if (!valid) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid aliased option %s",newname);
  ierr = PetscOptionsValidKey(oldname,&valid);CHKERRQ(ierr);
  if (!valid) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid aliasee option %s",oldname);

  n = options->Naliases;
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

/*@C
   PetscOptionsSetValue - Sets an option name-value pair in the options
   database, overriding whatever is already present.

   Logically Collective

   Input Parameters:
+  options - options database, use NULL for the default global database
.  name - name of option, this SHOULD have the - prepended
-  value - the option value (not used for all options, so can be NULL)

   Level: intermediate

   Note:
   This function can be called BEFORE PetscInitialize()

   The collectivity of this routine is complex; only the MPI processes that call this routine will
   have the affect of these options. If some processes that create objects call this routine and others do
   not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
   on different ranks.

   Developers Note: Uses malloc() directly because PETSc may not be initialized yet.

.seealso: PetscOptionsInsert(), PetscOptionsClearValue()
@*/
PetscErrorCode PetscOptionsSetValue(PetscOptions options,const char name[],const char value[])
{
  return PetscOptionsSetValue_Private(options,name,value,NULL);
}

static PetscErrorCode PetscOptionsSetValue_Private(PetscOptions options,const char name[],const char value[],int *pos)
{
  size_t         len;
  int            N,n,i;
  char           **names;
  char           fullname[MAXOPTNAME] = "";
  PetscBool      flg;
  PetscErrorCode ierr;

  if (!options) {
    ierr = PetscOptionsCreateDefault();if (ierr) return ierr;
    options = defaultoptions;
  }

  if (name[0] != '-') return PETSC_ERR_ARG_OUTOFRANGE;

  ierr = PetscOptionsSkipPrecedent(options,name,&flg);CHKERRQ(ierr);
  if (flg) return 0;

  name++; /* skip starting dash */

  if (options->prefixind > 0) {
    strncpy(fullname,options->prefix,sizeof(fullname));
    fullname[sizeof(fullname)-1] = 0;
    strncat(fullname,name,sizeof(fullname)-strlen(fullname)-1);
    fullname[sizeof(fullname)-1] = 0;
    name = fullname;
  }

  /* check against aliases */
  N = options->Naliases;
  for (i=0; i<N; i++) {
    int result = PetscOptNameCmp(options->aliases1[i],name);
    if (!result) { name = options->aliases2[i]; break; }
  }

  /* slow search */
  N = n = options->N;
  names = options->names;
  for (i=0; i<N; i++) {
    int result = PetscOptNameCmp(names[i],name);
    if (!result) {
      n = i; goto setvalue;
    } else if (result > 0) {
      n = i; break;
    }
  }
  if (N >= MAXOPTIONS) return PETSC_ERR_MEM;
  /* shift remaining values up 1 */
  for (i=N; i>n; i--) {
    options->names[i]  = options->names[i-1];
    options->values[i] = options->values[i-1];
    options->used[i]   = options->used[i-1];
  }
  options->names[n]  = NULL;
  options->values[n] = NULL;
  options->used[n]   = PETSC_FALSE;
  options->N++;

  /* destroy hash table */
  kh_destroy(HO,options->ht);
  options->ht = NULL;

  /* set new name */
  len = strlen(name);
  options->names[n] = (char*)malloc((len+1)*sizeof(char));
  if (!options->names[n]) return PETSC_ERR_MEM;
  strcpy(options->names[n],name);

setvalue:
  /* set new value */
  if (options->values[n]) free(options->values[n]);
  len = value ? strlen(value) : 0;
  if (len) {
    options->values[n] = (char*)malloc((len+1)*sizeof(char));
    if (!options->values[n]) return PETSC_ERR_MEM;
    strcpy(options->values[n],value);
  } else {
    options->values[n] = NULL;
  }

  if (PetscErrorHandlingInitialized) {
    ierr = PetscOptionsMonitor(options,name,value);CHKERRQ(ierr);
  }
  if (pos) *pos = n;
  return 0;
}

/*@C
   PetscOptionsClearValue - Clears an option name-value pair in the options
   database, overriding whatever is already present.

   Logically Collective

   Input Parameter:
+  options - options database, use NULL for the default global database
-  name - name of option, this SHOULD have the - prepended

   Level: intermediate

   The collectivity of this routine is complex; only the MPI processes that call this routine will
   have the affect of these options. If some processes that create objects call this routine and others do
   not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
   on different ranks.

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode PetscOptionsClearValue(PetscOptions options,const char name[])
{
  int            N,n,i;
  char           **names;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  if (name[0] != '-') SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Name must begin with '-': Instead %s",name);

  if (!PetscOptNameCmp(name,"-help")) options->help = PETSC_FALSE;

  name++; /* skip starting dash */

  /* slow search */
  N = n = options->N;
  names = options->names;
  for (i=0; i<N; i++) {
    int result = PetscOptNameCmp(names[i],name);
    if (!result) {
      n = i; break;
    } else if (result > 0) {
      n = N; break;
    }
  }
  if (n == N) PetscFunctionReturn(0); /* it was not present */

  /* remove name and value */
  if (options->names[n])  free(options->names[n]);
  if (options->values[n]) free(options->values[n]);
  /* shift remaining values down 1 */
  for (i=n; i<N-1; i++) {
    options->names[i]  = options->names[i+1];
    options->values[i] = options->values[i+1];
    options->used[i]   = options->used[i+1];
  }
  options->N--;

  /* destroy hash table */
  kh_destroy(HO,options->ht);
  options->ht = NULL;

  ierr = PetscOptionsMonitor(options,name,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsFindPair - Gets an option name-value pair from the options database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for the default global database
.  pre - the string to prepend to the name or NULL, this SHOULD NOT have the "-" prepended
-  name - name of option, this SHOULD have the "-" prepended

   Output Parameters:
+  value - the option value (optional, not used for all options)
-  set - whether the option is set (optional)

   Notes:
   Each process may find different values or no value depending on how options were inserted into the database

   Level: developer

.seealso: PetscOptionsSetValue(), PetscOptionsClearValue()
@*/
PetscErrorCode PetscOptionsFindPair(PetscOptions options,const char pre[],const char name[],const char *value[],PetscBool *set)
{
  char           buf[MAXOPTNAME];
  PetscBool      usehashtable = PETSC_TRUE;
  PetscBool      matchnumbers = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  if (pre && PetscUnlikely(pre[0] == '-')) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Prefix cannot begin with '-': Instead %s",pre);
  if (PetscUnlikely(name[0] != '-')) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Name must begin with '-': Instead %s",name);

  name++; /* skip starting dash */

  /* append prefix to name, if prefix="foo_" and option='--bar", prefixed option is --foo_bar */
  if (pre && pre[0]) {
    char *ptr = buf;
    if (name[0] == '-') { *ptr++ = '-';  name++; }
    ierr = PetscStrncpy(ptr,pre,buf+sizeof(buf)-ptr);CHKERRQ(ierr);
    ierr = PetscStrlcat(buf,name,sizeof(buf));CHKERRQ(ierr);
    name = buf;
  }

  if (PetscDefined(USE_DEBUG)) {
    PetscBool valid;
    char      key[MAXOPTNAME+1] = "-";
    ierr = PetscStrncpy(key+1,name,sizeof(key)-1);CHKERRQ(ierr);
    ierr = PetscOptionsValidKey(key,&valid);CHKERRQ(ierr);
    if (!valid) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid option '%s' obtained from pre='%s' and name='%s'",key,pre?pre:"",name);
  }

  if (!options->ht && usehashtable) {
    int i,ret;
    khiter_t it;
    khash_t(HO) *ht;
    ht = kh_init(HO);
    if (PetscUnlikely(!ht)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Hash table allocation failed");
    ret = kh_resize(HO,ht,options->N*2); /* twice the required size to reduce risk of collisions */
    if (PetscUnlikely(ret)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Hash table allocation failed");
    for (i=0; i<options->N; i++) {
      it = kh_put(HO,ht,options->names[i],&ret);
      if (PetscUnlikely(ret != 1)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Hash table allocation failed");
      kh_val(ht,it) = i;
    }
    options->ht = ht;
  }

  if (usehashtable)
  { /* fast search */
    khash_t(HO) *ht = options->ht;
    khiter_t it = kh_get(HO,ht,name);
    if (it != kh_end(ht)) {
      int i = kh_val(ht,it);
      options->used[i]  = PETSC_TRUE;
      if (value) *value = options->values[i];
      if (set)   *set   = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
  } else
  { /* slow search */
    int i, N = options->N;
    for (i=0; i<N; i++) {
      int result = PetscOptNameCmp(options->names[i],name);
      if (!result) {
        options->used[i]  = PETSC_TRUE;
        if (value) *value = options->values[i];
        if (set)   *set   = PETSC_TRUE;
        PetscFunctionReturn(0);
      } else if (result > 0) {
        break;
      }
    }
  }

  /*
   The following block slows down all lookups in the most frequent path (most lookups are unsuccessful).
   Maybe this special lookup mode should be enabled on request with a push/pop API.
   The feature of matching _%d_ used sparingly in the codebase.
   */
  if (matchnumbers) {
    int i,j,cnt = 0,locs[16],loce[16];
    /* determine the location and number of all _%d_ in the key */
    for (i=0; name[i]; i++) {
      if (name[i] == '_') {
        for (j=i+1; name[j]; j++) {
          if (name[j] >= '0' && name[j] <= '9') continue;
          if (name[j] == '_' && j > i+1) { /* found a number */
            locs[cnt]   = i+1;
            loce[cnt++] = j+1;
          }
          i = j-1;
          break;
        }
      }
    }
    for (i=0; i<cnt; i++) {
      PetscBool found;
      char      opt[MAXOPTNAME+1] = "-", tmp[MAXOPTNAME];
      ierr = PetscStrncpy(tmp,name,PetscMin((size_t)(locs[i]+1),sizeof(tmp)));CHKERRQ(ierr);
      ierr = PetscStrlcat(opt,tmp,sizeof(opt));CHKERRQ(ierr);
      ierr = PetscStrlcat(opt,name+loce[i],sizeof(opt));CHKERRQ(ierr);
      ierr = PetscOptionsFindPair(options,NULL,opt,value,&found);CHKERRQ(ierr);
      if (found) {if (set) *set = PETSC_TRUE; PetscFunctionReturn(0);}
    }
  }

  if (set) *set = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* Check whether any option begins with pre+name */
PETSC_EXTERN PetscErrorCode PetscOptionsFindPairPrefix_Private(PetscOptions options,const char pre[], const char name[],const char *value[],PetscBool *set)
{
  char           buf[MAXOPTNAME];
  int            numCnt = 0, locs[16],loce[16];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  if (pre && pre[0] == '-') SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Prefix cannot begin with '-': Instead %s",pre);
  if (name[0] != '-') SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Name must begin with '-': Instead %s",name);

  name++; /* skip starting dash */

  /* append prefix to name, if prefix="foo_" and option='--bar", prefixed option is --foo_bar */
  if (pre && pre[0]) {
    char *ptr = buf;
    if (name[0] == '-') { *ptr++ = '-';  name++; }
    ierr = PetscStrncpy(ptr,pre,sizeof(buf)+(size_t)(ptr-buf));CHKERRQ(ierr);
    ierr = PetscStrlcat(buf,name,sizeof(buf));CHKERRQ(ierr);
    name = buf;
  }

  if (PetscDefined(USE_DEBUG)) {
    PetscBool valid;
    char      key[MAXOPTNAME+1] = "-";
    ierr = PetscStrncpy(key+1,name,sizeof(key)-1);CHKERRQ(ierr);
    ierr = PetscOptionsValidKey(key,&valid);CHKERRQ(ierr);
    if (!valid) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid option '%s' obtained from pre='%s' and name='%s'",key,pre?pre:"",name);
  }

  /* determine the location and number of all _%d_ in the key */
  {
    int i,j;
    for (i=0; name[i]; i++) {
      if (name[i] == '_') {
        for (j=i+1; name[j]; j++) {
          if (name[j] >= '0' && name[j] <= '9') continue;
          if (name[j] == '_' && j > i+1) { /* found a number */
            locs[numCnt]   = i+1;
            loce[numCnt++] = j+1;
          }
          i = j-1;
          break;
        }
      }
    }
  }

  { /* slow search */
    int       c, i;
    size_t    len;
    PetscBool match;

    for (c = -1; c < numCnt; ++c) {
      char opt[MAXOPTNAME+1] = "", tmp[MAXOPTNAME];

      if (c < 0) {
        ierr = PetscStrcpy(opt,name);CHKERRQ(ierr);
      } else {
        ierr = PetscStrncpy(tmp,name,PetscMin((size_t)(locs[c]+1),sizeof(tmp)));CHKERRQ(ierr);
        ierr = PetscStrlcat(opt,tmp,sizeof(opt));CHKERRQ(ierr);
        ierr = PetscStrlcat(opt,name+loce[c],sizeof(opt));CHKERRQ(ierr);
      }
      ierr = PetscStrlen(opt,&len);CHKERRQ(ierr);
      for (i=0; i<options->N; i++) {
        ierr = PetscStrncmp(options->names[i],opt,len,&match);CHKERRQ(ierr);
        if (match) {
          options->used[i]  = PETSC_TRUE;
          if (value) *value = options->values[i];
          if (set)   *set   = PETSC_TRUE;
          PetscFunctionReturn(0);
        }
      }
    }
  }

  if (set) *set = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsReject - Generates an error if a certain option is given.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - the option prefix (may be NULL)
.  name - the option name one is seeking
-  mess - error message (may be NULL)

   Level: advanced

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),OptionsHasName(),
          PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsReject(PetscOptions options,const char pre[],const char name[],const char mess[])
{
  PetscErrorCode ierr;
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(options,pre,name,&flag);CHKERRQ(ierr);
  if (flag) {
    if (mess && mess[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Program has disabled option: -%s%s with %s",pre?pre:"",name+1,mess);
    else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Program has disabled option: -%s%s",pre?pre:"",name+1);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsHasHelp - Determines whether the "-help" option is in the database.

   Not Collective

   Input Parameters:
.  options - options database, use NULL for default global database

   Output Parameters:
.  set - PETSC_TRUE if found else PETSC_FALSE.

   Level: advanced

.seealso: PetscOptionsHasName()
@*/
PetscErrorCode PetscOptionsHasHelp(PetscOptions options,PetscBool *set)
{
  PetscFunctionBegin;
  PetscValidPointer(set,2);
  options = options ? options : defaultoptions;
  *set = options->help;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOptionsHasHelpIntro_Internal(PetscOptions options,PetscBool *set)
{
  PetscFunctionBegin;
  PetscValidPointer(set,2);
  options = options ? options : defaultoptions;
  *set = options->help_intro;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsHasName - Determines whether a certain option is given in the database. This returns true whether the option is a number, string or boolean, even
                      its value is set to false.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - string to prepend to the name or NULL
-  name - the option one is seeking

   Output Parameters:
.  set - PETSC_TRUE if found else PETSC_FALSE.

   Level: beginner

   Notes:
   In many cases you probably want to use PetscOptionsGetBool() instead of calling this, to allowing toggling values.

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
          PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsHasName(PetscOptions options,const char pre[],const char name[],PetscBool *set)
{
  const char     *value;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscOptionsFindPair(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (set) *set = flag;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetAll - Lists all the options the program was run with in a single string.

   Not Collective

   Input Parameter:
.  options - the options database, use NULL for the default global database

   Output Parameter:
.  copts - pointer where string pointer is stored

   Notes:
    The array and each entry in the array should be freed with PetscFree()
    Each process may have different values depending on how the options were inserted into the database

   Level: advanced

.seealso: PetscOptionsAllUsed(), PetscOptionsView(), PetscOptionsPush(), PetscOptionsPop()
@*/
PetscErrorCode PetscOptionsGetAll(PetscOptions options,char *copts[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  size_t         len = 1,lent = 0;
  char           *coptions = NULL;

  PetscFunctionBegin;
  PetscValidPointer(copts,2);
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
   PetscOptionsUsed - Indicates if PETSc has used a particular option set in the database

   Not Collective

   Input Parameter:
+  options - options database, use NULL for default global database
-  name - string name of option

   Output Parameter:
.  used - PETSC_TRUE if the option was used, otherwise false, including if option was not found in options database

   Level: advanced

   Notes:
   The value returned may be different on each process and depends on which options have been processed
   on the given process

.seealso: PetscOptionsView(), PetscOptionsLeft(), PetscOptionsAllUsed()
@*/
PetscErrorCode PetscOptionsUsed(PetscOptions options,const char *name,PetscBool *used)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidPointer(used,3);
  options = options ? options : defaultoptions;
  *used = PETSC_FALSE;
  for (i=0; i<options->N; i++) {
    ierr = PetscStrcmp(options->names[i],name,used);CHKERRQ(ierr);
    if (*used) {
      *used = options->used[i];
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscOptionsAllUsed - Returns a count of the number of options in the
   database that have never been selected.

   Not Collective

   Input Parameter:
.  options - options database, use NULL for default global database

   Output Parameter:
.  N - count of options not used

   Level: advanced

   Notes:
   The value returned may be different on each process and depends on which options have been processed
   on the given process

.seealso: PetscOptionsView()
@*/
PetscErrorCode PetscOptionsAllUsed(PetscOptions options,PetscInt *N)
{
  PetscInt     i,n = 0;

  PetscFunctionBegin;
  PetscValidIntPointer(N,2);
  options = options ? options : defaultoptions;
  for (i=0; i<options->N; i++) {
    if (!options->used[i]) n++;
  }
  *N = n;
  PetscFunctionReturn(0);
}

/*@
   PetscOptionsLeft - Prints to screen any options that were set and never used.

   Not Collective

   Input Parameter:
.  options - options database; use NULL for default global database

   Options Database Key:
.  -options_left - activates PetscOptionsAllUsed() within PetscFinalize()

   Notes:
      This is rarely used directly, it is called by PetscFinalize() in debug more or if -options_left
      is passed otherwise to help users determine possible mistakes in their usage of options. This
      only prints values on process zero of PETSC_COMM_WORLD. Other processes depending the objects
      used may have different options that are left unused.

   Level: advanced

.seealso: PetscOptionsAllUsed()
@*/
PetscErrorCode PetscOptionsLeft(PetscOptions options)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       cnt = 0;
  PetscOptions   toptions;

  PetscFunctionBegin;
  toptions = options ? options : defaultoptions;
  for (i=0; i<toptions->N; i++) {
    if (!toptions->used[i]) {
      if (toptions->values[i]) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s value: %s\n",toptions->names[i],toptions->values[i]);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: name:-%s (no value)\n",toptions->names[i]);CHKERRQ(ierr);
      }
    }
  }
  if (!options) {
    toptions = defaultoptions;
    while (toptions->previous) {
      cnt++;
      toptions = toptions->previous;
    }
    if (cnt) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Option left: You may have forgotten some calls to PetscOptionsPop(),\n             PetscOptionsPop() has been called %D less times than PetscOptionsPush()\n",cnt);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsLeftGet - Returns all options that were set and never used.

   Not Collective

   Input Parameter:
.  options - options database, use NULL for default global database

   Output Parameter:
+  N - count of options not used
.  names - names of options not used
-  values - values of options not used

   Level: advanced

   Notes:
   Users should call PetscOptionsLeftRestore() to free the memory allocated in this routine
   Notes: The value returned may be different on each process and depends on which options have been processed
   on the given process

.seealso: PetscOptionsAllUsed(), PetscOptionsLeft()
@*/
PetscErrorCode PetscOptionsLeftGet(PetscOptions options,PetscInt *N,char **names[],char **values[])
{
  PetscErrorCode ierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  if (N) PetscValidIntPointer(N,2);
  if (names) PetscValidPointer(names,3);
  if (values) PetscValidPointer(values,4);
  options = options ? options : defaultoptions;

  /* The number of unused PETSc options */
  n = 0;
  for (i=0; i<options->N; i++) {
    if (!options->used[i]) n++;
  }
  if (N) { *N = n; }
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

   Not Collective

   Input Parameter:
+  options - options database, use NULL for default global database
.  names - names of options not used
-  values - values of options not used

   Level: advanced

.seealso: PetscOptionsAllUsed(), PetscOptionsLeft(), PetscOptionsLeftGet()
@*/
PetscErrorCode PetscOptionsLeftRestore(PetscOptions options,PetscInt *N,char **names[],char **values[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (N) PetscValidIntPointer(N,2);
  if (names) PetscValidPointer(names,3);
  if (values) PetscValidPointer(values,4);
  if (N) { *N = 0; }
  if (names)  { ierr = PetscFree(*names);CHKERRQ(ierr); }
  if (values) { ierr = PetscFree(*values);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsMonitorDefault - Print all options set value events using the supplied PetscViewer.

   Logically Collective on ctx

   Input Parameters:
+  name  - option name string
.  value - option value string
-  ctx - an ASCII viewer or NULL

   Level: intermediate

   Notes:
     If ctx=NULL, PetscPrintf() is used.
     The first MPI rank in the PetscViewer viewer actually prints the values, other
     processes may have different values set

.seealso: PetscOptionsMonitorSet()
@*/
PetscErrorCode PetscOptionsMonitorDefault(const char name[],const char value[],void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx) {
    PetscViewer viewer = (PetscViewer)ctx;
    if (!value) {
      ierr = PetscViewerASCIIPrintf(viewer,"Removing option: %s\n",name,value);CHKERRQ(ierr);
    } else if (!value[0]) {
      ierr = PetscViewerASCIIPrintf(viewer,"Setting option: %s (no value)\n",name);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"Setting option: %s = %s\n",name,value);CHKERRQ(ierr);
    }
  } else {
    MPI_Comm comm = PETSC_COMM_WORLD;
    if (!value) {
      ierr = PetscPrintf(comm,"Removing option: %s\n",name,value);CHKERRQ(ierr);
    } else if (!value[0]) {
      ierr = PetscPrintf(comm,"Setting option: %s (no value)\n",name);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(comm,"Setting option: %s = %s\n",name,value);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsMonitorSet - Sets an ADDITIONAL function to be called at every method that
   modified the PETSc options database.

   Not Collective

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
   See PetscInitialize() for options related to option database monitoring.

   Notes:
   The default is to do nothing.  To print the name and value of options
   being inserted into the database, use PetscOptionsMonitorDefault() as the monitoring routine,
   with a null monitoring context.

   Several different monitoring routines may be set by calling
   PetscOptionsMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Level: intermediate

.seealso: PetscOptionsMonitorDefault(), PetscInitialize()
@*/
PetscErrorCode PetscOptionsMonitorSet(PetscErrorCode (*monitor)(const char name[], const char value[], void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscOptions options = defaultoptions;

  PetscFunctionBegin;
  if (options->monitorCancel) PetscFunctionReturn(0);
  if (options->numbermonitors >= MAXOPTIONSMONITORS) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscOptions monitors set");
  options->monitor[options->numbermonitors]          = monitor;
  options->monitordestroy[options->numbermonitors]   = monitordestroy;
  options->monitorcontext[options->numbermonitors++] = (void*)mctx;
  PetscFunctionReturn(0);
}

/*
   PetscOptionsStringToBool - Converts string to PetscBool, handles cases like "yes", "no", "true", "false", "0", "1", "off", "on".
     Empty string is considered as true.
*/
PetscErrorCode PetscOptionsStringToBool(const char value[],PetscBool *a)
{
  PetscBool      istrue,isfalse;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* PetscStrlen() returns 0 for NULL or "" */
  ierr = PetscStrlen(value,&len);CHKERRQ(ierr);
  if (!len)  {*a = PETSC_TRUE; PetscFunctionReturn(0);}
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
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown logical value: %s",value);
}

/*
   PetscOptionsStringToInt - Converts a string to an integer value. Handles special cases such as "default" and "decide"
*/
PetscErrorCode PetscOptionsStringToInt(const char name[],PetscInt *a)
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
PetscErrorCode PetscOptionsStringToReal(const char name[],PetscReal *a)
{
  size_t         len;
  PetscBool      match;
  char           *endptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"String of length zero has no numerical value");

  ierr = PetscStrcasecmp(name,"PETSC_DEFAULT",&match);CHKERRQ(ierr);
  if (!match) {
    ierr = PetscStrcasecmp(name,"DEFAULT",&match);CHKERRQ(ierr);
  }
  if (match) {*a = PETSC_DEFAULT; PetscFunctionReturn(0);}

  ierr = PetscStrcasecmp(name,"PETSC_DECIDE",&match);CHKERRQ(ierr);
  if (!match) {
    ierr = PetscStrcasecmp(name,"DECIDE",&match);CHKERRQ(ierr);
  }
  if (match) {*a = PETSC_DECIDE; PetscFunctionReturn(0);}

  ierr = PetscStrtod(name,a,&endptr);CHKERRQ(ierr);
  if ((size_t) (endptr - name) != len) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Input string %s has no numeric value",name);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscOptionsStringToScalar(const char name[],PetscScalar *a)
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

      If the option is given, but no value is provided, then ivalue and set are both given the value PETSC_TRUE. That is -requested_bool
     is equivalent to -requested_bool true

       If the user does not supply the option at all ivalue is NOT changed. Thus
     you should ALWAYS initialize the ivalue if you access it without first checking if the set flag is true.

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsGetInt(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetBool(PetscOptions options,const char pre[],const char name[],PetscBool *ivalue,PetscBool *set)
{
  const char     *value;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  if (ivalue) PetscValidIntPointer(ivalue,4);
  ierr = PetscOptionsFindPair(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (set) *set = PETSC_TRUE;
    ierr = PetscOptionsStringToBool(value, &flag);CHKERRQ(ierr);
    if (ivalue) *ivalue = flag;
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
-  ntext - number of choices

   Output Parameter:
+  value - the index of the value to return (defaults to zero if the option name is given but no choice is listed)
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes:
    If the user does not supply the option value is NOT changed. Thus
     you should ALWAYS initialize the ivalue if you access it without first checking if the set flag is true.

   See PetscOptionsFList() for when the choices are given in a PetscFunctionList()

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
          PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetEList(PetscOptions options,const char pre[],const char opt[],const char * const *list,PetscInt ntext,PetscInt *value,PetscBool *set)
{
  PetscErrorCode ierr;
  size_t         alen,len = 0, tlen = 0;
  char           *svalue;
  PetscBool      aset,flg = PETSC_FALSE;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidCharPointer(opt,3);
  for (i=0; i<ntext; i++) {
    ierr = PetscStrlen(list[i],&alen);CHKERRQ(ierr);
    if (alen > len) len = alen;
    tlen += len + 1;
  }
  len += 5; /* a little extra space for user mistypes */
  ierr = PetscMalloc1(len,&svalue);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(options,pre,opt,svalue,len,&aset);CHKERRQ(ierr);
  if (aset) {
    ierr = PetscEListFind(ntext,list,svalue,value,&flg);CHKERRQ(ierr);
    if (!flg) {
      char *avail,*pavl;

      ierr = PetscMalloc1(tlen,&avail);CHKERRQ(ierr);
      pavl = avail;
      for (i=0; i<ntext; i++) {
        ierr = PetscStrlen(list[i],&alen);CHKERRQ(ierr);
        ierr = PetscStrcpy(pavl,list[i]);CHKERRQ(ierr);
        pavl += alen;
        ierr = PetscStrcpy(pavl," ");CHKERRQ(ierr);
        pavl += 1;
      }
      ierr = PetscStrtolower(avail);CHKERRQ(ierr);
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown option %s for -%s%s. Available options: %s",svalue,pre ? pre : "",opt+1,avail);
    }
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

   Notes:
    If the user does not supply the option value is NOT changed. Thus
     you should ALWAYS initialize the ivalue if you access it without first checking if the set flag is true.

          List is usually something like PCASMTypes or some other predefined list of enum names

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(), PetscOptionsGetInt(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsGetEList(), PetscOptionsEnum()
@*/
PetscErrorCode PetscOptionsGetEnum(PetscOptions options,const char pre[],const char opt[],const char * const *list,PetscEnum *value,PetscBool *set)
{
  PetscErrorCode ierr;
  PetscInt       ntext = 0,tval;
  PetscBool      fset;

  PetscFunctionBegin;
  PetscValidCharPointer(opt,3);
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

   Notes:
   If the user does not supply the option ivalue is NOT changed. Thus
   you should ALWAYS initialize the ivalue if you access it without first checking if the set flag is true.

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetInt(PetscOptions options,const char pre[],const char name[],PetscInt *ivalue,PetscBool *set)
{
  const char     *value;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidIntPointer(ivalue,4);
  ierr = PetscOptionsFindPair(options,pre,name,&value,&flag);CHKERRQ(ierr);
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

   Notes:
    If the user does not supply the option dvalue is NOT changed. Thus
     you should ALWAYS initialize the ivalue if you access it without first checking if the set flag is true.

   Level: beginner

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
          PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(),PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetReal(PetscOptions options,const char pre[],const char name[],PetscReal *dvalue,PetscBool *set)
{
  const char     *value;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidRealPointer(dvalue,4);
  ierr = PetscOptionsFindPair(options,pre,name,&value,&flag);CHKERRQ(ierr);
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

   Notes:
    If the user does not supply the option dvalue is NOT changed. Thus
     you should ALWAYS initialize the ivalue if you access it without first checking if the set flag is true.

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
          PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetScalar(PetscOptions options,const char pre[],const char name[],PetscScalar *dvalue,PetscBool *set)
{
  const char     *value;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidScalarPointer(dvalue,4);
  ierr = PetscOptionsFindPair(options,pre,name,&value,&flag);CHKERRQ(ierr);
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

   Notes:
    if the option is given but no string is provided then an empty string is returned and set is given the value of PETSC_TRUE

           If the user does not use the option then the string is not changed. Thus
           you should ALWAYS initialize the string if you access it without first checking if the set flag is true.

    Note:
      Even if the user provided no string (for example -optionname -someotheroption) the flag is set to PETSC_TRUE (and the string is fulled with nulls).

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
          PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetString(PetscOptions options,const char pre[],const char name[],char string[],size_t len,PetscBool *set)
{
  const char     *value;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidCharPointer(string,4);
  ierr = PetscOptionsFindPair(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (!flag) {
    if (set) *set = PETSC_FALSE;
  } else {
    if (set) *set = PETSC_TRUE;
    if (value) {
      ierr = PetscStrncpy(string,value,len);CHKERRQ(ierr);
    } else {
      ierr = PetscArrayzero(string,len);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

char *PetscOptionsGetStringMatlab(PetscOptions options,const char pre[],const char name[])
{
  const char     *value;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsFindPair(options,pre,name,&value,&flag);if (ierr) PetscFunctionReturn(NULL);
  if (flag) PetscFunctionReturn((char*)value);
  else PetscFunctionReturn(NULL);
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
PetscErrorCode PetscOptionsGetBoolArray(PetscOptions options,const char pre[],const char name[],PetscBool dvalue[],PetscInt *nmax,PetscBool *set)
{
  const char     *svalue;
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscBool      flag;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidIntPointer(dvalue,4);
  PetscValidIntPointer(nmax,5);

  ierr = PetscOptionsFindPair(options,pre,name,&svalue,&flag);CHKERRQ(ierr);
  if (!flag || !svalue)  { if (set) *set = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (set) *set = PETSC_TRUE;
  ierr = PetscTokenCreate(svalue,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (value && n < *nmax) {
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
   PetscOptionsGetEnumArray - Gets an array of enum values for a particular option in the database.

   Not Collective

   Input Parameters:
+  options - options database, use NULL for default global database
.  pre - option prefix or NULL
.  name - option name
.  list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
-  nmax - maximum number of values to retrieve

   Output Parameters:
+  ivalue - the  enum values to return
.  nmax - actual number of values retreived
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

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
PetscErrorCode PetscOptionsGetEnumArray(PetscOptions options,const char pre[],const char name[],const char *const *list,PetscEnum ivalue[],PetscInt *nmax,PetscBool *set)
{
  const char     *svalue;
  char           *value;
  PetscInt       n = 0;
  PetscEnum      evalue;
  PetscBool      flag;
  PetscToken     token;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidPointer(list,4);
  PetscValidPointer(ivalue,5);
  PetscValidIntPointer(nmax,6);

  ierr = PetscOptionsFindPair(options,pre,name,&svalue,&flag);CHKERRQ(ierr);
  if (!flag || !svalue)  { if (set) *set = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (set) *set = PETSC_TRUE;
  ierr = PetscTokenCreate(svalue,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (value && n < *nmax) {
    ierr = PetscEnumFind(list,value,&evalue,&flag);CHKERRQ(ierr);
    if (!flag) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown enum value '%s' for -%s%s",svalue,pre ? pre : "",name+1);
    ivalue[n++] = evalue;
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
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
+  ivalue - the integer values to return
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

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
          PetscOptionsGetString(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetIntArray(PetscOptions options,const char pre[],const char name[],PetscInt ivalue[],PetscInt *nmax,PetscBool *set)
{
  const char     *svalue;
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0,i,j,start,end,inc,nvalues;
  size_t         len;
  PetscBool      flag,foundrange;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidIntPointer(ivalue,4);
  PetscValidIntPointer(nmax,5);

  ierr = PetscOptionsFindPair(options,pre,name,&svalue,&flag);CHKERRQ(ierr);
  if (!flag || !svalue)  { if (set) *set = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (set) *set = PETSC_TRUE;
  ierr = PetscTokenCreate(svalue,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (value && n < *nmax) {
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
            if (inc <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry,%s cannot have negative increment",n,value+j+1);
            break;
          }
        }
        ierr = PetscOptionsStringToInt(value+i+1,&end);CHKERRQ(ierr);
        if (end <= start) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry, %s-%s cannot have decreasing list",n,value,value+i+1);
        nvalues = (end-start)/inc + (end-start)%inc;
        if (n + nvalues  > *nmax) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"Error in %D-th array entry, not enough space left in array (%D) to contain entire range from %D to %D",n,*nmax-n,start,end);
        for (;start<end; start+=inc) {
          *ivalue = start; ivalue++;n++;
        }
        foundrange = PETSC_TRUE;
        break;
      }
    }
    if (!foundrange) {
      ierr = PetscOptionsStringToInt(value,ivalue);CHKERRQ(ierr);
      ivalue++;
      n++;
    }
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
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

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
          PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetRealArray(PetscOptions options,const char pre[],const char name[],PetscReal dvalue[],PetscInt *nmax,PetscBool *set)
{
  const char     *svalue;
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscBool      flag;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidRealPointer(dvalue,4);
  PetscValidIntPointer(nmax,5);

  ierr = PetscOptionsFindPair(options,pre,name,&svalue,&flag);CHKERRQ(ierr);
  if (!flag || !svalue)  { if (set) *set = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (set) *set = PETSC_TRUE;
  ierr = PetscTokenCreate(svalue,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (value && n < *nmax) {
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

.seealso: PetscOptionsGetInt(), PetscOptionsHasName(),
          PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetScalarArray(PetscOptions options,const char pre[],const char name[],PetscScalar dvalue[],PetscInt *nmax,PetscBool *set)
{
  const char     *svalue;
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscBool      flag;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidRealPointer(dvalue,4);
  PetscValidIntPointer(nmax,5);

  ierr = PetscOptionsFindPair(options,pre,name,&svalue,&flag);CHKERRQ(ierr);
  if (!flag || !svalue)  { if (set) *set = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (set) *set = PETSC_TRUE;
  ierr = PetscTokenCreate(svalue,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (value && n < *nmax) {
    ierr = PetscOptionsStringToScalar(value,dvalue++);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    n++;
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0);
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

   Output Parameters:
+  strings - location to copy strings
.  nmax - the number of strings found
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: beginner

   Notes:
   The nmax parameter is used for both input and output.

   The user should pass in an array of pointers to char, to hold all the
   strings returned by this function.

   The user is responsible for deallocating the strings that are
   returned. The Fortran interface for this routine is not supported.

.seealso: PetscOptionsGetInt(), PetscOptionsGetReal(),
          PetscOptionsHasName(), PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode PetscOptionsGetStringArray(PetscOptions options,const char pre[],const char name[],char *strings[],PetscInt *nmax,PetscBool *set)
{
  const char     *svalue;
  char           *value;
  PetscErrorCode ierr;
  PetscInt       n = 0;
  PetscBool      flag;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  PetscValidPointer(strings,4);
  PetscValidIntPointer(nmax,5);

  ierr = PetscOptionsFindPair(options,pre,name,&svalue,&flag);CHKERRQ(ierr);
  if (!flag || !svalue)  { if (set) *set = PETSC_FALSE; *nmax = 0; PetscFunctionReturn(0);}
  if (set) *set = PETSC_TRUE;
  ierr = PetscTokenCreate(svalue,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
  while (value && n < *nmax) {
    ierr = PetscStrallocpy(value,&strings[n]);CHKERRQ(ierr);
    ierr = PetscTokenFind(token,&value);CHKERRQ(ierr);
    n++;
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsDeprecated - mark an option as deprecated, optionally replacing it with a new one

   Prints a deprecation warning, unless an option is supplied to suppress.

   Logically Collective

   Input Parameters:
+  pre - string to prepend to name or NULL
.  oldname - the old, deprecated option
.  newname - the new option, or NULL if option is purely removed
.  version - a string describing the version of first deprecation, e.g. "3.9"
-  info - additional information string, or NULL.

   Options Database Keys:
. -options_suppress_deprecated_warnings - do not print deprecation warnings

   Notes:
   Must be called between PetscOptionsBegin() (or PetscObjectOptionsBegin()) and PetscOptionsEnd().
   Only the proces of rank zero that owns the PetscOptionsItems are argument (managed by PetscOptionsBegin() or
   PetscObjectOptionsBegin() prints the information
   If newname is provided, the old option is replaced. Otherwise, it remains
   in the options database.
   If an option is not replaced, the info argument should be used to advise the user
   on how to proceed.
   There is a limit on the length of the warning printed, so very long strings
   provided as info may be truncated.

   Level: developer

.seealso: PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsScalar(), PetscOptionsBool(), PetscOptionsString(), PetscOptionsSetValue()

@*/
PetscErrorCode PetscOptionsDeprecated_Private(PetscOptionItems *PetscOptionsObject,const char oldname[],const char newname[],const char version[],const char info[])
{
  PetscErrorCode     ierr;
  PetscBool          found,quiet;
  const char         *value;
  const char * const quietopt="-options_suppress_deprecated_warnings";
  char               msg[4096];
  char               *prefix = NULL;
  PetscOptions       options = NULL;
  MPI_Comm           comm = PETSC_COMM_SELF;

  PetscFunctionBegin;
  PetscValidCharPointer(oldname,2);
  PetscValidCharPointer(version,4);
  if (PetscOptionsObject) {
    prefix  = PetscOptionsObject->prefix;
    options = PetscOptionsObject->options;
    comm    = PetscOptionsObject->comm;
  }
  ierr = PetscOptionsFindPair(options,prefix,oldname,&value,&found);CHKERRQ(ierr);
  if (found) {
    if (newname) {
      if (prefix) {
        ierr = PetscOptionsPrefixPush(options,prefix);CHKERRQ(ierr);
      }
      ierr = PetscOptionsSetValue(options,newname,value);CHKERRQ(ierr);
      if (prefix) {
        ierr = PetscOptionsPrefixPop(options);CHKERRQ(ierr);
      }
      ierr = PetscOptionsClearValue(options,oldname);CHKERRQ(ierr);
    }
    quiet = PETSC_FALSE;
    ierr = PetscOptionsGetBool(options,NULL,quietopt,&quiet,NULL);CHKERRQ(ierr);
    if (!quiet) {
      ierr = PetscStrcpy(msg,"** PETSc DEPRECATION WARNING ** : the option ");CHKERRQ(ierr);
      ierr = PetscStrcat(msg,oldname);CHKERRQ(ierr);
      ierr = PetscStrcat(msg," is deprecated as of version ");CHKERRQ(ierr);
      ierr = PetscStrcat(msg,version);CHKERRQ(ierr);
      ierr = PetscStrcat(msg," and will be removed in a future release.");CHKERRQ(ierr);
      if (newname) {
        ierr = PetscStrcat(msg," Please use the option ");CHKERRQ(ierr);
        ierr = PetscStrcat(msg,newname);CHKERRQ(ierr);
        ierr = PetscStrcat(msg," instead.");CHKERRQ(ierr);
      }
      if (info) {
        ierr = PetscStrcat(msg," ");CHKERRQ(ierr);
        ierr = PetscStrcat(msg,info);CHKERRQ(ierr);
      }
      ierr = PetscStrcat(msg," (Silence this warning with ");CHKERRQ(ierr);
      ierr = PetscStrcat(msg,quietopt);CHKERRQ(ierr);
      ierr = PetscStrcat(msg,")\n");CHKERRQ(ierr);
      ierr = PetscPrintf(comm,msg);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
