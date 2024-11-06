/* Define Feature test macros to make sure atoll is available (SVr4, POSIX.1-2001, 4.3BSD, C99), not in (C89 and POSIX.1-1996) */
#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for atoll() */

/*
   These routines simplify the use of command line, file options, etc., and are used to manipulate the options database.
   This provides the low-level interface, the high level interface is in aoptions.c

   Some routines use regular malloc and free because it cannot know  what malloc is requested with the
   options database until it has already processed the input.
*/

#include <petsc/private/petscimpl.h> /*I  "petscsys.h"   I*/
#include <petscviewer.h>
#include <ctype.h>
#if defined(PETSC_HAVE_MALLOC_H)
  #include <malloc.h>
#endif
#if defined(PETSC_HAVE_STRINGS_H)
  #include <strings.h> /* strcasecmp */
#endif

#if defined(PETSC_HAVE_STRCASECMP)
  #define PetscOptNameCmp(a, b) strcasecmp(a, b)
#elif defined(PETSC_HAVE_STRICMP)
  #define PetscOptNameCmp(a, b) stricmp(a, b)
#else
  #define PetscOptNameCmp(a, b) Error_strcasecmp_not_found
#endif

#include <petsc/private/hashtable.h>

/* This assumes ASCII encoding and ignores locale settings */
/* Using tolower() is about 2X slower in microbenchmarks   */
static inline int PetscToLower(int c)
{
  return ((c >= 'A') & (c <= 'Z')) ? c + 'a' - 'A' : c;
}

/* Bob Jenkins's one at a time hash function (case-insensitive) */
static inline unsigned int PetscOptHash(const char key[])
{
  unsigned int hash = 0;
  while (*key) {
    hash += PetscToLower(*key++);
    hash += hash << 10;
    hash ^= hash >> 6;
  }
  hash += hash << 3;
  hash ^= hash >> 11;
  hash += hash << 15;
  return hash;
}

static inline int PetscOptEqual(const char a[], const char b[])
{
  return !PetscOptNameCmp(a, b);
}

KHASH_INIT(HO, kh_cstr_t, int, 1, PetscOptHash, PetscOptEqual)

#define MAXPREFIXES        25
#define MAXOPTIONSMONITORS 5

const char *PetscOptionSources[] = {"code", "command line", "file", "environment"};

// This table holds all the options set by the user
struct _n_PetscOptions {
  PetscOptions previous;

  int                N;      /* number of options */
  int                Nalloc; /* number of allocated options */
  char             **names;  /* option names */
  char             **values; /* option values */
  PetscBool         *used;   /* flag option use */
  PetscOptionSource *source; /* source for option value */
  PetscBool          precedentProcessed;

  /* Hash table */
  khash_t(HO) *ht;

  /* Prefixes */
  int  prefixind;
  int  prefixstack[MAXPREFIXES];
  char prefix[PETSC_MAX_OPTION_NAME];

  /* Aliases */
  int    Na;       /* number or aliases */
  int    Naalloc;  /* number of allocated aliases */
  char **aliases1; /* aliased */
  char **aliases2; /* aliasee */

  /* Help */
  PetscBool help;       /* flag whether "-help" is in the database */
  PetscBool help_intro; /* flag whether "-help intro" is in the database */

  /* Monitors */
  PetscBool monitorFromOptions, monitorCancel;
  PetscErrorCode (*monitor[MAXOPTIONSMONITORS])(const char[], const char[], PetscOptionSource, void *); /* returns control to user after */
  PetscErrorCode (*monitordestroy[MAXOPTIONSMONITORS])(void **);                                        /* callback for monitor destruction */
  void    *monitorcontext[MAXOPTIONSMONITORS];                                                          /* to pass arbitrary user data into monitor */
  PetscInt numbermonitors;                                                                              /* to, for instance, detect options being set */
};

static PetscOptions defaultoptions = NULL; /* the options database routines query this object for options */

/* list of options which precede others, i.e., are processed in PetscOptionsProcessPrecedentFlags() */
/* these options can only take boolean values, the code will crash if given a non-boolean value */
static const char *precedentOptions[] = {"-petsc_ci", "-options_monitor", "-options_monitor_cancel", "-help", "-skip_petscrc"};
enum PetscPrecedentOption {
  PO_CI_ENABLE,
  PO_OPTIONS_MONITOR,
  PO_OPTIONS_MONITOR_CANCEL,
  PO_HELP,
  PO_SKIP_PETSCRC,
  PO_NUM
};

PETSC_INTERN PetscErrorCode PetscOptionsSetValue_Private(PetscOptions, const char[], const char[], int *, PetscOptionSource);
PETSC_INTERN PetscErrorCode PetscOptionsInsertStringYAML_Private(PetscOptions, const char[], PetscOptionSource);

/*
    Options events monitor
*/
static PetscErrorCode PetscOptionsMonitor(PetscOptions options, const char name[], const char value[], PetscOptionSource source)
{
  PetscFunctionBegin;
  if (options->monitorFromOptions) PetscCall(PetscOptionsMonitorDefault(name, value, source, NULL));
  for (PetscInt i = 0; i < options->numbermonitors; i++) PetscCall((*options->monitor[i])(name, value, source, options->monitorcontext[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsCreate - Creates an empty options database.

  Logically Collective

  Output Parameter:
. options - Options database object

  Level: advanced

  Note:
  Though PETSc has a concept of multiple options database the current code uses a single default `PetscOptions` object

  Developer Notes:
  We may want eventually to pass a `MPI_Comm` to determine the ownership of the object

  This object never got developed after being introduced, it is not clear that supporting multiple `PetscOptions` objects is useful

.seealso: `PetscOptionsDestroy()`, `PetscOptionsPush()`, `PetscOptionsPop()`, `PetscOptionsInsert()`, `PetscOptionsSetValue()`
@*/
PetscErrorCode PetscOptionsCreate(PetscOptions *options)
{
  PetscFunctionBegin;
  PetscAssertPointer(options, 1);
  *options = (PetscOptions)calloc(1, sizeof(**options));
  PetscCheck(*options, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate the options database");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsDestroy - Destroys an option database.

  Logically Collective on whatever communicator was associated with the call to `PetscOptionsCreate()`

  Input Parameter:
. options - the `PetscOptions` object

  Level: advanced

.seealso: `PetscOptionsInsert()`, `PetscOptionsPush()`, `PetscOptionsPop()`, `PetscOptionsSetValue()`
@*/
PetscErrorCode PetscOptionsDestroy(PetscOptions *options)
{
  PetscFunctionBegin;
  PetscAssertPointer(options, 1);
  if (!*options) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(!(*options)->previous, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "You are destroying an option that has been used with PetscOptionsPush() but does not have a corresponding PetscOptionsPop()");
  PetscCall(PetscOptionsClear(*options));
  /* XXX what about monitors ? */
  free(*options);
  *options = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    PetscOptionsCreateDefault - Creates the default global options database
*/
PetscErrorCode PetscOptionsCreateDefault(void)
{
  PetscFunctionBegin;
  if (PetscUnlikely(!defaultoptions)) PetscCall(PetscOptionsCreate(&defaultoptions));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsPush - Push a new `PetscOptions` object as the default provider of options
  Allows using different parts of a code to use different options databases

  Logically Collective

  Input Parameter:
. opt - the options obtained with `PetscOptionsCreate()`

  Level: advanced

  Notes:
  Use `PetscOptionsPop()` to return to the previous default options database

  The collectivity of this routine is complex; only the MPI ranks that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

  Developer Notes:
  Though this functionality has been provided it has never been used in PETSc and might be removed.

.seealso: `PetscOptionsPop()`, `PetscOptionsCreate()`, `PetscOptionsInsert()`, `PetscOptionsSetValue()`, `PetscOptionsLeft()`
@*/
PetscErrorCode PetscOptionsPush(PetscOptions opt)
{
  PetscFunctionBegin;
  PetscCall(PetscOptionsCreateDefault());
  opt->previous  = defaultoptions;
  defaultoptions = opt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsPop - Pop the most recent `PetscOptionsPush()` to return to the previous default options

  Logically Collective on whatever communicator was associated with the call to `PetscOptionsCreate()`

  Level: advanced

.seealso: `PetscOptionsCreate()`, `PetscOptionsInsert()`, `PetscOptionsSetValue()`, `PetscOptionsLeft()`
@*/
PetscErrorCode PetscOptionsPop(void)
{
  PetscOptions current = defaultoptions;

  PetscFunctionBegin;
  PetscCheck(defaultoptions, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing default options");
  PetscCheck(defaultoptions->previous, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscOptionsPop() called too many times");
  defaultoptions    = defaultoptions->previous;
  current->previous = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    PetscOptionsDestroyDefault - Destroys the default global options database
*/
PetscErrorCode PetscOptionsDestroyDefault(void)
{
  PetscFunctionBegin;
  if (!defaultoptions) PetscFunctionReturn(PETSC_SUCCESS);
  /* Destroy any options that the user forgot to pop */
  while (defaultoptions->previous) {
    PetscOptions tmp = defaultoptions;

    PetscCall(PetscOptionsPop());
    PetscCall(PetscOptionsDestroy(&tmp));
  }
  PetscCall(PetscOptionsDestroy(&defaultoptions));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsValidKey - PETSc Options database keys must begin with one or two dashes (-) followed by a letter.

  Not Collective

  Input Parameter:
. key - string to check if valid

  Output Parameter:
. valid - `PETSC_TRUE` if a valid key

  Level: intermediate

.seealso: `PetscOptionsCreate()`, `PetscOptionsInsert()`
@*/
PetscErrorCode PetscOptionsValidKey(const char key[], PetscBool *valid)
{
  char               *ptr;
  PETSC_UNUSED double d;

  PetscFunctionBegin;
  if (key) PetscAssertPointer(key, 1);
  PetscAssertPointer(valid, 2);
  *valid = PETSC_FALSE;
  if (!key) PetscFunctionReturn(PETSC_SUCCESS);
  if (key[0] != '-') PetscFunctionReturn(PETSC_SUCCESS);
  if (key[1] == '-') key++;
  if (!isalpha((int)key[1])) PetscFunctionReturn(PETSC_SUCCESS);
  d = strtod(key, &ptr);
  if (ptr != key && !(*ptr == '_' || isalnum((int)*ptr))) PetscFunctionReturn(PETSC_SUCCESS);
  *valid = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscOptionsInsertString_Private(PetscOptions options, const char in_str[], PetscOptionSource source)
{
  char      *first, *second;
  PetscToken token;

  PetscFunctionBegin;
  PetscCall(PetscTokenCreate(in_str, ' ', &token));
  PetscCall(PetscTokenFind(token, &first));
  while (first) {
    PetscBool isfile, isfileyaml, isstringyaml, ispush, ispop, key;

    PetscCall(PetscStrcasecmp(first, "-options_file", &isfile));
    PetscCall(PetscStrcasecmp(first, "-options_file_yaml", &isfileyaml));
    PetscCall(PetscStrcasecmp(first, "-options_string_yaml", &isstringyaml));
    PetscCall(PetscStrcasecmp(first, "-prefix_push", &ispush));
    PetscCall(PetscStrcasecmp(first, "-prefix_pop", &ispop));
    PetscCall(PetscOptionsValidKey(first, &key));
    if (!key) {
      PetscCall(PetscTokenFind(token, &first));
    } else if (isfile) {
      PetscCall(PetscTokenFind(token, &second));
      PetscCall(PetscOptionsInsertFile(PETSC_COMM_SELF, options, second, PETSC_TRUE));
      PetscCall(PetscTokenFind(token, &first));
    } else if (isfileyaml) {
      PetscCall(PetscTokenFind(token, &second));
      PetscCall(PetscOptionsInsertFileYAML(PETSC_COMM_SELF, options, second, PETSC_TRUE));
      PetscCall(PetscTokenFind(token, &first));
    } else if (isstringyaml) {
      PetscCall(PetscTokenFind(token, &second));
      PetscCall(PetscOptionsInsertStringYAML_Private(options, second, source));
      PetscCall(PetscTokenFind(token, &first));
    } else if (ispush) {
      PetscCall(PetscTokenFind(token, &second));
      PetscCall(PetscOptionsPrefixPush(options, second));
      PetscCall(PetscTokenFind(token, &first));
    } else if (ispop) {
      PetscCall(PetscOptionsPrefixPop(options));
      PetscCall(PetscTokenFind(token, &first));
    } else {
      PetscCall(PetscTokenFind(token, &second));
      PetscCall(PetscOptionsValidKey(second, &key));
      if (!key) {
        PetscCall(PetscOptionsSetValue_Private(options, first, second, NULL, source));
        PetscCall(PetscTokenFind(token, &first));
      } else {
        PetscCall(PetscOptionsSetValue_Private(options, first, NULL, NULL, source));
        first = second;
      }
    }
  }
  PetscCall(PetscTokenDestroy(&token));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsInsertString - Inserts options into the database from a string

  Logically Collective

  Input Parameters:
+ options - options object
- in_str  - string that contains options separated by blanks

  Level: intermediate

  The collectivity of this routine is complex; only the MPI processes that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

   Contributed by Boyana Norris

.seealso: `PetscOptionsSetValue()`, `PetscOptionsView()`, `PetscOptionsHasName()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetReal()`, `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsInsertFile()`
@*/
PetscErrorCode PetscOptionsInsertString(PetscOptions options, const char in_str[])
{
  PetscFunctionBegin;
  PetscCall(PetscOptionsInsertString_Private(options, in_str, PETSC_OPT_CODE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Returns a line (ended by a \n, \r or null character of any length. Result should be freed with free()
*/
static char *Petscgetline(FILE *f)
{
  size_t size = 0;
  size_t len  = 0;
  size_t last = 0;
  char  *buf  = NULL;

  if (feof(f)) return NULL;
  do {
    size += 1024;                             /* BUFSIZ is defined as "the optimal read size for this platform" */
    buf = (char *)realloc((void *)buf, size); /* realloc(NULL,n) is the same as malloc(n) */
    /* Actually do the read. Note that fgets puts a terminal '\0' on the
    end of the string, so we make sure we overwrite this */
    if (!fgets(buf + len, 1024, f)) buf[len] = 0;
    PetscCallAbort(PETSC_COMM_SELF, PetscStrlen(buf, &len));
    last = len - 1;
  } while (!feof(f) && buf[last] != '\n' && buf[last] != '\r');
  if (len) return buf;
  free(buf);
  return NULL;
}

static PetscErrorCode PetscOptionsFilename(MPI_Comm comm, const char file[], char filename[PETSC_MAX_PATH_LEN], PetscBool *yaml)
{
  char fname[PETSC_MAX_PATH_LEN + 8], path[PETSC_MAX_PATH_LEN + 8], *tail;

  PetscFunctionBegin;
  *yaml = PETSC_FALSE;
  PetscCall(PetscStrreplace(comm, file, fname, sizeof(fname)));
  PetscCall(PetscFixFilename(fname, path));
  PetscCall(PetscStrendswith(path, ":yaml", yaml));
  if (*yaml) {
    PetscCall(PetscStrrchr(path, ':', &tail));
    tail[-1] = 0; /* remove ":yaml" suffix from path */
  }
  PetscCall(PetscStrncpy(filename, path, PETSC_MAX_PATH_LEN));
  /* check for standard YAML and JSON filename extensions */
  if (!*yaml) PetscCall(PetscStrendswith(filename, ".yaml", yaml));
  if (!*yaml) PetscCall(PetscStrendswith(filename, ".yml", yaml));
  if (!*yaml) PetscCall(PetscStrendswith(filename, ".json", yaml));
  if (!*yaml) { /* check file contents */
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (rank == 0) {
      FILE *fh = fopen(filename, "r");
      if (fh) {
        char buf[6] = "";
        if (fread(buf, 1, 6, fh) > 0) {
          PetscCall(PetscStrncmp(buf, "%YAML ", 6, yaml));          /* check for '%YAML' tag */
          if (!*yaml) PetscCall(PetscStrncmp(buf, "---", 3, yaml)); /* check for document start */
        }
        (void)fclose(fh);
      }
    }
    PetscCallMPI(MPI_Bcast(yaml, 1, MPIU_BOOL, 0, comm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscOptionsInsertFilePetsc(MPI_Comm comm, PetscOptions options, const char file[], PetscBool require)
{
  char       *string, *vstring = NULL, *astring = NULL, *packed = NULL;
  char       *tokens[4];
  PetscCount  bytes;
  size_t      len;
  FILE       *fd;
  PetscToken  token = NULL;
  int         err;
  char       *cmatch = NULL;
  const char  cmt    = '#';
  PetscInt    line   = 1;
  PetscMPIInt rank, cnt = 0, acnt = 0, counts[2];
  PetscBool   isdir, alias = PETSC_FALSE, valid;

  PetscFunctionBegin;
  PetscCall(PetscMemzero(tokens, sizeof(tokens)));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    char fpath[PETSC_MAX_PATH_LEN];
    char fname[PETSC_MAX_PATH_LEN];

    PetscCall(PetscStrreplace(PETSC_COMM_SELF, file, fpath, sizeof(fpath)));
    PetscCall(PetscFixFilename(fpath, fname));

    fd = fopen(fname, "r");
    PetscCall(PetscTestDirectory(fname, 'r', &isdir));
    PetscCheck(!isdir || !require, PETSC_COMM_SELF, PETSC_ERR_USER, "Specified options file %s is a directory", fname);
    if (fd && !isdir) {
      PetscSegBuffer vseg, aseg;
      PetscCall(PetscSegBufferCreate(1, 4000, &vseg));
      PetscCall(PetscSegBufferCreate(1, 2000, &aseg));

      /* the following line will not work when opening initial files (like .petscrc) since info is not yet set */
      PetscCall(PetscInfo(NULL, "Opened options file %s\n", file));

      while ((string = Petscgetline(fd))) {
        /* eliminate comments from each line */
        PetscCall(PetscStrchr(string, cmt, &cmatch));
        if (cmatch) *cmatch = 0;
        PetscCall(PetscStrlen(string, &len));
        /* replace tabs, ^M, \n with " " */
        for (size_t i = 0; i < len; i++) {
          if (string[i] == '\t' || string[i] == '\r' || string[i] == '\n') string[i] = ' ';
        }
        PetscCall(PetscTokenCreate(string, ' ', &token));
        PetscCall(PetscTokenFind(token, &tokens[0]));
        if (!tokens[0]) {
          goto destroy;
        } else if (!tokens[0][0]) { /* if token 0 is empty (string begins with spaces), redo */
          PetscCall(PetscTokenFind(token, &tokens[0]));
        }
        for (PetscInt i = 1; i < 4; i++) PetscCall(PetscTokenFind(token, &tokens[i]));
        if (!tokens[0]) {
          goto destroy;
        } else if (tokens[0][0] == '-') {
          PetscCall(PetscOptionsValidKey(tokens[0], &valid));
          PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Error in options file %s line %" PetscInt_FMT ": invalid option %s", fname, line, tokens[0]);
          PetscCall(PetscStrlen(tokens[0], &len));
          PetscCall(PetscSegBufferGet(vseg, len + 1, &vstring));
          PetscCall(PetscArraycpy(vstring, tokens[0], len));
          vstring[len] = ' ';
          if (tokens[1]) {
            PetscCall(PetscOptionsValidKey(tokens[1], &valid));
            PetscCheck(!valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Error in options file %s line %" PetscInt_FMT ": cannot specify two options per line (%s %s)", fname, line, tokens[0], tokens[1]);
            PetscCall(PetscStrlen(tokens[1], &len));
            PetscCall(PetscSegBufferGet(vseg, len + 3, &vstring));
            vstring[0] = '"';
            PetscCall(PetscArraycpy(vstring + 1, tokens[1], len));
            vstring[len + 1] = '"';
            vstring[len + 2] = ' ';
          }
        } else {
          PetscCall(PetscStrcasecmp(tokens[0], "alias", &alias));
          if (alias) {
            PetscCall(PetscOptionsValidKey(tokens[1], &valid));
            PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Error in options file %s line %" PetscInt_FMT ": invalid aliased option %s", fname, line, tokens[1]);
            PetscCheck(tokens[2], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Error in options file %s line %" PetscInt_FMT ": alias missing for %s", fname, line, tokens[1]);
            PetscCall(PetscOptionsValidKey(tokens[2], &valid));
            PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Error in options file %s line %" PetscInt_FMT ": invalid aliasee option %s", fname, line, tokens[2]);
            PetscCall(PetscStrlen(tokens[1], &len));
            PetscCall(PetscSegBufferGet(aseg, len + 1, &astring));
            PetscCall(PetscArraycpy(astring, tokens[1], len));
            astring[len] = ' ';

            PetscCall(PetscStrlen(tokens[2], &len));
            PetscCall(PetscSegBufferGet(aseg, len + 1, &astring));
            PetscCall(PetscArraycpy(astring, tokens[2], len));
            astring[len] = ' ';
          } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown first token in options file %s line %" PetscInt_FMT ": %s", fname, line, tokens[0]);
        }
        {
          const char *extraToken = alias ? tokens[3] : tokens[2];
          PetscCheck(!extraToken, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Error in options file %s line %" PetscInt_FMT ": extra token %s", fname, line, extraToken);
        }
      destroy:
        free(string);
        PetscCall(PetscTokenDestroy(&token));
        alias = PETSC_FALSE;
        line++;
      }
      err = fclose(fd);
      PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file %s", fname);
      PetscCall(PetscSegBufferGetSize(aseg, &bytes)); /* size without null termination */
      PetscCall(PetscMPIIntCast(bytes, &acnt));
      PetscCall(PetscSegBufferGet(aseg, 1, &astring));
      astring[0] = 0;
      PetscCall(PetscSegBufferGetSize(vseg, &bytes)); /* size without null termination */
      PetscCall(PetscMPIIntCast(bytes, &cnt));
      PetscCall(PetscSegBufferGet(vseg, 1, &vstring));
      vstring[0] = 0;
      PetscCall(PetscMalloc1(2 + acnt + cnt, &packed));
      PetscCall(PetscSegBufferExtractTo(aseg, packed));
      PetscCall(PetscSegBufferExtractTo(vseg, packed + acnt + 1));
      PetscCall(PetscSegBufferDestroy(&aseg));
      PetscCall(PetscSegBufferDestroy(&vseg));
    } else PetscCheck(!require, PETSC_COMM_SELF, PETSC_ERR_USER, "Unable to open options file %s", fname);
  }

  counts[0] = acnt;
  counts[1] = cnt;
  err       = MPI_Bcast(counts, 2, MPI_INT, 0, comm);
  PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in first MPI collective call, could be caused by using an incorrect mpiexec or a network problem, it can be caused by having VPN running: see https://petsc.org/release/faq/");
  acnt = counts[0];
  cnt  = counts[1];
  if (rank) PetscCall(PetscMalloc1(2 + acnt + cnt, &packed));
  if (acnt || cnt) {
    PetscCallMPI(MPI_Bcast(packed, 2 + acnt + cnt, MPI_CHAR, 0, comm));
    astring = packed;
    vstring = packed + acnt + 1;
  }

  if (acnt) {
    PetscCall(PetscTokenCreate(astring, ' ', &token));
    PetscCall(PetscTokenFind(token, &tokens[0]));
    while (tokens[0]) {
      PetscCall(PetscTokenFind(token, &tokens[1]));
      PetscCall(PetscOptionsSetAlias(options, tokens[0], tokens[1]));
      PetscCall(PetscTokenFind(token, &tokens[0]));
    }
    PetscCall(PetscTokenDestroy(&token));
  }

  if (cnt) PetscCall(PetscOptionsInsertString_Private(options, vstring, PETSC_OPT_FILE));
  PetscCall(PetscFree(packed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsInsertFile - Inserts options into the database from a file.

  Collective

  Input Parameters:
+ comm    - the processes that will share the options (usually `PETSC_COMM_WORLD`)
. options - options database, use `NULL` for default global database
. file    - name of file,
           ".yml" and ".yaml" filename extensions are inserted as YAML options,
           append ":yaml" to filename to force YAML options.
- require - if `PETSC_TRUE` will generate an error if the file does not exist

  Level: developer

  Notes:
  Use  # for lines that are comments and which should be ignored.
  Usually, instead of using this command, one should list the file name in the call to `PetscInitialize()`, this insures that certain options
  such as `-log_view` or `-malloc_debug` are processed properly. This routine only sets options into the options database that will be processed by later
  calls to `XXXSetFromOptions()`, it should not be used for options listed under PetscInitialize().
  The collectivity of this routine is complex; only the MPI processes in comm will
  have the effect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

.seealso: `PetscOptionsSetValue()`, `PetscOptionsView()`, `PetscOptionsHasName()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetReal()`, `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsInsertFile(MPI_Comm comm, PetscOptions options, const char file[], PetscBool require)
{
  char      filename[PETSC_MAX_PATH_LEN];
  PetscBool yaml;

  PetscFunctionBegin;
  PetscCall(PetscOptionsFilename(comm, file, filename, &yaml));
  if (yaml) {
    PetscCall(PetscOptionsInsertFileYAML(comm, options, filename, require));
  } else {
    PetscCall(PetscOptionsInsertFilePetsc(comm, options, filename, require));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsInsertArgs - Inserts options into the database from a array of strings

  Logically Collective

  Input Parameters:
+ options - options object
. argc    - the array length
- args    - the string array

  Level: intermediate

.seealso: `PetscOptions`, `PetscOptionsInsertString()`, `PetscOptionsInsertFile()`
@*/
PetscErrorCode PetscOptionsInsertArgs(PetscOptions options, int argc, char *args[])
{
  MPI_Comm     comm  = PETSC_COMM_WORLD;
  int          left  = PetscMax(argc, 0);
  char *const *eargs = args;

  PetscFunctionBegin;
  while (left) {
    PetscBool isfile, isfileyaml, isstringyaml, ispush, ispop, key;
    PetscCall(PetscStrcasecmp(eargs[0], "-options_file", &isfile));
    PetscCall(PetscStrcasecmp(eargs[0], "-options_file_yaml", &isfileyaml));
    PetscCall(PetscStrcasecmp(eargs[0], "-options_string_yaml", &isstringyaml));
    PetscCall(PetscStrcasecmp(eargs[0], "-prefix_push", &ispush));
    PetscCall(PetscStrcasecmp(eargs[0], "-prefix_pop", &ispop));
    PetscCall(PetscOptionsValidKey(eargs[0], &key));
    if (!key) {
      eargs++;
      left--;
    } else if (isfile) {
      PetscCheck(left > 1 && eargs[1][0] != '-', PETSC_COMM_SELF, PETSC_ERR_USER, "Missing filename for -options_file filename option");
      PetscCall(PetscOptionsInsertFile(comm, options, eargs[1], PETSC_TRUE));
      eargs += 2;
      left -= 2;
    } else if (isfileyaml) {
      PetscCheck(left > 1 && eargs[1][0] != '-', PETSC_COMM_SELF, PETSC_ERR_USER, "Missing filename for -options_file_yaml filename option");
      PetscCall(PetscOptionsInsertFileYAML(comm, options, eargs[1], PETSC_TRUE));
      eargs += 2;
      left -= 2;
    } else if (isstringyaml) {
      PetscCheck(left > 1 && eargs[1][0] != '-', PETSC_COMM_SELF, PETSC_ERR_USER, "Missing string for -options_string_yaml string option");
      PetscCall(PetscOptionsInsertStringYAML_Private(options, eargs[1], PETSC_OPT_CODE));
      eargs += 2;
      left -= 2;
    } else if (ispush) {
      PetscCheck(left > 1, PETSC_COMM_SELF, PETSC_ERR_USER, "Missing prefix for -prefix_push option");
      PetscCheck(eargs[1][0] != '-', PETSC_COMM_SELF, PETSC_ERR_USER, "Missing prefix for -prefix_push option (prefixes cannot start with '-')");
      PetscCall(PetscOptionsPrefixPush(options, eargs[1]));
      eargs += 2;
      left -= 2;
    } else if (ispop) {
      PetscCall(PetscOptionsPrefixPop(options));
      eargs++;
      left--;
    } else {
      PetscBool nextiskey = PETSC_FALSE;
      if (left >= 2) PetscCall(PetscOptionsValidKey(eargs[1], &nextiskey));
      if (left < 2 || nextiskey) {
        PetscCall(PetscOptionsSetValue_Private(options, eargs[0], NULL, NULL, PETSC_OPT_COMMAND_LINE));
        eargs++;
        left--;
      } else {
        PetscCall(PetscOptionsSetValue_Private(options, eargs[0], eargs[1], NULL, PETSC_OPT_COMMAND_LINE));
        eargs += 2;
        left -= 2;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscOptionsStringToBoolIfSet_Private(enum PetscPrecedentOption opt, const char *val[], const PetscBool set[], PetscBool *flg)
{
  PetscFunctionBegin;
  if (set[opt]) {
    PetscCall(PetscOptionsStringToBool(val[opt], flg));
  } else *flg = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Process options with absolute precedence, these are only processed from the command line, not the environment or files */
static PetscErrorCode PetscOptionsProcessPrecedentFlags(PetscOptions options, int argc, char *args[], PetscBool *skip_petscrc, PetscBool *skip_petscrc_set)
{
  const char *const *opt = precedentOptions;
  const size_t       n   = PO_NUM;
  size_t             o;
  int                a;
  const char       **val;
  char             **cval;
  PetscBool         *set, unneeded;

  PetscFunctionBegin;
  PetscCall(PetscCalloc2(n, &cval, n, &set));
  val = (const char **)cval;

  /* Look for options possibly set using PetscOptionsSetValue beforehand */
  for (o = 0; o < n; o++) PetscCall(PetscOptionsFindPair(options, NULL, opt[o], &val[o], &set[o]));

  /* Loop through all args to collect last occurring value of each option */
  for (a = 1; a < argc; a++) {
    PetscBool valid, eq;

    PetscCall(PetscOptionsValidKey(args[a], &valid));
    if (!valid) continue;
    for (o = 0; o < n; o++) {
      PetscCall(PetscStrcasecmp(args[a], opt[o], &eq));
      if (eq) {
        set[o] = PETSC_TRUE;
        if (a == argc - 1 || !args[a + 1] || !args[a + 1][0] || args[a + 1][0] == '-') val[o] = NULL;
        else val[o] = args[a + 1];
        break;
      }
    }
  }

  /* Process flags */
  PetscCall(PetscStrcasecmp(val[PO_HELP], "intro", &options->help_intro));
  if (options->help_intro) options->help = PETSC_TRUE;
  else PetscCall(PetscOptionsStringToBoolIfSet_Private(PO_HELP, val, set, &options->help));
  PetscCall(PetscOptionsStringToBoolIfSet_Private(PO_CI_ENABLE, val, set, &unneeded));
  /* need to manage PO_CI_ENABLE option before the PetscOptionsMonitor is turned on, so its setting is not monitored */
  if (set[PO_CI_ENABLE]) PetscCall(PetscOptionsSetValue_Private(options, opt[PO_CI_ENABLE], val[PO_CI_ENABLE], &a, PETSC_OPT_COMMAND_LINE));
  PetscCall(PetscOptionsStringToBoolIfSet_Private(PO_OPTIONS_MONITOR_CANCEL, val, set, &options->monitorCancel));
  PetscCall(PetscOptionsStringToBoolIfSet_Private(PO_OPTIONS_MONITOR, val, set, &options->monitorFromOptions));
  PetscCall(PetscOptionsStringToBoolIfSet_Private(PO_SKIP_PETSCRC, val, set, skip_petscrc));
  *skip_petscrc_set = set[PO_SKIP_PETSCRC];

  /* Store precedent options in database and mark them as used */
  for (o = 1; o < n; o++) {
    if (set[o]) {
      PetscCall(PetscOptionsSetValue_Private(options, opt[o], val[o], &a, PETSC_OPT_COMMAND_LINE));
      options->used[a] = PETSC_TRUE;
    }
  }
  PetscCall(PetscFree2(cval, set));
  options->precedentProcessed = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscOptionsSkipPrecedent(PetscOptions options, const char name[], PetscBool *flg)
{
  PetscFunctionBegin;
  PetscAssertPointer(flg, 3);
  *flg = PETSC_FALSE;
  if (options->precedentProcessed) {
    for (int i = 0; i < PO_NUM; ++i) {
      if (!PetscOptNameCmp(precedentOptions[i], name)) {
        /* check if precedent option has been set already */
        PetscCall(PetscOptionsFindPair(options, NULL, name, NULL, flg));
        if (*flg) break;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsInsert - Inserts into the options database from the command line,
  the environmental variable and a file.

  Collective on `PETSC_COMM_WORLD`

  Input Parameters:
+ options - options database or `NULL` for the default global database
. argc    - count of number of command line arguments
. args    - the command line arguments
- file    - [optional] PETSc database file, append ":yaml" to filename to specify YAML options format.
          Use `NULL` or empty string to not check for code specific file.
          Also checks ~/.petscrc, .petscrc and petscrc.
          Use -skip_petscrc in the code specific file (or command line) to skip ~/.petscrc, .petscrc and petscrc files.

  Options Database Keys:
+ -options_file <filename>      - read options from a file
- -options_file_yaml <filename> - read options from a YAML file

  Level: advanced

  Notes:
  Since `PetscOptionsInsert()` is automatically called by `PetscInitialize()`,
  the user does not typically need to call this routine. `PetscOptionsInsert()`
  can be called several times, adding additional entries into the database.

  See `PetscInitialize()` for options related to option database monitoring.

.seealso: `PetscOptionsDestroy()`, `PetscOptionsView()`, `PetscOptionsInsertString()`, `PetscOptionsInsertFile()`,
          `PetscInitialize()`
@*/
PetscErrorCode PetscOptionsInsert(PetscOptions options, int *argc, char ***args, const char file[])
{
  MPI_Comm    comm = PETSC_COMM_WORLD;
  PetscMPIInt rank;
  PetscBool   hasArgs     = (argc && *argc) ? PETSC_TRUE : PETSC_FALSE;
  PetscBool   skipPetscrc = PETSC_FALSE, skipPetscrcSet = PETSC_FALSE;
  char       *eoptions = NULL;
  size_t      len      = 0;

  PetscFunctionBegin;
  PetscCheck(!hasArgs || (args && *args), comm, PETSC_ERR_ARG_NULL, "*argc > 1 but *args not given");
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (!options) {
    PetscCall(PetscOptionsCreateDefault());
    options = defaultoptions;
  }
  if (hasArgs) {
    /* process options with absolute precedence */
    PetscCall(PetscOptionsProcessPrecedentFlags(options, *argc, *args, &skipPetscrc, &skipPetscrcSet));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-petsc_ci", &PetscCIEnabled, NULL));
  }
  if (file && file[0]) {
    PetscCall(PetscOptionsInsertFile(comm, options, file, PETSC_TRUE));
    /* if -skip_petscrc has not been set from command line, check whether it has been set in the file */
    if (!skipPetscrcSet) PetscCall(PetscOptionsGetBool(options, NULL, "-skip_petscrc", &skipPetscrc, NULL));
  }
  if (!skipPetscrc) {
    char filename[PETSC_MAX_PATH_LEN];

    PetscCall(PetscGetHomeDirectory(filename, sizeof(filename)));
    PetscCallMPI(MPI_Bcast(filename, (int)sizeof(filename), MPI_CHAR, 0, comm));
    if (filename[0]) PetscCall(PetscStrlcat(filename, "/.petscrc", sizeof(filename)));
    PetscCall(PetscOptionsInsertFile(comm, options, filename, PETSC_FALSE));
    PetscCall(PetscOptionsInsertFile(comm, options, ".petscrc", PETSC_FALSE));
    PetscCall(PetscOptionsInsertFile(comm, options, "petscrc", PETSC_FALSE));
  }

  /* insert environment options */
  if (rank == 0) {
    eoptions = (char *)getenv("PETSC_OPTIONS");
    PetscCall(PetscStrlen(eoptions, &len));
  }
  PetscCallMPI(MPI_Bcast(&len, 1, MPIU_SIZE_T, 0, comm));
  if (len) {
    if (rank) PetscCall(PetscMalloc1(len + 1, &eoptions));
    PetscCallMPI(MPI_Bcast(eoptions, (PetscMPIInt)len, MPI_CHAR, 0, comm));
    if (rank) eoptions[len] = 0;
    PetscCall(PetscOptionsInsertString_Private(options, eoptions, PETSC_OPT_ENVIRONMENT));
    if (rank) PetscCall(PetscFree(eoptions));
  }

  /* insert YAML environment options */
  if (rank == 0) {
    eoptions = (char *)getenv("PETSC_OPTIONS_YAML");
    PetscCall(PetscStrlen(eoptions, &len));
  }
  PetscCallMPI(MPI_Bcast(&len, 1, MPIU_SIZE_T, 0, comm));
  if (len) {
    if (rank) PetscCall(PetscMalloc1(len + 1, &eoptions));
    PetscCallMPI(MPI_Bcast(eoptions, (PetscMPIInt)len, MPI_CHAR, 0, comm));
    if (rank) eoptions[len] = 0;
    PetscCall(PetscOptionsInsertStringYAML_Private(options, eoptions, PETSC_OPT_ENVIRONMENT));
    if (rank) PetscCall(PetscFree(eoptions));
  }

  /* insert command line options here because they take precedence over arguments in petscrc/environment */
  if (hasArgs) PetscCall(PetscOptionsInsertArgs(options, *argc - 1, *args + 1));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-petsc_ci_portable_error_output", &PetscCIEnabledPortableErrorOutput, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* These options are not printed with PetscOptionsView() or PetscOptionsMonitor() when PetscCIEnabled is on */
/* TODO: get the list from the test harness, do not have it hardwired here. Maybe from gmakegentest.py */
static const char *PetscCIOptions[] = {"malloc_debug", "malloc_dump", "malloc_test", "malloc", "nox", "nox_warning", "display", "saws_port_auto_select", "saws_port_auto_select_silent", "vecscatter_mpi1", "check_pointer_intensity", "cuda_initialize", "error_output_stdout", "use_gpu_aware_mpi", "checkfunctionlist", "fp_trap", "petsc_ci", "petsc_ci_portable_error_output", "options_left"};

static PetscBool PetscCIOption(const char *name)
{
  PetscInt  idx;
  PetscBool found;

  if (!PetscCIEnabled) return PETSC_FALSE;
  PetscCallAbort(PETSC_COMM_SELF, PetscEListFind(PETSC_STATIC_ARRAY_LENGTH(PetscCIOptions), PetscCIOptions, name, &idx, &found));
  return found;
}

/*@
  PetscOptionsView - Prints the options that have been loaded. This is
  useful for debugging purposes.

  Logically Collective, No Fortran Support

  Input Parameters:
+ options - options database, use `NULL` for default global database
- viewer  - must be an `PETSCVIEWERASCII` viewer

  Options Database Key:
. -options_view - Activates `PetscOptionsView()` within `PetscFinalize()`

  Level: advanced

  Note:
  Only the MPI rank 0 of the `MPI_Comm` used to create view prints the option values. Other processes
  may have different values but they are not printed.

.seealso: `PetscOptionsAllUsed()`
@*/
PetscErrorCode PetscOptionsView(PetscOptions options, PetscViewer viewer)
{
  PetscInt  i, N = 0;
  PetscBool isascii;

  PetscFunctionBegin;
  if (viewer) PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  options = options ? options : defaultoptions;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_WORLD;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(isascii, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Only supports ASCII viewer");

  for (i = 0; i < options->N; i++) {
    if (PetscCIOption(options->names[i])) continue;
    N++;
  }

  if (!N) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "#No PETSc Option Table entries\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscViewerASCIIPrintf(viewer, "#PETSc Option Table entries:\n"));
  for (i = 0; i < options->N; i++) {
    if (PetscCIOption(options->names[i])) continue;
    if (options->values[i]) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "-%s %s", options->names[i], options->values[i]));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "-%s", options->names[i]));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, " # (source: %s)\n", PetscOptionSources[options->source[i]]));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "#End of PETSc Option Table entries\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Called by error handlers to print options used in run
*/
PetscErrorCode PetscOptionsLeftError(void)
{
  PetscInt i, nopt = 0;

  for (i = 0; i < defaultoptions->N; i++) {
    if (!defaultoptions->used[i]) {
      if (PetscCIOption(defaultoptions->names[i])) continue;
      nopt++;
    }
  }
  if (nopt) {
    PetscCall((*PetscErrorPrintf)("WARNING! There are unused option(s) set! Could be the program crashed before usage or a spelling mistake, etc!\n"));
    for (i = 0; i < defaultoptions->N; i++) {
      if (!defaultoptions->used[i]) {
        if (PetscCIOption(defaultoptions->names[i])) continue;
        if (defaultoptions->values[i]) PetscCall((*PetscErrorPrintf)("  Option left: name:-%s value: %s source: %s\n", defaultoptions->names[i], defaultoptions->values[i], PetscOptionSources[defaultoptions->source[i]]));
        else PetscCall((*PetscErrorPrintf)("  Option left: name:-%s (no value) source: %s\n", defaultoptions->names[i], PetscOptionSources[defaultoptions->source[i]]));
      }
    }
  }
  return PETSC_SUCCESS;
}

PETSC_EXTERN PetscErrorCode PetscOptionsViewError(void)
{
  PetscInt     i, N = 0;
  PetscOptions options = defaultoptions;

  for (i = 0; i < options->N; i++) {
    if (PetscCIOption(options->names[i])) continue;
    N++;
  }

  if (N) {
    PetscCall((*PetscErrorPrintf)("PETSc Option Table entries:\n"));
  } else {
    PetscCall((*PetscErrorPrintf)("No PETSc Option Table entries\n"));
  }
  for (i = 0; i < options->N; i++) {
    if (PetscCIOption(options->names[i])) continue;
    if (options->values[i]) {
      PetscCall((*PetscErrorPrintf)("-%s %s (source: %s)\n", options->names[i], options->values[i], PetscOptionSources[options->source[i]]));
    } else {
      PetscCall((*PetscErrorPrintf)("-%s (source: %s)\n", options->names[i], PetscOptionSources[options->source[i]]));
    }
  }
  return PETSC_SUCCESS;
}

/*@
  PetscOptionsPrefixPush - Designate a prefix to be used by all options insertions to follow.

  Logically Collective

  Input Parameters:
+ options - options database, or `NULL` for the default global database
- prefix  - The string to append to the existing prefix

  Options Database Keys:
+ -prefix_push <some_prefix_> - push the given prefix
- -prefix_pop                 - pop the last prefix

  Level: advanced

  Notes:
  It is common to use this in conjunction with `-options_file` as in
.vb
 -prefix_push system1_ -options_file system1rc -prefix_pop -prefix_push system2_ -options_file system2rc -prefix_pop
.ve
  where the files no longer require all options to be prefixed with `-system2_`.

  The collectivity of this routine is complex; only the MPI processes that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

.seealso: `PetscOptionsPrefixPop()`, `PetscOptionsPush()`, `PetscOptionsPop()`, `PetscOptionsCreate()`, `PetscOptionsSetValue()`
@*/
PetscErrorCode PetscOptionsPrefixPush(PetscOptions options, const char prefix[])
{
  size_t    n;
  PetscInt  start;
  char      key[PETSC_MAX_OPTION_NAME + 1];
  PetscBool valid;

  PetscFunctionBegin;
  PetscAssertPointer(prefix, 2);
  options = options ? options : defaultoptions;
  PetscCheck(options->prefixind < MAXPREFIXES, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Maximum depth of prefix stack %d exceeded, recompile src/sys/objects/options.c with larger value for MAXPREFIXES", MAXPREFIXES);
  key[0] = '-'; /* keys must start with '-' */
  PetscCall(PetscStrncpy(key + 1, prefix, sizeof(key) - 1));
  PetscCall(PetscOptionsValidKey(key, &valid));
  if (!valid && options->prefixind > 0 && isdigit((int)prefix[0])) valid = PETSC_TRUE; /* If the prefix stack is not empty, make numbers a valid prefix */
  PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_USER, "Given prefix \"%s\" not valid (the first character must be a letter%s, do not include leading '-')", prefix, options->prefixind ? " or digit" : "");
  start = options->prefixind ? options->prefixstack[options->prefixind - 1] : 0;
  PetscCall(PetscStrlen(prefix, &n));
  PetscCheck(n + 1 <= sizeof(options->prefix) - start, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Maximum prefix length %zu exceeded", sizeof(options->prefix));
  PetscCall(PetscArraycpy(options->prefix + start, prefix, n + 1));
  options->prefixstack[options->prefixind++] = (int)(start + n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsPrefixPop - Remove the latest options prefix, see `PetscOptionsPrefixPush()` for details

  Logically Collective on the `MPI_Comm` used when called `PetscOptionsPrefixPush()`

  Input Parameter:
. options - options database, or `NULL` for the default global database

  Level: advanced

.seealso: `PetscOptionsPrefixPush()`, `PetscOptionsPush()`, `PetscOptionsPop()`, `PetscOptionsCreate()`, `PetscOptionsSetValue()`
@*/
PetscErrorCode PetscOptionsPrefixPop(PetscOptions options)
{
  PetscInt offset;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  PetscCheck(options->prefixind >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "More prefixes popped than pushed");
  options->prefixind--;
  offset                  = options->prefixind ? options->prefixstack[options->prefixind - 1] : 0;
  options->prefix[offset] = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsClear - Removes all options form the database leaving it empty.

  Logically Collective

  Input Parameter:
. options - options database, use `NULL` for the default global database

  Level: developer

  Note:
  The collectivity of this routine is complex; only the MPI processes that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

  Developer Note:
  Uses `free()` directly because the current option values were set with `malloc()`

.seealso: `PetscOptionsInsert()`
@*/
PetscErrorCode PetscOptionsClear(PetscOptions options)
{
  PetscInt i;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  if (!options) PetscFunctionReturn(PETSC_SUCCESS);

  for (i = 0; i < options->N; i++) {
    if (options->names[i]) free(options->names[i]);
    if (options->values[i]) free(options->values[i]);
  }
  options->N = 0;
  free(options->names);
  free(options->values);
  free(options->used);
  free(options->source);
  options->names  = NULL;
  options->values = NULL;
  options->used   = NULL;
  options->source = NULL;
  options->Nalloc = 0;

  for (i = 0; i < options->Na; i++) {
    free(options->aliases1[i]);
    free(options->aliases2[i]);
  }
  options->Na = 0;
  free(options->aliases1);
  free(options->aliases2);
  options->aliases1 = options->aliases2 = NULL;
  options->Naalloc                      = 0;

  /* destroy hash table */
  kh_destroy(HO, options->ht);
  options->ht = NULL;

  options->prefixind  = 0;
  options->prefix[0]  = 0;
  options->help       = PETSC_FALSE;
  options->help_intro = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsSetAlias - Makes a key and alias for another key

  Logically Collective

  Input Parameters:
+ options - options database, or `NULL` for default global database
. newname - the alias
- oldname - the name that alias will refer to

  Level: advanced

  Note:
  The collectivity of this routine is complex; only the MPI processes that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

  Developer Note:
  Uses `malloc()` directly because PETSc may not be initialized yet.

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`,
          `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsSetAlias(PetscOptions options, const char newname[], const char oldname[])
{
  size_t    len;
  PetscBool valid;

  PetscFunctionBegin;
  PetscAssertPointer(newname, 2);
  PetscAssertPointer(oldname, 3);
  options = options ? options : defaultoptions;
  PetscCall(PetscOptionsValidKey(newname, &valid));
  PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid aliased option %s", newname);
  PetscCall(PetscOptionsValidKey(oldname, &valid));
  PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid aliasee option %s", oldname);

  if (options->Na == options->Naalloc) {
    char **tmpA1, **tmpA2;

    options->Naalloc = PetscMax(4, options->Naalloc * 2);
    tmpA1            = (char **)malloc(options->Naalloc * sizeof(char *));
    tmpA2            = (char **)malloc(options->Naalloc * sizeof(char *));
    for (int i = 0; i < options->Na; ++i) {
      tmpA1[i] = options->aliases1[i];
      tmpA2[i] = options->aliases2[i];
    }
    free(options->aliases1);
    free(options->aliases2);
    options->aliases1 = tmpA1;
    options->aliases2 = tmpA2;
  }
  newname++;
  oldname++;
  PetscCall(PetscStrlen(newname, &len));
  options->aliases1[options->Na] = (char *)malloc((len + 1) * sizeof(char));
  PetscCall(PetscStrncpy(options->aliases1[options->Na], newname, len + 1));
  PetscCall(PetscStrlen(oldname, &len));
  options->aliases2[options->Na] = (char *)malloc((len + 1) * sizeof(char));
  PetscCall(PetscStrncpy(options->aliases2[options->Na], oldname, len + 1));
  ++options->Na;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsSetValue - Sets an option name-value pair in the options
  database, overriding whatever is already present.

  Logically Collective

  Input Parameters:
+ options - options database, use `NULL` for the default global database
. name    - name of option, this SHOULD have the - prepended
- value   - the option value (not used for all options, so can be `NULL`)

  Level: intermediate

  Note:
  This function can be called BEFORE `PetscInitialize()`

  The collectivity of this routine is complex; only the MPI processes that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

  Developer Note:
  Uses `malloc()` directly because PETSc may not be initialized yet.

.seealso: `PetscOptionsInsert()`, `PetscOptionsClearValue()`
@*/
PetscErrorCode PetscOptionsSetValue(PetscOptions options, const char name[], const char value[])
{
  PetscFunctionBegin;
  PetscCall(PetscOptionsSetValue_Private(options, name, value, NULL, PETSC_OPT_CODE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsSetValue_Private(PetscOptions options, const char name[], const char value[], int *pos, PetscOptionSource source)
{
  size_t    len;
  int       n, i;
  char    **names;
  char      fullname[PETSC_MAX_OPTION_NAME] = "";
  PetscBool flg;

  PetscFunctionBegin;
  if (!options) {
    PetscCall(PetscOptionsCreateDefault());
    options = defaultoptions;
  }
  PetscCheck(name[0] == '-', PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "name %s must start with '-'", name);

  PetscCall(PetscOptionsSkipPrecedent(options, name, &flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  name++; /* skip starting dash */

  if (options->prefixind > 0) {
    strncpy(fullname, options->prefix, sizeof(fullname));
    fullname[sizeof(fullname) - 1] = 0;
    strncat(fullname, name, sizeof(fullname) - strlen(fullname) - 1);
    fullname[sizeof(fullname) - 1] = 0;
    name                           = fullname;
  }

  /* check against aliases */
  for (i = 0; i < options->Na; i++) {
    int result = PetscOptNameCmp(options->aliases1[i], name);
    if (!result) {
      name = options->aliases2[i];
      break;
    }
  }

  /* slow search */
  n     = options->N;
  names = options->names;
  for (i = 0; i < options->N; i++) {
    int result = PetscOptNameCmp(names[i], name);
    if (!result) {
      n = i;
      goto setvalue;
    } else if (result > 0) {
      n = i;
      break;
    }
  }
  if (options->N == options->Nalloc) {
    char             **names, **values;
    PetscBool         *used;
    PetscOptionSource *source;

    options->Nalloc = PetscMax(10, options->Nalloc * 2);
    names           = (char **)malloc(options->Nalloc * sizeof(char *));
    values          = (char **)malloc(options->Nalloc * sizeof(char *));
    used            = (PetscBool *)malloc(options->Nalloc * sizeof(PetscBool));
    source          = (PetscOptionSource *)malloc(options->Nalloc * sizeof(PetscOptionSource));
    for (int i = 0; i < options->N; ++i) {
      names[i]  = options->names[i];
      values[i] = options->values[i];
      used[i]   = options->used[i];
      source[i] = options->source[i];
    }
    free(options->names);
    free(options->values);
    free(options->used);
    free(options->source);
    options->names  = names;
    options->values = values;
    options->used   = used;
    options->source = source;
  }

  /* shift remaining values up 1 */
  for (i = options->N; i > n; i--) {
    options->names[i]  = options->names[i - 1];
    options->values[i] = options->values[i - 1];
    options->used[i]   = options->used[i - 1];
    options->source[i] = options->source[i - 1];
  }
  options->names[n]  = NULL;
  options->values[n] = NULL;
  options->used[n]   = PETSC_FALSE;
  options->source[n] = PETSC_OPT_CODE;
  options->N++;

  /* destroy hash table */
  kh_destroy(HO, options->ht);
  options->ht = NULL;

  /* set new name */
  len               = strlen(name);
  options->names[n] = (char *)malloc((len + 1) * sizeof(char));
  PetscCheck(options->names[n], PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate option name");
  strcpy(options->names[n], name);

setvalue:
  /* set new value */
  if (options->values[n]) free(options->values[n]);
  len = value ? strlen(value) : 0;
  if (len) {
    options->values[n] = (char *)malloc((len + 1) * sizeof(char));
    if (!options->values[n]) return PETSC_ERR_MEM;
    strcpy(options->values[n], value);
    options->values[n][len] = '\0';
  } else {
    options->values[n] = NULL;
  }
  options->source[n] = source;

  /* handle -help so that it can be set from anywhere */
  if (!PetscOptNameCmp(name, "help")) {
    options->help       = PETSC_TRUE;
    options->help_intro = (value && !PetscOptNameCmp(value, "intro")) ? PETSC_TRUE : PETSC_FALSE;
    options->used[n]    = PETSC_TRUE;
  }

  PetscCall(PetscOptionsMonitor(options, name, value ? value : "", source));
  if (pos) *pos = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsClearValue - Clears an option name-value pair in the options
  database, overriding whatever is already present.

  Logically Collective

  Input Parameters:
+ options - options database, use `NULL` for the default global database
- name    - name of option, this SHOULD have the - prepended

  Level: intermediate

  Note:
  The collectivity of this routine is complex; only the MPI processes that call this routine will
  have the affect of these options. If some processes that create objects call this routine and others do
  not the code may fail in complicated ways because the same parallel solvers may incorrectly use different options
  on different ranks.

  Developer Note:
  Uses `free()` directly because the options have been set with `malloc()`

.seealso: `PetscOptionsInsert()`
@*/
PetscErrorCode PetscOptionsClearValue(PetscOptions options, const char name[])
{
  int    N, n, i;
  char **names;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  PetscCheck(name[0] == '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Name must begin with '-': Instead %s", name);
  if (!PetscOptNameCmp(name, "-help")) options->help = options->help_intro = PETSC_FALSE;

  name++; /* skip starting dash */

  /* slow search */
  N = n = options->N;
  names = options->names;
  for (i = 0; i < N; i++) {
    int result = PetscOptNameCmp(names[i], name);
    if (!result) {
      n = i;
      break;
    } else if (result > 0) {
      n = N;
      break;
    }
  }
  if (n == N) PetscFunctionReturn(PETSC_SUCCESS); /* it was not present */

  /* remove name and value */
  if (options->names[n]) free(options->names[n]);
  if (options->values[n]) free(options->values[n]);
  /* shift remaining values down 1 */
  for (i = n; i < N - 1; i++) {
    options->names[i]  = options->names[i + 1];
    options->values[i] = options->values[i + 1];
    options->used[i]   = options->used[i + 1];
    options->source[i] = options->source[i + 1];
  }
  options->N--;

  /* destroy hash table */
  kh_destroy(HO, options->ht);
  options->ht = NULL;

  PetscCall(PetscOptionsMonitor(options, name, NULL, PETSC_OPT_CODE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsFindPair - Gets an option name-value pair from the options database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for the default global database
. pre     - the string to prepend to the name or `NULL`, this SHOULD NOT have the "-" prepended
- name    - name of option, this SHOULD have the "-" prepended

  Output Parameters:
+ value - the option value (optional, not used for all options)
- set   - whether the option is set (optional)

  Level: developer

  Note:
  Each process may find different values or no value depending on how options were inserted into the database

.seealso: `PetscOptionsSetValue()`, `PetscOptionsClearValue()`
@*/
PetscErrorCode PetscOptionsFindPair(PetscOptions options, const char pre[], const char name[], const char *value[], PetscBool *set)
{
  char      buf[PETSC_MAX_OPTION_NAME];
  PetscBool usehashtable = PETSC_TRUE;
  PetscBool matchnumbers = PETSC_TRUE;

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  PetscCheck(!pre || !PetscUnlikely(pre[0] == '-'), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Prefix cannot begin with '-': Instead %s", pre);
  PetscCheck(name[0] == '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Name must begin with '-': Instead %s", name);

  name++; /* skip starting dash */

  /* append prefix to name, if prefix="foo_" and option='--bar", prefixed option is --foo_bar */
  if (pre && pre[0]) {
    char *ptr = buf;
    if (name[0] == '-') {
      *ptr++ = '-';
      name++;
    }
    PetscCall(PetscStrncpy(ptr, pre, buf + sizeof(buf) - ptr));
    PetscCall(PetscStrlcat(buf, name, sizeof(buf)));
    name = buf;
  }

  if (PetscDefined(USE_DEBUG)) {
    PetscBool valid;
    char      key[PETSC_MAX_OPTION_NAME + 1] = "-";
    PetscCall(PetscStrncpy(key + 1, name, sizeof(key) - 1));
    PetscCall(PetscOptionsValidKey(key, &valid));
    PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid option '%s' obtained from pre='%s' and name='%s'", key, pre ? pre : "", name);
  }

  if (!options->ht && usehashtable) {
    int          i, ret;
    khiter_t     it;
    khash_t(HO) *ht;
    ht = kh_init(HO);
    PetscCheck(ht, PETSC_COMM_SELF, PETSC_ERR_MEM, "Hash table allocation failed");
    ret = kh_resize(HO, ht, options->N * 2); /* twice the required size to reduce risk of collisions */
    PetscCheck(!ret, PETSC_COMM_SELF, PETSC_ERR_MEM, "Hash table allocation failed");
    for (i = 0; i < options->N; i++) {
      it = kh_put(HO, ht, options->names[i], &ret);
      PetscCheck(ret == 1, PETSC_COMM_SELF, PETSC_ERR_MEM, "Hash table allocation failed");
      kh_val(ht, it) = i;
    }
    options->ht = ht;
  }

  if (usehashtable) { /* fast search */
    khash_t(HO) *ht = options->ht;
    khiter_t     it = kh_get(HO, ht, name);
    if (it != kh_end(ht)) {
      int i            = kh_val(ht, it);
      options->used[i] = PETSC_TRUE;
      if (value) *value = options->values[i];
      if (set) *set = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else { /* slow search */
    int i, N = options->N;
    for (i = 0; i < N; i++) {
      int result = PetscOptNameCmp(options->names[i], name);
      if (!result) {
        options->used[i] = PETSC_TRUE;
        if (value) *value = options->values[i];
        if (set) *set = PETSC_TRUE;
        PetscFunctionReturn(PETSC_SUCCESS);
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
    int i, j, cnt = 0, locs[16], loce[16];
    /* determine the location and number of all _%d_ in the key */
    for (i = 0; name[i]; i++) {
      if (name[i] == '_') {
        for (j = i + 1; name[j]; j++) {
          if (name[j] >= '0' && name[j] <= '9') continue;
          if (name[j] == '_' && j > i + 1) { /* found a number */
            locs[cnt]   = i + 1;
            loce[cnt++] = j + 1;
          }
          i = j - 1;
          break;
        }
      }
    }
    for (i = 0; i < cnt; i++) {
      PetscBool found;
      char      opt[PETSC_MAX_OPTION_NAME + 1] = "-", tmp[PETSC_MAX_OPTION_NAME];
      PetscCall(PetscStrncpy(tmp, name, PetscMin((size_t)(locs[i] + 1), sizeof(tmp))));
      PetscCall(PetscStrlcat(opt, tmp, sizeof(opt)));
      PetscCall(PetscStrlcat(opt, name + loce[i], sizeof(opt)));
      PetscCall(PetscOptionsFindPair(options, NULL, opt, value, &found));
      if (found) {
        if (set) *set = PETSC_TRUE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
  }

  if (set) *set = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check whether any option begins with pre+name */
PETSC_EXTERN PetscErrorCode PetscOptionsFindPairPrefix_Private(PetscOptions options, const char pre[], const char name[], const char *option[], const char *value[], PetscBool *set)
{
  char buf[PETSC_MAX_OPTION_NAME];
  int  numCnt = 0, locs[16], loce[16];

  PetscFunctionBegin;
  options = options ? options : defaultoptions;
  PetscCheck(!pre || pre[0] != '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Prefix cannot begin with '-': Instead %s", pre);
  PetscCheck(name[0] == '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Name must begin with '-': Instead %s", name);

  name++; /* skip starting dash */

  /* append prefix to name, if prefix="foo_" and option='--bar", prefixed option is --foo_bar */
  if (pre && pre[0]) {
    char *ptr = buf;
    if (name[0] == '-') {
      *ptr++ = '-';
      name++;
    }
    PetscCall(PetscStrncpy(ptr, pre, sizeof(buf) - ((ptr == buf) ? 0 : 1)));
    PetscCall(PetscStrlcat(buf, name, sizeof(buf)));
    name = buf;
  }

  if (PetscDefined(USE_DEBUG)) {
    PetscBool valid;
    char      key[PETSC_MAX_OPTION_NAME + 1] = "-";
    PetscCall(PetscStrncpy(key + 1, name, sizeof(key) - 1));
    PetscCall(PetscOptionsValidKey(key, &valid));
    PetscCheck(valid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid option '%s' obtained from pre='%s' and name='%s'", key, pre ? pre : "", name);
  }

  /* determine the location and number of all _%d_ in the key */
  {
    int i, j;
    for (i = 0; name[i]; i++) {
      if (name[i] == '_') {
        for (j = i + 1; name[j]; j++) {
          if (name[j] >= '0' && name[j] <= '9') continue;
          if (name[j] == '_' && j > i + 1) { /* found a number */
            locs[numCnt]   = i + 1;
            loce[numCnt++] = j + 1;
          }
          i = j - 1;
          break;
        }
      }
    }
  }

  /* slow search */
  for (int c = -1; c < numCnt; ++c) {
    char   opt[PETSC_MAX_OPTION_NAME + 2] = "";
    size_t len;

    if (c < 0) {
      PetscCall(PetscStrncpy(opt, name, sizeof(opt)));
    } else {
      PetscCall(PetscStrncpy(opt, name, PetscMin((size_t)(locs[c] + 1), sizeof(opt))));
      PetscCall(PetscStrlcat(opt, name + loce[c], sizeof(opt) - 1));
    }
    PetscCall(PetscStrlen(opt, &len));
    for (int i = 0; i < options->N; i++) {
      PetscBool match;

      PetscCall(PetscStrncmp(options->names[i], opt, len, &match));
      if (match) {
        options->used[i] = PETSC_TRUE;
        if (option) *option = options->names[i];
        if (value) *value = options->values[i];
        if (set) *set = PETSC_TRUE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
  }

  if (set) *set = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsReject - Generates an error if a certain option is given.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - the option prefix (may be `NULL`)
. name    - the option name one is seeking
- mess    - error message (may be `NULL`)

  Level: advanced

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`,
          `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsReject(PetscOptions options, const char pre[], const char name[], const char mess[])
{
  PetscBool flag = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHasName(options, pre, name, &flag));
  if (flag) {
    PetscCheck(!mess || !mess[0], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Program has disabled option: -%s%s with %s", pre ? pre : "", name + 1, mess);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Program has disabled option: -%s%s", pre ? pre : "", name + 1);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsHasHelp - Determines whether the "-help" option is in the database.

  Not Collective

  Input Parameter:
. options - options database, use `NULL` for default global database

  Output Parameter:
. set - `PETSC_TRUE` if found else `PETSC_FALSE`.

  Level: advanced

.seealso: `PetscOptionsHasName()`
@*/
PetscErrorCode PetscOptionsHasHelp(PetscOptions options, PetscBool *set)
{
  PetscFunctionBegin;
  PetscAssertPointer(set, 2);
  options = options ? options : defaultoptions;
  *set    = options->help;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsHasHelpIntro_Internal(PetscOptions options, PetscBool *set)
{
  PetscFunctionBegin;
  PetscAssertPointer(set, 2);
  options = options ? options : defaultoptions;
  *set    = options->help_intro;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsHasName - Determines whether a certain option is given in the database. This returns true whether the option is a number, string or Boolean, even
  if its value is set to false.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to the name or `NULL`
- name    - the option one is seeking

  Output Parameter:
. set - `PETSC_TRUE` if found else `PETSC_FALSE`.

  Level: beginner

  Note:
  In many cases you probably want to use `PetscOptionsGetBool()` instead of calling this, to allowing toggling values.

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsHasName(PetscOptions options, const char pre[], const char name[], PetscBool *set)
{
  const char *value;
  PetscBool   flag;

  PetscFunctionBegin;
  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (set) *set = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetAll - Lists all the options the program was run with in a single string.

  Not Collective

  Input Parameter:
. options - the options database, use `NULL` for the default global database

  Output Parameter:
. copts - pointer where string pointer is stored

  Level: advanced

  Notes:
  The array and each entry in the array should be freed with `PetscFree()`

  Each process may have different values depending on how the options were inserted into the database

.seealso: `PetscOptionsAllUsed()`, `PetscOptionsView()`, `PetscOptionsPush()`, `PetscOptionsPop()`
@*/
PetscErrorCode PetscOptionsGetAll(PetscOptions options, char *copts[])
{
  PetscInt i;
  size_t   len = 1, lent = 0;
  char    *coptions = NULL;

  PetscFunctionBegin;
  PetscAssertPointer(copts, 2);
  options = options ? options : defaultoptions;
  /* count the length of the required string */
  for (i = 0; i < options->N; i++) {
    PetscCall(PetscStrlen(options->names[i], &lent));
    len += 2 + lent;
    if (options->values[i]) {
      PetscCall(PetscStrlen(options->values[i], &lent));
      len += 1 + lent;
    }
  }
  PetscCall(PetscMalloc1(len, &coptions));
  coptions[0] = 0;
  for (i = 0; i < options->N; i++) {
    PetscCall(PetscStrlcat(coptions, "-", len));
    PetscCall(PetscStrlcat(coptions, options->names[i], len));
    PetscCall(PetscStrlcat(coptions, " ", len));
    if (options->values[i]) {
      PetscCall(PetscStrlcat(coptions, options->values[i], len));
      PetscCall(PetscStrlcat(coptions, " ", len));
    }
  }
  *copts = coptions;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsUsed - Indicates if PETSc has used a particular option set in the database

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
- name    - string name of option

  Output Parameter:
. used - `PETSC_TRUE` if the option was used, otherwise false, including if option was not found in options database

  Level: advanced

  Note:
  The value returned may be different on each process and depends on which options have been processed
  on the given process

.seealso: `PetscOptionsView()`, `PetscOptionsLeft()`, `PetscOptionsAllUsed()`
@*/
PetscErrorCode PetscOptionsUsed(PetscOptions options, const char *name, PetscBool *used)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscAssertPointer(name, 2);
  PetscAssertPointer(used, 3);
  options = options ? options : defaultoptions;
  *used   = PETSC_FALSE;
  for (i = 0; i < options->N; i++) {
    PetscCall(PetscStrcasecmp(options->names[i], name, used));
    if (*used) {
      *used = options->used[i];
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsAllUsed - Returns a count of the number of options in the
  database that have never been selected.

  Not Collective

  Input Parameter:
. options - options database, use `NULL` for default global database

  Output Parameter:
. N - count of options not used

  Level: advanced

  Note:
  The value returned may be different on each process and depends on which options have been processed
  on the given process

.seealso: `PetscOptionsView()`
@*/
PetscErrorCode PetscOptionsAllUsed(PetscOptions options, PetscInt *N)
{
  PetscInt i, n = 0;

  PetscFunctionBegin;
  PetscAssertPointer(N, 2);
  options = options ? options : defaultoptions;
  for (i = 0; i < options->N; i++) {
    if (!options->used[i]) n++;
  }
  *N = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsLeft - Prints to screen any options that were set and never used.

  Not Collective

  Input Parameter:
. options - options database; use `NULL` for default global database

  Options Database Key:
. -options_left - activates `PetscOptionsAllUsed()` within `PetscFinalize()`

  Level: advanced

  Notes:
  This is rarely used directly, it is called by `PetscFinalize()` in debug more or if -options_left
  is passed otherwise to help users determine possible mistakes in their usage of options. This
  only prints values on process zero of `PETSC_COMM_WORLD`.

  Other processes depending the objects
  used may have different options that are left unused.

.seealso: `PetscOptionsAllUsed()`
@*/
PetscErrorCode PetscOptionsLeft(PetscOptions options)
{
  PetscInt     i;
  PetscInt     cnt = 0;
  PetscOptions toptions;

  PetscFunctionBegin;
  toptions = options ? options : defaultoptions;
  for (i = 0; i < toptions->N; i++) {
    if (!toptions->used[i]) {
      if (PetscCIOption(toptions->names[i])) continue;
      if (toptions->values[i]) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Option left: name:-%s value: %s source: %s\n", toptions->names[i], toptions->values[i], PetscOptionSources[toptions->source[i]]));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Option left: name:-%s (no value) source: %s\n", toptions->names[i], PetscOptionSources[toptions->source[i]]));
      }
    }
  }
  if (!options) {
    toptions = defaultoptions;
    while (toptions->previous) {
      cnt++;
      toptions = toptions->previous;
    }
    if (cnt) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Option left: You may have forgotten some calls to PetscOptionsPop(),\n             PetscOptionsPop() has been called %" PetscInt_FMT " less times than PetscOptionsPush()\n", cnt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsLeftGet - Returns all options that were set and never used.

  Not Collective

  Input Parameter:
. options - options database, use `NULL` for default global database

  Output Parameters:
+ N      - count of options not used
. names  - names of options not used
- values - values of options not used

  Level: advanced

  Notes:
  Users should call `PetscOptionsLeftRestore()` to free the memory allocated in this routine

  The value returned may be different on each process and depends on which options have been processed
  on the given process

.seealso: `PetscOptionsAllUsed()`, `PetscOptionsLeft()`
@*/
PetscErrorCode PetscOptionsLeftGet(PetscOptions options, PetscInt *N, char **names[], char **values[])
{
  PetscInt i, n;

  PetscFunctionBegin;
  if (N) PetscAssertPointer(N, 2);
  if (names) PetscAssertPointer(names, 3);
  if (values) PetscAssertPointer(values, 4);
  options = options ? options : defaultoptions;

  /* The number of unused PETSc options */
  n = 0;
  for (i = 0; i < options->N; i++) {
    if (PetscCIOption(options->names[i])) continue;
    if (!options->used[i]) n++;
  }
  if (N) *N = n;
  if (names) PetscCall(PetscMalloc1(n, names));
  if (values) PetscCall(PetscMalloc1(n, values));

  n = 0;
  if (names || values) {
    for (i = 0; i < options->N; i++) {
      if (!options->used[i]) {
        if (PetscCIOption(options->names[i])) continue;
        if (names) (*names)[n] = options->names[i];
        if (values) (*values)[n] = options->values[i];
        n++;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsLeftRestore - Free memory for the unused PETSc options obtained using `PetscOptionsLeftGet()`.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. N       - count of options not used
. names   - names of options not used
- values  - values of options not used

  Level: advanced

  Notes:
  The user should pass the same pointer to `N` as they did when calling `PetscOptionsLeftGet()`

.seealso: `PetscOptionsAllUsed()`, `PetscOptionsLeft()`, `PetscOptionsLeftGet()`
@*/
PetscErrorCode PetscOptionsLeftRestore(PetscOptions options, PetscInt *N, char **names[], char **values[])
{
  PetscFunctionBegin;
  (void)options;
  if (N) PetscAssertPointer(N, 2);
  if (names) PetscAssertPointer(names, 3);
  if (values) PetscAssertPointer(values, 4);
  if (N) *N = 0;
  if (names) PetscCall(PetscFree(*names));
  if (values) PetscCall(PetscFree(*values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsMonitorDefault - Print all options set value events using the supplied `PetscViewer`.

  Logically Collective

  Input Parameters:
+ name   - option name string
. value  - option value string
. source - The source for the option
- ctx    - a `PETSCVIEWERASCII` or `NULL`

  Level: intermediate

  Notes:
  If ctx is `NULL`, `PetscPrintf()` is used.
  The first MPI process in the `PetscViewer` viewer actually prints the values, other
  processes may have different values set

  If `PetscCIEnabled` then do not print the test harness options

.seealso: `PetscOptionsMonitorSet()`
@*/
PetscErrorCode PetscOptionsMonitorDefault(const char name[], const char value[], PetscOptionSource source, void *ctx)
{
  PetscFunctionBegin;
  if (PetscCIOption(name)) PetscFunctionReturn(PETSC_SUCCESS);

  if (ctx) {
    PetscViewer viewer = (PetscViewer)ctx;
    if (!value) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Removing option: %s\n", name));
    } else if (!value[0]) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Setting option: %s (no value) (source: %s)\n", name, PetscOptionSources[source]));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Setting option: %s = %s (source: %s)\n", name, value, PetscOptionSources[source]));
    }
  } else {
    MPI_Comm comm = PETSC_COMM_WORLD;
    if (!value) {
      PetscCall(PetscPrintf(comm, "Removing option: %s\n", name));
    } else if (!value[0]) {
      PetscCall(PetscPrintf(comm, "Setting option: %s (no value) (source: %s)\n", name, PetscOptionSources[source]));
    } else {
      PetscCall(PetscPrintf(comm, "Setting option: %s = %s (source: %s)\n", name, value, PetscOptionSources[source]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsMonitorSet - Sets an ADDITIONAL function to be called at every method that
  modified the PETSc options database.

  Not Collective

  Input Parameters:
+ monitor        - pointer to function (if this is `NULL`, it turns off monitoring
. mctx           - [optional] context for private data for the monitor routine (use `NULL` if
                   no context is desired)
- monitordestroy - [optional] routine that frees monitor context (may be `NULL`)

  Calling sequence of `monitor`:
+ name   - option name string
. value  - option value string, a value of `NULL` indicates the option is being removed from the database. A value
           of "" indicates the option is in the database but has no value.
. source - option source
- mctx   - optional monitoring context, as set by `PetscOptionsMonitorSet()`

  Calling sequence of `monitordestroy`:
. mctx - [optional] pointer to context to destroy with

  Options Database Keys:
+ -options_monitor <viewer> - turn on default monitoring
- -options_monitor_cancel   - turn off any option monitors except the default monitor obtained with `-options_monitor`

  Level: intermediate

  Notes:
  See `PetscInitialize()` for options related to option database monitoring.

  The default is to do no monitoring.  To print the name and value of options
  being inserted into the database, use `PetscOptionsMonitorDefault()` as the monitoring routine,
  with a `NULL` monitoring context. Or use the option `-options_monitor` <viewer>.

  Several different monitoring routines may be set by calling
  `PetscOptionsMonitorSet()` multiple times; all will be called in the
  order in which they were set.

.seealso: `PetscOptionsMonitorDefault()`, `PetscInitialize()`
@*/
PetscErrorCode PetscOptionsMonitorSet(PetscErrorCode (*monitor)(const char name[], const char value[], PetscOptionSource source, void *mctx), void *mctx, PetscErrorCode (*monitordestroy)(void **mctx))
{
  PetscOptions options = defaultoptions;

  PetscFunctionBegin;
  if (options->monitorCancel) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(options->numbermonitors < MAXOPTIONSMONITORS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many PetscOptions monitors set");
  options->monitor[options->numbermonitors]          = monitor;
  options->monitordestroy[options->numbermonitors]   = monitordestroy;
  options->monitorcontext[options->numbermonitors++] = (void *)mctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PetscOptionsStringToBool - Converts string to PetscBool, handles cases like "yes", "no", "true", "false", "0", "1", "off", "on".
     Empty string is considered as true.
*/
PetscErrorCode PetscOptionsStringToBool(const char value[], PetscBool *a)
{
  PetscBool istrue, isfalse;
  size_t    len;

  PetscFunctionBegin;
  /* PetscStrlen() returns 0 for NULL or "" */
  PetscCall(PetscStrlen(value, &len));
  if (!len) {
    *a = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrcasecmp(value, "TRUE", &istrue));
  if (istrue) {
    *a = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrcasecmp(value, "YES", &istrue));
  if (istrue) {
    *a = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrcasecmp(value, "1", &istrue));
  if (istrue) {
    *a = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrcasecmp(value, "on", &istrue));
  if (istrue) {
    *a = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrcasecmp(value, "FALSE", &isfalse));
  if (isfalse) {
    *a = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrcasecmp(value, "NO", &isfalse));
  if (isfalse) {
    *a = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrcasecmp(value, "0", &isfalse));
  if (isfalse) {
    *a = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrcasecmp(value, "off", &isfalse));
  if (isfalse) {
    *a = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown logical value: %s", value);
}

/*
   PetscOptionsStringToInt - Converts a string to an integer value. Handles special cases such as "default" and "decide"
*/
PetscErrorCode PetscOptionsStringToInt(const char name[], PetscInt *a)
{
  size_t    len;
  PetscBool decide, tdefault, mouse, unlimited;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &len));
  PetscCheck(len, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "character string of length zero has no numerical value");

  PetscCall(PetscStrcasecmp(name, "PETSC_DEFAULT", &tdefault));
  if (!tdefault) PetscCall(PetscStrcasecmp(name, "DEFAULT", &tdefault));
  PetscCall(PetscStrcasecmp(name, "PETSC_DECIDE", &decide));
  if (!decide) PetscCall(PetscStrcasecmp(name, "DECIDE", &decide));
  if (!decide) PetscCall(PetscStrcasecmp(name, "PETSC_DETERMINE", &decide));
  if (!decide) PetscCall(PetscStrcasecmp(name, "DETERMINE", &decide));
  PetscCall(PetscStrcasecmp(name, "PETSC_UNLIMITED", &unlimited));
  if (!unlimited) PetscCall(PetscStrcasecmp(name, "UNLIMITED", &unlimited));
  PetscCall(PetscStrcasecmp(name, "mouse", &mouse));

  if (tdefault) *a = PETSC_DEFAULT;
  else if (decide) *a = PETSC_DECIDE;
  else if (unlimited) *a = PETSC_UNLIMITED;
  else if (mouse) *a = -1;
  else {
    char *endptr;
    long  strtolval;

    strtolval = strtol(name, &endptr, 10);
    PetscCheck((size_t)(endptr - name) == len, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input string %s has no integer value (do not include . in it)", name);

#if defined(PETSC_USE_64BIT_INDICES) && defined(PETSC_HAVE_ATOLL)
    (void)strtolval;
    *a = atoll(name);
#elif defined(PETSC_USE_64BIT_INDICES) && defined(PETSC_HAVE___INT64)
    (void)strtolval;
    *a = _atoi64(name);
#else
    *a = (PetscInt)strtolval;
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_REAL___FLOAT128)
  #include <quadmath.h>
#endif

static PetscErrorCode PetscStrtod(const char name[], PetscReal *a, char **endptr)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_REAL___FLOAT128)
  *a = strtoflt128(name, endptr);
#else
  *a = (PetscReal)strtod(name, endptr);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStrtoz(const char name[], PetscScalar *a, char **endptr, PetscBool *isImaginary)
{
  PetscBool hasi = PETSC_FALSE;
  char     *ptr;
  PetscReal strtoval;

  PetscFunctionBegin;
  PetscCall(PetscStrtod(name, &strtoval, &ptr));
  if (ptr == name) {
    strtoval = 1.;
    hasi     = PETSC_TRUE;
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
  *endptr      = ptr;
  *isImaginary = hasi;
  if (hasi) {
#if !defined(PETSC_USE_COMPLEX)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input string %s contains imaginary but complex not supported ", name);
#else
    *a = PetscCMPLX(0., strtoval);
#endif
  } else {
    *a = strtoval;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Converts a string to PetscReal value. Handles special cases like "default" and "decide"
*/
PetscErrorCode PetscOptionsStringToReal(const char name[], PetscReal *a)
{
  size_t    len;
  PetscBool match;
  char     *endptr;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &len));
  PetscCheck(len, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "String of length zero has no numerical value");

  PetscCall(PetscStrcasecmp(name, "PETSC_DEFAULT", &match));
  if (!match) PetscCall(PetscStrcasecmp(name, "DEFAULT", &match));
  if (match) {
    *a = PETSC_DEFAULT;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrcasecmp(name, "PETSC_DECIDE", &match));
  if (!match) PetscCall(PetscStrcasecmp(name, "DECIDE", &match));
  if (match) {
    *a = PETSC_DECIDE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrcasecmp(name, "PETSC_DETERMINE", &match));
  if (!match) PetscCall(PetscStrcasecmp(name, "DETERMINE", &match));
  if (match) {
    *a = PETSC_DETERMINE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrcasecmp(name, "PETSC_UNLIMITED", &match));
  if (!match) PetscCall(PetscStrcasecmp(name, "UNLIMITED", &match));
  if (match) {
    *a = PETSC_UNLIMITED;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrtod(name, a, &endptr));
  PetscCheck((size_t)(endptr - name) == len, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input string %s has no numeric value", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscOptionsStringToScalar(const char name[], PetscScalar *a)
{
  PetscBool   imag1;
  size_t      len;
  PetscScalar val = 0.;
  char       *ptr = NULL;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &len));
  PetscCheck(len, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "character string of length zero has no numerical value");
  PetscCall(PetscStrtoz(name, &val, &ptr, &imag1));
#if defined(PETSC_USE_COMPLEX)
  if ((size_t)(ptr - name) < len) {
    PetscBool   imag2;
    PetscScalar val2;

    PetscCall(PetscStrtoz(ptr, &val2, &ptr, &imag2));
    if (imag1) PetscCheck(imag2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input string %s: must specify imaginary component second", name);
    val = PetscCMPLX(PetscRealPart(val), PetscImaginaryPart(val2));
  }
#endif
  PetscCheck((size_t)(ptr - name) == len, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input string %s has no numeric value ", name);
  *a = val;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetBool - Gets the Logical (true or false) value for a particular
  option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - the string to prepend to the name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ ivalue - the logical value to return
- set    - `PETSC_TRUE`  if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  TRUE, true, YES, yes, ON, on, nostring, and 1 all translate to `PETSC_TRUE`
  FALSE, false, NO, no, OFF, off and 0 all translate to `PETSC_FALSE`

  If the option is given, but no value is provided, then `ivalue` and `set` are both given the value `PETSC_TRUE`. That is `-requested_bool`
  is equivalent to `-requested_bool true`

  If the user does not supply the option at all `ivalue` is NOT changed. Thus
  you should ALWAYS initialize `ivalue` if you access it without first checking that the `set` flag is true.

.seealso: `PetscOptionsGetBool3()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetInt()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetBool(PetscOptions options, const char pre[], const char name[], PetscBool *ivalue, PetscBool *set)
{
  const char *value;
  PetscBool   flag;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  if (ivalue) PetscAssertPointer(ivalue, 4);
  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (flag) {
    if (set) *set = PETSC_TRUE;
    PetscCall(PetscOptionsStringToBool(value, &flag));
    if (ivalue) *ivalue = flag;
  } else {
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetBool3 - Gets the ternary logical (true, false or unknown) value for a particular
  option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - the string to prepend to the name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ ivalue - the ternary logical value to return
- set    - `PETSC_TRUE`  if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  TRUE, true, YES, yes, ON, on, nostring and 1 all translate to `PETSC_BOOL3_TRUE`
  FALSE, false, NO, no, OFF, off and 0 all translate to `PETSC_BOOL3_FALSE`
  UNKNOWN, unknown, AUTO and auto all translate to `PETSC_BOOL3_UNKNOWN`

  If the option is given, but no value is provided, then `ivalue` will be set to `PETSC_BOOL3_TRUE` and `set` will be set to `PETSC_TRUE`. That is `-requested_bool3`
  is equivalent to `-requested_bool3 true`

  If the user does not supply the option at all `ivalue` is NOT changed. Thus
  you should ALWAYS initialize `ivalue` if you access it without first checking that the `set` flag is true.

.seealso: `PetscOptionsGetBool()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetInt()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetBool3(PetscOptions options, const char pre[], const char name[], PetscBool3 *ivalue, PetscBool *set)
{
  const char *value;
  PetscBool   flag;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  if (ivalue) PetscAssertPointer(ivalue, 4);
  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (flag) { // found the option
    PetscBool isAUTO = PETSC_FALSE, isUNKNOWN = PETSC_FALSE;

    if (set) *set = PETSC_TRUE;
    PetscCall(PetscStrcasecmp("AUTO", value, &isAUTO));                    // auto or AUTO
    if (!isAUTO) PetscCall(PetscStrcasecmp("UNKNOWN", value, &isUNKNOWN)); // unknown or UNKNOWN
    if (isAUTO || isUNKNOWN) {
      if (ivalue) *ivalue = PETSC_BOOL3_UNKNOWN;
    } else { // handle boolean values (if no value is given, it returns true)
      PetscCall(PetscOptionsStringToBool(value, &flag));
      if (ivalue) *ivalue = PetscBoolToBool3(flag);
    }
  } else {
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetEList - Puts a list of option values that a single one may be selected from

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - the string to prepend to the name or `NULL`
. opt     - option name
. list    - the possible choices (one of these must be selected, anything else is invalid)
- ntext   - number of choices

  Output Parameters:
+ value - the index of the value to return (defaults to zero if the option name is given but no choice is listed)
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: intermediate

  Notes:
  If the user does not supply the option `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that the `set` flag is true.

  See `PetscOptionsFList()` for when the choices are given in a `PetscFunctionList`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetEList(PetscOptions options, const char pre[], const char opt[], const char *const *list, PetscInt ntext, PetscInt *value, PetscBool *set)
{
  size_t    alen, len = 0, tlen = 0;
  char     *svalue;
  PetscBool aset, flg = PETSC_FALSE;
  PetscInt  i;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 3);
  for (i = 0; i < ntext; i++) {
    PetscCall(PetscStrlen(list[i], &alen));
    if (alen > len) len = alen;
    tlen += len + 1;
  }
  len += 5; /* a little extra space for user mistypes */
  PetscCall(PetscMalloc1(len, &svalue));
  PetscCall(PetscOptionsGetString(options, pre, opt, svalue, len, &aset));
  if (aset) {
    PetscCall(PetscEListFind(ntext, list, svalue, value, &flg));
    if (!flg) {
      char *avail;

      PetscCall(PetscMalloc1(tlen, &avail));
      avail[0] = '\0';
      for (i = 0; i < ntext; i++) {
        PetscCall(PetscStrlcat(avail, list[i], tlen));
        PetscCall(PetscStrlcat(avail, " ", tlen));
      }
      PetscCall(PetscStrtolower(avail));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Unknown option %s for -%s%s. Available options: %s", svalue, pre ? pre : "", opt + 1, avail);
    }
    if (set) *set = PETSC_TRUE;
  } else if (set) *set = PETSC_FALSE;
  PetscCall(PetscFree(svalue));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetEnum - Gets the enum value for a particular option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - option prefix or `NULL`
. opt     - option name
- list    - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null

  Output Parameters:
+ value - the value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  If the user does not supply the option `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that the `set` flag is true.

  `list` is usually something like `PCASMTypes` or some other predefined list of enum names

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsGetEList()`, `PetscOptionsEnum()`
@*/
PetscErrorCode PetscOptionsGetEnum(PetscOptions options, const char pre[], const char opt[], const char *const *list, PetscEnum *value, PetscBool *set)
{
  PetscInt  ntext = 0, tval;
  PetscBool fset;

  PetscFunctionBegin;
  PetscAssertPointer(opt, 3);
  while (list[ntext++]) PetscCheck(ntext <= 50, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "List argument appears to be wrong or have more than 50 entries");
  PetscCheck(ntext >= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "List argument must have at least two entries: typename and type prefix");
  ntext -= 3;
  PetscCall(PetscOptionsGetEList(options, pre, opt, list, ntext, &tval, &fset));
  /* with PETSC_USE_64BIT_INDICES sizeof(PetscInt) != sizeof(PetscEnum) */
  if (fset) *value = (PetscEnum)tval;
  if (set) *set = fset;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetInt - Gets the integer value for a particular option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - the string to prepend to the name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ ivalue - the integer value to return
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  If the user does not supply the option `ivalue` is NOT changed. Thus
  you should ALWAYS initialize the `ivalue` if you access it without first checking that the `set` flag is true.

  Accepts the special values `determine`, `decide` and `unlimited`.

  Accepts the deprecated value `default`.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetInt(PetscOptions options, const char pre[], const char name[], PetscInt *ivalue, PetscBool *set)
{
  const char *value;
  PetscBool   flag;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(ivalue, 4);
  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (flag) {
    if (!value) {
      if (set) *set = PETSC_FALSE;
    } else {
      if (set) *set = PETSC_TRUE;
      PetscCall(PetscOptionsStringToInt(value, ivalue));
    }
  } else {
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetMPIInt - Gets the MPI integer value for a particular option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - the string to prepend to the name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ ivalue - the MPI integer value to return
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  If the user does not supply the option `ivalue` is NOT changed. Thus
  you should ALWAYS initialize the `ivalue` if you access it without first checking that the `set` flag is true.

  Accepts the special values `determine`, `decide` and `unlimited`.

  Accepts the deprecated value `default`.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetMPIInt(PetscOptions options, const char pre[], const char name[], PetscMPIInt *ivalue, PetscBool *set)
{
  PetscInt  value;
  PetscBool flag;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetInt(options, pre, name, &value, &flag));
  if (flag) PetscCall(PetscMPIIntCast(value, ivalue));
  if (set) *set = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetReal - Gets the double precision value for a particular
  option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to each name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ dvalue - the double value to return
- set    - `PETSC_TRUE` if found, `PETSC_FALSE` if not found

  Level: beginner

  Notes:
  Accepts the special values `determine`, `decide` and `unlimited`.

  Accepts the deprecated value `default`

  If the user does not supply the option `dvalue` is NOT changed. Thus
  you should ALWAYS initialize `dvalue` if you access it without first checking that the `set` flag is true.

.seealso: `PetscOptionsGetInt()`, `PetscOptionsHasName()`,
          `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetReal(PetscOptions options, const char pre[], const char name[], PetscReal *dvalue, PetscBool *set)
{
  const char *value;
  PetscBool   flag;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(dvalue, 4);
  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (flag) {
    if (!value) {
      if (set) *set = PETSC_FALSE;
    } else {
      if (set) *set = PETSC_TRUE;
      PetscCall(PetscOptionsStringToReal(value, dvalue));
    }
  } else {
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetScalar - Gets the scalar value for a particular
  option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to each name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ dvalue - the scalar value to return
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Example Usage:
  A complex number 2+3i must be specified with NO spaces

  Note:
  If the user does not supply the option `dvalue` is NOT changed. Thus
  you should ALWAYS initialize `dvalue` if you access it without first checking if the `set` flag is true.

.seealso: `PetscOptionsGetInt()`, `PetscOptionsHasName()`,
          `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetScalar(PetscOptions options, const char pre[], const char name[], PetscScalar *dvalue, PetscBool *set)
{
  const char *value;
  PetscBool   flag;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(dvalue, 4);
  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (flag) {
    if (!value) {
      if (set) *set = PETSC_FALSE;
    } else {
#if !defined(PETSC_USE_COMPLEX)
      PetscCall(PetscOptionsStringToReal(value, dvalue));
#else
      PetscCall(PetscOptionsStringToScalar(value, dvalue));
#endif
      if (set) *set = PETSC_TRUE;
    }
  } else { /* flag */
    if (set) *set = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetString - Gets the string value for a particular option in
  the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to name or `NULL`
. name    - the option one is seeking
- len     - maximum length of the string including null termination

  Output Parameters:
+ string - location to copy string
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Note:
  if the option is given but no string is provided then an empty string is returned and `set` is given the value of `PETSC_TRUE`

  If the user does not use the option then `string` is not changed. Thus
  you should ALWAYS initialize `string` if you access it without first checking that the `set` flag is true.

  Fortran Notes:
  The Fortran interface is slightly different from the C/C++
  interface (len is not used).  Sample usage in Fortran follows
.vb
      character *20    string
      PetscErrorCode   ierr
      PetscBool        set
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-s',string,set,ierr)
.ve

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetString(PetscOptions options, const char pre[], const char name[], char string[], size_t len, PetscBool *set)
{
  const char *value;
  PetscBool   flag;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(string, 4);
  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (!flag) {
    if (set) *set = PETSC_FALSE;
  } else {
    if (set) *set = PETSC_TRUE;
    if (value) PetscCall(PetscStrncpy(string, value, len));
    else PetscCall(PetscArrayzero(string, len));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetBoolArray - Gets an array of Logical (true or false) values for a particular
  option in the database.  The values must be separated with commas with no intervening spaces.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to each name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ dvalue - the Boolean values to return
. nmax   - On input maximum number of values to retrieve, on output the actual number of values retrieved
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Note:
  TRUE, true, YES, yes, nostring, and 1 all translate to `PETSC_TRUE`. FALSE, false, NO, no, and 0 all translate to `PETSC_FALSE`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsHasName()`,
          `PetscOptionsGetString()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetBoolArray(PetscOptions options, const char pre[], const char name[], PetscBool dvalue[], PetscInt *nmax, PetscBool *set)
{
  const char *svalue;
  char       *value;
  PetscInt    n = 0;
  PetscBool   flag;
  PetscToken  token;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(dvalue, 4);
  PetscAssertPointer(nmax, 5);

  PetscCall(PetscOptionsFindPair(options, pre, name, &svalue, &flag));
  if (!flag || !svalue) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (set) *set = PETSC_TRUE;
  PetscCall(PetscTokenCreate(svalue, ',', &token));
  PetscCall(PetscTokenFind(token, &value));
  while (value && n < *nmax) {
    PetscCall(PetscOptionsStringToBool(value, dvalue));
    PetscCall(PetscTokenFind(token, &value));
    dvalue++;
    n++;
  }
  PetscCall(PetscTokenDestroy(&token));
  *nmax = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetEnumArray - Gets an array of enum values for a particular option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - option prefix or `NULL`
. name    - option name
- list    - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null

  Output Parameters:
+ ivalue - the  enum values to return
. nmax   - On input maximum number of values to retrieve, on output the actual number of values retrieved
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  The array must be passed as a comma separated list with no spaces between the items.

  `list` is usually something like `PCASMTypes` or some other predefined list of enum names.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetEnum()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsName()`,
          `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`, `PetscOptionsStringArray()`, `PetscOptionsRealArray()`,
          `PetscOptionsScalar()`, `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsGetEList()`, `PetscOptionsEnum()`
@*/
PetscErrorCode PetscOptionsGetEnumArray(PetscOptions options, const char pre[], const char name[], const char *const *list, PetscEnum ivalue[], PetscInt *nmax, PetscBool *set)
{
  const char *svalue;
  char       *value;
  PetscInt    n = 0;
  PetscEnum   evalue;
  PetscBool   flag;
  PetscToken  token;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(list, 4);
  PetscAssertPointer(ivalue, 5);
  PetscAssertPointer(nmax, 6);

  PetscCall(PetscOptionsFindPair(options, pre, name, &svalue, &flag));
  if (!flag || !svalue) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (set) *set = PETSC_TRUE;
  PetscCall(PetscTokenCreate(svalue, ',', &token));
  PetscCall(PetscTokenFind(token, &value));
  while (value && n < *nmax) {
    PetscCall(PetscEnumFind(list, value, &evalue, &flag));
    PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_USER, "Unknown enum value '%s' for -%s%s", svalue, pre ? pre : "", name + 1);
    ivalue[n++] = evalue;
    PetscCall(PetscTokenFind(token, &value));
  }
  PetscCall(PetscTokenDestroy(&token));
  *nmax = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetIntArray - Gets an array of integer values for a particular option in the database.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to each name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ ivalue - the integer values to return
. nmax   - On input maximum number of values to retrieve, on output the actual number of values retrieved
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  The array can be passed as
+  a comma separated list -                                 0,1,2,3,4,5,6,7
.  a range (start\-end+1) -                                 0-8
.  a range with given increment (start\-end+1:inc) -        0-7:2
-  a combination of values and ranges separated by commas - 0,1-8,8-15:2

  There must be no intervening spaces between the values.

.seealso: `PetscOptionsGetInt()`, `PetscOptionsHasName()`,
          `PetscOptionsGetString()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetIntArray(PetscOptions options, const char pre[], const char name[], PetscInt ivalue[], PetscInt *nmax, PetscBool *set)
{
  const char *svalue;
  char       *value;
  PetscInt    n = 0, i, j, start, end, inc, nvalues;
  size_t      len;
  PetscBool   flag, foundrange;
  PetscToken  token;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(ivalue, 4);
  PetscAssertPointer(nmax, 5);

  PetscCall(PetscOptionsFindPair(options, pre, name, &svalue, &flag));
  if (!flag || !svalue) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (set) *set = PETSC_TRUE;
  PetscCall(PetscTokenCreate(svalue, ',', &token));
  PetscCall(PetscTokenFind(token, &value));
  while (value && n < *nmax) {
    /* look for form  d-D where d and D are integers */
    foundrange = PETSC_FALSE;
    PetscCall(PetscStrlen(value, &len));
    if (value[0] == '-') i = 2;
    else i = 1;
    for (; i < (int)len; i++) {
      if (value[i] == '-') {
        PetscCheck(i != (int)len - 1, PETSC_COMM_SELF, PETSC_ERR_USER, "Error in %" PetscInt_FMT "-th array entry %s", n, value);
        value[i] = 0;

        PetscCall(PetscOptionsStringToInt(value, &start));
        inc = 1;
        j   = i + 1;
        for (; j < (int)len; j++) {
          if (value[j] == ':') {
            value[j] = 0;

            PetscCall(PetscOptionsStringToInt(value + j + 1, &inc));
            PetscCheck(inc > 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Error in %" PetscInt_FMT "-th array entry,%s cannot have negative increment", n, value + j + 1);
            break;
          }
        }
        PetscCall(PetscOptionsStringToInt(value + i + 1, &end));
        PetscCheck(end > start, PETSC_COMM_SELF, PETSC_ERR_USER, "Error in %" PetscInt_FMT "-th array entry, %s-%s cannot have decreasing list", n, value, value + i + 1);
        nvalues = (end - start) / inc + (end - start) % inc;
        PetscCheck(n + nvalues <= *nmax, PETSC_COMM_SELF, PETSC_ERR_USER, "Error in %" PetscInt_FMT "-th array entry, not enough space left in array (%" PetscInt_FMT ") to contain entire range from %" PetscInt_FMT " to %" PetscInt_FMT, n, *nmax - n, start, end);
        for (; start < end; start += inc) {
          *ivalue = start;
          ivalue++;
          n++;
        }
        foundrange = PETSC_TRUE;
        break;
      }
    }
    if (!foundrange) {
      PetscCall(PetscOptionsStringToInt(value, ivalue));
      ivalue++;
      n++;
    }
    PetscCall(PetscTokenFind(token, &value));
  }
  PetscCall(PetscTokenDestroy(&token));
  *nmax = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetRealArray - Gets an array of double precision values for a
  particular option in the database.  The values must be separated with commas with no intervening spaces.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to each name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ dvalue - the double values to return
. nmax   - On input maximum number of values to retrieve, on output the actual number of values retrieved
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

.seealso: `PetscOptionsGetInt()`, `PetscOptionsHasName()`,
          `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetRealArray(PetscOptions options, const char pre[], const char name[], PetscReal dvalue[], PetscInt *nmax, PetscBool *set)
{
  const char *svalue;
  char       *value;
  PetscInt    n = 0;
  PetscBool   flag;
  PetscToken  token;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(dvalue, 4);
  PetscAssertPointer(nmax, 5);

  PetscCall(PetscOptionsFindPair(options, pre, name, &svalue, &flag));
  if (!flag || !svalue) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (set) *set = PETSC_TRUE;
  PetscCall(PetscTokenCreate(svalue, ',', &token));
  PetscCall(PetscTokenFind(token, &value));
  while (value && n < *nmax) {
    PetscCall(PetscOptionsStringToReal(value, dvalue++));
    PetscCall(PetscTokenFind(token, &value));
    n++;
  }
  PetscCall(PetscTokenDestroy(&token));
  *nmax = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetScalarArray - Gets an array of scalars for a
  particular option in the database.  The values must be separated with commas with no intervening spaces.

  Not Collective

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to each name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ dvalue - the scalar values to return
. nmax   - On input maximum number of values to retrieve, on output the actual number of values retrieved
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

.seealso: `PetscOptionsGetInt()`, `PetscOptionsHasName()`,
          `PetscOptionsGetString()`, `PetscOptionsGetIntArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetScalarArray(PetscOptions options, const char pre[], const char name[], PetscScalar dvalue[], PetscInt *nmax, PetscBool *set)
{
  const char *svalue;
  char       *value;
  PetscInt    n = 0;
  PetscBool   flag;
  PetscToken  token;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(dvalue, 4);
  PetscAssertPointer(nmax, 5);

  PetscCall(PetscOptionsFindPair(options, pre, name, &svalue, &flag));
  if (!flag || !svalue) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (set) *set = PETSC_TRUE;
  PetscCall(PetscTokenCreate(svalue, ',', &token));
  PetscCall(PetscTokenFind(token, &value));
  while (value && n < *nmax) {
    PetscCall(PetscOptionsStringToScalar(value, dvalue++));
    PetscCall(PetscTokenFind(token, &value));
    n++;
  }
  PetscCall(PetscTokenDestroy(&token));
  *nmax = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsGetStringArray - Gets an array of string values for a particular
  option in the database. The values must be separated with commas with no intervening spaces.

  Not Collective; No Fortran Support

  Input Parameters:
+ options - options database, use `NULL` for default global database
. pre     - string to prepend to name or `NULL`
- name    - the option one is seeking

  Output Parameters:
+ strings - location to copy strings
. nmax    - On input maximum number of strings, on output the actual number of strings found
- set     - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  The `nmax` parameter is used for both input and output.

  The user should pass in an array of pointers to char, to hold all the
  strings returned by this function.

  The user is responsible for deallocating the strings that are
  returned.

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode PetscOptionsGetStringArray(PetscOptions options, const char pre[], const char name[], char *strings[], PetscInt *nmax, PetscBool *set)
{
  const char *svalue;
  char       *value;
  PetscInt    n = 0;
  PetscBool   flag;
  PetscToken  token;

  PetscFunctionBegin;
  PetscAssertPointer(name, 3);
  PetscAssertPointer(strings, 4);
  PetscAssertPointer(nmax, 5);

  PetscCall(PetscOptionsFindPair(options, pre, name, &svalue, &flag));
  if (!flag || !svalue) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (set) *set = PETSC_TRUE;
  PetscCall(PetscTokenCreate(svalue, ',', &token));
  PetscCall(PetscTokenFind(token, &value));
  while (value && n < *nmax) {
    PetscCall(PetscStrallocpy(value, &strings[n]));
    PetscCall(PetscTokenFind(token, &value));
    n++;
  }
  PetscCall(PetscTokenDestroy(&token));
  *nmax = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscOptionsDeprecated_Private - mark an option as deprecated, optionally replacing it with `newname`

  Prints a deprecation warning, unless an option is supplied to suppress.

  Logically Collective

  Input Parameters:
+ PetscOptionsObject - string to prepend to name or `NULL`
. oldname            - the old, deprecated option
. newname            - the new option, or `NULL` if option is purely removed
. version            - a string describing the version of first deprecation, e.g. "3.9"
- info               - additional information string, or `NULL`.

  Options Database Key:
. -options_suppress_deprecated_warnings - do not print deprecation warnings

  Level: developer

  Notes:
  If `newname` is provided then the options database will automatically check the database for `oldname`.

  The old call `PetscOptionsXXX`(`oldname`) should be removed from the source code when both (1) the call to `PetscOptionsDeprecated()` occurs before the
  new call to `PetscOptionsXXX`(`newname`) and (2) the argument handling of the new call to `PetscOptionsXXX`(`newname`) is identical to the previous call.
  See `PTScotch_PartGraph_Seq()` for an example of when (1) fails and `SNESTestJacobian()` where an example of (2) fails.

  Must be called between `PetscOptionsBegin()` (or `PetscObjectOptionsBegin()`) and `PetscOptionsEnd()`.
  Only the process of rank zero that owns the `PetscOptionsItems` are argument (managed by `PetscOptionsBegin()` or
  `PetscObjectOptionsBegin()` prints the information
  If newname is provided, the old option is replaced. Otherwise, it remains
  in the options database.
  If an option is not replaced, the info argument should be used to advise the user
  on how to proceed.
  There is a limit on the length of the warning printed, so very long strings
  provided as info may be truncated.

.seealso: `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsScalar()`, `PetscOptionsBool()`, `PetscOptionsString()`, `PetscOptionsSetValue()`
@*/
PetscErrorCode PetscOptionsDeprecated_Private(PetscOptionItems *PetscOptionsObject, const char oldname[], const char newname[], const char version[], const char info[])
{
  PetscBool         found, quiet;
  const char       *value;
  const char *const quietopt = "-options_suppress_deprecated_warnings";
  char              msg[4096];
  char             *prefix  = NULL;
  PetscOptions      options = NULL;
  MPI_Comm          comm    = PETSC_COMM_SELF;

  PetscFunctionBegin;
  PetscAssertPointer(oldname, 2);
  PetscAssertPointer(version, 4);
  if (PetscOptionsObject) {
    prefix  = PetscOptionsObject->prefix;
    options = PetscOptionsObject->options;
    comm    = PetscOptionsObject->comm;
  }
  PetscCall(PetscOptionsFindPair(options, prefix, oldname, &value, &found));
  if (found) {
    if (newname) {
      if (prefix) PetscCall(PetscOptionsPrefixPush(options, prefix));
      PetscCall(PetscOptionsSetValue(options, newname, value));
      if (prefix) PetscCall(PetscOptionsPrefixPop(options));
      PetscCall(PetscOptionsClearValue(options, oldname));
    }
    quiet = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(options, NULL, quietopt, &quiet, NULL));
    if (!quiet) {
      PetscCall(PetscStrncpy(msg, "** PETSc DEPRECATION WARNING ** : the option ", sizeof(msg)));
      PetscCall(PetscStrlcat(msg, oldname, sizeof(msg)));
      PetscCall(PetscStrlcat(msg, " is deprecated as of version ", sizeof(msg)));
      PetscCall(PetscStrlcat(msg, version, sizeof(msg)));
      PetscCall(PetscStrlcat(msg, " and will be removed in a future release.\n", sizeof(msg)));
      if (newname) {
        PetscCall(PetscStrlcat(msg, "   Use the option ", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, newname, sizeof(msg)));
        PetscCall(PetscStrlcat(msg, " instead.", sizeof(msg)));
      }
      if (info) {
        PetscCall(PetscStrlcat(msg, " ", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, info, sizeof(msg)));
      }
      PetscCall(PetscStrlcat(msg, " (Silence this warning with ", sizeof(msg)));
      PetscCall(PetscStrlcat(msg, quietopt, sizeof(msg)));
      PetscCall(PetscStrlcat(msg, ")\n", sizeof(msg)));
      PetscCall(PetscPrintf(comm, "%s", msg));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
