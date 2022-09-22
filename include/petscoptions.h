/*
   Routines to determine options set in the options database.
*/
#ifndef PETSCOPTIONS_H
#define PETSCOPTIONS_H

#include <petscsys.h>
#include <petscviewertypes.h>

/* SUBMANSEC = Sys */

#define PETSC_MAX_OPTION_NAME 512
typedef struct _n_PetscOptions *PetscOptions;
PETSC_EXTERN PetscErrorCode     PetscOptionsCreate(PetscOptions *);
PETSC_EXTERN PetscErrorCode     PetscOptionsPush(PetscOptions);
PETSC_EXTERN PetscErrorCode     PetscOptionsPop(void);
PETSC_EXTERN PetscErrorCode     PetscOptionsDestroy(PetscOptions *);
PETSC_EXTERN PetscErrorCode     PetscOptionsCreateDefault(void);
PETSC_EXTERN PetscErrorCode     PetscOptionsDestroyDefault(void);

PETSC_EXTERN PetscErrorCode PetscOptionsHasHelp(PetscOptions, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsHasName(PetscOptions, const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetBool(PetscOptions, const char[], const char[], PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetInt(PetscOptions, const char[], const char[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetEnum(PetscOptions, const char[], const char[], const char *const *, PetscEnum *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetEList(PetscOptions, const char[], const char[], const char *const *, PetscInt, PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetReal(PetscOptions, const char[], const char[], PetscReal *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetScalar(PetscOptions, const char[], const char[], PetscScalar *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetString(PetscOptions, const char[], const char[], char[], size_t, PetscBool *);

PETSC_EXTERN PetscErrorCode PetscOptionsGetBoolArray(PetscOptions, const char[], const char[], PetscBool[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetEnumArray(PetscOptions, const char[], const char[], const char *const *, PetscEnum *, PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetIntArray(PetscOptions, const char[], const char[], PetscInt[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetRealArray(PetscOptions, const char[], const char[], PetscReal[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetScalarArray(PetscOptions, const char[], const char[], PetscScalar[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetStringArray(PetscOptions, const char[], const char[], char *[], PetscInt *, PetscBool *);

PETSC_EXTERN PetscErrorCode PetscOptionsValidKey(const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsSetAlias(PetscOptions, const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsSetValue(PetscOptions, const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsClearValue(PetscOptions, const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsFindPair(PetscOptions, const char[], const char[], const char *[], PetscBool *);

PETSC_EXTERN PetscErrorCode PetscOptionsGetAll(PetscOptions, char *[]);
PETSC_EXTERN PetscErrorCode PetscOptionsAllUsed(PetscOptions, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscOptionsUsed(PetscOptions, const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsLeft(PetscOptions);
PETSC_EXTERN PetscErrorCode PetscOptionsLeftGet(PetscOptions, PetscInt *, char ***, char ***);
PETSC_EXTERN PetscErrorCode PetscOptionsLeftRestore(PetscOptions, PetscInt *, char ***, char ***);
PETSC_EXTERN PetscErrorCode PetscOptionsView(PetscOptions, PetscViewer);

PETSC_EXTERN PetscErrorCode PetscOptionsReject(PetscOptions, const char[], const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsInsert(PetscOptions, int *, char ***, const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsInsertFile(MPI_Comm, PetscOptions, const char[], PetscBool);
PETSC_EXTERN PetscErrorCode PetscOptionsInsertFileYAML(MPI_Comm, PetscOptions, const char[], PetscBool);
PETSC_EXTERN PetscErrorCode PetscOptionsInsertString(PetscOptions, const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsInsertStringYAML(PetscOptions, const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsInsertArgs(PetscOptions, int, char **);
PETSC_EXTERN PetscErrorCode PetscOptionsClear(PetscOptions);
PETSC_EXTERN PetscErrorCode PetscOptionsPrefixPush(PetscOptions, const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsPrefixPop(PetscOptions);

PETSC_EXTERN PetscErrorCode PetscOptionsGetenv(MPI_Comm, const char[], char[], size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToBool(const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToInt(const char[], PetscInt *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToReal(const char[], PetscReal *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToScalar(const char[], PetscScalar *);

PETSC_EXTERN PetscErrorCode PetscOptionsMonitorSet(PetscErrorCode (*)(const char[], const char[], void *), void *, PetscErrorCode (*)(void **));
PETSC_EXTERN PetscErrorCode PetscOptionsMonitorDefault(const char[], const char[], void *);

PETSC_EXTERN PetscErrorCode PetscObjectSetOptions(PetscObject, PetscOptions);
PETSC_EXTERN PetscErrorCode PetscObjectGetOptions(PetscObject, PetscOptions *);

PETSC_EXTERN PetscBool PetscOptionsPublish;

/*
    See manual page for PetscOptionsBegin()

    PetscOptionsItem and PetscOptionsItems are a single option (such as ksp_type) and a collection of such single
  options being handled with a PetscOptionsBegin/End()

*/
typedef enum {
  OPTION_INT,
  OPTION_BOOL,
  OPTION_REAL,
  OPTION_FLIST,
  OPTION_STRING,
  OPTION_REAL_ARRAY,
  OPTION_SCALAR_ARRAY,
  OPTION_HEAD,
  OPTION_INT_ARRAY,
  OPTION_ELIST,
  OPTION_BOOL_ARRAY,
  OPTION_STRING_ARRAY
} PetscOptionType;
typedef struct _n_PetscOptionItem *PetscOptionItem;
struct _n_PetscOptionItem {
  char              *option;
  char              *text;
  void              *data;  /* used to hold the default value and then any value it is changed to by GUI */
  PetscFunctionList  flist; /* used for available values for PetscOptionsList() */
  const char *const *list;  /* used for available values for PetscOptionsEList() */
  char               nlist; /* number of entries in list */
  char              *man;
  size_t             arraylength; /* number of entries in data in the case that it is an array (of PetscInt etc) */
  PetscBool          set;         /* the user has changed this value in the GUI */
  PetscOptionType    type;
  PetscOptionItem    next;
  char              *pman;
  void              *edata;
};

typedef struct _p_PetscOptionItems {
  PetscInt        count;
  PetscOptionItem next;
  char           *prefix, *pprefix;
  char           *title;
  MPI_Comm        comm;
  PetscBool       printhelp, changedmethod, alreadyprinted;
  PetscObject     object;
  PetscOptions    options;
} PetscOptionItems;

#if defined(PETSC_CLANG_STATIC_ANALYZER)
extern PetscOptionItems *PetscOptionsObject; /* declare this so that the PetscOptions stubs work */
PetscErrorCode           PetscOptionsBegin(MPI_Comm, const char *, const char *, const char *);
PetscErrorCode           PetscObjectOptionsBegin(PetscObject);
PetscErrorCode           PetscOptionsEnd(void);
#else
  /*MC
    PetscOptionsBegin - Begins a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively. Often one should
     use PetscObjectOptionsBegin() rather than this call.

   Synopsis:
    #include <petscoptions.h>
    PetscErrorCode PetscOptionsBegin(MPI_Comm comm,const char prefix[],const char title[],const char mansec[])

    Collective

  Input Parameters:
+   comm - communicator that shares GUI
.   prefix - options prefix for all options displayed on window (optional)
.   title - short descriptive text, for example "Krylov Solver Options"
-   mansec - section of manual pages for options, for example `KSP` (optional)

  Level: intermediate

  Notes:
    This is a macro that handles its own error checking, it does not return an error code.

    The set of queries needs to be ended by a call to `PetscOptionsEnd()`.

    One can add subheadings with `PetscOptionsHeadBegin()`.

  Developer Notes:
      PetscOptionsPublish is set in PetscOptionsCheckInitial_Private() with -saws_options. When PetscOptionsPublish is set the
      loop between PetscOptionsBegin() and PetscOptionsEnd() is run THREE times with PetscOptionsPublishCount of values -1,0,1.
      Otherwise the loop is run ONCE with a PetscOptionsPublishCount of 1.
+      \-1 - PetscOptionsInt() etc. just call PetscOptionsGetInt() etc.
.      0  - The GUI objects are created in PetscOptionsInt() etc. and displayed in PetscOptionsEnd() and the options
               database updated with user changes; PetscOptionsGetInt() etc. are also called.
-      1 - PetscOptionsInt() etc. again call PetscOptionsGetInt() etc. (possibly getting new values), in addition the help message and
              default values are printed if -help was given.
      When PetscOptionsObject.changedmethod is set this causes PetscOptionsPublishCount to be reset to -2 (so in the next loop iteration it is -1)
      and the whole process is repeated. This is to handle when, for example, the KSPType is changed thus changing the list of
      options available so they need to be redisplayed so the user can change the. Changing PetscOptionsObjects.changedmethod is never
      currently set.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscObjectOptionsBegin()`

M*/
  #define PetscOptionsBegin(comm, prefix, mess, sec) \
    do { \
      PetscOptionItems  PetscOptionsObjectBase; \
      PetscOptionItems *PetscOptionsObject = &PetscOptionsObjectBase; \
      PetscCall(PetscMemzero(PetscOptionsObject, sizeof(*PetscOptionsObject))); \
      for (PetscOptionsObject->count = (PetscOptionsPublish ? -1 : 1); PetscOptionsObject->count < 2; PetscOptionsObject->count++) { \
        PetscCall(PetscOptionsBegin_Private(PetscOptionsObject, comm, prefix, mess, sec))

  /*MC
    PetscObjectOptionsBegin - Begins a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

   Synopsis:
    #include <petscoptions.h>
    PetscErrorCode PetscObjectOptionsBegin(PetscObject obj)

    Collective on obj

  Input Parameters:
.   obj - object to set options for

  Level: intermediate

  Notes:
    This is a macro that handles its own error checking, it does not return an error code.

    Needs to be ended by a call the `PetscOptionsEnd()`

    Can add subheadings with `PetscOptionsHeadBegin()`

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`

M*/
  #define PetscObjectOptionsBegin(obj) \
    do { \
      PetscOptionItems  PetscOptionsObjectBase; \
      PetscOptionItems *PetscOptionsObject = &PetscOptionsObjectBase; \
      PetscOptionsObject->options          = ((PetscObject)obj)->options; \
      for (PetscOptionsObject->count = (PetscOptionsPublish ? -1 : 1); PetscOptionsObject->count < 2; PetscOptionsObject->count++) { \
        PetscCall(PetscObjectOptionsBegin_Private(obj, PetscOptionsObject))

  /*MC
    PetscOptionsEnd - Ends a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

    Collective on the comm used in `PetscOptionsBegin()` or obj used in `PetscObjectOptionsBegin()`

   Synopsis:
     #include <petscoptions.h>
     PetscErrorCode PetscOptionsEnd(void)

  Level: intermediate

  Notes:
    Needs to be preceded by a call to `PetscOptionsBegin()` or `PetscObjectOptionsBegin()`

    This is a macro that handles its own error checking, it does not return an error code.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscObjectOptionsBegin()`

M*/
  #define PetscOptionsEnd() \
    PetscCall(PetscOptionsEnd_Private(PetscOptionsObject)); \
    } \
    } \
    while (0)
#endif /* PETSC_CLANG_STATIC_ANALYZER */

PETSC_EXTERN PetscErrorCode PetscOptionsBegin_Private(PetscOptionItems *, MPI_Comm, const char[], const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectOptionsBegin_Private(PetscObject, PetscOptionItems *);
PETSC_EXTERN PetscErrorCode PetscOptionsEnd_Private(PetscOptionItems *);
PETSC_EXTERN PetscErrorCode PetscOptionsHeadBegin(PetscOptionItems *, const char[]);

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename... T>
void PetscOptionsHeadBegin(T...);
void PetscOptionsHeadEnd(void);
template <typename... T>
PetscErrorCode PetscOptionsEnum(T...);
template <typename... T>
PetscErrorCode PetscOptionsInt(T...);
template <typename... T>
PetscErrorCode PetscOptionsBoundedInt(T...);
template <typename... T>
PetscErrorCode PetscOptionsRangeInt(T...);
template <typename... T>
PetscErrorCode PetscOptionsReal(T...);
template <typename... T>
PetscErrorCode PetscOptionsScalar(T...);
template <typename... T>
PetscErrorCode PetscOptionsName(T...);
template <typename... T>
PetscErrorCode PetscOptionsString(T...);
template <typename... T>
PetscErrorCode PetscOptionsBool(T...);
template <typename... T>
PetscErrorCode PetscOptionsBoolGroupBegin(T...);
template <typename... T>
PetscErrorCode PetscOptionsBoolGroup(T...);
template <typename... T>
PetscErrorCode PetscOptionsBoolGroupEnd(T...);
template <typename... T>
PetscErrorCode PetscOptionsFList(T...);
template <typename... T>
PetscErrorCode PetscOptionsEList(T...);
template <typename... T>
PetscErrorCode PetscOptionsRealArray(T...);
template <typename... T>
PetscErrorCode PetscOptionsScalarArray(T...);
template <typename... T>
PetscErrorCode PetscOptionsIntArray(T...);
template <typename... T>
PetscErrorCode PetscOptionsStringArray(T...);
template <typename... T>
PetscErrorCode PetscOptionsBoolArray(T...);
template <typename... T>
PetscErrorCode PetscOptionsEnumArray(T...);
template <typename... T>
PetscErrorCode PetscOptionsDeprecated(T...);
template <typename... T>
PetscErrorCode PetscOptionsDeprecatedNoObject(T...);
#else
  /*MC
     PetscOptionsHeadBegin - Puts a heading before listing any more published options. Used, for example,
            in KSPSetFromOptions_GMRES().

   Logically Collective on the communicator passed in PetscOptionsBegin()

   Input Parameter:
.   head - the heading text

   Level: developer

   Notes:
    Handles errors directly, hence does not return an error code

    Must be between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`, and PetscOptionsObject created in `PetscOptionsBegin()` should be the first argument

    Must be followed by a call to `PetscOptionsHeadEnd()` in the same function.

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
  #define PetscOptionsHeadBegin(PetscOptionsObject, head) \
    do { \
      if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  %s\n", head)); \
    } while (0)

  #define PetscOptionsHead(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscOptionsHeadBegin() (since version 3.18)\"") PetscOptionsHeadBegin(__VA_ARGS__)

  /*MC
     PetscOptionsHeadEnd - Ends a section of options begun with `PetscOptionsHeadBegin()`
            See, for example, KSPSetFromOptions_GMRES().

    Collective on the comm used in `PetscOptionsBegin()` or obj used in `PetscObjectOptionsBegin()`

   Synopsis:
     #include <petscoptions.h>
     PetscErrorCode PetscOptionsHeadEnd(void)

  Level: intermediate

   Notes:
    Must be between a `PetscOptionsBegin()` or `PetscObjectOptionsBegin()` and a `PetscOptionsEnd()`

    Must be preceded by a call to `PetscOptionsHeadBegin()` in the same function.

    This needs to be used only if the code below `PetscOptionsHeadEnd()` can be run ONLY once.
    See, for example, `PCSetFromOptions_Composite()`. This is a return(0) in it for early exit
    from the function.

          This is only for use with the PETSc options GUI

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsEnum()`
M*/
  #define PetscOptionsHeadEnd() \
    do { \
      if (PetscOptionsObject->count != 1) PetscFunctionReturn(0); \
    } while (0)

  #define PetscOptionsTail(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscOptionsHeadEnd() (since version 3.18)\"") PetscOptionsHeadEnd(__VA_ARGS__)

  #define PetscOptionsEnum(a, b, c, d, e, f, g)        PetscOptionsEnum_Private(PetscOptionsObject, a, b, c, d, e, f, g)
  #define PetscOptionsInt(a, b, c, d, e, f)            PetscOptionsInt_Private(PetscOptionsObject, a, b, c, d, e, f, PETSC_MIN_INT, PETSC_MAX_INT)
  #define PetscOptionsBoundedInt(a, b, c, d, e, f, g)  PetscOptionsInt_Private(PetscOptionsObject, a, b, c, d, e, f, g, PETSC_MAX_INT)
  #define PetscOptionsRangeInt(a, b, c, d, e, f, g, h) PetscOptionsInt_Private(PetscOptionsObject, a, b, c, d, e, f, g, h)
  #define PetscOptionsReal(a, b, c, d, e, f)           PetscOptionsReal_Private(PetscOptionsObject, a, b, c, d, e, f)
  #define PetscOptionsScalar(a, b, c, d, e, f)         PetscOptionsScalar_Private(PetscOptionsObject, a, b, c, d, e, f)
  #define PetscOptionsName(a, b, c, d)                 PetscOptionsName_Private(PetscOptionsObject, a, b, c, d)
  #define PetscOptionsString(a, b, c, d, e, f, g)      PetscOptionsString_Private(PetscOptionsObject, a, b, c, d, e, f, g)
  #define PetscOptionsBool(a, b, c, d, e, f)           PetscOptionsBool_Private(PetscOptionsObject, a, b, c, d, e, f)
  #define PetscOptionsBoolGroupBegin(a, b, c, d)       PetscOptionsBoolGroupBegin_Private(PetscOptionsObject, a, b, c, d)
  #define PetscOptionsBoolGroup(a, b, c, d)            PetscOptionsBoolGroup_Private(PetscOptionsObject, a, b, c, d)
  #define PetscOptionsBoolGroupEnd(a, b, c, d)         PetscOptionsBoolGroupEnd_Private(PetscOptionsObject, a, b, c, d)
  #define PetscOptionsFList(a, b, c, d, e, f, g, h)    PetscOptionsFList_Private(PetscOptionsObject, a, b, c, d, e, f, g, h)
  #define PetscOptionsEList(a, b, c, d, e, f, g, h)    PetscOptionsEList_Private(PetscOptionsObject, a, b, c, d, e, f, g, h)
  #define PetscOptionsRealArray(a, b, c, d, e, f)      PetscOptionsRealArray_Private(PetscOptionsObject, a, b, c, d, e, f)
  #define PetscOptionsScalarArray(a, b, c, d, e, f)    PetscOptionsScalarArray_Private(PetscOptionsObject, a, b, c, d, e, f)
  #define PetscOptionsIntArray(a, b, c, d, e, f)       PetscOptionsIntArray_Private(PetscOptionsObject, a, b, c, d, e, f)
  #define PetscOptionsStringArray(a, b, c, d, e, f)    PetscOptionsStringArray_Private(PetscOptionsObject, a, b, c, d, e, f)
  #define PetscOptionsBoolArray(a, b, c, d, e, f)      PetscOptionsBoolArray_Private(PetscOptionsObject, a, b, c, d, e, f)
  #define PetscOptionsEnumArray(a, b, c, d, e, f, g)   PetscOptionsEnumArray_Private(PetscOptionsObject, a, b, c, d, e, f, g)
  #define PetscOptionsDeprecated(a, b, c, d)           PetscOptionsDeprecated_Private(PetscOptionsObject, a, b, c, d)
  #define PetscOptionsDeprecatedNoObject(a, b, c, d)   PetscOptionsDeprecated_Private(NULL, a, b, c, d)
#endif /* PETSC_CLANG_STATIC_ANALYZER */

PETSC_EXTERN PetscErrorCode PetscOptionsEnum_Private(PetscOptionItems *, const char[], const char[], const char[], const char *const *, PetscEnum, PetscEnum *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsInt_Private(PetscOptionItems *, const char[], const char[], const char[], PetscInt, PetscInt *, PetscBool *, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscOptionsReal_Private(PetscOptionItems *, const char[], const char[], const char[], PetscReal, PetscReal *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsScalar_Private(PetscOptionItems *, const char[], const char[], const char[], PetscScalar, PetscScalar *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsName_Private(PetscOptionItems *, const char[], const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsString_Private(PetscOptionItems *, const char[], const char[], const char[], const char[], char *, size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBool_Private(PetscOptionItems *, const char[], const char[], const char[], PetscBool, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroupBegin_Private(PetscOptionItems *, const char[], const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroup_Private(PetscOptionItems *, const char[], const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroupEnd_Private(PetscOptionItems *, const char[], const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsFList_Private(PetscOptionItems *, const char[], const char[], const char[], PetscFunctionList, const char[], char[], size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsEList_Private(PetscOptionItems *, const char[], const char[], const char[], const char *const *, PetscInt, const char[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsRealArray_Private(PetscOptionItems *, const char[], const char[], const char[], PetscReal[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsScalarArray_Private(PetscOptionItems *, const char[], const char[], const char[], PetscScalar[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsIntArray_Private(PetscOptionItems *, const char[], const char[], const char[], PetscInt[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringArray_Private(PetscOptionItems *, const char[], const char[], const char[], char *[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolArray_Private(PetscOptionItems *, const char[], const char[], const char[], PetscBool[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsEnumArray_Private(PetscOptionItems *, const char[], const char[], const char[], const char *const *, PetscEnum[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsDeprecated_Private(PetscOptionItems *, const char[], const char[], const char[], const char[]);

PETSC_EXTERN PetscErrorCode PetscOptionsSAWsDestroy(void);

PETSC_EXTERN PetscErrorCode PetscObjectAddOptionsHandler(PetscObject, PetscErrorCode (*)(PetscObject, PetscOptionItems *, void *), PetscErrorCode (*)(PetscObject, void *), void *);
PETSC_EXTERN PetscErrorCode PetscObjectProcessOptionsHandlers(PetscObject, PetscOptionItems *);
PETSC_EXTERN PetscErrorCode PetscObjectDestroyOptionsHandlers(PetscObject);

PETSC_EXTERN PetscErrorCode PetscOptionsLeftError(void);
#endif
