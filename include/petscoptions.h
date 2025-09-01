/*
   Routines to determine options set in the options database.
*/
#pragma once

#include <petscsys.h>
#include <petscviewertypes.h>

/* SUBMANSEC = Sys */

typedef enum {
  PETSC_OPT_CODE,
  PETSC_OPT_COMMAND_LINE,
  PETSC_OPT_FILE,
  PETSC_OPT_ENVIRONMENT,
  NUM_PETSC_OPT_SOURCE
} PetscOptionSource;

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
PETSC_EXTERN PetscErrorCode PetscOptionsGetBool3(PetscOptions, const char[], const char[], PetscBool3 *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetInt(PetscOptions, const char[], const char[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsGetMPIInt(PetscOptions, const char[], const char[], PetscMPIInt *, PetscBool *);
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
PETSC_EXTERN PetscErrorCode PetscOptionsInsertArgs(PetscOptions, int, const char *const *);
PETSC_EXTERN PetscErrorCode PetscOptionsClear(PetscOptions);
PETSC_EXTERN PetscErrorCode PetscOptionsPrefixPush(PetscOptions, const char[]);
PETSC_EXTERN PetscErrorCode PetscOptionsPrefixPop(PetscOptions);

PETSC_EXTERN PetscErrorCode PetscOptionsGetenv(MPI_Comm, const char[], char[], size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToBool(const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToInt(const char[], PetscInt *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToReal(const char[], PetscReal *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringToScalar(const char[], PetscScalar *);

PETSC_EXTERN PetscErrorCode PetscOptionsMonitorSet(PetscErrorCode (*)(const char[], const char[], PetscOptionSource, void *), void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode PetscOptionsMonitorDefault(const char[], const char[], PetscOptionSource, void *);

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
  PetscInt           arraylength; /* number of entries in data in the case that it is an array (of PetscInt etc), never a giant value */
  PetscBool          set;         /* the user has changed this value in the GUI */
  PetscOptionType    type;
  PetscOptionItem    next;
  char              *pman;
  void              *edata;
};

typedef struct _n_PetscOptionItems *PetscOptionItems;
struct _n_PetscOptionItems {
  PetscInt        count;
  PetscOptionItem next;
  char           *prefix, *pprefix;
  char           *title;
  MPI_Comm        comm;
  PetscBool       printhelp, changedmethod, alreadyprinted;
  PetscObject     object;
  PetscOptions    options;
};

#if defined(PETSC_CLANG_STATIC_ANALYZER)
extern PetscOptionItems PetscOptionsObject; /* declare this so that the PetscOptions stubs work */
PetscErrorCode          PetscOptionsBegin(MPI_Comm, const char *, const char *, const char *);
PetscErrorCode          PetscObjectOptionsBegin(PetscObject);
PetscErrorCode          PetscOptionsEnd(void);
#else
  /*MC
    PetscOptionsBegin - Begins a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively. Often one should
     use `PetscObjectOptionsBegin()` rather than this call.

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
    `PetscOptionsPublish` is set in `PetscOptionsCheckInitial_Private()` with `-saws_options`. When `PetscOptionsPublish` is set the
    loop between `PetscOptionsBegin()` and `PetscOptionsEnd()` is run THREE times with `PetscOptionsPublishCount` of values -1,0,1.
     Otherwise the loop is run ONCE with a `PetscOptionsPublishCount` of 1.
+      \-1 - `PetscOptionsInt()` etc. just call `PetscOptionsGetInt()` etc.
.      0   - The GUI objects are created in `PetscOptionsInt()` etc. and displayed in `PetscOptionsEnd()` and the options
              database updated with user changes; `PetscOptionsGetInt()` etc. are also called.
-      1   - `PetscOptionsInt()` etc. again call `PetscOptionsGetInt()` etc. (possibly getting new values), in addition the help message and
              default values are printed if -help was given.
     When `PetscOptionsObject.changedmethod` is set this causes `PetscOptionsPublishCount` to be reset to -2 (so in the next loop iteration it is -1)
     and the whole process is repeated. This is to handle when, for example, the `KSPType` is changed thus changing the list of
     options available so they need to be redisplayed so the user can change the. Changing `PetscOptionsObjects.changedmethod` is never
     currently set.

     Fortran Note:
     Returns ierr error code as the final argument per PETSc Fortran API

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
      struct _n_PetscOptionItems PetscOptionsObjectBase; \
      PetscOptionItems           PetscOptionsObject = &PetscOptionsObjectBase; \
      PetscCall(PetscMemzero(PetscOptionsObject, sizeof(*PetscOptionsObject))); \
      for (PetscOptionsObject->count = (PetscOptionsPublish ? -1 : 1); PetscOptionsObject->count < 2; PetscOptionsObject->count++) { \
        PetscCall(PetscOptionsBegin_Private(PetscOptionsObject, comm, prefix, mess, sec))

  /*MC
    PetscObjectOptionsBegin - Begins a set of queries on the options database that are related and should be
    displayed on the same window of a GUI that allows the user to set the options interactively.

    Synopsis:
    #include <petscoptions.h>
    PetscErrorCode PetscObjectOptionsBegin(PetscObject obj)

    Collective

    Input Parameter:
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
      struct _n_PetscOptionItems PetscOptionsObjectBase; \
      PetscOptionItems           PetscOptionsObject = &PetscOptionsObjectBase; \
      PetscOptionsObject->options                   = ((PetscObject)obj)->options; \
      for (PetscOptionsObject->count = (PetscOptionsPublish ? -1 : 1); PetscOptionsObject->count < 2; PetscOptionsObject->count++) { \
        PetscCall(PetscObjectOptionsBegin_Private(obj, PetscOptionsObject))

  /*MC
    PetscOptionsEnd - Ends a set of queries on the options database that are related and should be
     displayed on the same window of a GUI that allows the user to set the options interactively.

    Synopsis:
     #include <petscoptions.h>
     PetscErrorCode PetscOptionsEnd(void)

    Collective on the comm used in `PetscOptionsBegin()` or obj used in `PetscObjectOptionsBegin()`

    Level: intermediate

    Notes:
    Needs to be preceded by a call to `PetscOptionsBegin()` or `PetscObjectOptionsBegin()`

    This is a macro that handles its own error checking, it does not return an error code.

    Fortran Note:
    Returns ierr error code as the final argument per PETSc Fortran API

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

PETSC_EXTERN PetscErrorCode PetscOptionsBegin_Private(PetscOptionItems, MPI_Comm, const char[], const char[], const char[]);
PETSC_EXTERN PetscErrorCode PetscObjectOptionsBegin_Private(PetscObject, PetscOptionItems);
PETSC_EXTERN PetscErrorCode PetscOptionsEnd_Private(PetscOptionItems);
PETSC_EXTERN PetscErrorCode PetscOptionsHeadBegin(PetscOptionItems, const char[]);

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
   in `KSPSetFromOptions_GMRES()`.

   Logically Collective on the communicator passed in `PetscOptionsBegin()`

   Input Parameter:
.  head - the heading text

   Level: developer

   Notes:
   Handles errors directly, hence does not return an error code

   Must be between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`, and `PetscOptionsObject` created in `PetscOptionsBegin()` should be the first argument

   Must be followed by a call to `PetscOptionsHeadEnd()` in the same function.

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsHeadBegin(PetscOptionsObject, head) \
    do { \
      if (PetscOptionsObject->printhelp && PetscOptionsObject->count == 1 && !PetscOptionsObject->alreadyprinted) PetscCall((*PetscHelpPrintf)(PetscOptionsObject->comm, "  %s\n", head)); \
    } while (0)

  #define PetscOptionsHead(...) PETSC_DEPRECATED_MACRO(3, 18, 0, "PetscOptionsHeadBegin()", ) PetscOptionsHeadBegin(__VA_ARGS__)

  /*MC
     PetscOptionsHeadEnd - Ends a section of options begun with `PetscOptionsHeadBegin()`
     See, for example, `KSPSetFromOptions_GMRES()`.

     Synopsis:
     #include <petscoptions.h>
     PetscErrorCode PetscOptionsHeadEnd(void)

     Collective on the comm used in `PetscOptionsBegin()` or obj used in `PetscObjectOptionsBegin()`

     Level: intermediate

     Notes:
     Must be between a `PetscOptionsBegin()` or `PetscObjectOptionsBegin()` and a `PetscOptionsEnd()`

     Must be preceded by a call to `PetscOptionsHeadBegin()` in the same function.

     This needs to be used only if the code below `PetscOptionsHeadEnd()` can be run ONLY once.
     See, for example, `PCSetFromOptions_Composite()`. This is a `return(0)` in it for early exit
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
      if (PetscOptionsObject->count != 1) PetscFunctionReturn(PETSC_SUCCESS); \
    } while (0)

  #define PetscOptionsTail(...)                                                     PETSC_DEPRECATED_MACRO(3, 18, 0, "PetscOptionsHeadEnd()", ) PetscOptionsHeadEnd(__VA_ARGS__)

/*MC
  PetscOptionsEnum - Gets the enum value for a particular option in the database.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsEnum(const char opt[], const char text[], const char man[], const char *const *list, PetscEnum currentvalue, PetscEnum *value, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. text         - short string that describes the option
. man          - manual page with additional information on option
. list         - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
- currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
                 PetscOptionsEnum(..., obj->value,&object->value,...) or
                 value = defaultvalue
                 PetscOptionsEnum(..., value,&value,&set);
                 if (set) {
.ve

  Output Parameters:
+ value - the  value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  Must be between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

  `list` is usually something like `PCASMTypes` or some other predefined list of enum names

  If the user does not supply the option at all `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

  The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsEnum(opt, text, man, list, currentvalue, value, set)          PetscOptionsEnum_Private(PetscOptionsObject, opt, text, man, list, currentvalue, value, set)

/*MC
  PetscOptionsInt - Gets the integer value for a particular option in the database.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsInt(const char opt[], const char text[], const char man[], PetscInt currentvalue, PetscInt *value, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. text         - short string that describes the option
. man          - manual page with additional information on option
- currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
                 PetscOptionsInt(..., obj->value, &obj->value, ...) or
                 value = defaultvalue
                 PetscOptionsInt(..., value, &value, &set);
                 if (set) {
.ve

  Output Parameters:
+ value - the integer value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  If the user does not supply the option at all `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

  The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

  Must be between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsBoundedInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsRangeInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsBoundedReal()`, `PetscOptionsRangeReal()`
M*/
  #define PetscOptionsInt(opt, text, man, currentvalue, value, set)                 PetscOptionsInt_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set, PETSC_INT_MIN, PETSC_INT_MAX)

/*MC
  PetscOptionsMPIInt - Gets the MPI integer value for a particular option in the database.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsMPIInt(const char opt[], const char text[], const char man[], PetscMPIInt currentvalue, PetscMPIInt *value, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. text         - short string that describes the option
. man          - manual page with additional information on option
- currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
                 PetscOptionsInt(..., obj->value, &obj->value, ...) or
                 value = defaultvalue
                 PetscOptionsInt(..., value, &value, &set);
                 if (set) {
.ve

  Output Parameters:
+ value - the MPI integer value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  If the user does not supply the option at all `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

  The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

  Must be between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsBoundedInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsRangeInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsBoundedReal()`, `PetscOptionsRangeReal()`
M*/
  #define PetscOptionsMPIInt(opt, text, man, currentvalue, value, set)              PetscOptionsMPIInt_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set, PETSC_MPI_INT_MIN, PETSC_MPI_INT_MAX)

/*MC
   PetscOptionsBoundedInt - Gets an integer value greater than or equal to a given bound for a particular option in the database.

   Synopsis:
   #include <petscoptions.h>
   PetscErrorCode  PetscOptionsBoundedInt(const char opt[], const char text[], const char man[], PetscInt currentvalue, PetscInt *value, PetscBool *set, PetscInt bound)

   Logically Collective on the communicator passed in `PetscOptionsBegin()`

   Input Parameters:
+  opt          - option name
.  text         - short string that describes the option
.  man          - manual page with additional information on option
.  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
  PetscOptionsBoundedInt(..., obj->value, &obj->value, ...)
.ve
or
.vb
  value = defaultvalue
  PetscOptionsBoundedInt(..., value, &value, &set, ...);
  if (set) {
.ve
-  bound - the requested value should be greater than or equal to this bound or an error is generated

   Output Parameters:
+  value - the integer value to return
-  set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

   Level: beginner

   Notes:
   If the user does not supply the option at all `value` is NOT changed. Thus
   you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

   The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

   Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsRangeInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsBoundedReal()`, `PetscOptionsRangeReal()`
M*/
  #define PetscOptionsBoundedInt(opt, text, man, currentvalue, value, set, lb)      PetscOptionsInt_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set, lb, PETSC_INT_MAX)

/*MC
   PetscOptionsRangeInt - Gets an integer value within a range of values for a particular option in the database.

   Synopsis:
   #include <petscoptions.h>
   PetscErrorCode PetscOptionsRangeInt(const char opt[], const char text[], const char man[], PetscInt currentvalue, PetscInt *value, PetscBool *set, PetscInt lb, PetscInt ub)

   Logically Collective on the communicator passed in `PetscOptionsBegin()`

   Input Parameters:
+  opt          - option name
.  text         - short string that describes the option
.  man          - manual page with additional information on option
.  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
  PetscOptionsRangeInt(..., obj->value, &obj->value, ...)
.ve
or
.vb
  value = defaultvalue
  PetscOptionsRangeInt(..., value, &value, &set, ...);
  if (set) {
.ve
.  lb - the lower bound, provided value must be greater than or equal to this value or an error is generated
-  ub - the upper bound, provided value must be less than or equal to this value or an error is generated

   Output Parameters:
+  value - the integer value to return
-  set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

   Level: beginner

   Notes:
   If the user does not supply the option at all `value` is NOT changed. Thus
   you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

   The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

   Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsBoundedInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsBoundedReal()`, `PetscOptionsRangeReal()`
M*/
  #define PetscOptionsRangeInt(opt, text, man, currentvalue, value, set, lb, ub)    PetscOptionsInt_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set, lb, ub)

/*MC
  PetscOptionsReal - Gets a `PetscReal` value for a particular option in the database.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsReal(const char opt[], const char text[], const char man[], PetscReal currentvalue, PetscReal *value, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. text         - short string that describes the option
. man          - manual page with additional information on option
- currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
                 PetscOptionsReal(..., obj->value,&obj->value,...) or
                 value = defaultvalue
                 PetscOptionsReal(..., value,&value,&set);
                 if (set) {
.ve

  Output Parameters:
+ value - the value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  If the user does not supply the option at all `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

  The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsBoundedReal()`, `PetscOptionsRangeReal()`
M*/
  #define PetscOptionsReal(opt, text, man, currentvalue, value, set)                PetscOptionsReal_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set, PETSC_MIN_REAL, PETSC_MAX_REAL)

/*MC
   PetscOptionsBoundedReal - Gets a `PetscReal` value greater than or equal to a given bound for a particular option in the database.

   Synopsis:
   #include <petscoptions.h>
   PetscErrorCode  PetscOptionsBoundedReal(const char opt[], const char text[], const char man[], PetscReal currentvalue, PetscReal *value, PetscBool *set, PetscReal bound)

   Logically Collective on the communicator passed in `PetscOptionsBegin()`

   Input Parameters:
+  opt          - option name
.  text         - short string that describes the option
.  man          - manual page with additional information on option
.  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
  PetscOptionsBoundedReal(..., obj->value, &obj->value, ...)
.ve
or
.vb
  value = defaultvalue
  PetscOptionsBoundedReal(..., value, &value, &set, ...);
  if (set) {
.ve
-  bound - the requested value should be greater than or equal to this bound or an error is generated

   Output Parameters:
+  value - the real value to return
-  set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

   Level: beginner

   Notes:
   If the user does not supply the option at all `value` is NOT changed. Thus
   you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

   The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

   Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsRangeInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsBoundedInt()`, `PetscOptionsRangeReal()`
M*/
  #define PetscOptionsBoundedReal(opt, text, man, currentvalue, value, set, lb)     PetscOptionsReal_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set, lb, PETSC_MAX_REAL)

/*MC
   PetscOptionsRangeReal - Gets a `PetscReal` value within a range of values for a particular option in the database.

   Synopsis:
   #include <petscoptions.h>
   PetscErrorCode PetscOptionsRangeReal(const char opt[], const char text[], const char man[], PetscReal currentvalue, PetscReal *value, PetscBool *set, PetscReal lb, PetscReal ub)

   Logically Collective on the communicator passed in `PetscOptionsBegin()`

   Input Parameters:
+  opt          - option name
.  text         - short string that describes the option
.  man          - manual page with additional information on option
.  currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
  PetscOptionsRangeReal(..., obj->value, &obj->value, ...)
.ve
or
.vb
  value = defaultvalue
  PetscOptionsRangeReal(..., value, &value, &set, ...);
  if (set) {
.ve
.  lb - the lower bound, provided value must be greater than or equal to this value or an error is generated
-  ub - the upper bound, provided value must be less than or equal to this value or an error is generated

   Output Parameters:
+  value - the value to return
-  set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

   Level: beginner

   Notes:
   If the user does not supply the option at all `value` is NOT changed. Thus
   you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

   The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

   Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsInt()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`, `PetscOptionsBoundedInt()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsRangeInt()`, `PetscOptionsBoundedReal()`
M*/
  #define PetscOptionsRangeReal(opt, text, man, currentvalue, value, set, lb, ub)   PetscOptionsReal_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set, lb, ub)

/*MC
  PetscOptionsScalar - Gets the `PetscScalar` value for a particular option in the database.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsScalar(const char opt[], const char text[], const char man[], PetscScalar currentvalue, PetscScalar *value, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. text         - short string that describes the option
. man          - manual page with additional information on option
- currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with either
.vb
                 PetscOptionsScalar(..., obj->value,&obj->value,...) or
                 value = defaultvalue
                 PetscOptionsScalar(..., value,&value,&set);
                 if (set) {
.ve

  Output Parameters:
+ value - the value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  If the user does not supply the option at all `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

  The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsScalar(opt, text, man, currentvalue, value, set)              PetscOptionsScalar_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set)

/*MC
  PetscOptionsName - Determines if a particular option has been set in the database. This returns true whether the option is a number, string or boolean, even
  its value is set to false.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsName(const char opt[], const char text[], const char man[], PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - option name
. text - short string that describes the option
- man  - manual page with additional information on option

  Output Parameter:
. set - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Note:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsName(opt, text, man, set)                                     PetscOptionsName_Private(PetscOptionsObject, opt, text, man, set)

/*MC
  PetscOptionsString - Gets the string value for a particular option in the database.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsString(const char opt[], const char text[], const char man[], const char currentvalue[], char value[], size_t len, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. text         - short string that describes the option
. man          - manual page with additional information on option
. currentvalue - the current value; caller is responsible for setting this value correctly. This is not used to set value
- len          - length of the result string including null terminator

  Output Parameters:
+ value - the value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

  If the user provided no string (for example `-optionname` `-someotheroption`) `set` is set to `PETSC_TRUE` (and the string is filled with nulls).

  If the user does not supply the option at all `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that `set` is `PETSC_TRUE`.

  The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsString(opt, text, man, currentvalue, value, len, set)         PetscOptionsString_Private(PetscOptionsObject, opt, text, man, currentvalue, value, len, set)

/*MC
  PetscOptionsBool - Determines if a particular option is in the database with a true or false

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsBool(const char opt[], const char text[], const char man[], PetscBool currentvalue, PetscBool *flg, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. text         - short string that describes the option
. man          - manual page with additional information on option
- currentvalue - the current value

  Output Parameters:
+ flg - `PETSC_TRUE` or `PETSC_FALSE`
- set - `PETSC_TRUE` if found, else `PETSC_FALSE`, pass `NULL` if not needed

  Level: beginner

  Notes:
  TRUE, true, YES, yes, nostring, and 1 all translate to `PETSC_TRUE`
  FALSE, false, NO, no, and 0 all translate to `PETSC_FALSE`

  If the option is given, but no value is provided, then `flg` and `set` are both given the value `PETSC_TRUE`. That is `-requested_bool`
  is equivalent to `-requested_bool true`

  If the user does not supply the option at all `flg` is NOT changed. Thus
  you should ALWAYS initialize the `flg` variable if you access it without first checking that the `set` flag is `PETSC_TRUE`.

  Must be between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsBool(opt, text, man, currentvalue, value, set)                PetscOptionsBool_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set)

/*MC
  PetscOptionsBool3 - Determines if a particular option is in the database with a true, false, or unknown

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsBool3(const char opt[], const char text[], const char man[], PetscBool currentvalue, PetscBool3 *flg, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. text         - short string that describes the option
. man          - manual page with additional information on option
- currentvalue - the current value

  Output Parameters:
+ flg - `PETSC_BOOL3_TRUE`, `PETSC_BOOL3_FALSE`, or `PETSC_BOOL3_UNKNOWN`
- set - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  TRUE, true, YES, yes, nostring, and 1 all translate to `PETSC_TRUE`
  FALSE, false, NO, no, and 0 all translate to `PETSC_FALSE`

  If the option is given, but no value is provided, then `flg` and `set` are both given the value `PETSC_BOOL3_TRUE`. That is `-requested_bool`
  is equivalent to `-requested_bool true`

  If the user does not supply the option at all `flg` is NOT changed. Thus
  you should ALWAYS initialize the `flg` variable if you access it without first checking that the `set` flag is `PETSC_TRUE`.

  Must be between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`, `PetscOptionsGetInt()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsBool3(opt, text, man, currentvalue, value, set)               PetscOptionsBool3_Private(PetscOptionsObject, opt, text, man, currentvalue, value, set)

/*MC
  PetscOptionsBoolGroupBegin - First in a series of logical queries on the options database for
  which at most a single value can be true.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsBoolGroupBegin(const char opt[], const char text[], const char man[], PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - option name
. text - short string that describes the option
- man  - manual page with additional information on option

  Output Parameter:
. set - whether that option was set or not

  Level: intermediate

  Notes:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

  Must be followed by 0 or more `PetscOptionsBoolGroup()`s and `PetscOptionsBoolGroupEnd()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsBoolGroupBegin(opt, text, man, set)                           PetscOptionsBoolGroupBegin_Private(PetscOptionsObject, opt, text, man, set)

/*MC
  PetscOptionsBoolGroup - One in a series of logical queries on the options database for
  which at most a single value can be true.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsBoolGroup(const char opt[], const char text[], const char man[], PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - option name
. text - short string that describes the option
- man  - manual page with additional information on option

  Output Parameter:
. set - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: intermediate

  Notes:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

  Must follow a `PetscOptionsBoolGroupBegin()` and preceded a `PetscOptionsBoolGroupEnd()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsBoolGroup(opt, text, man, set)                                PetscOptionsBoolGroup_Private(PetscOptionsObject, opt, text, man, set)

/*MC
  PetscOptionsBoolGroupEnd - Last in a series of logical queries on the options database for
  which at most a single value can be true.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsBoolGroupEnd(const char opt[], const char text[], const char man[], PetscBool  *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - option name
. text - short string that describes the option
- man  - manual page with additional information on option

  Output Parameter:
. set - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: intermediate

  Notes:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

  Must follow a `PetscOptionsBoolGroupBegin()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsBoolGroupEnd(opt, text, man, set)                             PetscOptionsBoolGroupEnd_Private(PetscOptionsObject, opt, text, man, set)

/*MC
  PetscOptionsFList - Puts a list of option values that a single one may be selected from

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsFList(const char opt[], const char ltext[], const char man[], PetscFunctionList list, const char currentvalue[], char value[], size_t len, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. ltext        - short string that describes the option
. man          - manual page with additional information on option
. list         - the possible choices
. currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with
.vb
                 PetscOptionsFlist(..., obj->value,value,len,&set);
                 if (set) {
.ve
- len          - the length of the character array value

  Output Parameters:
+ value - the value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: intermediate

  Notes:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

  If the user does not supply the option at all `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that the `set` flag is `PETSC_TRUE`.

  The `currentvalue` passed into this routine does not get transferred to the output `value` variable automatically.

  See `PetscOptionsEList()` for when the choices are given in a string array

  To get a listing of all currently specified options,
  see `PetscOptionsView()` or `PetscOptionsGetAll()`

  Developer Note:
  This cannot check for invalid selection because of things like `MATAIJ` that are not included in the list

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsEnum()`
M*/
  #define PetscOptionsFList(opt, ltext, man, list, currentvalue, value, len, set)   PetscOptionsFList_Private(PetscOptionsObject, opt, ltext, man, list, currentvalue, value, len, set)

/*MC
  PetscOptionsEList - Puts a list of option values that a single one may be selected from

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsEList(const char opt[], const char ltext[], const char man[], const char *const *list, PetscInt ntext, const char currentvalue[], PetscInt *value, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt          - option name
. ltext        - short string that describes the option
. man          - manual page with additional information on option
. list         - the possible choices (one of these must be selected, anything else is invalid)
. ntext        - number of choices
- currentvalue - the current value; caller is responsible for setting this value correctly. Normally this is done with
.vb
                 PetscOptionsEList(..., obj->value,&value,&set);
.ve                 if (set) {

  Output Parameters:
+ value - the index of the value to return
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: intermediate

  Notes:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

  If the user does not supply the option at all `value` is NOT changed. Thus
  you should ALWAYS initialize `value` if you access it without first checking that the `set` flag is `PETSC_TRUE`.

  See `PetscOptionsFList()` for when the choices are given in a `PetscFunctionList()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEnum()`
M*/
  #define PetscOptionsEList(opt, ltext, man, list, ntext, currentvalue, value, set) PetscOptionsEList_Private(PetscOptionsObject, opt, ltext, man, list, ntext, currentvalue, value, set)

/*MC
  PetscOptionsRealArray - Gets an array of double values for a particular
  option in the database. The values must be separated with commas with
  no intervening spaces.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsRealArray(const char opt[], const char text[], const char man[], PetscReal value[], PetscInt *n, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - the option one is seeking
. text - short string describing option
. man  - manual page for option
- n    - maximum number of values that value has room for

  Output Parameters:
+ value - location to copy values
. n     - actual number of values found
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Note:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsRealArray(opt, text, man, value, n, set)                      PetscOptionsRealArray_Private(PetscOptionsObject, opt, text, man, value, n, set)

/*MC
  PetscOptionsScalarArray - Gets an array of `PetscScalar` values for a particular
  option in the database. The values must be separated with commas with
  no intervening spaces.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsScalarArray(const char opt[], const char text[], const char man[], PetscScalar value[], PetscInt *n, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - the option one is seeking
. text - short string describing option
. man  - manual page for option
- n    - maximum number of values allowed in the value array

  Output Parameters:
+ value - location to copy values
. n     - actual number of values found
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Note:
  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsScalarArray(opt, text, man, value, n, set)                    PetscOptionsScalarArray_Private(PetscOptionsObject, opt, text, man, value, n, set)

/*MC
  PetscOptionsIntArray - Gets an array of integers for a particular
  option in the database.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsIntArray(const char opt[], const char text[], const char man[], PetscInt value[], PetscInt *n, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - the option one is seeking
. text - short string describing option
. man  - manual page for option
- n    - maximum number of values

  Output Parameters:
+ value - location to copy values
. n     - actual number of values found
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  The array can be passed as
+   a comma separated list -                                  0,1,2,3,4,5,6,7
.   a range (start\-end+1) -                                  0-8
.   a range with given increment (start\-end+1:inc) -         0-7:2
-   a combination of values and ranges separated by commas -  0,1-8,8-15:2

  There must be no intervening spaces between the values.

  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsIntArray(opt, text, man, value, n, set)                       PetscOptionsIntArray_Private(PetscOptionsObject, opt, text, man, value, n, set)

/*MC
  PetscOptionsStringArray - Gets an array of string values for a particular
  option in the database. The values must be separated with commas with
  no intervening spaces.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsStringArray(const char opt[], const char text[], const char man[], char *value[], PetscInt *nmax, PetscBool  *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`; No Fortran Support

  Input Parameters:
+ opt  - the option one is seeking
. text - short string describing option
. man  - manual page for option
- n    - maximum number of strings

  Output Parameters:
+ value - location to copy strings
. n     - actual number of strings found
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  The user should pass in an array of pointers to char, to hold all the
  strings returned by this function.

  The user is responsible for deallocating the strings that are
  returned.

  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsStringArray(opt, text, man, value, n, set)                    PetscOptionsStringArray_Private(PetscOptionsObject, opt, text, man, value, n, set)

/*MC
  PetscOptionsBoolArray - Gets an array of logical values (true or false) for a particular
  option in the database. The values must be separated with commas with
  no intervening spaces.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsBoolArray(const char opt[], const char text[], const char man[], PetscBool value[], PetscInt *n, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - the option one is seeking
. text - short string describing option
. man  - manual page for option
- n    - maximum number of values allowed in the value array

  Output Parameters:
+ value - location to copy values
. n     - actual number of values found
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  The user should pass in an array of `PetscBool`

  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsBoolArray(opt, text, man, value, n, set)                      PetscOptionsBoolArray_Private(PetscOptionsObject, opt, text, man, value, n, set)

/*MC
  PetscOptionsEnumArray - Gets an array of enum values for a particular
  option in the database.

  Synopsis:
  #include <petscoptions.h>
  PetscErrorCode PetscOptionsEnumArray(const char opt[], const char text[], const char man[], const char *const *list, PetscEnum value[], PetscInt *n, PetscBool *set)

  Logically Collective on the communicator passed in `PetscOptionsBegin()`

  Input Parameters:
+ opt  - the option one is seeking
. text - short string describing option
. man  - manual page for option
. list - array containing the list of choices, followed by the enum name, followed by the enum prefix, followed by a null
- n    - maximum number of values allowed in the value array

  Output Parameters:
+ value - location to copy values
. n     - actual number of values found
- set   - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: beginner

  Notes:
  The array must be passed as a comma separated list.

  There must be no intervening spaces between the values.

  Must be used between a `PetscOptionsBegin()` and a `PetscOptionsEnd()`

.seealso: `PetscOptionsGetInt()`, `PetscOptionsGetReal()`,
          `PetscOptionsHasName()`, `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsGetBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
M*/
  #define PetscOptionsEnumArray(opt, text, man, list, value, n, set)                PetscOptionsEnumArray_Private(PetscOptionsObject, opt, text, man, list, value, n, set)

/*MC
  PetscOptionsDeprecated - mark an option as deprecated, optionally replacing it with `newname`

  Prints a deprecation warning, unless an option is supplied to suppress.

  Logically Collective

  Input Parameters:
+ oldname - the old, deprecated option
. newname - the new option, or `NULL` if option is purely removed
. version - a string describing the version of first deprecation, e.g. "3.9"
- info    - additional information string, or `NULL`.

  Options Database Key:
. -options_suppress_deprecated_warnings - do not print deprecation warnings

  Level: developer

  Notes:
  If `newname` is provided then the options database will automatically check the database for `oldname`.

  The old call `PetscOptionsXXX`(`oldname`) should be removed from the source code when both (1) the call to `PetscOptionsDeprecated()` occurs before the
  new call to `PetscOptionsXXX`(`newname`) and (2) the argument handling of the new call to `PetscOptionsXXX`(`newname`) is identical to the previous call.
  See `PTScotch_PartGraph_Seq()` for an example of when (1) fails and `SNESTestJacobian()` where an example of (2) fails.

  Must be called between `PetscOptionsBegin()` (or `PetscObjectOptionsBegin()`) and `PetscOptionsEnd()`.
  Only the process of MPI rank zero that owns the `PetscOptionsItems` are argument (managed by `PetscOptionsBegin()` or `PetscObjectOptionsBegin()` prints the information
  If newname is provided, the old option is replaced. Otherwise, it remains in the options database.
  If an option is not replaced, the info argument should be used to advise the user on how to proceed.
  There is a limit on the length of the warning printed, so very long strings provided as info may be truncated.

.seealso: `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsScalar()`, `PetscOptionsBool()`, `PetscOptionsString()`, `PetscOptionsSetValue()`
M*/
  #define PetscOptionsDeprecated(opt, text, man, info)                              PetscOptionsDeprecated_Private(PetscOptionsObject, opt, text, man, info)

/*MC
  PetscOptionsDeprecatedMoObject - mark an option as deprecated in the global PetscOptionsObject, optionally replacing it with `newname`

  Prints a deprecation warning, unless an option is supplied to suppress.

  Logically Collective

  Input Parameters:
+ oldname - the old, deprecated option
. newname - the new option, or `NULL` if option is purely removed
. version - a string describing the version of first deprecation, e.g. "3.9"
- info    - additional information string, or `NULL`.

  Options Database Key:
. -options_suppress_deprecated_warnings - do not print deprecation warnings

  Level: developer

  Notes:
  If `newname` is provided then the options database will automatically check the database for `oldname`.

  The old call `PetscOptionsXXX`(`oldname`) should be removed from the source code when both (1) the call to `PetscOptionsDeprecated()` occurs before the
  new call to `PetscOptionsXXX`(`newname`) and (2) the argument handling of the new call to `PetscOptionsXXX`(`newname`) is identical to the previous call.
  See `PTScotch_PartGraph_Seq()` for an example of when (1) fails and `SNESTestJacobian()` where an example of (2) fails.

  Only the process of MPI rank zero that owns the `PetscOptionsItems` are argument (managed by `PetscOptionsBegin()` or `PetscObjectOptionsBegin()` prints the information
  If newname is provided, the old option is replaced. Otherwise, it remains in the options database.
  If an option is not replaced, the info argument should be used to advise the user on how to proceed.
  There is a limit on the length of the warning printed, so very long strings provided as info may be truncated.

.seealso: `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsScalar()`, `PetscOptionsBool()`, `PetscOptionsString()`, `PetscOptionsSetValue()`
M*/
  #define PetscOptionsDeprecatedNoObject(opt, text, man, info)                      PetscOptionsDeprecated_Private(NULL, opt, text, man, info)
#endif /* PETSC_CLANG_STATIC_ANALYZER */

PETSC_EXTERN PetscErrorCode PetscOptionsEnum_Private(PetscOptionItems, const char[], const char[], const char[], const char *const *, PetscEnum, PetscEnum *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsInt_Private(PetscOptionItems, const char[], const char[], const char[], PetscInt, PetscInt *, PetscBool *, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscOptionsMPIInt_Private(PetscOptionItems, const char[], const char[], const char[], PetscMPIInt, PetscMPIInt *, PetscBool *, PetscMPIInt, PetscMPIInt);
PETSC_EXTERN PetscErrorCode PetscOptionsReal_Private(PetscOptionItems, const char[], const char[], const char[], PetscReal, PetscReal *, PetscBool *, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscOptionsScalar_Private(PetscOptionItems, const char[], const char[], const char[], PetscScalar, PetscScalar *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsName_Private(PetscOptionItems, const char[], const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsString_Private(PetscOptionItems, const char[], const char[], const char[], const char[], char *, size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBool_Private(PetscOptionItems, const char[], const char[], const char[], PetscBool, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBool3_Private(PetscOptionItems, const char[], const char[], const char[], PetscBool3, PetscBool3 *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroupBegin_Private(PetscOptionItems, const char[], const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroup_Private(PetscOptionItems, const char[], const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolGroupEnd_Private(PetscOptionItems, const char[], const char[], const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsFList_Private(PetscOptionItems, const char[], const char[], const char[], PetscFunctionList, const char[], char[], size_t, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsEList_Private(PetscOptionItems, const char[], const char[], const char[], const char *const *, PetscInt, const char[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsRealArray_Private(PetscOptionItems, const char[], const char[], const char[], PetscReal[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsScalarArray_Private(PetscOptionItems, const char[], const char[], const char[], PetscScalar[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsIntArray_Private(PetscOptionItems, const char[], const char[], const char[], PetscInt[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsStringArray_Private(PetscOptionItems, const char[], const char[], const char[], char *[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsBoolArray_Private(PetscOptionItems, const char[], const char[], const char[], PetscBool[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsEnumArray_Private(PetscOptionItems, const char[], const char[], const char[], const char *const *, PetscEnum[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscOptionsDeprecated_Private(PetscOptionItems, const char[], const char[], const char[], const char[]);

PETSC_EXTERN PetscErrorCode PetscObjectAddOptionsHandler(PetscObject, PetscErrorCode (*)(PetscObject, PetscOptionItems, void *), PetscErrorCode (*)(PetscObject, void *), void *);
PETSC_EXTERN PetscErrorCode PetscObjectProcessOptionsHandlers(PetscObject, PetscOptionItems);
PETSC_EXTERN PetscErrorCode PetscObjectDestroyOptionsHandlers(PetscObject);

PETSC_EXTERN PetscErrorCode PetscOptionsLeftError(void);
