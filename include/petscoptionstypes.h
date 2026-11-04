/*
   Types for the PETSc options database. Include petscoptions.h for the functions that operate on them.
*/
#pragma once

#include <petscsystypes.h>

/* SUBMANSEC = Sys */

/*E
   PetscOptionSource - Records where a value in the PETSc options database came from, passed to options-database monitors

   Values:
+   `PETSC_OPT_CODE`         - the option was set by a call from inside source code, for example `PetscOptionsSetValue()`
.   `PETSC_OPT_COMMAND_LINE` - the option came from the command-line arguments of the program
.   `PETSC_OPT_FILE`         - the option came from an options file processed by `PetscOptionsInsertFile()` (or its YAML counterpart)
.   `PETSC_OPT_ENVIRONMENT`  - the option came from the `PETSC_OPTIONS` (or related) environment variable
-   `NUM_PETSC_OPT_SOURCE`   - sentinel; equals the number of valid option sources

   Level: developer

.seealso: `PetscOptions`, `PetscOptionsMonitorSet()`, `PetscOptionsMonitorDefault()`
E*/
typedef enum {
  PETSC_OPT_CODE,
  PETSC_OPT_COMMAND_LINE,
  PETSC_OPT_FILE,
  PETSC_OPT_ENVIRONMENT,
  NUM_PETSC_OPT_SOURCE
} PetscOptionSource;

#define PETSC_MAX_OPTION_NAME 512
/*S
  PetscOptions - PETSc's runtime options database object; the holder of all PETSc command-line and configuration options for a session, looked up via `PetscOptionsGet*()`

  Level: beginner

  Notes:
  Most PETSc API calls accept `NULL` for `PetscOptions`, meaning "the default global options database". Use `PetscOptionsCreate()` / `PetscOptionsPush()` to manage non-default databases (e.g. when reading options from a file).

  Each `PetscObject` may also carry its own non-default options through `PetscObjectSetOptions()`.

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscOptionsPush()`, `PetscOptionsPop()`, `PetscOptionsGetBool()`,
          `PetscOptionsGetInt()`, `PetscOptionsGetReal()`, `PetscOptionsGetString()`, `PetscOptionsSetValue()`, `PetscOptionsView()`
S*/
typedef struct _n_PetscOptions *PetscOptions;

/*
    See manual page for PetscOptionsBegin()

    PetscOptionsItem and PetscOptionsItems are a single option (such as ksp_type) and a collection of such single
  options being handled with a PetscOptionsBegin/End()

*/
/*E
   PetscOptionType - Identifies the kind of value held by a `PetscOptionItem` inside a `PetscOptionsBegin()`/`PetscOptionsEnd()` block

   Values:
+   `OPTION_INT`          - a single `PetscInt`
.   `OPTION_BOOL`         - a single `PetscBool`
.   `OPTION_REAL`         - a single `PetscReal`
.   `OPTION_FLIST`        - a selection from a registered `PetscFunctionList`
.   `OPTION_STRING`       - a single string
.   `OPTION_REAL_ARRAY`   - an array of `PetscReal`
.   `OPTION_SCALAR_ARRAY` - an array of `PetscScalar`
.   `OPTION_HEAD`         - a section heading inserted with `PetscOptionsHead()`
.   `OPTION_INT_ARRAY`    - an array of `PetscInt`
.   `OPTION_ELIST`        - a selection from an enumerated list of strings
.   `OPTION_BOOL_ARRAY`   - an array of `PetscBool`
-   `OPTION_STRING_ARRAY` - an array of strings

   Level: developer

.seealso: `PetscOptions`, `PetscOptionItem`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsInt()`, `PetscOptionsReal()`
E*/
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

/*S
  PetscOptionItem - Internal record describing a single option (such as `-ksp_type`) inside a `PetscOptionsBegin()` / `PetscOptionsEnd()` block, holding its option name, help text, default value, and selected value

  Level: developer

.seealso: `PetscOptions`, `PetscOptionItems`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsInt()`, `PetscOptionsBool()`
S*/
typedef struct _n_PetscOptionItem *PetscOptionItem;

/*S
  PetscOptionItems - Internal context object representing the set of options being processed inside a `PetscOptionsBegin()` / `PetscOptionsEnd()` block; holds a linked list of `PetscOptionItem`s, the option prefix and the owning `PetscObject`

  Level: developer

.seealso: `PetscOptions`, `PetscOptionItem`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscObjectOptionsBegin()`
S*/
typedef struct _n_PetscOptionItems *PetscOptionItems;
