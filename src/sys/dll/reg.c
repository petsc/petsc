
/*
    Provides a general mechanism to allow one to register new routines in
    dynamic libraries for many of the PETSc objects (including, e.g., KSP and PC).
*/
#include <petsc/private/petscimpl.h> /*I "petscsys.h" I*/
#include <petscviewer.h>

#include <petsc/private/hashmap.h>
/*
    This is the default list used by PETSc with the PetscDLLibrary register routines
*/
PetscDLLibrary PetscDLLibrariesLoaded = NULL;

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)

static PetscErrorCode PetscLoadDynamicLibrary(const char *name, PetscBool *found)
{
  char libs[PETSC_MAX_PATH_LEN], dlib[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(PetscStrncpy(libs, "${PETSC_LIB_DIR}/libpetsc", sizeof(libs)));
  PetscCall(PetscStrlcat(libs, name, sizeof(libs)));
  PetscCall(PetscDLLibraryRetrieve(PETSC_COMM_WORLD, libs, dlib, 1024, found));
  if (*found) {
    PetscCall(PetscDLLibraryAppend(PETSC_COMM_WORLD, &PetscDLLibrariesLoaded, dlib));
  } else {
    PetscCall(PetscStrncpy(libs, "${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc", sizeof(libs)));
    PetscCall(PetscStrlcat(libs, name, sizeof(libs)));
    PetscCall(PetscDLLibraryRetrieve(PETSC_COMM_WORLD, libs, dlib, 1024, found));
    if (*found) PetscCall(PetscDLLibraryAppend(PETSC_COMM_WORLD, &PetscDLLibrariesLoaded, dlib));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#if defined(PETSC_USE_SINGLE_LIBRARY) && !(defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES))
PETSC_EXTERN PetscErrorCode AOInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscSFInitializePackage(void);
  #if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN PetscErrorCode CharacteristicInitializePackage(void);
  #endif
PETSC_EXTERN PetscErrorCode ISInitializePackage(void);
PETSC_EXTERN PetscErrorCode VecInitializePackage(void);
PETSC_EXTERN PetscErrorCode MatInitializePackage(void);
PETSC_EXTERN PetscErrorCode DMInitializePackage(void);
PETSC_EXTERN PetscErrorCode PCInitializePackage(void);
PETSC_EXTERN PetscErrorCode KSPInitializePackage(void);
PETSC_EXTERN PetscErrorCode SNESInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSInitializePackage(void);
PETSC_EXTERN PetscErrorCode TaoInitializePackage(void);
#endif

/*
    PetscInitialize_DynamicLibraries - Adds the default dynamic link libraries to the
    search path.
*/
PETSC_INTERN PetscErrorCode PetscInitialize_DynamicLibraries(void)
{
  char     *libname[32];
  PetscInt  nmax, i;
  PetscBool preload = PETSC_FALSE;
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscBool PetscInitialized = PetscInitializeCalled;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_THREADSAFETY)
  /* These must be all initialized here because it is not safe for individual threads to call these initialize routines */
  preload = PETSC_TRUE;
#endif

  nmax = 32;
  PetscCall(PetscOptionsGetStringArray(NULL, NULL, "-dll_prepend", libname, &nmax, NULL));
  for (i = 0; i < nmax; i++) {
    PetscCall(PetscDLLibraryPrepend(PETSC_COMM_WORLD, &PetscDLLibrariesLoaded, libname[i]));
    PetscCall(PetscFree(libname[i]));
  }

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-library_preload", &preload, NULL));
  if (!preload) {
    PetscCall(PetscSysInitializePackage());
  } else {
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    PetscBool found;
  #if defined(PETSC_USE_SINGLE_LIBRARY)
    PetscCall(PetscLoadDynamicLibrary("", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc dynamic library \n You cannot move the dynamic libraries!");
  #else
    PetscCall(PetscLoadDynamicLibrary("sys", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc dynamic library \n You cannot move the dynamic libraries!");
    PetscCall(PetscLoadDynamicLibrary("vec", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc Vec dynamic library \n You cannot move the dynamic libraries!");
    PetscCall(PetscLoadDynamicLibrary("mat", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc Mat dynamic library \n You cannot move the dynamic libraries!");
    PetscCall(PetscLoadDynamicLibrary("dm", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc DM dynamic library \n You cannot move the dynamic libraries!");
    PetscCall(PetscLoadDynamicLibrary("ksp", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc KSP dynamic library \n You cannot move the dynamic libraries!");
    PetscCall(PetscLoadDynamicLibrary("snes", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc SNES dynamic library \n You cannot move the dynamic libraries!");
    PetscCall(PetscLoadDynamicLibrary("ts", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc TS dynamic library \n You cannot move the dynamic libraries!");
    PetscCall(PetscLoadDynamicLibrary("tao", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate Tao dynamic library \n You cannot move the dynamic libraries!");
  #endif
#else /* defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES) */
  #if defined(PETSC_USE_SINGLE_LIBRARY)
    PetscCall(AOInitializePackage());
    PetscCall(PetscSFInitializePackage());
    #if !defined(PETSC_USE_COMPLEX)
    PetscCall(CharacteristicInitializePackage());
    #endif
    PetscCall(ISInitializePackage());
    PetscCall(VecInitializePackage());
    PetscCall(MatInitializePackage());
    PetscCall(DMInitializePackage());
    PetscCall(PCInitializePackage());
    PetscCall(KSPInitializePackage());
    PetscCall(SNESInitializePackage());
    PetscCall(TSInitializePackage());
    PetscCall(TaoInitializePackage());
  #else
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Cannot use -library_preload with multiple static PETSc libraries");
  #endif
#endif /* defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES) */
  }

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES) && defined(PETSC_HAVE_BAMG)
  {
    PetscBool found;
    PetscCall(PetscLoadDynamicLibrary("bamg", &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to locate PETSc BAMG dynamic library \n You cannot move the dynamic libraries!");
  }
#endif

  nmax = 32;
  PetscCall(PetscOptionsGetStringArray(NULL, NULL, "-dll_append", libname, &nmax, NULL));
  for (i = 0; i < nmax; i++) {
    PetscCall(PetscDLLibraryAppend(PETSC_COMM_WORLD, &PetscDLLibrariesLoaded, libname[i]));
    PetscCall(PetscFree(libname[i]));
  }

#if defined(PETSC_HAVE_ELEMENTAL)
  /* in Fortran, PetscInitializeCalled is set to PETSC_TRUE before PetscInitialize_DynamicLibraries() */
  /* in C, it is not the case, but the value is forced to PETSC_TRUE so that PetscRegisterFinalize() is called */
  PetscInitializeCalled = PETSC_TRUE;
  PetscCall(PetscElementalInitializePackage());
  PetscInitializeCalled = PetscInitialized;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PetscFinalize_DynamicLibraries - Closes the opened dynamic libraries.
*/
PETSC_INTERN PetscErrorCode PetscFinalize_DynamicLibraries(void)
{
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dll_view", &flg, NULL));
  if (flg) PetscCall(PetscDLLibraryPrintPath(PetscDLLibrariesLoaded));
  PetscCall(PetscDLLibraryClose(PetscDLLibrariesLoaded));
  PetscDLLibrariesLoaded = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------------------*/
PETSC_HASH_MAP(HMapFunc, const char *, PetscVoidFunction, kh_str_hash_func, kh_str_hash_equal, NULL)

struct _n_PetscFunctionList {
  PetscHMapFunc map;
};

/* Keep a linked list of PetscFunctionLists so that we can destroy all the left-over ones. */
typedef struct n_PetscFunctionListDLAll *PetscFunctionListDLAll;
struct n_PetscFunctionListDLAll {
  PetscFunctionList      data;
  PetscFunctionListDLAll next;
};

static PetscFunctionListDLAll dlallhead = NULL;

static PetscErrorCode PetscFunctionListDLAllPush_Private(PetscFunctionList fl)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)) {
    PetscFunctionListDLAll head;

    PetscCall(PetscNew(&head));
    head->data = fl;
    head->next = dlallhead;
    dlallhead  = head;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFunctionListDLAllPop_Private(PetscFunctionList fl)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)) {
    PetscFunctionListDLAll current = dlallhead, prev = NULL;

    /* Remove this entry from the main DL list (if it is in it) */
    while (current) {
      const PetscFunctionListDLAll next = current->next;

      if (current->data == fl) {
        if (prev) {
          // somewhere in the middle (or end) of the list
          prev->next = next;
        } else {
          // prev = NULL implies current = dlallhead, so front of list
          dlallhead = next;
        }
        PetscCall(PetscFree(current));
        break;
      }
      prev    = current;
      current = next;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscHMapFuncInsert_Private(PetscHMapFunc map, const char name[], PetscVoidFunction fnc)
{
  PetscHashIter it;
  PetscBool     found;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 2);
  if (fnc) PetscValidFunction(fnc, 3);
  PetscCall(PetscHMapFuncFind(map, name, &it, &found));
  if (fnc) {
    if (found) {
      PetscCall(PetscHMapFuncIterSet(map, it, fnc));
    } else {
      char *tmp_name;

      PetscCall(PetscStrallocpy(name, &tmp_name));
      PetscCall(PetscHMapFuncSet(map, tmp_name, fnc));
    }
  } else if (found) {
    const char *tmp_name;

    PetscHashIterGetKey(map, it, tmp_name);
    PetscCall(PetscFree(tmp_name));
    PetscCall(PetscHMapFuncIterDel(map, it));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFunctionListCreate_Private(PetscInt size, PetscFunctionList *fl)
{
  PetscFunctionBegin;
  if (*fl) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscNew(fl));
  PetscCall(PetscHMapFuncCreateWithSize(size, &(*fl)->map));
  PetscCall(PetscFunctionListDLAllPush_Private(*fl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PetscFunctionListAdd - Given a routine and a string id, saves that routine in the
   specified registry.

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode PetscFunctionListAdd(PetscFunctionList *flist,const char name[],void (*fptr)(void))

   Not Collective

   Input Parameters:
+  flist - pointer to function list object
.  name - string to identify routine
-  fptr - function pointer

   Notes:
   To remove a registered routine, pass in a NULL fptr.

   Users who wish to register new classes for use by a particular PETSc
   component (e.g., `SNES`) should generally call the registration routine
   for that particular component (e.g., `SNESRegister()`) instead of
   calling `PetscFunctionListAdd()` directly.

    Level: developer

.seealso: `PetscFunctionListDestroy()`, `SNESRegister()`, `KSPRegister()`,
          `PCRegister()`, `TSRegister()`, `PetscFunctionList`, `PetscObjectComposeFunction()`
M*/
PetscErrorCode PetscFunctionListAdd_Private(PetscFunctionList *fl, const char name[], PetscVoidFunction fnc)
{
  PetscFunctionBegin;
  PetscValidPointer(fl, 1);
  if (name) PetscValidCharPointer(name, 2);
  if (fnc) PetscValidFunction(fnc, 3);
  PetscCall(PetscFunctionListCreate_Private(0, fl));
  PetscCall(PetscHMapFuncInsert_Private((*fl)->map, name, fnc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscFunctionListDestroy - Destroys a list of registered routines.

    Input Parameter:
.   fl  - pointer to list

    Level: developer

.seealso: `PetscFunctionListAdd()`, `PetscFunctionList`, `PetscFunctionListClear()`
@*/
PetscErrorCode PetscFunctionListDestroy(PetscFunctionList *fl)
{
  PetscFunctionBegin;
  if (!*fl) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFunctionListDLAllPop_Private(*fl));
  /* free this list */
  PetscCall(PetscFunctionListClear(*fl));
  PetscCall(PetscHMapFuncDestroy(&(*fl)->map));
  PetscCall(PetscFree(*fl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PetscHMapFuncForEach(__func_list__, __key_name__, __val_name__, ...) \
  do { \
    const PetscHMapFunc phmfi_map_ = (__func_list__)->map; \
    PetscHashIter       phmfi_iter_; \
\
    PetscHashIterBegin(phmfi_map_, phmfi_iter_); \
    while (!PetscHashIterAtEnd(phmfi_map_, phmfi_iter_)) { \
      const char *PETSC_UNUSED       __key_name__; \
      PetscVoidFunction PETSC_UNUSED __val_name__; \
\
      PetscHashIterGetKey(phmfi_map_, phmfi_iter_, __key_name__); \
      PetscHashIterGetVal(phmfi_map_, phmfi_iter_, __val_name__); \
      { \
        __VA_ARGS__; \
      } \
      PetscHashIterNext(phmfi_map_, phmfi_iter_); \
    } /* end while */ \
  } while (0)

/*@
  PetscFunctionListClear - Clear a `PetscFunctionList`

  Not Collective

  Input Parameter:
. fl - The `PetscFunctionList` to clear

  Notes:
  This clears the contents of `fl` but does not deallocate the entries themselves.

  Level: developer

.seealso: `PetscFunctionList`, `PetscFunctionListDestroy()`, `PetscFunctionListAdd()`
@*/
PetscErrorCode PetscFunctionListClear(PetscFunctionList fl)
{
  PetscFunctionBegin;
  if (fl) {
    PetscHMapFuncForEach(fl, name, func, PetscCall(PetscFree(name)));
    PetscCall(PetscHMapFuncClear(fl->map));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Print registered PetscFunctionLists
*/
PetscErrorCode PetscFunctionListPrintAll(void)
{
  PetscFunctionListDLAll current = dlallhead;

  PetscFunctionBegin;
  if (current) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Registered PetscFunctionLists\n", PetscGlobalRank));
  while (current) {
    PetscCall(PetscFunctionListPrintNonEmpty(current->data));
    current = current->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
    PetscFunctionListNonEmpty - Print composed names for non null function pointers

    Input Parameter:
.   flist   - pointer to list

    Level: developer

.seealso: `PetscFunctionListAdd()`, `PetscFunctionList`, `PetscObjectQueryFunction()`
M*/
PetscErrorCode PetscFunctionListPrintNonEmpty(PetscFunctionList fl)
{
  PetscFunctionBegin;
  if (fl) {
    // clang-format off
    PetscHMapFuncForEach(
      fl,
      name, func,
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, PETSC_STDOUT, "[%d] function name: %s\n", PetscGlobalRank, name));
    );
    // clang-format on
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
    PetscFunctionListFind - Find function registered under given name

    Synopsis:
    #include <petscsys.h>
    PetscErrorCode PetscFunctionListFind(PetscFunctionList flist,const char name[],void (**fptr)(void))

    Input Parameters:
+   flist   - pointer to list
-   name - name registered for the function

    Output Parameter:
.   fptr - the function pointer if name was found, else NULL

    Level: developer

.seealso: `PetscFunctionListAdd()`, `PetscFunctionList`, `PetscObjectQueryFunction()`
M*/
PetscErrorCode PetscFunctionListFind_Private(PetscFunctionList fl, const char name[], PetscVoidFunction *r)
{
  PetscFunctionBegin;
  PetscValidCharPointer(name, 2);
  PetscValidPointer(r, 3);
  *r = NULL;
  if (fl) PetscCall(PetscHMapFuncGet(fl->map, name, r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscFunctionListView - prints out contents of a `PetscFunctionList`

   Collective

   Input Parameters:
+  list - the list of functions
-  viewer - the `PetscViewer` used to view the `PetscFunctionList`

   Level: developer

.seealso: `PetscFunctionListAdd()`, `PetscFunctionListPrintTypes()`, `PetscFunctionList`
@*/
PetscErrorCode PetscFunctionListView(PetscFunctionList list, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidPointer(list, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCheck(iascii, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only ASCII viewer supported");
  {
    PetscInt size;

    PetscCall(PetscHMapFuncGetSize(list->map, &size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscFunctionList Object:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "size: %" PetscInt_FMT "\n", size));
    if (size) {
      PetscInt count = 0;

      PetscCall(PetscViewerASCIIPrintf(viewer, "functions:\n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscHMapFuncForEach(list, name, func, PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT ": %s\n", ++count, name)));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscFunctionListGet - Gets an array the contains the entries in `PetscFunctionList`, this is used
         by help etc.

   Not Collective

   Input Parameter:
.  list   - list of types

   Output Parameters:
+  array - array of names
-  n - length of array

   Note:
       This allocates the array so that must be freed. BUT the individual entries are
    not copied so should not be freed.

   Level: developer

.seealso: `PetscFunctionListAdd()`, `PetscFunctionList`
@*/
PetscErrorCode PetscFunctionListGet(PetscFunctionList list, const char ***array, int *n)
{
  PetscInt size = 0;

  PetscFunctionBegin;
  PetscValidPointer(array, 2);
  *array = NULL;
  if (list) {
    const PetscHMapFunc map = list->map;
    PetscInt            off = 0;

    PetscCall(PetscHMapFuncGetSize(map, &size));
    PetscCall(PetscMalloc1(size, (char ***)array));
    PetscCall(PetscHMapFuncGetKeys(map, &off, *array));
  }
  *n = (int)size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscFunctionListPrintTypes - Prints the methods available in a list of functions

   Collective over MPI_Comm

   Input Parameters:
+  comm   - the communicator (usually `MPI_COMM_WORLD`)
.  fd     - file to print to, usually stdout
.  prefix - prefix to prepend to name (optional)
.  name   - option string (for example, "-ksp_type")
.  text - short description of the object (for example, "Krylov solvers")
.  man - name of manual page that discusses the object (for example, "KSPCreate")
.  list   - list of types
.  def - default (current) value
-  newv - new value

   Level: developer

.seealso: `PetscFunctionListAdd()`, `PetscFunctionList`
@*/
PetscErrorCode PetscFunctionListPrintTypes(MPI_Comm comm, FILE *fd, const char prefix[], const char name[], const char text[], const char man[], PetscFunctionList list, const char def[], const char newv[])
{
  char p[64];

  PetscFunctionBegin;
  (void)fd;
  PetscCall(PetscStrncpy(p, "-", sizeof(p)));
  if (prefix) PetscCall(PetscStrlcat(p, prefix, sizeof(p)));
  PetscCall((*PetscHelpPrintf)(comm, "  %s%s <now %s : formerly %s>: %s (one of)", p, name + 1, newv, def, text));

  if (list) PetscHMapFuncForEach(list, name, func, PetscCall((*PetscHelpPrintf)(comm, " %s", name)));
  PetscCall((*PetscHelpPrintf)(comm, " (%s)\n", man));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscFunctionListDuplicate - Creates a new list from a given object list.

    Input Parameter:
.   fl   - pointer to list

    Output Parameter:
.   nl - the new list (should point to 0 to start, otherwise appends)

    Level: developer

.seealso: `PetscFunctionList`, `PetscFunctionListAdd()`, `PetscFlistDestroy()`
@*/
PetscErrorCode PetscFunctionListDuplicate(PetscFunctionList fl, PetscFunctionList *nl)
{
  PetscFunctionBegin;
  if (fl) {
    PetscHMapFunc dup_map;

    if (!*nl) {
      PetscInt n;

      PetscCall(PetscHMapFuncGetSize(fl->map, &n));
      PetscCall(PetscFunctionListCreate_Private(n, nl));
    }
    dup_map = (*nl)->map;
    PetscHMapFuncForEach(fl, name, func, PetscCall(PetscHMapFuncInsert_Private(dup_map, name, func)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
