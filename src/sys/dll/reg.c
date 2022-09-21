
/*
    Provides a general mechanism to allow one to register new routines in
    dynamic libraries for many of the PETSc objects (including, e.g., KSP and PC).
*/
#include <petsc/private/petscimpl.h> /*I "petscsys.h" I*/
#include <petscviewer.h>

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
  PetscFunctionReturn(0);
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
#if defined(PETSC_HAVE_THREADSAFETY)
static MPI_Comm PETSC_COMM_WORLD_INNER = 0, PETSC_COMM_SELF_INNER = 0;
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

#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscCommDuplicate(PETSC_COMM_SELF, &PETSC_COMM_SELF_INNER, NULL));
  PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &PETSC_COMM_WORLD_INNER, NULL));
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  /* in Fortran, PetscInitializeCalled is set to PETSC_TRUE before PetscInitialize_DynamicLibraries() */
  /* in C, it is not the case, but the value is forced to PETSC_TRUE so that PetscRegisterFinalize() is called */
  PetscInitializeCalled = PETSC_TRUE;
  PetscCall(PetscElementalInitializePackage());
  PetscInitializeCalled = PetscInitialized;
#endif
  PetscFunctionReturn(0);
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

#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscCommDestroy(&PETSC_COMM_SELF_INNER));
  PetscCall(PetscCommDestroy(&PETSC_COMM_WORLD_INNER));
#endif

  PetscDLLibrariesLoaded = NULL;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
struct _n_PetscFunctionList {
  void (*routine)(void);       /* the routine */
  char             *name;      /* string to identify routine */
  PetscFunctionList next;      /* next pointer */
  PetscFunctionList next_list; /* used to maintain list of all lists for freeing */
};

/*
     Keep a linked list of PetscFunctionLists so that we can destroy all the left-over ones.
*/
static PetscFunctionList dlallhead = NULL;

static PetscErrorCode PetscFunctionListCreateNode_Private(PetscFunctionList *entry, const char name[], void (*func)(void))
{
  PetscFunctionBegin;
  PetscCall(PetscNew(entry));
  PetscCall(PetscStrallocpy(name, &(*entry)->name));
  (*entry)->routine = func;
  (*entry)->next    = NULL;
  PetscFunctionReturn(0);
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
PETSC_EXTERN PetscErrorCode PetscFunctionListAdd_Private(PetscFunctionList *fl, const char name[], void (*fnc)(void))
{
  PetscFunctionBegin;
  PetscValidPointer(fl, 1);
  if (name) PetscValidCharPointer(name, 2);
  if (fnc) PetscValidFunction(fnc, 3);
  if (*fl) {
    /* search list to see if it is already there */
    PetscFunctionList empty_node = NULL;
    PetscFunctionList ne         = *fl;

    while (1) {
      PetscBool founddup;

      PetscCall(PetscStrcmp(ne->name, name, &founddup));
      if (founddup) {
        /* found duplicate, clear it */
        ne->routine = fnc;
        if (!fnc) PetscCall(PetscFree(ne->name));
        PetscFunctionReturn(0);
      }

      if (!empty_node && !ne->routine && !ne->name) {
        /* save the empty node for later */
        empty_node = ne;
      }

      if (!ne->next) break; /* end of list */
      ne = ne->next;
    }

    /* there was an empty entry we could grab, fill it and bail */
    if (empty_node) {
      empty_node->routine = fnc;
      PetscCall(PetscStrallocpy(name, &empty_node->name));
    } else {
      /* create new entry at the end of list */
      PetscCall(PetscFunctionListCreateNode_Private(&ne->next, name, fnc));
    }
    PetscFunctionReturn(0);
  }

  /* we didn't have a list */
  PetscCall(PetscFunctionListCreateNode_Private(fl, name, fnc));
  if (PetscDefined(USE_DEBUG)) {
    const PetscFunctionList head = dlallhead;

    /* add this new list to list of all lists */
    dlallhead        = *fl;
    (*fl)->next_list = head;
  }
  PetscFunctionReturn(0);
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
  PetscFunctionList next, entry, tmp = dlallhead;

  PetscFunctionBegin;
  if (!*fl) PetscFunctionReturn(0);

  /*
       Remove this entry from the main DL list (if it is in it)
  */
  if (dlallhead == *fl) {
    if (dlallhead->next_list) dlallhead = dlallhead->next_list;
    else dlallhead = NULL;
  } else if (tmp) {
    while (tmp->next_list != *fl) {
      tmp = tmp->next_list;
      if (!tmp->next_list) break;
    }
    if (tmp->next_list) tmp->next_list = tmp->next_list->next_list;
  }

  /* free this list */
  entry = *fl;
  while (entry) {
    next = entry->next;
    PetscCall(PetscFree(entry->name));
    PetscCall(PetscFree(entry));
    entry = next;
  }
  *fl = NULL;
  PetscFunctionReturn(0);
}

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
  /* free the names and clear the routine but don't deallocate the node */
  while (fl) {
    PetscCall(PetscFree(fl->name));
    fl->routine = NULL;
    fl          = fl->next;
  }
  PetscFunctionReturn(0);
}

/*
   Print registered PetscFunctionLists
*/
PetscErrorCode PetscFunctionListPrintAll(void)
{
  PetscFunctionList tmp = dlallhead;

  PetscFunctionBegin;
  if (tmp) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Registered PetscFunctionLists\n", PetscGlobalRank));
  while (tmp) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d]   %s\n", PetscGlobalRank, tmp->name));
    tmp = tmp->next_list;
  }
  PetscFunctionReturn(0);
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
  while (fl) {
    PetscFunctionList next = fl->next;
    if (fl->routine) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] function name: %s\n", PetscGlobalRank, fl->name));
    fl = next;
  }
  PetscFunctionReturn(0);
}

/*MC
    PetscFunctionListFind - Find function registered under given name

    Synopsis:
    #include <petscsys.h>
    PetscErrorCode PetscFunctionListFind(PetscFunctionList flist,const char name[],void (**fptr)(void))

    Input Parameters:
+   flist   - pointer to list
-   name - name registered for the function

    Output Parameters:
.   fptr - the function pointer if name was found, else NULL

    Level: developer

.seealso: `PetscFunctionListAdd()`, `PetscFunctionList`, `PetscObjectQueryFunction()`
M*/
PETSC_EXTERN PetscErrorCode PetscFunctionListFind_Private(PetscFunctionList fl, const char name[], void (**r)(void))
{
  PetscFunctionList entry = fl;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 2);
  PetscValidPointer(r, 3);
  while (entry) {
    PetscBool flg;

    PetscCall(PetscStrcmp(name, entry->name, &flg));
    if (flg) {
      *r = entry->routine;
      PetscFunctionReturn(0);
    }
    entry = entry->next;
  }
  *r = NULL;
  PetscFunctionReturn(0);
}

/*@
   PetscFunctionListView - prints out contents of an PetscFunctionList

   Collective over viewer

   Input Parameters:
+  list - the list of functions
-  viewer - currently ignored

   Level: developer

.seealso: `PetscFunctionListAdd()`, `PetscFunctionListPrintTypes()`, `PetscFunctionList`
@*/
PetscErrorCode PetscFunctionListView(PetscFunctionList list, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidPointer(list, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCheck(iascii, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only ASCII viewer supported");

  while (list) {
    PetscCall(PetscViewerASCIIPrintf(viewer, " %s\n", list->name));
    list = list->next;
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscFunctionReturn(0);
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
  PetscInt          count = 0;
  PetscFunctionList klist = list;

  PetscFunctionBegin;
  while (list) {
    list = list->next;
    count++;
  }
  PetscCall(PetscMalloc1(count + 1, (char ***)array));
  count = 0;
  while (klist) {
    (*array)[count] = klist->name;
    klist           = klist->next;
    count++;
  }
  (*array)[count] = NULL;
  *n              = count + 1;
  PetscFunctionReturn(0);
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
  if (!fd) fd = PETSC_STDOUT;

  PetscCall(PetscStrncpy(p, "-", sizeof(p)));
  if (prefix) PetscCall(PetscStrlcat(p, prefix, sizeof(p)));
  PetscCall(PetscFPrintf(comm, fd, "  %s%s <now %s : formerly %s>: %s (one of)", p, name + 1, newv, def, text));

  while (list) {
    PetscCall(PetscFPrintf(comm, fd, " %s", list->name));
    list = list->next;
  }
  PetscCall(PetscFPrintf(comm, fd, " (%s)\n", man));
  PetscFunctionReturn(0);
}

/*@
    PetscFunctionListDuplicate - Creates a new list from a given object list.

    Input Parameters:
.   fl   - pointer to list

    Output Parameters:
.   nl - the new list (should point to 0 to start, otherwise appends)

    Level: developer

.seealso: `PetscFunctionList`, `PetscFunctionListAdd()`, `PetscFlistDestroy()`
@*/
PetscErrorCode PetscFunctionListDuplicate(PetscFunctionList fl, PetscFunctionList *nl)
{
  PetscFunctionBegin;
  while (fl) {
    PetscCall(PetscFunctionListAdd(nl, fl->name, fl->routine));
    fl = fl->next;
  }
  PetscFunctionReturn(0);
}
