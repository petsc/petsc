
/*
    Provides a general mechanism to allow one to register new routines in
    dynamic libraries for many of the PETSc objects (including, e.g., KSP and PC).
*/
#include <petsc/private/petscimpl.h>           /*I "petscsys.h" I*/
#include <petscviewer.h>

/*
    This is the default list used by PETSc with the PetscDLLibrary register routines
*/
PetscDLLibrary PetscDLLibrariesLoaded = NULL;

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)

static PetscErrorCode  PetscLoadDynamicLibrary(const char *name,PetscBool  *found)
{
  char           libs[PETSC_MAX_PATH_LEN],dlib[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  CHKERRQ(PetscStrncpy(libs,"${PETSC_LIB_DIR}/libpetsc",sizeof(libs)));
  CHKERRQ(PetscStrlcat(libs,name,sizeof(libs)));
  CHKERRQ(PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,found));
  if (*found) {
    CHKERRQ(PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,dlib));
  } else {
    CHKERRQ(PetscStrncpy(libs,"${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc",sizeof(libs)));
    CHKERRQ(PetscStrlcat(libs,name,sizeof(libs)));
    CHKERRQ(PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,found));
    if (*found) {
      CHKERRQ(PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,dlib));
    }
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
static MPI_Comm PETSC_COMM_WORLD_INNER = 0,PETSC_COMM_SELF_INNER = 0;
#endif

/*
    PetscInitialize_DynamicLibraries - Adds the default dynamic link libraries to the
    search path.
*/
PETSC_INTERN PetscErrorCode PetscInitialize_DynamicLibraries(void)
{
  char           *libname[32];
  PetscInt       nmax,i;
  PetscBool      preload = PETSC_FALSE;
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscBool      PetscInitialized = PetscInitializeCalled;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_THREADSAFETY)
  /* These must be all initialized here because it is not safe for individual threads to call these initialize routines */
  preload = PETSC_TRUE;
#endif

  nmax = 32;
  CHKERRQ(PetscOptionsGetStringArray(NULL,NULL,"-dll_prepend",libname,&nmax,NULL));
  for (i=0; i<nmax; i++) {
    CHKERRQ(PetscDLLibraryPrepend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,libname[i]));
    CHKERRQ(PetscFree(libname[i]));
  }

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-library_preload",&preload,NULL));
  if (!preload) {
    CHKERRQ(PetscSysInitializePackage());
  } else {
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
    PetscBool found;
#if defined(PETSC_USE_SINGLE_LIBRARY)
    CHKERRQ(PetscLoadDynamicLibrary("",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc dynamic library \n You cannot move the dynamic libraries!");
#else
    CHKERRQ(PetscLoadDynamicLibrary("sys",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc dynamic library \n You cannot move the dynamic libraries!");
    CHKERRQ(PetscLoadDynamicLibrary("vec",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc Vec dynamic library \n You cannot move the dynamic libraries!");
    CHKERRQ(PetscLoadDynamicLibrary("mat",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc Mat dynamic library \n You cannot move the dynamic libraries!");
    CHKERRQ(PetscLoadDynamicLibrary("dm",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc DM dynamic library \n You cannot move the dynamic libraries!");
    CHKERRQ(PetscLoadDynamicLibrary("ksp",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc KSP dynamic library \n You cannot move the dynamic libraries!");
    CHKERRQ(PetscLoadDynamicLibrary("snes",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc SNES dynamic library \n You cannot move the dynamic libraries!");
    CHKERRQ(PetscLoadDynamicLibrary("ts",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc TS dynamic library \n You cannot move the dynamic libraries!");
    CHKERRQ(PetscLoadDynamicLibrary("tao",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate Tao dynamic library \n You cannot move the dynamic libraries!");
#endif
#else /* defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES) */
#if defined(PETSC_USE_SINGLE_LIBRARY)
  CHKERRQ(AOInitializePackage());
  CHKERRQ(PetscSFInitializePackage());
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(CharacteristicInitializePackage());
#endif
  CHKERRQ(ISInitializePackage());
  CHKERRQ(VecInitializePackage());
  CHKERRQ(MatInitializePackage());
  CHKERRQ(DMInitializePackage());
  CHKERRQ(PCInitializePackage());
  CHKERRQ(KSPInitializePackage());
  CHKERRQ(SNESInitializePackage());
  CHKERRQ(TSInitializePackage());
  CHKERRQ(TaoInitializePackage());
#else
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Cannot use -library_preload with multiple static PETSc libraries");
#endif
#endif /* defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES) */
  }

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES) && defined(PETSC_HAVE_BAMG)
  {
    PetscBool found;
    CHKERRQ(PetscLoadDynamicLibrary("bamg",&found));
    PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc BAMG dynamic library \n You cannot move the dynamic libraries!");
  }
#endif

  nmax = 32;
  CHKERRQ(PetscOptionsGetStringArray(NULL,NULL,"-dll_append",libname,&nmax,NULL));
  for (i=0; i<nmax; i++) {
    CHKERRQ(PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,libname[i]));
    CHKERRQ(PetscFree(libname[i]));
  }

#if defined(PETSC_HAVE_THREADSAFETY)
  CHKERRQ(PetscCommDuplicate(PETSC_COMM_SELF,&PETSC_COMM_SELF_INNER,NULL));
  CHKERRQ(PetscCommDuplicate(PETSC_COMM_WORLD,&PETSC_COMM_WORLD_INNER,NULL));
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  /* in Fortran, PetscInitializeCalled is set to PETSC_TRUE before PetscInitialize_DynamicLibraries() */
  /* in C, it is not the case, but the value is forced to PETSC_TRUE so that PetscRegisterFinalize() is called */
  PetscInitializeCalled = PETSC_TRUE;
  CHKERRQ(PetscElementalInitializePackage());
  PetscInitializeCalled = PetscInitialized;
#endif
  PetscFunctionReturn(0);
}

/*
     PetscFinalize_DynamicLibraries - Closes the opened dynamic libraries.
*/
PETSC_INTERN PetscErrorCode PetscFinalize_DynamicLibraries(void)
{
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-dll_view",&flg,NULL));
  if (flg) CHKERRQ(PetscDLLibraryPrintPath(PetscDLLibrariesLoaded));
  CHKERRQ(PetscDLLibraryClose(PetscDLLibrariesLoaded));

#if defined(PETSC_HAVE_THREADSAFETY)
  CHKERRQ(PetscCommDestroy(&PETSC_COMM_SELF_INNER));
  CHKERRQ(PetscCommDestroy(&PETSC_COMM_WORLD_INNER));
#endif

  PetscDLLibrariesLoaded = NULL;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
struct _n_PetscFunctionList {
  void              (*routine)(void);    /* the routine */
  char              *name;               /* string to identify routine */
  PetscFunctionList next;                /* next pointer */
  PetscFunctionList next_list;           /* used to maintain list of all lists for freeing */
};

/*
     Keep a linked list of PetscFunctionLists so that we can destroy all the left-over ones.
*/
static PetscFunctionList dlallhead = NULL;

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
   component (e.g., SNES) should generally call the registration routine
   for that particular component (e.g., SNESRegister()) instead of
   calling PetscFunctionListAdd() directly.

    Level: developer

.seealso: PetscFunctionListDestroy(), SNESRegister(), KSPRegister(),
          PCRegister(), TSRegister(), PetscFunctionList, PetscObjectComposeFunction()
M*/
PETSC_EXTERN PetscErrorCode PetscFunctionListAdd_Private(PetscFunctionList *fl,const char name[],void (*fnc)(void))
{
  PetscFunctionList entry,ne;

  PetscFunctionBegin;
  if (!*fl) {
    CHKERRQ(PetscNew(&entry));
    CHKERRQ(PetscStrallocpy(name,&entry->name));
    entry->routine = fnc;
    entry->next    = NULL;
    *fl            = entry;

    if (PetscDefined(USE_DEBUG)) {
      /* add this new list to list of all lists */
      if (!dlallhead) {
        dlallhead        = *fl;
        (*fl)->next_list = NULL;
      } else {
        ne               = dlallhead;
        dlallhead        = *fl;
        (*fl)->next_list = ne;
      }
    }

  } else {
    /* search list to see if it is already there */
    ne = *fl;
    while (ne) {
      PetscBool founddup;

      CHKERRQ(PetscStrcmp(ne->name,name,&founddup));
      if (founddup) { /* found duplicate */
        ne->routine = fnc;
        PetscFunctionReturn(0);
      }
      if (ne->next) ne = ne->next;
      else break;
    }
    /* create new entry and add to end of list */
    CHKERRQ(PetscNew(&entry));
    CHKERRQ(PetscStrallocpy(name,&entry->name));
    entry->routine = fnc;
    entry->next    = NULL;
    ne->next       = entry;
  }
  PetscFunctionReturn(0);
}

/*@
    PetscFunctionListDestroy - Destroys a list of registered routines.

    Input Parameter:
.   fl  - pointer to list

    Level: developer

.seealso: PetscFunctionListAdd(), PetscFunctionList
@*/
PetscErrorCode  PetscFunctionListDestroy(PetscFunctionList *fl)
{
  PetscFunctionList next,entry,tmp = dlallhead;

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
    next  = entry->next;
    CHKERRQ(PetscFree(entry->name));
    CHKERRQ(PetscFree(entry));
    entry = next;
  }
  *fl = NULL;
  PetscFunctionReturn(0);
}

/*
   Print any PetscFunctionLists that have not be destroyed
*/
PetscErrorCode  PetscFunctionListPrintAll(void)
{
  PetscFunctionList tmp = dlallhead;

  PetscFunctionBegin;
  if (tmp) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The following PetscFunctionLists were not destroyed\n"));
  }
  while (tmp) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s \n",tmp->name));
    tmp = tmp->next_list;
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

.seealso: PetscFunctionListAdd(), PetscFunctionList, PetscObjectQueryFunction()
M*/
PETSC_EXTERN PetscErrorCode PetscFunctionListFind_Private(PetscFunctionList fl,const char name[],void (**r)(void))
{
  PetscFunctionList entry = fl;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCheck(name,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to find routine with null name");

  *r = NULL;
  while (entry) {
    CHKERRQ(PetscStrcmp(name,entry->name,&flg));
    if (flg) {
      *r   = entry->routine;
      PetscFunctionReturn(0);
    }
    entry = entry->next;
  }
  PetscFunctionReturn(0);
}

/*@
   PetscFunctionListView - prints out contents of an PetscFunctionList

   Collective over MPI_Comm

   Input Parameters:
+  list - the list of functions
-  viewer - currently ignored

   Level: developer

.seealso: PetscFunctionListAdd(), PetscFunctionListPrintTypes(), PetscFunctionList
@*/
PetscErrorCode  PetscFunctionListView(PetscFunctionList list,PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidPointer(list,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCheck(iascii,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ASCII viewer supported");

  while (list) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," %s\n",list->name));
    list = list->next;
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n"));
  PetscFunctionReturn(0);
}

/*@C
   PetscFunctionListGet - Gets an array the contains the entries in PetscFunctionList, this is used
         by help etc.

   Not Collective

   Input Parameter:
.  list   - list of types

   Output Parameters:
+  array - array of names
-  n - length of array

   Notes:
       This allocates the array so that must be freed. BUT the individual entries are
    not copied so should not be freed.

   Level: developer

.seealso: PetscFunctionListAdd(), PetscFunctionList
@*/
PetscErrorCode  PetscFunctionListGet(PetscFunctionList list,const char ***array,int *n)
{
  PetscInt          count = 0;
  PetscFunctionList klist = list;

  PetscFunctionBegin;
  while (list) {
    list = list->next;
    count++;
  }
  CHKERRQ(PetscMalloc1(count+1,(char***)array));
  count = 0;
  while (klist) {
    (*array)[count] = klist->name;
    klist           = klist->next;
    count++;
  }
  (*array)[count] = NULL;
  *n              = count+1;
  PetscFunctionReturn(0);
}

/*@C
   PetscFunctionListPrintTypes - Prints the methods available.

   Collective over MPI_Comm

   Input Parameters:
+  comm   - the communicator (usually MPI_COMM_WORLD)
.  fd     - file to print to, usually stdout
.  prefix - prefix to prepend to name (optional)
.  name   - option string (for example, "-ksp_type")
.  text - short description of the object (for example, "Krylov solvers")
.  man - name of manual page that discusses the object (for example, "KSPCreate")
.  list   - list of types
.  def - default (current) value
-  newv - new value

   Level: developer

.seealso: PetscFunctionListAdd(), PetscFunctionList
@*/
PetscErrorCode  PetscFunctionListPrintTypes(MPI_Comm comm,FILE *fd,const char prefix[],const char name[],const char text[],const char man[],PetscFunctionList list,const char def[],const char newv[])
{
  char           p[64];

  PetscFunctionBegin;
  if (!fd) fd = PETSC_STDOUT;

  CHKERRQ(PetscStrncpy(p,"-",sizeof(p)));
  if (prefix) CHKERRQ(PetscStrlcat(p,prefix,sizeof(p)));
  CHKERRQ(PetscFPrintf(comm,fd,"  %s%s <now %s : formerly %s>: %s (one of)",p,name+1,newv,def,text));

  while (list) {
    CHKERRQ(PetscFPrintf(comm,fd," %s",list->name));
    list = list->next;
  }
  CHKERRQ(PetscFPrintf(comm,fd," (%s)\n",man));
  PetscFunctionReturn(0);
}

/*@
    PetscFunctionListDuplicate - Creates a new list from a given object list.

    Input Parameters:
.   fl   - pointer to list

    Output Parameters:
.   nl - the new list (should point to 0 to start, otherwise appends)

    Level: developer

.seealso: PetscFunctionList, PetscFunctionListAdd(), PetscFlistDestroy()

@*/
PetscErrorCode  PetscFunctionListDuplicate(PetscFunctionList fl,PetscFunctionList *nl)
{
  PetscFunctionBegin;
  while (fl) {
    CHKERRQ(PetscFunctionListAdd(nl,fl->name,fl->routine));
    fl   = fl->next;
  }
  PetscFunctionReturn(0);
}
