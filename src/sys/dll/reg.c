
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

PetscErrorCode  PetscLoadDynamicLibrary(const char *name,PetscBool  *found)
{
  char           libs[PETSC_MAX_PATH_LEN],dlib[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(libs,"${PETSC_LIB_DIR}/libpetsc",sizeof(libs));CHKERRQ(ierr);
  ierr = PetscStrlcat(libs,name,sizeof(libs));CHKERRQ(ierr);
  ierr = PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,found);CHKERRQ(ierr);
  if (*found) {
    ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,dlib);CHKERRQ(ierr);
  } else {
    ierr = PetscStrncpy(libs,"${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc",sizeof(libs));CHKERRQ(ierr);
    ierr = PetscStrlcat(libs,name,sizeof(libs));CHKERRQ(ierr);
    ierr = PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,found);CHKERRQ(ierr);
    if (*found) {
      ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,dlib);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_THREADSAFETY)
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
static MPI_Comm PETSC_COMM_WORLD_INNER = 0,PETSC_COMM_SELF_INNER = 0;
#endif

/*
    PetscInitialize_DynamicLibraries - Adds the default dynamic link libraries to the
    search path.
*/
PETSC_INTERN PetscErrorCode PetscInitialize_DynamicLibraries(void)
{
  char           *libname[32];
  PetscErrorCode ierr;
  PetscInt       nmax,i;
#if defined(PETSC_USE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
  PetscBool      preload;
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscBool      PetscInitialized = PetscInitializeCalled;
#endif

  PetscFunctionBegin;
  nmax = 32;
  ierr = PetscOptionsGetStringArray(NULL,NULL,"-dll_prepend",libname,&nmax,NULL);CHKERRQ(ierr);
  for (i=0; i<nmax; i++) {
    ierr = PetscDLLibraryPrepend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    ierr = PetscFree(libname[i]);CHKERRQ(ierr);
  }

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES) || !defined(PETSC_USE_SHARED_LIBRARIES)
  /*
      This just initializes the most basic PETSc stuff.

    The classes, from PetscDraw to PetscTS, are initialized the first
    time an XXCreate() is called.
  */
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
#else
  preload = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-dynamic_library_preload",&preload,NULL);CHKERRQ(ierr);
  if (preload) {
    PetscBool found;
#if defined(PETSC_USE_SINGLE_LIBRARY)
    ierr = PetscLoadDynamicLibrary("",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc dynamic library \n You cannot move the dynamic libraries!");
#else
    ierr = PetscLoadDynamicLibrary("sys",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc dynamic library \n You cannot move the dynamic libraries!");
    ierr = PetscLoadDynamicLibrary("vec",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc Vec dynamic library \n You cannot move the dynamic libraries!");
    ierr = PetscLoadDynamicLibrary("mat",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc Mat dynamic library \n You cannot move the dynamic libraries!");
    ierr = PetscLoadDynamicLibrary("dm",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc DM dynamic library \n You cannot move the dynamic libraries!");
    ierr = PetscLoadDynamicLibrary("ksp",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc KSP dynamic library \n You cannot move the dynamic libraries!");
    ierr = PetscLoadDynamicLibrary("snes",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc SNES dynamic library \n You cannot move the dynamic libraries!");
    ierr = PetscLoadDynamicLibrary("ts",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc TS dynamic library \n You cannot move the dynamic libraries!");
#endif
  }
#endif
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
#if defined(PETSC_HAVE_BAMG)
  {
    PetscBool found;
    ierr = PetscLoadDynamicLibrary("bamg",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc BAMG dynamic library \n You cannot move the dynamic libraries!");
  }
#endif
#endif

  nmax = 32;
  ierr = PetscOptionsGetStringArray(NULL,NULL,"-dll_append",libname,&nmax,NULL);CHKERRQ(ierr);
  for (i=0; i<nmax; i++) {
    ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    ierr = PetscFree(libname[i]);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_THREADSAFETY)
  /* These must be done here because it is not safe for individual threads to call these initialize routines */
  ierr = AOInitializePackage();CHKERRQ(ierr);
  ierr = PetscSFInitializePackage();CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = CharacteristicInitializePackage();CHKERRQ(ierr);
#endif
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = VecInitializePackage();CHKERRQ(ierr);
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = DMInitializePackage();CHKERRQ(ierr);
  ierr = PCInitializePackage();CHKERRQ(ierr);
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = SNESInitializePackage();CHKERRQ(ierr);
  ierr = TSInitializePackage();CHKERRQ(ierr);
  ierr = PetscCommDuplicate(PETSC_COMM_SELF,&PETSC_COMM_SELF_INNER,NULL);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&PETSC_COMM_WORLD_INNER,NULL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  /* in Fortran, PetscInitializeCalled is set to PETSC_TRUE before PetscInitialize_DynamicLibraries() */
  /* in C, it is not the case, but the value is forced to PETSC_TRUE so that PetscRegisterFinalize() is called */
  PetscInitializeCalled = PETSC_TRUE;
  ierr = PetscElementalInitializePackage();CHKERRQ(ierr);
  PetscInitializeCalled = PetscInitialized;
#endif
  PetscFunctionReturn(0);
}

/*
     PetscFinalize_DynamicLibraries - Closes the opened dynamic libraries.
*/
PETSC_INTERN PetscErrorCode PetscFinalize_DynamicLibraries(void)
{
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-dll_view",&flg,NULL);CHKERRQ(ierr);
  if (flg) { ierr = PetscDLLibraryPrintPath(PetscDLLibrariesLoaded);CHKERRQ(ierr); }
  ierr = PetscDLLibraryClose(PetscDLLibrariesLoaded);CHKERRQ(ierr);

#if defined(PETSC_HAVE_THREADSAFETY)
  ierr = PetscCommDestroy(&PETSC_COMM_SELF_INNER);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&PETSC_COMM_WORLD_INNER);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!*fl) {
    ierr           = PetscNew(&entry);CHKERRQ(ierr);
    ierr           = PetscStrallocpy(name,&entry->name);CHKERRQ(ierr);
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

      ierr = PetscStrcmp(ne->name,name,&founddup);CHKERRQ(ierr);
      if (founddup) { /* found duplicate */
        ne->routine = fnc;
        PetscFunctionReturn(0);
      }
      if (ne->next) ne = ne->next;
      else break;
    }
    /* create new entry and add to end of list */
    ierr           = PetscNew(&entry);CHKERRQ(ierr);
    ierr           = PetscStrallocpy(name,&entry->name);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!*fl) PetscFunctionReturn(0);

  /*
       Remove this entry from the master DL list (if it is in it)
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
    ierr  = PetscFree(entry->name);CHKERRQ(ierr);
    ierr  = PetscFree(entry);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (tmp) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The following PetscFunctionLists were not destroyed\n");CHKERRQ(ierr);
  }
  while (tmp) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%s \n",tmp->name);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscBool         flg;

  PetscFunctionBegin;
  if (!name) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to find routine with null name");

  *r = NULL;
  while (entry) {
    ierr = PetscStrcmp(name,entry->name,&flg);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidPointer(list,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (!iascii) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ASCII viewer supported");

  while (list) {
    ierr = PetscViewerASCIIPrintf(viewer," %s\n",list->name);CHKERRQ(ierr);
    list = list->next;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscFunctionListGet - Gets an array the contains the entries in PetscFunctionList, this is used
         by help etc.

   Not Collective

   Input Parameter:
.  list   - list of types

   Output Parameter:
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
  PetscErrorCode    ierr;
  PetscInt          count = 0;
  PetscFunctionList klist = list;

  PetscFunctionBegin;
  while (list) {
    list = list->next;
    count++;
  }
  ierr  = PetscMalloc1(count+1,(char***)array);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  char           p[64];

  PetscFunctionBegin;
  if (!fd) fd = PETSC_STDOUT;

  ierr = PetscStrncpy(p,"-",sizeof(p));CHKERRQ(ierr);
  if (prefix) {ierr = PetscStrlcat(p,prefix,sizeof(p));CHKERRQ(ierr);}
  ierr = PetscFPrintf(comm,fd,"  %s%s <now %s : formerly %s>: %s (one of)",p,name+1,newv,def,text);CHKERRQ(ierr);

  while (list) {
    ierr = PetscFPrintf(comm,fd," %s",list->name);CHKERRQ(ierr);
    list = list->next;
  }
  ierr = PetscFPrintf(comm,fd," (%s)\n",man);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (fl) {
    ierr = PetscFunctionListAdd(nl,fl->name,fl->routine);CHKERRQ(ierr);
    fl   = fl->next;
  }
  PetscFunctionReturn(0);
}
