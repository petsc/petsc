
/*
    Provides a general mechanism to allow one to register new routines in
    dynamic libraries for many of the PETSc objects (including, e.g., KSP and PC).
*/
#include <petscsys.h>           /*I "petscsys.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscFListGetPathAndFunction"
PetscErrorCode  PetscFListGetPathAndFunction(const char name[],char *path[],char *function[])
{
  PetscErrorCode ierr;
  char           work[PETSC_MAX_PATH_LEN],*lfunction;

  PetscFunctionBegin;
  ierr = PetscStrncpy(work,name,sizeof(work));CHKERRQ(ierr);
  work[sizeof(work) - 1] = 0;
  ierr = PetscStrchr(work,':',&lfunction);CHKERRQ(ierr);
  if (lfunction != work && lfunction && lfunction[1] != ':') {
    lfunction[0] = 0;
    ierr = PetscStrallocpy(work,path);CHKERRQ(ierr);
    ierr = PetscStrallocpy(lfunction+1,function);CHKERRQ(ierr);
  } else {
    *path = 0;
    ierr = PetscStrallocpy(name,function);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    This is the default list used by PETSc with the PetscDLLibrary register routines
*/
PetscDLLibrary PetscDLLibrariesLoaded = 0;

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)

#undef __FUNCT__
#define __FUNCT__ "PetscLoadDynamicLibrary"
static PetscErrorCode  PetscLoadDynamicLibrary(const char *name,PetscBool  *found)
{
  char           libs[PETSC_MAX_PATH_LEN],dlib[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(libs,"${PETSC_LIB_DIR}/libpetsc");CHKERRQ(ierr);
  ierr = PetscStrcat(libs,name);CHKERRQ(ierr);
  ierr = PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,found);CHKERRQ(ierr);
  if (*found) {
    ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,dlib);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcpy(libs,"${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc");CHKERRQ(ierr);
    ierr = PetscStrcat(libs,name);CHKERRQ(ierr);
    ierr = PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,found);CHKERRQ(ierr);
    if (*found) {
      ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,dlib);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "PetscInitialize_DynamicLibraries"
/*
    PetscInitialize_DynamicLibraries - Adds the default dynamic link libraries to the
    search path.
*/
PetscErrorCode  PetscInitialize_DynamicLibraries(void)
{
  char           *libname[32];
  PetscErrorCode ierr;
  PetscInt       nmax,i;
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  PetscBool      found;
#endif

  PetscFunctionBegin;
  nmax = 32;
  ierr = PetscOptionsGetStringArray(PETSC_NULL,"-dll_prepend",libname,&nmax,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<nmax; i++) {
    ierr = PetscDLLibraryPrepend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    ierr = PetscFree(libname[i]);CHKERRQ(ierr);
  }

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  /*
      This just initializes the most basic PETSc stuff.

    The classes, from PetscDraw to PetscTS, are initialized the first
    time an XXCreate() is called.
  */
  ierr = PetscSysInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#else
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
  ierr = PetscLoadDynamicLibrary("characteristic",&found);CHKERRQ(ierr);
  if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc Characteristic dynamic library \n You cannot move the dynamic libraries!");
  ierr = PetscLoadDynamicLibrary("ksp",&found);CHKERRQ(ierr);
  if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc KSP dynamic library \n You cannot move the dynamic libraries!");
  ierr = PetscLoadDynamicLibrary("snes",&found);CHKERRQ(ierr);
  if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc SNES dynamic library \n You cannot move the dynamic libraries!");
  ierr = PetscLoadDynamicLibrary("ts",&found);CHKERRQ(ierr);
  if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate PETSc TS dynamic library \n You cannot move the dynamic libraries!");
#endif

  ierr = PetscLoadDynamicLibrary("mesh",&found);CHKERRQ(ierr);
  ierr = PetscLoadDynamicLibrary("contrib",&found);CHKERRQ(ierr);
#endif

  nmax = 32;
  ierr = PetscOptionsGetStringArray(PETSC_NULL,"-dll_append",libname,&nmax,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<nmax; i++) {
    ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    ierr = PetscFree(libname[i]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFinalize_DynamicLibraries"
/*
     PetscFinalize_DynamicLibraries - Closes the opened dynamic libraries.
*/
PetscErrorCode PetscFinalize_DynamicLibraries(void)
{
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-dll_view",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) { ierr = PetscDLLibraryPrintPath(PetscDLLibrariesLoaded);CHKERRQ(ierr); }
  ierr = PetscDLLibraryClose(PetscDLLibrariesLoaded);CHKERRQ(ierr);
  PetscDLLibrariesLoaded = 0;
  PetscFunctionReturn(0);
}



/* ------------------------------------------------------------------------------*/
struct _n_PetscFList {
  void        (*routine)(void);   /* the routine */
  char        *path;              /* path of link library containing routine */
  char        *name;              /* string to identify routine */
  char        *rname;             /* routine name in dynamic library */
  PetscFList  next;               /* next pointer */
  PetscFList  next_list;          /* used to maintain list of all lists for freeing */
};

/*
     Keep a linked list of PetscFLists so that we can destroy all the left-over ones.
*/
static PetscFList   dlallhead = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscFListAdd"
/*@C
   PetscFListAdd - Given a routine and a string id, saves that routine in the
   specified registry.

     Not Collective

   Input Parameters:
+  fl    - pointer registry
.  name  - string to identify routine
.  rname - routine name in dynamic library
-  fnc   - function pointer (optional if using dynamic libraries)

   Notes:
   To remove a registered routine, pass in a PETSC_NULL rname and fnc().

   Users who wish to register new classes for use by a particular PETSc
   component (e.g., SNES) should generally call the registration routine
   for that particular component (e.g., SNESRegisterDynamic()) instead of
   calling PetscFListAdd() directly.

   ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, or ${any environmental variable}
  occuring in pathname will be replaced with appropriate values.

   Level: developer

.seealso: PetscFListDestroy(), SNESRegisterDynamic(), KSPRegisterDynamic(),
          PCRegisterDynamic(), TSRegisterDynamic(), PetscFList
@*/
PetscErrorCode  PetscFListAdd(PetscFList *fl,const char name[],const char rname[],void (*fnc)(void))
{
  PetscFList     entry,ne;
  PetscErrorCode ierr;
  char           *fpath,*fname;

  PetscFunctionBegin;
  if (!*fl) {
    ierr           = PetscNew(struct _n_PetscFList,&entry);CHKERRQ(ierr);
    ierr           = PetscStrallocpy(name,&entry->name);CHKERRQ(ierr);
    ierr           = PetscFListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
    entry->path    = fpath;
    entry->rname   = fname;
    entry->routine = fnc;
    entry->next    = 0;
    *fl = entry;

    /* add this new list to list of all lists */
    if (!dlallhead) {
      dlallhead        = *fl;
      (*fl)->next_list = 0;
    } else {
      ne               = dlallhead;
      dlallhead        = *fl;
      (*fl)->next_list = ne;
    }
  } else {
    /* search list to see if it is already there */
    ne = *fl;
    while (ne) {
      PetscBool  founddup;

      ierr = PetscStrcmp(ne->name,name,&founddup);CHKERRQ(ierr);
      if (founddup) { /* found duplicate */
        ierr = PetscFListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
        ierr = PetscFree(ne->path);CHKERRQ(ierr);
        ierr = PetscFree(ne->rname);CHKERRQ(ierr);
        ne->path    = fpath;
        ne->rname   = fname;
        ne->routine = fnc;
        PetscFunctionReturn(0);
      }
      if (ne->next) ne = ne->next; else break;
    }
    /* create new entry and add to end of list */
    ierr           = PetscNew(struct _n_PetscFList,&entry);CHKERRQ(ierr);
    ierr           = PetscStrallocpy(name,&entry->name);CHKERRQ(ierr);
    ierr           = PetscFListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
    entry->path    = fpath;
    entry->rname   = fname;
    entry->routine = fnc;
    entry->next    = 0;
    ne->next       = entry;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFListDestroy"
/*@
    PetscFListDestroy - Destroys a list of registered routines.

    Input Parameter:
.   fl  - pointer to list

    Level: developer

.seealso: PetscFListAddDynamic(), PetscFList
@*/
PetscErrorCode  PetscFListDestroy(PetscFList *fl)
{
  PetscFList     next,entry,tmp = dlallhead;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*fl) PetscFunctionReturn(0);
  if (!dlallhead) PetscFunctionReturn(0);

  /*
       Remove this entry from the master DL list (if it is in it)
  */
  if (dlallhead == *fl) {
    if (dlallhead->next_list) {
      dlallhead = dlallhead->next_list;
    } else {
      dlallhead = 0;
    }
  } else {
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
    ierr = PetscFree(entry->path);CHKERRQ(ierr);
    ierr = PetscFree(entry->name);CHKERRQ(ierr);
    ierr = PetscFree(entry->rname);CHKERRQ(ierr);
    ierr = PetscFree(entry);CHKERRQ(ierr);
    entry = next;
  }
  *fl = 0;
  PetscFunctionReturn(0);
}

/*
   Destroys all the function lists that anyone has every registered, such as KSPList, VecList, etc.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscFListDestroyAll"
PetscErrorCode  PetscFListDestroyAll(void)
{
  PetscFList     tmp2,tmp1 = dlallhead;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (tmp1) {
    tmp2 = tmp1->next_list;
    ierr = PetscFListDestroy(&tmp1);CHKERRQ(ierr);
    tmp1 = tmp2;
  }
  dlallhead = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFListFind"
/*@C
    PetscFListFind - Given a name, finds the matching routine.

    Input Parameters:
+   fl   - pointer to list
.   comm - processors looking for routine
.   name - name string
-   searchlibraries - if not found in the list then search the dynamic libraries and executable for the symbol

    Output Parameters:
.   r - the routine

    Level: developer

.seealso: PetscFListAddDynamic(), PetscFList
@*/
PetscErrorCode  PetscFListFind(PetscFList fl,MPI_Comm comm,const char name[],PetscBool searchlibraries,void (**r)(void))
{
  PetscFList     entry = fl;
  PetscErrorCode ierr;
  char           *function,*path;
  PetscBool      flg,f1,f2,f3;
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  char           *newpath;
#endif

  PetscFunctionBegin;
  if (!name) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Trying to find routine with null name");

  *r = 0;
  ierr = PetscFListGetPathAndFunction(name,&path,&function);CHKERRQ(ierr);

  /*
        If path then append it to search libraries
  */
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  if (path) {
    ierr = PetscDLLibraryAppend(comm,&PetscDLLibrariesLoaded,path);CHKERRQ(ierr);
  }
#endif

  while (entry) {
    flg = PETSC_FALSE;
    if (path && entry->path) {
      ierr = PetscStrcmp(path,entry->path,&f1);CHKERRQ(ierr);
      ierr = PetscStrcmp(function,entry->rname,&f2);CHKERRQ(ierr);
      ierr = PetscStrcmp(function,entry->name,&f3);CHKERRQ(ierr);
      flg =  (PetscBool) ((f1 && f2) || (f1 && f3));
    } else if (!path) {
      ierr = PetscStrcmp(function,entry->name,&f1);CHKERRQ(ierr);
      ierr = PetscStrcmp(function,entry->rname,&f2);CHKERRQ(ierr);
      flg =  (PetscBool) (f1 || f2);
    } else {
      ierr = PetscStrcmp(function,entry->name,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscFree(function);CHKERRQ(ierr);
        ierr = PetscStrallocpy(entry->rname,&function);CHKERRQ(ierr);
      } else {
        ierr = PetscStrcmp(function,entry->rname,&flg);CHKERRQ(ierr);
      }
    }

    if (flg) {
      if (entry->routine) {
        *r   = entry->routine;
        ierr = PetscFree(path);CHKERRQ(ierr);
        ierr = PetscFree(function);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      if (!(entry->rname && entry->rname[0])) { /* The entry has been cleared */
        ierr = PetscFree(function);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      if ((path && entry->path && f3) || (!path && f1)) { /* convert name of function (alias) to actual function name */
        ierr = PetscFree(function);CHKERRQ(ierr);
        ierr = PetscStrallocpy(entry->rname,&function);CHKERRQ(ierr);
      }

      /* it is not yet in memory so load from dynamic library */
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
      newpath = path;
      if (!path) newpath = entry->path;
      ierr = PetscDLLibrarySym(comm,&PetscDLLibrariesLoaded,newpath,entry->rname,(void **)r);CHKERRQ(ierr);
      if (*r) {
        entry->routine = *r;
        ierr = PetscFree(path);CHKERRQ(ierr);
        ierr = PetscFree(function);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
#endif
    }
    entry = entry->next;
  }

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  if (searchlibraries) {
    /* Function never registered; try for it anyway */
    ierr = PetscDLLibrarySym(comm,&PetscDLLibrariesLoaded,path,function,(void **)r);CHKERRQ(ierr);
    ierr = PetscFree(path);CHKERRQ(ierr);
    if (*r) {
      ierr = PetscFListAdd(&fl,name,name,*r);CHKERRQ(ierr);
    }
  }
#endif
  ierr = PetscFree(function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFListView"
/*@
   PetscFListView - prints out contents of an PetscFList

   Collective over MPI_Comm

   Input Parameters:
+  list - the list of functions
-  viewer - currently ignored

   Level: developer

.seealso: PetscFListAddDynamic(), PetscFListPrintTypes(), PetscFList
@*/
PetscErrorCode  PetscFListView(PetscFList list,PetscViewer viewer)
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
    if (list->path) {
      ierr = PetscViewerASCIIPrintf(viewer," %s %s %s\n",list->path,list->name,list->rname);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer," %s %s\n",list->name,list->rname);CHKERRQ(ierr);
    }
    list = list->next;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFListGet"
/*@C
   PetscFListGet - Gets an array the contains the entries in PetscFList, this is used
         by help etc.

   Collective over MPI_Comm

   Input Parameter:
.  list   - list of types

   Output Parameter:
+  array - array of names
-  n - length of array

   Notes:
       This allocates the array so that must be freed. BUT the individual entries are
    not copied so should not be freed.

   Level: developer

.seealso: PetscFListAddDynamic(), PetscFList
@*/
PetscErrorCode  PetscFListGet(PetscFList list,const char ***array,int *n)
{
  PetscErrorCode ierr;
  PetscInt       count = 0;
  PetscFList     klist = list;

  PetscFunctionBegin;
  while (list) {
    list = list->next;
    count++;
  }
  ierr  = PetscMalloc((count+1)*sizeof(char *),array);CHKERRQ(ierr);
  count = 0;
  while (klist) {
    (*array)[count] = klist->name;
    klist = klist->next;
    count++;
  }
  (*array)[count] = 0;
  *n = count+1;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscFListPrintTypes"
/*@C
   PetscFListPrintTypes - Prints the methods available.

   Collective over MPI_Comm

   Input Parameters:
+  comm   - the communicator (usually MPI_COMM_WORLD)
.  fd     - file to print to, usually stdout
.  prefix - prefix to prepend to name (optional)
.  name   - option string (for example, "-ksp_type")
.  text - short description of the object (for example, "Krylov solvers")
.  man - name of manual page that discusses the object (for example, "KSPCreate")
.  list   - list of types
-  def - default (current) value

   Level: developer

.seealso: PetscFListAddDynamic(), PetscFList
@*/
PetscErrorCode  PetscFListPrintTypes(MPI_Comm comm,FILE *fd,const char prefix[],const char name[],const char text[],const char man[],PetscFList list,const char def[])
{
  PetscErrorCode ierr;
  PetscInt       count = 0;
  char           p[64];

  PetscFunctionBegin;
  if (!fd) fd = PETSC_STDOUT;

  ierr = PetscStrcpy(p,"-");CHKERRQ(ierr);
  if (prefix) {ierr = PetscStrcat(p,prefix);CHKERRQ(ierr);}
  ierr = PetscFPrintf(comm,fd,"  %s%s <%s>: %s (one of)",p,name+1,def,text);CHKERRQ(ierr);

  while (list) {
    ierr = PetscFPrintf(comm,fd," %s",list->name);CHKERRQ(ierr);
    list = list->next;
    count++;
    if (count == 8) {ierr = PetscFPrintf(comm,fd,"\n     ");CHKERRQ(ierr);}
  }
  ierr = PetscFPrintf(comm,fd," (%s)\n",man);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFListDuplicate"
/*@
    PetscFListDuplicate - Creates a new list from a given object list.

    Input Parameters:
.   fl   - pointer to list

    Output Parameters:
.   nl - the new list (should point to 0 to start, otherwise appends)

    Level: developer

.seealso: PetscFList, PetscFListAdd(), PetscFlistDestroy()

@*/
PetscErrorCode  PetscFListDuplicate(PetscFList fl,PetscFList *nl)
{
  PetscErrorCode ierr;
  char           path[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  while (fl) {
    /* this is silly, rebuild the complete pathname */
    if (fl->path) {
      ierr = PetscStrcpy(path,fl->path);CHKERRQ(ierr);
      ierr = PetscStrcat(path,":");CHKERRQ(ierr);
      ierr = PetscStrcat(path,fl->name);CHKERRQ(ierr);
    } else {
      ierr = PetscStrcpy(path,fl->name);CHKERRQ(ierr);
    }
    ierr = PetscFListAdd(nl,path,fl->rname,fl->routine);CHKERRQ(ierr);
    fl   = fl->next;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscFListConcat"
/*
    PetscFListConcat - joins name of a libary, and the path where it is located
    into a single string.

    Input Parameters:
.   path   - path to the library name.
.   name   - name of the library

    Output Parameters:
.   fullname - the name that is the union of the path and the library name,
               delimited by a semicolon, i.e., path:name

    Notes:
    If the path is NULL, assumes that the name, specified also includes
    the path as path:name

*/
PetscErrorCode  PetscFListConcat(const char path[],const char name[],char fullname[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (path) {
    ierr = PetscStrcpy(fullname,path);CHKERRQ(ierr);
    ierr = PetscStrcat(fullname,":");CHKERRQ(ierr);
    ierr = PetscStrcat(fullname,name);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcpy(fullname,name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



/* ------------------------------------------------------------------------------*/
struct _n_PetscOpFList {
  char                 *op;                /* op name */
  PetscInt             numArgs;            /* number of arguments to the operation */
  char                 **argTypes;         /* list of argument types */
  PetscVoidFunction    routine;            /* the routine */
  char                 *url;               /* url naming the link library and the routine */
  char                 *path;              /* path of link library containing routine */
  char                 *name;              /* routine name in dynamic library */
  PetscOpFList         next;              /* next pointer */
  PetscOpFList         next_list;         /* used to maintain list of all lists for freeing */
};

/*
     Keep a linked list of PetscOfFLists so that we can destroy all the left-over ones.
*/
static PetscOpFList   opallhead = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscOpFListAdd"
/*@C
   PetscOpFListAdd - Given a routine and a string id, saves that routine in the
   specified registry.

   Formally collective on comm.

   Input Parameters:
+  comm     - processors adding the op
.  fl       - list of known ops
.  url      - routine locator  (optional, if not using dynamic libraries and a nonempty fnc)
.  fnc      - function pointer (optional, if using dynamic libraries and a nonempty url)
.  op       - operation name
.  numArgs  - number of op arguments
-  argTypes - list of argument type names (const char*)

   Notes:
   To remove a registered routine, pass in a PETSC_NULL url and fnc().

   url can be of the form  [/path/libname[.so.1.0]:]functionname[()]  where items in [] denote optional

   ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, or ${any environment variable}
   occuring in url will be replaced with appropriate values.

   Level: developer

.seealso: PetscOpFListDestroy(),PetscOpFList,  PetscFListAdd(), PetscFList
@*/
PetscErrorCode  PetscOpFListAdd(MPI_Comm comm, PetscOpFList *fl,const char url[],PetscVoidFunction fnc,const char op[], PetscInt numArgs, char* argTypes[])
{
  PetscOpFList   entry,e,ne;
  PetscErrorCode ierr;
  char           *fpath,*fname;
  PetscInt       i;

  PetscFunctionBegin;
  if (!*fl) {
    ierr           = PetscNew(struct _n_PetscOpFList,&entry); CHKERRQ(ierr);
    ierr           = PetscStrallocpy(op,&entry->op);          CHKERRQ(ierr);
    ierr           = PetscStrallocpy(url,&(entry->url));      CHKERRQ(ierr);
    ierr           = PetscFListGetPathAndFunction(url,&fpath,&fname);CHKERRQ(ierr);
    entry->path    = fpath;
    entry->name    = fname;
    entry->routine = fnc;
    entry->numArgs = numArgs;
    if (numArgs) {
      ierr = PetscMalloc(sizeof(char*)*numArgs, &(entry->argTypes));    CHKERRQ(ierr);
      for (i = 0; i < numArgs; ++i) {
        ierr = PetscStrallocpy(argTypes[i], &(entry->argTypes[i]));         CHKERRQ(ierr);
      }
    }
    entry->next    = 0;
    *fl = entry;

    /* add this new list to list of all lists */
    if (!opallhead) {
      opallhead       = *fl;
      (*fl)->next_list = 0;
    } else {
      ne               = opallhead;
      opallhead        = *fl;
      (*fl)->next_list = ne;
    }
  } else {
    /* search list to see if it is already there */
    e  = PETSC_NULL;
    ne = *fl;
    while (ne) {
      PetscBool  match;
      ierr = PetscStrcmp(ne->op,op,&match);CHKERRQ(ierr);
      if (!match) goto next;
      if (numArgs == ne->numArgs)
        match = PETSC_TRUE;
      else
        match = PETSC_FALSE;
      if (!match) goto next;
      if (numArgs) {
        for (i = 0; i < numArgs; ++i) {
          ierr = PetscStrcmp(argTypes[i], ne->argTypes[i], &match);  CHKERRQ(ierr);
          if (!match) goto next;
        }
      }
      if (!url && !fnc) {
        /* remove this record */
        if (e) e->next = ne->next;
        ierr = PetscFree(ne->op);    CHKERRQ(ierr);
        ierr = PetscFree(ne->url);   CHKERRQ(ierr);
        ierr = PetscFree(ne->path);  CHKERRQ(ierr);
        ierr = PetscFree(ne->name);  CHKERRQ(ierr);
        if (numArgs) {
          for (i = 0; i < numArgs; ++i) {
            ierr = PetscFree(ne->argTypes[i]);  CHKERRQ(ierr);
          }
          ierr = PetscFree(ne->argTypes);       CHKERRQ(ierr);
        }
        ierr = PetscFree(ne);                   CHKERRQ(ierr);
      }
      else {
        /* Replace url, fpath, fname and fnc. */
        ierr = PetscStrallocpy(url, &(ne->url)); CHKERRQ(ierr);
        ierr = PetscFListGetPathAndFunction(url,&fpath,&fname);CHKERRQ(ierr);
        ierr = PetscFree(ne->path);CHKERRQ(ierr);
        ierr = PetscFree(ne->name);CHKERRQ(ierr);
        ne->path    = fpath;
        ne->name    = fname;
        ne->routine = fnc;
      }
      PetscFunctionReturn(0);
      next: {e = ne; ne = ne->next;}
    }
    /* create new entry and add to end of list */
    ierr           = PetscNew(struct _n_PetscOpFList,&entry);           CHKERRQ(ierr);
    ierr           = PetscStrallocpy(op,&entry->op);                    CHKERRQ(ierr);
    entry->numArgs = numArgs;
    if (numArgs) {
      ierr = PetscMalloc(sizeof(char*)*numArgs, &(entry->argTypes));    CHKERRQ(ierr);
      for (i = 0; i < numArgs; ++i) {
        ierr = PetscStrallocpy(argTypes[i], &(entry->argTypes[i]));         CHKERRQ(ierr);
      }
    }
    ierr = PetscStrallocpy(url, &(entry->url));                         CHKERRQ(ierr);
    ierr           = PetscFListGetPathAndFunction(url,&fpath,&fname);   CHKERRQ(ierr);
    entry->path    = fpath;
    entry->name    = fname;
    entry->routine = fnc;
    entry->next    = 0;
    ne->next       = entry;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOpFListDestroy"
/*@C
    PetscOpFListDestroy - Destroys a list of registered op routines.

    Input Parameter:
.   fl  - pointer to list

    Level: developer

.seealso: PetscOpFListAdd(), PetscOpFList
@*/
PetscErrorCode  PetscOpFListDestroy(PetscOpFList *fl)
{
  PetscOpFList     next,entry,tmp;
  PetscErrorCode   ierr;
  PetscInt         i;

  PetscFunctionBegin;
  if (!*fl) PetscFunctionReturn(0);
  if (!opallhead) PetscFunctionReturn(0);

  /*
       Remove this entry from the master Op list (if it is in it)
  */
  if (opallhead == *fl) {
    if (opallhead->next_list) {
      opallhead = opallhead->next_list;
    } else {
      opallhead = 0;
    }
  } else {
    tmp = opallhead;
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
    ierr = PetscFree(entry->op);  CHKERRQ(ierr);
    for (i = 0; i < entry->numArgs; ++i) {
      ierr = PetscFree(entry->argTypes[i]); CHKERRQ(ierr);
    }
    ierr = PetscFree(entry->argTypes);  CHKERRQ(ierr);
    ierr = PetscFree(entry->url);CHKERRQ(ierr);
    ierr = PetscFree(entry->path);CHKERRQ(ierr);
    ierr = PetscFree(entry->name);CHKERRQ(ierr);
    ierr = PetscFree(entry);CHKERRQ(ierr);
    entry = next;
  }
  *fl = 0;
  PetscFunctionReturn(0);
}

/*
   Destroys all the function lists that anyone has every registered, such as MatOpList, etc.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscOpFListDestroyAll"
PetscErrorCode  PetscOpFListDestroyAll(void)
{
  PetscOpFList     tmp2,tmp1 = opallhead;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (tmp1) {
    tmp2 = tmp1->next_list;
    ierr = PetscOpFListDestroy(&tmp1);CHKERRQ(ierr);
    tmp1 = tmp2;
  }
  opallhead = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOpFListFind"
/*@C
    PetscOpFListFind - Given a name, finds the matching op routine.
    Formally collective on comm.

    Input Parameters:
+   comm     - processes looking for the op
.   fl       - pointer to list of known ops
.   op       - operation name
.   numArgs  - number of op arguments
-   argTypes - list of argument type names


    Output Parameters:
.   r       - routine implementing op with the given arg types

    Level: developer

.seealso: PetscOpFListAdd(), PetscOpFList
@*/
PetscErrorCode  PetscOpFListFind(MPI_Comm comm, PetscOpFList fl,PetscVoidFunction *r, const char* op, PetscInt numArgs, char* argTypes[])
{
  PetscOpFList   entry;
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(r,3);
  if (!op) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Attempting to find operation with null name");
  *r = PETSC_NULL;
  match = PETSC_FALSE;
  entry = fl;
  while (entry) {
    ierr = PetscStrcmp(entry->op,op,&match); CHKERRQ(ierr);
    if (!match) goto next;
    if (numArgs == entry->numArgs)
      match = PETSC_TRUE;
    else
      match = PETSC_FALSE;
    if (!match) goto next;
    if (numArgs) {
      for (i = 0; i < numArgs; ++i) {
        ierr = PetscStrcmp(argTypes[i], entry->argTypes[i], &match);  CHKERRQ(ierr);
        if (!match) goto next;
      }
    }
    break;
    next: entry = entry->next;
  }
  if (match) {
    if (entry->routine) {
      *r   = entry->routine;
    }
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
    else {
      /* it is not yet in memory so load from dynamic library */
      ierr = PetscDLLibrarySym(comm,&PetscDLLibrariesLoaded,entry->path,entry->name,(void **)r);CHKERRQ(ierr);
      if (*r) {
        entry->routine = *r;
      }
    }
#endif
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOpFListView"
/*@C
   PetscOpFListView - prints out contents of a PetscOpFList

   Collective on viewer

   Input Parameters:
+  list   - the list of functions
-  viewer - ASCII viewer   Level: developer

.seealso: PetscOpFListAdd(), PetscOpFList
@*/
PetscErrorCode  PetscOpFListView(PetscOpFList list,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;
  PetscInt       i;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidPointer(list,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (!iascii) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ASCII viewer supported");

  while (list) {
    if (list->url) {
      ierr = PetscViewerASCIIPrintf(viewer," %s: ",list->url); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "%s(", list->op);    CHKERRQ(ierr);
    for (i = 0; i < list->numArgs;++i) {
      if (i > 0) {
        ierr = PetscViewerASCIIPrintf(viewer, ", "); CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "%s", list->argTypes[i]);    CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, ")\n");    CHKERRQ(ierr);
    list = list->next;
  }
  PetscFunctionReturn(0);
}
