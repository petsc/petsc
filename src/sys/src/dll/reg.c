/*$Id: reg.c,v 1.50 1999/11/10 03:17:56 bsmith Exp bsmith $*/
/*
    Provides a general mechanism to allow one to register new routines in
    dynamic libraries for many of the PETSc objects (including, e.g., KSP and PC).
*/
#include "petsc.h"
#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "FListGetPathAndFunction"
int FListGetPathAndFunction(const char name[],char *path[],char *function[])
{
  char work[256],*lfunction,ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(work,name,256);CHKERRQ(ierr);
  ierr = PetscStrrchr(work,':',&lfunction);CHKERRQ(ierr);
  if (lfunction != work) {
    lfunction[-1] = 0;
    ierr = PetscStrallocpy(work,path);CHKERRQ(ierr);
  } else {
    *path = 0;
  }
  ierr = PetscStrallocpy(lfunction,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)

/*
    This is the list used by the DLRegister routines
*/
DLLibraryList DLLibrariesLoaded = 0;

#undef __FUNC__  
#define __FUNC__ "PetscInitialize_DynamicLibraries"
/*
    PetscInitialize_DynamicLibraries - Adds the default dynamic link libraries to the 
    search path.
*/ 
int PetscInitialize_DynamicLibraries(void)
{
  char       *libname[32],libs[256],dlib[1024];
  int        nmax,i,ierr;
  PetscTruth found;

  PetscFunctionBegin;

  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_prepend",libname,&nmax,PETSC_NULL);CHKERRQ(ierr);
  for ( i=0; i<nmax; i++ ) {
    ierr = DLLibraryPrepend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    ierr = PetscFree(libname[i]);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetsc");CHKERRQ(ierr);
  ierr = DLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Unable to locate PETSc dynamic library %s \n You cannot move the dynamic libraries!\n or remove USE_DYNAMIC_LIBRARIES from $PETSC_DIR/bmake/$PETSC_ARCH/petscconf.h\n and rebuild libraries before moving",libs);
  }


  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscvec");CHKERRQ(ierr);
  ierr = DLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscmat");CHKERRQ(ierr);
  ierr = DLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscdm");CHKERRQ(ierr);
  ierr = DLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscsles");CHKERRQ(ierr);
  ierr = DLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscsnes");CHKERRQ(ierr);
  ierr = DLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscts");CHKERRQ(ierr);
  ierr = DLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);CHKERRQ(ierr);
  if (found) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);
  }

  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_append",libname,&nmax,PETSC_NULL);CHKERRQ(ierr);
  for ( i=0; i<nmax; i++ ) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    ierr = PetscFree(libname[i]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscFinalize_DynamicLibraries"
/*
     PetscFinalize_DynamicLibraries - Closes the opened dynamic libraries.
*/ 
int PetscFinalize_DynamicLibraries(void)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DLLibraryClose(DLLibrariesLoaded);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else /* not using dynamic libraries */

extern int DLLibraryRegister_Petsc(char *);

#undef __FUNC__  
#define __FUNC__ "PetscInitalize_DynamicLibraries"
int PetscInitialize_DynamicLibraries(void)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DLLibraryRegister_Petsc(PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "PetscFinalize_DynamicLibraries"
int PetscFinalize_DynamicLibraries(void)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
#endif

/* ------------------------------------------------------------------------------*/
struct _FList {
  int    (*routine)(void *); /* the routine */
  char   *path;              /* path of link library containing routine */
  char   *name;              /* string to identify routine */
  char   *rname;             /* routine name in dynamic library */
  FList  next;               /* next pointer */
  FList  next_list;          /* used to maintain list of all lists for freeing */
};

/*
     Keep a linked list of FLists so that we can destroy all the left-over ones.
*/
static FList   dlallhead = 0;

/*
   FListAddDynamic - Given a routine and a string id, saves that routine in the
   specified registry.

   Synopsis:
   int FListAddDynamic(FList *fl, char *name, char *rname,int (*fnc)(void *))

   Input Parameters:
+  fl    - pointer registry
.  name  - string to identify routine
.  rname - routine name in dynamic library
-  fnc   - function pointer (optional if using dynamic libraries)

   Notes:
   Users who wish to register new methods for use by a particular PETSc
   component (e.g., SNES) should generally call the registration routine
   for that particular component (e.g., SNESRegisterDynamic()) instead of
   calling FListAddDynamic() directly.

   $PETSC_ARCH, $PETSC_DIR, $PETSC_LDIR, and $BOPT occuring in pathname will be replaced with appropriate values.

.seealso: FListDestroy(), SNESRegisterDynamic(), KSPRegisterDynamic(),
          PCRegisterDynamic(), TSRegisterDynamic()
*/

#undef __FUNC__  
#define __FUNC__ "FListAdd"
int FListAdd( FList *fl,const char name[],const char rname[],int (*fnc)(void *))
{
  FList   entry,ne;
  int      ierr;
  char     *fpath,*fname;

  PetscFunctionBegin;

  if (!*fl) {
    entry          = (FList) PetscMalloc(sizeof(struct _FList));CHKPTRQ(entry);
    ierr           = PetscStrallocpy(name,&entry->name);CHKERRQ(ierr);
    ierr = FListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
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
      PetscTruth founddup;

      ierr = PetscStrcmp(ne->name,name,&founddup);CHKERRQ(ierr);
      if (founddup) { /* found duplicate */
        ierr = FListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
        ierr = PetscStrfree(ne->path);CHKERRQ(ierr);
        ierr = PetscStrfree(ne->rname);CHKERRQ(ierr);
        ne->path    = fpath;
        ne->rname   = fname;
        ne->routine = fnc;
        PetscFunctionReturn(0);
      }
      if (ne->next) ne = ne->next; else break;
    }
    /* create new entry and add to end of list */
    entry          = (FList) PetscMalloc(sizeof(struct _FList));CHKPTRQ(entry);
    ierr           = PetscStrallocpy(name,&entry->name);
    ierr           = FListGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);
    entry->path    = fpath;
    entry->rname   = fname;
    entry->routine = fnc;
    entry->next    = 0;
    ne->next = entry;
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListDestroy"
/*
    FListDestroy - Destroys a list of registered routines.

    Input Parameter:
.   fl  - pointer to list

.seealso: FListAddDynamic()
*/
int FListDestroy(FList fl)
{
  FList   next,entry,tmp = dlallhead;
  int     ierr;

  PetscFunctionBegin;
  if (!fl) PetscFunctionReturn(0);

  if (!dlallhead) {
    SETERRQ(1,1,"Internal PETSc error, function registration corrupted");
  }

  /*
       Remove this entry from the master DL list 
  */
  if (dlallhead == fl) {
    if (dlallhead->next_list) {
      dlallhead = dlallhead->next_list;
    } else {
      dlallhead = 0;
    }
  } else {
    while (tmp->next_list != fl) {
      tmp = tmp->next_list;
      if (!tmp->next_list) SETERRQ(1,1,"Internal PETSc error, function registration corrupted");
    }
    tmp->next_list = tmp->next_list->next_list;
  }

  /* free this list */
  entry = fl;
  while (entry) {
    next = entry->next;
    ierr = PetscStrfree(entry->path);CHKERRQ(ierr);
    ierr = PetscFree( entry->name );CHKERRQ(ierr);
    ierr = PetscFree( entry->rname );CHKERRQ(ierr);
    ierr = PetscFree( entry );CHKERRQ(ierr);
    entry = next;
  }

 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListDestroyAll"
int FListDestroyAll(void)
{
  FList tmp2,tmp1 = dlallhead;
  int    ierr;

  PetscFunctionBegin;
  while (tmp1) {
    tmp2 = tmp1->next_list;
    ierr = FListDestroy(tmp1);CHKERRQ(ierr);
    tmp1 = tmp2;
  }
  dlallhead = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListFind"
/*
    FListFind - Given a name, finds the matching routine.

    Input Parameters:
+   comm - processors looking for routine
.   fl   - pointer to list
-   name - name string

    Output Parameters:
.   r - the routine

    Notes:
    The routine's id or name MUST have been registered with the FList via
    FListAddDynamic() before FListFind() can be called.

.seealso: FListAddDynamic()
*/
int FListFind(MPI_Comm comm,FList fl,const char name[], int (**r)(void *))
{
  FList        entry = fl;
  int          ierr;
  char         *function, *path;
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  char         *newpath;
#endif
  PetscTruth   flg,f1,f2,f3;
 
  PetscFunctionBegin;
  *r = 0;
  ierr = FListGetPathAndFunction(name,&path,&function);CHKERRQ(ierr);

  /*
        If path then append it to search libraries
  */
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  if (path) {
    ierr = DLLibraryAppend(comm,&DLLibrariesLoaded,path);CHKERRQ(ierr);
  }
#endif

  while (entry) {
    flg = PETSC_FALSE;
    if (path && entry->path) {
      ierr = PetscStrcmp(path,entry->path,&f1);CHKERRQ(ierr);
      ierr = PetscStrcmp(function,entry->rname,&f2);CHKERRQ(ierr);
      ierr = PetscStrcmp(function,entry->name,&f3);CHKERRQ(ierr);
      flg =  (PetscTruth) ((f1 && f2) || (f1 && f3));
    } else if (!path) {
      ierr = PetscStrcmp(function,entry->name,&f1);CHKERRQ(ierr);
      ierr = PetscStrcmp(function,entry->rname,&f2);CHKERRQ(ierr);
      flg =  (PetscTruth) (f1 || f2);
    }

    if (flg) {

      if (entry->routine) {
        *r = entry->routine; 
        ierr = PetscStrfree(path);CHKERRQ(ierr);
        ierr = PetscFree(function);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }

      /* it is not yet in memory so load from dynamic library */
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
      newpath = path;
      if (!path) newpath = entry->path;
      ierr = DLLibrarySym(comm,&DLLibrariesLoaded,newpath,entry->rname,(void **)r);CHKERRQ(ierr);
      if (*r) {
        entry->routine = *r;
        ierr = PetscStrfree(path);CHKERRQ(ierr);
        ierr = PetscFree(function);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      } else {
        PetscErrorPrintf("Registered function name: %s\n",entry->rname);
        ierr = DLLibraryPrintPath();CHKERRQ(ierr);
        SETERRQ(1,1,"Unable to find function: either it is mis-spelled or dynamic library is not in path");
      }
#endif
    }
    entry = entry->next;
  }

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  /* Function never registered; try for it anyway */
  ierr = DLLibrarySym(comm,&DLLibrariesLoaded,path,function,(void **)r);CHKERRQ(ierr);
  ierr = PetscStrfree(path);CHKERRQ(ierr);
  if (*r) {
    ierr = FListAddDynamic(&fl,name,name,r);CHKERRQ(ierr);
    ierr = PetscFree(function);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  /*
       Don't generate error, just end
  PetscErrorPrintf("Function name: %s\n",function);
  ierr = DLLibraryPrintPath();CHKERRQ(ierr);
  SETERRQ(1,1,"Unable to find function: either it is mis-spelled or dynamic library is not in path");
  */

  ierr = PetscFree(function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListView"
/*
   FListView - prints out contents of an FList

   Collective over MPI_Comm

   Input Parameters:
+  flist - the list of functions
-  viewer - currently ignored

.seealso: FListAddDynamic(), FListPrintTypes()
*/
int FListView(FList list,Viewer viewer)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  if (!viewer) viewer = VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  PetscValidPointer(list);
  
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (!isascii) SETERRQ(1,1,"Only ASCII viewer supported");

  while (list) {
    if (list->path) {
      ierr = ViewerASCIIPrintf(viewer," %s %s %s\n",list->path,list->name,list->rname);CHKERRQ(ierr);
    } else {
      ierr = ViewerASCIIPrintf(viewer," %s %s\n",list->name,list->rname);CHKERRQ(ierr);
    }
    list = list->next;
  }
  ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "FListPrintTypes"
/*
   FListPrintTypes - Prints the methods available.

   Collective over MPI_Comm

   Input Parameters:
+  comm   - the communicator (usually MPI_COMM_WORLD)
.  fd     - file to print to, usually stdout
.  prefix - prefix to prepend to name (optional)
.  name   - option string
-  list   - list of types

.seealso: FListAddDynamic()
*/
int FListPrintTypes(MPI_Comm comm,FILE *fd,const char prefix[],const char name[],FList list)
{
  int      ierr, count = 0;
  char     p[64];

  PetscFunctionBegin;
  if (!fd) fd = stdout;

  ierr = PetscStrcpy(p,"-");CHKERRQ(ierr);
  if (prefix) {ierr = PetscStrcat(p,prefix);CHKERRQ(ierr);}
  ierr = PetscFPrintf(comm,fd,"  %s%s (one of)",p,name);CHKERRQ(ierr);

  while (list) {
    ierr = PetscFPrintf(comm,fd," %s",list->name);CHKERRQ(ierr);
    list = list->next;
    count++;
    if (count == 8) {ierr = PetscFPrintf(comm,fd,"\n     ");CHKERRQ(ierr);}
  }
  ierr = PetscFPrintf(comm,fd,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "FListDuplicate"
/*
    FListDuplicate - Creates a new list from a given object list.

    Input Parameters:
.   fl   - pointer to list

    Output Parameters:
.   nl - the new list (should point to 0 to start, otherwise appends)


*/
int FListDuplicate(FList fl, FList *nl)
{
  int  ierr;
  char path[1024];

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
    ierr = FListAddDynamic(nl,path,fl->rname,fl->routine);CHKERRQ(ierr);
    fl   = fl->next;
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "FListConcat"
/*
    FListConcat - joins name of a libary, and the path where it is located
    into a single string.

    Input Parameters:
.   path   - path to the library name.
.   name   - name of the library

    Output Parameters:
.   fullname - the name that is the union of the path and the library name,
               delimited by a semicolon. i.e path:name

    Notes:
    If the path is NULL, assumes that the name, specified also includes
    the path as path:name

*/
int FListConcat(const char path[],const char name[], char fullname[])
{
  int ierr;
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
