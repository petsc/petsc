
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: reg.c,v 1.17 1998/05/15 17:20:46 bsmith Exp bsmith $";
#endif
/*
    Provides a general mechanism to allow one to register new routines in
    dynamic libraries for many of the PETSc objects (including, e.g., KSP and PC).
*/
#include "petsc.h"
#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "DLRegisterGetPathAndFunction"
int DLRegisterGetPathAndFunction(char *name,char **path,char **function)
{
  char work[256],*lfunction;

  PetscFunctionBegin;
  PetscStrncpy(work,name,256);
  lfunction = PetscStrrchr(work,':');
  if (lfunction != work) {
    lfunction[-1] = 0;
    *path = (char *) PetscMalloc( (PetscStrlen(work) + 1)*sizeof(char));CHKPTRQ(*path);
    PetscStrcpy(*path,work);
  } else {
    *path = 0;
  }
  *function = (char *) PetscMalloc((PetscStrlen(lfunction)+1)*sizeof(char));CHKPTRQ(*function);
  PetscStrcpy(*function,lfunction);
  PetscFunctionReturn(0);
}

#if defined(USE_DYNAMIC_LIBRARIES)

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
  char *libname[32],libs[256];
  int  nmax,i,ierr,flg;

  PetscFunctionBegin;

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscts"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscsnes"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscsles"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscmat"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscvec"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_prepend",libname,&nmax,&flg);CHKERRQ(ierr);
  for ( i=nmax-1; i>=0; i-- ) {
    ierr = DLLibraryPrepend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    PetscFree(libname[i]);
  }
  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_append",libname,&nmax,&flg);CHKERRQ(ierr);
  for ( i=0; i<nmax; i++ ) {
    ierr = DLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    PetscFree(libname[i]);
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

#else

#undef __FUNC__  
#define __FUNC__ "PetscInitalize_DynamicLibraries"
int PetscInitialize_DynamicLibraries(void)
{
  PetscFunctionBegin;

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
typedef struct _FuncList *FuncList;
struct _FuncList {
  int      (*routine)(void *); /* the routine */
  char     *path;              /* path of link library containing routine */
  char     *name;              /* string to identify routine */
  char     *rname;             /* routine name in dynamic library */
  FuncList next;               /* next pointer */
};

struct _DLList {
    FuncList head, tail;   /* head and tail of this DLList */
    char     *regname;       /* registration type name */
    DLList   next;           /* next DLList */
};

/*
     Keep a linked list of DLLists so that we can destroy all the left-over ones.
*/
static DLList dlallhead = 0;

#undef __FUNC__  
#define __FUNC__ "DLRegisterCreate"
/*
  DLRegisterCreate - Creates a name registry.

.seealso: DLRegister(), DLRegisterDestroy()
*/
int DLRegisterCreate(DLList *fl )
{
  PetscFunctionBegin;
  *fl                = PetscNew(struct _DLList);CHKPTRQ(*fl);
  (*fl)->head        = 0;
  (*fl)->tail        = 0;
  
  /* 
      Add list to front of nasty-global lists of lists
  */
  if (!dlallhead) {
    dlallhead       = *fl;
    (*fl)->next     = 0;
  } else {
    DLList tmp = dlallhead;
    
    dlallhead   = *fl;
    (*fl)->next = tmp;
  }
  PetscFunctionReturn(0);
}

/*
   DLRegister - Given a routine and a string id, saves that routine in the
   specified registry.

   Synopsis:
   int DLRegister(DLList *fl, char *name, char *rname,int (*fnc)(void *))

   Input Parameters:
+  fl    - pointer registry
.  name  - string to identify routine
.  rname - routine name in dynamic library
-  fnc   - function pointer (optional if using dynamic libraries)

   Notes:
   Users who wish to register new methods for use by a particular PETSc
   component (e.g., SNES) should generally call the registration routine
   for that particular component (e.g., SNESRegister()) instead of
   calling DLRegister() directly.

.seealso: DLRegisterCreate(), DLRegisterDestroy(), SNESRegister(), KSPRegister(),
          PCRegister(), TSRegister()
*/

#undef __FUNC__  
#define __FUNC__ "DLRegister_Private"
int DLRegister_Private( DLList *fl, char *name, char *rname,int (*fnc)(void *))
{
  FuncList entry;
  int      ierr;
  char     *fpath,*fname;

  PetscFunctionBegin;
  entry          = (FuncList) PetscMalloc(sizeof(struct _FuncList));CHKPTRQ(entry);
  entry->name    = (char *)PetscMalloc( PetscStrlen(name) + 1 ); CHKPTRQ(entry->name);
  PetscStrcpy( entry->name, name );

  ierr = DLRegisterGetPathAndFunction(rname,&fpath,&fname);CHKERRQ(ierr);

  entry->path    = fpath;
  entry->rname   = fname;
  entry->routine = fnc;

  if (!*fl) {
    ierr = DLRegisterCreate(fl);CHKERRQ(ierr);
  }

  entry->next = 0;
  if ((*fl)->tail) (*fl)->tail->next = entry;
  else             (*fl)->head       = entry;
  (*fl)->tail = entry;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLRegisterDestroy"
/*
    DLRegisterDestroy - Destroys a list of registered routines.

    Input Parameter:
.   fl  - pointer to list

.seealso: DLRegisterCreate(), DLRegister()
*/
int DLRegisterDestroy(DLList fl)
{
  FuncList entry, next;
  DLList   tmp = dlallhead;

  PetscFunctionBegin;
  if (!fl) PetscFunctionReturn(0);

  entry = fl->head;
  while (entry) {
    next = entry->next;
    if (entry->path) PetscFree(entry->path);
    PetscFree( entry->name );
    PetscFree( entry->rname );
    PetscFree( entry );
    entry = next;
  }

  /*
       Remove this entry from the master DL list 
  */
  if (dlallhead == fl) {
    if (dlallhead->next) {
      dlallhead = dlallhead->next;
    } else {
      dlallhead = 0;
    }
  } else {
    while (tmp->next != fl) {
      tmp = tmp->next;
      if (!tmp->next) SETERRQ(1,1,"Internal PETSc error, function registration corrupted");
    }
    tmp->next = tmp->next->next;
  }
 
  PetscFree( fl );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLRegisterDestroyAll"
int DLRegisterDestroyAll(void)
{
  DLList tmp2,tmp1 = dlallhead;

  PetscFunctionBegin;
  while (tmp1) {
    tmp2 = tmp1->next;
    DLRegisterDestroy(tmp1);
    tmp1 = tmp2;
  }
  dlallhead = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLRegisterFind"
/*
    DLRegisterFind - Given a name, finds the matching routine.

    Input Parameters:
+   comm - processors looking for routine
.   fl   - pointer to list
-   name - name string

    Output Parameters:
.   r - the routine

    Notes:
    The routine's id or name MUST have been registered with the DLList via
    DLRegister() before DLRegisterFind() can be called.

.seealso: DLRegister()
*/
int DLRegisterFind(MPI_Comm comm,DLList fl,char *name, int (**r)(void *))
{
  FuncList entry = fl->head;
  char     *function, *path;
  int      ierr;
  
  PetscFunctionBegin;
  ierr = DLRegisterGetPathAndFunction(name,&path,&function);CHKERRQ(ierr);

  /*
        If path then append it to search libraries
  */
#if defined(USE_DYNAMIC_LIBRARIES)
  if (path) {
    ierr = DLLibraryAppend(comm,&DLLibrariesLoaded,path); CHKERRQ(ierr);
  }
#endif

  while (entry) {
    if ((path && entry->path && !PetscStrcmp(path,entry->path) && !PetscStrcmp(function,entry->rname)) ||
        (path && entry->path && !PetscStrcmp(path,entry->path) && !PetscStrcmp(function,entry->name)) ||
        (!path &&  !PetscStrcmp(function,entry->name)) || 
        (!path &&  !PetscStrcmp(function,entry->rname))) {

      if (entry->routine) {
        *r = entry->routine; 
         if (path) PetscFree(path);
         PetscFree(function);
         PetscFunctionReturn(0);
      }

      /* it is not yet in memory so load from dynamic library */
#if defined(USE_DYNAMIC_LIBRARIES)
      ierr = DLLibrarySym(comm,&DLLibrariesLoaded,path,entry->rname,(void **)r);CHKERRQ(ierr);
      if (*r) {
        entry->routine = *r;
        if (path) PetscFree(path);
        PetscFree(function);
        PetscFunctionReturn(0);
      } else {
        PetscErrorPrintf("Registered function name: %s\n",entry->rname);
        SETERRQ(1,1,"Unable to find function: either it is mis-spelled or dynamic library is not in path");
      }
#endif
    }
    entry = entry->next;
  }

#if defined(USE_DYNAMIC_LIBRARIES)
  /* Function never registered; try for it anyway */
  ierr = DLLibrarySym(comm,&DLLibrariesLoaded,path,function,(void **)r);CHKERRQ(ierr);
  if (path) PetscFree(path);
  if (r) {
    ierr = DLRegister(&fl,name,name,r); CHKERRQ(ierr);
    PetscFree(function);
    PetscFunctionReturn(0);
  }
#endif

  PetscFree(function);

  if (name) PetscErrorPrintf("Function name: %s\n",name);
  SETERRQ(1,1,"Unable to find function: either it is mis-spelled or dynamic library is not in path");
#if !defined(USE_PETSC_DEBUG)
  PetscFunctionReturn(0);
#endif
}

#undef __FUNC__  
#define __FUNC__ "DLRegisterPrintTypes"
/*
   DLRegisterPrintTypes - Prints the methods available.

   Collective over MPI_Comm

   Input Parameters:
+  comm   - the communicator (usually MPI_COMM_WORLD)
.  fd     - file to print to, usually stdout
.  prefix - prefix to prepend to name (optional)
.  name   - option string
-  list   - list of types

.seealso: DLRegister()
*/
int DLRegisterPrintTypes(MPI_Comm comm,FILE *fd,char *prefix,char *name,DLList list)
{
  FuncList entry;
  int      count = 0;
  char     p[64];

  PetscFunctionBegin;
  PetscStrcpy(p,"-");
  if (prefix) PetscStrcat(p,prefix);
  PetscPrintf(comm,"  %s%s (one of)",p,name);

  entry = list->head;
  while (entry) {
    PetscFPrintf(comm,fd," %s",entry->name);
    entry = entry->next;
    count++;
    if (count == 8) PetscFPrintf(comm,fd,"\n     ");
  }
  PetscFPrintf(comm,fd,"\n");
  PetscFunctionReturn(0);
}






