
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: reg.c,v 1.11 1998/03/06 00:12:03 bsmith Exp bsmith $";
#endif
/*
         Provides a general mechanism to allow one to register
    new routines in dynamic libraries for many of the PETSc objects including KSP and PC.
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
int PetscInitialize_DynamicLibraries()
{
  char *libname[32],libs[256];
  int  nmax,i,ierr,flg;

  PetscFunctionBegin;

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscts"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscsnes"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscsles"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscmat"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  ierr = PetscStrcpy(libs,PETSC_LDIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libpetscvec"); CHKERRQ(ierr);
  ierr = DLLibraryAppend(&DLLibrariesLoaded,libs);CHKERRQ(ierr);

  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_prepend",libname,&nmax,&flg);CHKERRQ(ierr);
  for ( i=nmax-1; i>=0; i-- ) {
    ierr = DLLibraryPrepend(&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    PetscFree(libname[i]);
  }
  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_append",libname,&nmax,&flg);CHKERRQ(ierr);
  for ( i=0; i<nmax; i++ ) {
    ierr = DLLibraryAppend(&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    PetscFree(libname[i]);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscFinalize_DynamicLibraries"
/*
      PetscFinalize_DynamicLibraries - Closes the opened dynamic libraries
*/ 
int PetscFinalize_DynamicLibraries()
{
  int ierr;

  PetscFunctionBegin;
  ierr = DLLibraryClose(DLLibrariesLoaded);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ "PetscInitalize_DynamicLibraries"
int PetscInitialize_DynamicLibraries()
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "PetscFinalize_DynamicLibraries"
int PetscFinalize_DynamicLibraries()
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
#endif

/* ------------------------------------------------------------------------------*/
struct FuncList_struct {
  int                    (*routine)(void *);
  char                   *path;
  char                   *name;               
  char                   *rname;            /* name of create function in link library */
  struct FuncList_struct *next;
};
typedef struct FuncList_struct FuncList;

struct _DLList {
    FuncList *head, *tail;
    char     *regname;        /* registration type name */
    DLList   next;
};

/*
     Keep a linked list of DLLists so that we may destroy all the left-over ones
*/
static DLList dlallhead = 0;

#undef __FUNC__  
#define __FUNC__ "DLRegisterCreate"
/*
  DLRegisterCreate - create a name registry.

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


/*M
   DLRegister - Given a routine and a string id, 
                save that routine in the specified registry
    Input Parameters:
.      fl       - pointer registry
.      name     - string for routine
.      rname    - routine name in dynamic library
.      fnc      - function pointer (optional if using dynamic libraries)


   Synopsis:
    int DLRegister(DLList fl, char *name, char *rname,int (*fnc)(void *))

*/

#undef __FUNC__  
#define __FUNC__ "DLRegister_Private"
int DLRegister_Private( DLList *fl, char *name, char *rname,int (*fnc)(void *))
{
  FuncList *entry;
  int      ierr;
  char     *fpath,*fname;

  PetscFunctionBegin;
  entry          = (FuncList*) PetscMalloc(sizeof(FuncList));CHKPTRQ(entry);
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
    DLRegisterDestroy - Destroy a list of registered routines

    Input Parameter:
.   fl   - pointer to list
*/
int DLRegisterDestroy(DLList fl )
{
  FuncList *entry, *next;
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
int DLRegisterDestroyAll()
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
    DLRegisterFind - givn a name, find the matching routine

    Input Parameters:
.   fl   - pointer to list
.   name - name string

    The id or name must have been registered with the DLList before calling this 
    routine.
*/
int DLRegisterFind(DLList fl, char *name, int (**r)(void *))
{
  FuncList *entry = fl->head;
  char     *function, *path;
  int      ierr;
  
  PetscFunctionBegin;
  ierr = DLRegisterGetPathAndFunction(name,&path,&function);CHKERRQ(ierr);

  /*
        If path then append it to search libraries
  */
#if defined(USE_DYNAMIC_LIBRARIES)
  if (path) {
    ierr = DLLibraryAppend(&DLLibrariesLoaded,path); CHKERRQ(ierr);
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
      ierr = DLLibrarySym(&DLLibrariesLoaded,path,entry->rname,(void **)r);CHKERRQ(ierr);
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
  /* Function never registered; try for it anyways */
  ierr = DLLibrarySym(&DLLibrariesLoaded,path,function,(void **)r);CHKERRQ(ierr);
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

   Input Parameters:
.  comm   - The communicator (usually MPI_COMM_WORLD)
.  fd     - file to print to, usually stdout
.  list   - list of types

*/
int DLRegisterPrintTypes(MPI_Comm comm,FILE *fd,char *prefix,char *name,DLList list)
{
  FuncList *entry;
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






