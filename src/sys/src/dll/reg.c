#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: reg.c,v 1.4 1998/01/03 01:25:00 bsmith Exp bsmith $";
#endif
/*
         Provides a general mechanism to allow one to register
    new routines in dynamic libraries for many of the PETSc objects including KSP and PC.
*/
#include "petsc.h"
#include "sys.h"


/* ------------------------------------------------------------------------------*/
/*
      Code to maintain a list of opened dynamic libraries
*/
#if defined(USE_DYNAMIC_LIBRARIES)
#include <dlfcn.h>

struct _DLLibraryList {
  DLLibraryList next;
  void          *handle;
};


#undef __FUNC__  
#define __FUNC__ "DLOpen"
/*
     DLOpen - Opens a dynamic link library

   Input Parameter:
    libname - name of the library, can be relative or absolute

   Output Paramter:
    handle - returned from dlopen

   Notes:
    libname may contain option $BOPT and $PETSC_ARCH that are replaced with 
        appropriate values
    libname may omit the suffix and a .so.1.0 will automatically be appended

    Should also support http:// and ftp:// prefixes

*/
int DLOpen(char *libname,void **handle)
{
  char       *par2,ierr,len,*par3,arch[10];
  PetscTruth foundlibrary;

  PetscFunctionBegin;

  /* 
     make copy of library name and replace $PETSC_ARCH and $BOPT and 
     so we can add to the end of it to look for something like .so.1.0 etc.
  */
  len   = PetscStrlen(libname);
  par2  = (char *) PetscMalloc((16+len+1)*sizeof(char));CHKPTRQ(par2);
  ierr  = PetscStrcpy(par2,libname);CHKERRQ(ierr);
  
  par3 = PetscStrstr(par2,"$PETSC_ARCH");
  while (par3) {
    *par3  =  0;
    par3  += 11;
    ierr   = PetscGetArchType(arch,10);
    PetscStrcat(par2,arch);
    PetscStrcat(par2,par3);
    par3 = PetscStrstr(par2,"$PETSC_ARCH");
  }

  par3 = PetscStrstr(par2,"$BOPT");
  while (par3) {
    *par3  =  0;
    par3  += 5;
    PetscStrcat(par2,PETSC_BOPT);
    PetscStrcat(par2,par3);
    par3 = PetscStrstr(par2,"$BOPT");
  }

  /* first check original given name */
  ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) {

    /* strip out .a from it if user put it in by mistake */
    len    = PetscStrlen(par2);
    if (par2[len-1] == 'a' && par2[len-2] == '.') par2[len-2] = 0;

    /* try appending .so.1.0 */
    PetscStrcat(par2,".so.1.0");
    ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
    if (!foundlibrary) {
      PetscErrorPrintf("Library name %s\n",par2);
      SETERRQ(1,1,"Unable to locate dynamic library");
    }
  }

  *handle = dlopen(par2,1);    
  if (!*handle) {
    PetscErrorPrintf("Library name %s\n",libname);
    SETERRQ(1,1,"Unable to locate dynamic library");
  }
  PetscFree(par2);
  PetscFunctionReturn(0);
}

/*
     DLSym - Load a symbol from the dynamic link libraries.

  Input Parameter:
.  insymbol - name of symbol

  Output Parameter:
.  value 

  Notes: Symbol can be of the form

        [/path/libname[.so.1.0]:]functionname[()] where items in [] denote optional 

*/
#undef __FUNC__  
#define __FUNC__ "DLSym"
int DLSym(DLLibraryList list,char *insymbol, void **value)
{
  char          *par1,*symbol;
  int           ierr,len;

  PetscFunctionBegin;
  *value = 0;

  /* make copy of symbol so we can edit it in place */
  len    = PetscStrlen(insymbol);
  symbol = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(symbol);
  ierr   = PetscStrcpy(symbol,insymbol);CHKERRQ(ierr);

  /* 
      If symbol contains () then replace with a NULL, to support functionname() 
  */
  par1 = PetscStrchr(symbol,'(');
  if (par1) *par1 = 0;

  /* 
     check if library path is given in function name 
  */
  par1 = PetscStrchr(symbol,':');
  if (par1) {
    void *handle;

    *par1++ = 0; 
    ierr    = DLOpen(symbol,&handle);CHKERRQ(ierr);
    *value  =  dlsym(handle,par1);
    if (!*value) {
      PetscErrorPrintf("Library path and function name %s\n",insymbol);
      SETERRQ(1,1,"Unable to locate function in dynamic library");
    }
    PLogInfo(0,"DLSym:Loading function %s from dynamic library\n",par1);
  /* 
     look for symbol in predefined path of libraries 
  */
  } else {
    while (list) {
      *value =  dlsym(list->handle,symbol);
      if (*value) {
        PLogInfo(0,"DLSym:Loading function %s from dynamic library\n",symbol);
        break;
      }
      list = list->next;
    }
  }

  PetscFree(symbol);
  PetscFunctionReturn(0);
}

/*
     DLAppend - Appends another dynamic link library to the seach list, to the end
                of the search path.

     Notes: if library is already in path will not add it.
*/
#undef __FUNC__  
#define __FUNC__ "DLAppend"
int DLAppend(DLLibraryList *outlist,char *libname)
{
  DLLibraryList list,next;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;
  ierr = DLOpen(libname,&handle);CHKERRQ(ierr);

  list = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
  list->next   = 0;
  list->handle = handle;


  if (!*outlist) {
    *outlist = list;
  } else {
    next = *outlist;
    if (next->handle == handle) {
      PetscFree(list);
      PetscFunctionReturn(0); /* it is already listed */
    }
    while (next->next) {
      next = next->next;
      if (next->handle == handle) {
        PetscFree(list);
        PetscFunctionReturn(0); /* it is already listed */
      }
    }
    next->next = list;
  }
  PLogInfo(0,"DLAppend:Appending %s to dynamic library search path\n",libname);
  PetscFunctionReturn(0);
}

/*
     DLPrepend - Add another dynamic library to search for symbols to the beginning of
                 the search path.

     Notes: If library is already in path will remove old reference.

*/
#undef __FUNC__  
#define __FUNC__ "DLPrepend"
int DLPrepend(DLLibraryList *outlist,char *libname)
{
  DLLibraryList list,next,prev;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;
  ierr = DLOpen(libname,&handle);CHKERRQ(ierr);

  PLogInfo(0,"DLPrepend:Prepending %s to dynamic library search path\n",libname);

  list = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
  list->handle = handle;

  list->next        = *outlist;
  *outlist          = list;

  /* check if library was previously open, if so remove duplicate reference */
  next = list->next;
  prev = list;
  while (next) {
    if (next->handle == handle) {
      prev->next = next->next;
      PetscFree(next);
      PetscFunctionReturn(0);
    }
    prev = next;
    next = next->next;
  }
  PetscFunctionReturn(0);
}

/*
     DLClose - Destroys the search path of dynamic libraries and closes the libraries.

*/
#undef __FUNC__  
#define __FUNC__ "DLClose"
int DLClose(DLLibraryList next)
{
  DLLibraryList prev;

  PetscFunctionBegin;

  while (next) {
    prev = next;
    next = next->next;
    dlclose(prev->handle);
    PetscFree(prev);
  }
  PetscFunctionReturn(0);
}

/*
    This is the list used by the DLRegister routines
*/
DLLibraryList DLLibrariesLoaded = 0;

#endif

/* ------------------------------------------------------------------------------*/
struct FuncList_struct {
  int                    id;
  int                    (*routine)(void *);
  char                   *name;               
  char                   *rname;            /* name of create function in link library */
  struct FuncList_struct *next;
};
typedef struct FuncList_struct FuncList;


struct _DLList {
    FuncList *head, *tail;
    int      nextid;          /* next id available */
    int      nextidflag;      /* value passed to DLListRegister() to request id */
    char     *regname;        /* registration type name, for example, KSPRegister */
};


static int    NumberRegisters = 0;
static DLList *Registers[10];


#undef __FUNC__  
#define __FUNC__ "DLCreate"
/*
  DLCreate - create a name registry.

  Input Parameter:
.    preallocated - the number of pre-defined ids for this type object;
                    for example, KSPNEW.

.seealso: DLRegister(), DLDestroy()
*/
int DLCreate(int preallocated,DLList *fl )
{
  PetscFunctionBegin;
  *fl                = PetscNew(struct _DLList);CHKPTRQ(*fl);
  (*fl)->head        = 0;
  (*fl)->tail        = 0;
  (*fl)->nextid      = preallocated;
  (*fl)->nextidflag  = preallocated;

  Registers[NumberRegisters++] = fl;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLDestroyAll"
/*
   DLDestroyAll - Destroys all registers. Should be called only 
           when you know none of the methods are needed.

@*/
int DLDestroyAll()
{
  int i,ierr;

  PetscFunctionBegin;
  for ( i=0; i<NumberRegisters; i++ ) {
    ierr = DLDestroy(*Registers[i]);CHKERRQ(ierr);
    *Registers[i] = 0;
  }
#if defined(USE_DYNAMIC_LIBRARIES)
  ierr = DLClose(DLLibrariesLoaded);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*M
   DLRegister - Given a routine and two ids (an int and a string), 
                save that routine in the specified registry
 
   Input Parameters:
.      fl       - pointer registry
.      id       - integer (or enum) for routine
.      name     - string for routine
.      rname    - routine name in dynamic library
.      fnc      - function pointer (optional if using dynamic libraries)

   Output Parameters:
.      idout    - id assigned to function, same as id for predefined functions

*/

#undef __FUNC__  
#define __FUNC__ "DLRegister_Private"
int DLRegister_Private( DLList fl, int id, char *name, char *rname,int (*fnc)(void *),int *idout)
{
  FuncList *entry;

  PetscFunctionBegin;
  entry          = (FuncList*) PetscMalloc(sizeof(FuncList));CHKPTRQ(entry);
  entry->name    = (char *)PetscMalloc( PetscStrlen(name) + 1 ); CHKPTRQ(entry->name);
  PetscStrcpy( entry->name, name );
  entry->rname   = (char *)PetscMalloc( PetscStrlen(rname) + 1 ); CHKPTRQ(entry->rname);
  PetscStrcpy( entry->rname, rname );
  entry->routine = fnc;

  entry->next = 0;
  if (fl->tail) fl->tail->next = entry;
  else          fl->head       = entry;
  fl->tail = entry;
  
  if (id == fl->nextidflag) {
    entry->id  = fl->nextid++;
  } else {
    entry->id  = id;
  }
  if (idout) *idout = id;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLDestroy"
/*
    DLDestroy - Destroy a list of registered routines

    Input Parameter:
.   fl   - pointer to list
*/
int DLDestroy(DLList fl )
{
  FuncList *entry = fl->head, *next;

  PetscFunctionBegin;
  while (entry) {
    next = entry->next;
    PetscFree( entry->name );
    PetscFree( entry->rname );
    PetscFree( entry );
    entry = next;
  }
  PetscFree( fl );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLFindRoutine"
/*
    DLFindRoutine - given an id or name, find the matching routine

    Input Parameters:
.   fl   - pointer to list
.   id   - id (-1 for ignore)
.   name - name string.  (Null for ignore)

    The id or name must have been registered with the DLList before calling this 
    routine.
*/
int DLFindRoutine(DLList fl, int id, char *name, int (**r)(void *))
{
  FuncList *entry = fl->head;
  
  PetscFunctionBegin;
  while (entry) {
    if ((id >= 0 && entry->id == id) || (name && !PetscStrcmp(name,entry->name))) {
      /* found it */
      if (entry->routine) {*r =entry->routine;  PetscFunctionReturn(0);}
      /* it is not yet in memory so load from dynamic library */
#if defined(USE_DYNAMIC_LIBRARIES)
      { int ierr;
        ierr = DLSym(DLLibrariesLoaded,entry->rname,(void **)r);CHKERRQ(ierr);
      }
      if (*r) {
        entry->routine = *r;
        PetscFunctionReturn(0);
      } else {
        PetscErrorPrintf("Registered function name: %s\n",entry->rname);
        SETERRQ(1,1,"Unable to find function: either it is mis-spelled or dynamic library is not in path");
      }
#endif
    }
    entry = entry->next;
  }

  if (name) PetscErrorPrintf("Function name: %s\n",name);
  SETERRQ(1,1,"Unable to find function: either it is mis-spelled or dynamic library is not in path");
#if !defined(USE_PETSC_DEBUG)
  PetscFunctionReturn(0);
#endif
}

#undef __FUNC__  
#define __FUNC__ "DLFindID"
/*
    DLFindID - Given a name, find the corresponding id

    Input Parameters:
.   fl   - pointer to list
.   name - name string

    Returns:
    id - id associate with name, -1 if name not found

*/
int DLFindID( DLList fl, char *name, int *id )
{
  FuncList *entry = fl->head;

  PetscFunctionBegin;
  *id = -1;
  while (entry) {
    if (!PetscStrcmp(name,entry->name)) {*id = entry->id; PetscFunctionReturn(0);}
    entry = entry->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLFindName"
/*
    DLFindName - Given an id, find the corresponding name

    Input Parameters:
.   fl   - pointer to list
.   id   - id 

    Output Parameter:
.   name - pointer to name of object type. You should NOT free this.

*/
int DLFindName( DLList fl, int id, char **name )
{
  FuncList *entry = fl->head;

  PetscFunctionBegin;
  while (entry) {
    if (id == entry->id) {*name = entry->name; PetscFunctionReturn(0);}
    entry = entry->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLPrintTypes"
/*
   DLPrintTypes - Prints the methods available.

   Input Parameters:
.  comm   - The communicator (usually MPI_COMM_WORLD)
.  fd     - file to print to, usually stdout
.  list   - list of types

*/
int DLPrintTypes(MPI_Comm comm,FILE *fd,char *prefix,char *name,DLList list)
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

#undef __FUNC__  
#define __FUNC__ "DLGetTypeFromOptions" 
/*
   DLGetTypeFromOptions

   Input Parameter:
.  prefix - optional database prefix
.  name - type name
.  list - list of registered types

   Output Parameter:
.  type -  method
.  flag - 1 if type found
.  oname - if not PETSC_NULL, copies option name
.  len  - if oname is given, this is its length

.keywords: 

.seealso: 
*/
int DLGetTypeFromOptions(char *prefix,char *name,DLList list,int *type,char *oname,int len,int *flag)
{
  char sbuf[256];
  int  ierr,itype;
  
  PetscFunctionBegin;
  ierr = OptionsGetString(prefix,name, sbuf, 256,flag); CHKERRQ(ierr);
  if (*flag) {
    ierr = DLFindID( list, sbuf,&itype ); CHKERRQ(ierr);
    if (name) {
      PetscStrncpy(oname,sbuf,len);
    }
    *(int *)type = itype;
  }
  PetscFunctionReturn(0);
}




