
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: reg.c,v 1.7 1998/01/14 02:39:22 bsmith Exp bsmith $";
#endif
/*
         Provides a general mechanism to allow one to register
    new routines in dynamic libraries for many of the PETSc objects including KSP and PC.
*/
#include "petsc.h"
#include "sys.h"

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
  char *libname[32];
  int  nmax,i,ierr,flg;

  PetscFunctionBegin;


  ierr = DLAppend(&DLLibrariesLoaded,PETSC_DEFAULT_DYNAMIC_LIBRARY);CHKERRQ(ierr);

  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_prepend",libname,&nmax,&flg);CHKERRQ(ierr);
  for ( i=nmax-1; i>=0; i-- ) {
    ierr = DLPrepend(&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    PetscFree(libname[i]);
  }
  nmax = 32;
  ierr = OptionsGetStringArray(PETSC_NULL,"-dll_append",libname,&nmax,&flg);CHKERRQ(ierr);
  for ( i=0; i<nmax; i++ ) {
    ierr = DLAppend(&DLLibrariesLoaded,libname[i]);CHKERRQ(ierr);
    PetscFree(libname[i]);
  }

  PetscFunctionReturn(0);
}
#else
int PetscInitialize_DynamicLibraries()
{
 PetscFunctionBegin;

  PetscFunctionReturn(0);
}
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

/*
      These are kept so that we may free the space used for all registered systems
   from a single, centralized location.
*/
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

   Synopsis:
    int DLRegister(DLList fl, int id, char *name, char *rname,int (*fnc)(void *),int *idout)

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




