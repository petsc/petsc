#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: reg.c,v 1.1 1997/12/12 04:18:45 bsmith Exp bsmith $";
#endif
/*
         Provides a general mechanism to allow one to register
    new routines in dynamic libraries for many of the PETSc objects including KSP and PC.
*/
#include "petsc.h"

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
};

extern int    DLCreate(int,DLList *);
extern int    DLRegister(DLList,int,char*,char*,int*);
extern int    DLDestroy(DLList);
extern int    DLFindRoutine(DLList,int,char*,int (**)(void*));
extern int    DLFindID(DLList,char*,int *);
extern int    DLFindName(DLList,int,char**);
extern int    DLDestroyAll();
extern int    DLPrintTypes(MPI_Comm,FILE*,char*,char *,DLList);
extern int    DLGetTypeFromOptions(char *,char *,DLList,int *,int *);


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
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLRegister"
/*
   DLRegister - Given a routine and two ids (an int and a string), 
                save that routine in the specified registry
 
   Input Parameters:
.      fl       - pointer registry
.      id       - integer (or enum) for routine
.      name     - string for routine
.      rname    - routine name in dynamic library

   Output Parameters:
.      idout    - id assigned to function, same as id for predefined functions

*/
int DLRegister( DLList fl, int id, char *name, char *rname,int *idout)
{
  FuncList *entry;

  PetscFunctionBegin;
  entry          = (FuncList*) PetscMalloc(sizeof(FuncList));CHKPTRQ(entry);
  entry->name    = (char *)PetscMalloc( PetscStrlen(name) + 1 ); CHKPTRQ(entry->name);
  PetscStrcpy( entry->name, name );
  entry->rname   = (char *)PetscMalloc( PetscStrlen(rname) + 1 ); CHKPTRQ(entry->rname);
  PetscStrcpy( entry->rname, rname );
  entry->routine = 0;

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


*/
int DLFindRoutine(DLList fl, int id, char *name, int (**r)(void *))
{
  FuncList *entry = fl->head;
  
  PetscFunctionBegin;
  while (entry) {
    if ((id >= 0 && entry->id == id) || (name && !PetscStrcmp(name,entry->name))) {
      if (entry->routine) {*r =entry->routine;  PetscFunctionReturn(0);}
      ;
    }
    entry = entry->next;
  }
  PetscFunctionReturn(0);
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
  while (entry) {
    if (!PetscStrcmp(name,entry->name)) PetscFunctionReturn(entry->id);
    entry = entry->next;
  }
  PetscFunctionReturn(-1);
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
.  flag - if type found

.keywords: 

.seealso: 
*/
int DLGetTypeFromOptions(char *prefix,char *name,DLList list,int *type,int *flag)
{
  char sbuf[50];
  int  ierr,itype;
  
  PetscFunctionBegin;
  ierr = OptionsGetString(prefix,name, sbuf, 50,flag); CHKERRQ(ierr);
  if (*flag) {
    ierr = DLFindID( list, sbuf,&itype ); CHKERRQ(ierr);
    if (itype == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Invalid type name given");
    *(int *)type = itype;
  }
  PetscFunctionReturn(0);
}

