#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: nreg.c,v 1.23 1997/12/01 01:53:22 bsmith Exp $";
#endif
/*
         This provides a general mechanism to allow one to register
    new routines for many of the PETSc operations including KSP and PC.
*/
#include "petsc.h"
#include "src/sys/nreg.h"  

static int    NumberRegisters = 0;
static NRList **Registers[10];

/*
   This file contains a simple system to register functions by 
   name and number
 */

#undef __FUNC__  
#define __FUNC__ "NRCreate"
/*
  NRCreate - create a name registry

  Note:
  Use NRRegister to add names to the registry
*/
int NRCreate(NRList **fl )
{
  *fl            = (NRList *) PetscMalloc(sizeof(NRList));  if (!fl) PetscFunctionReturn(0);

  PetscFunctionBegin;
  (*fl)->head    = 0;
  (*fl)->tail    = 0;
  Registers[NumberRegisters++] = fl;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "NRDestroyAll"
/*
   NRDestroyAll - Destroys all registers. Should be called only 
           when you know none of the methods are needed.

@*/
int NRDestroyAll()
{
  int i;

  PetscFunctionBegin;
  for ( i=0; i<NumberRegisters; i++ ) {
    NRDestroy(*Registers[i]);
    *Registers[i] = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "NRRegister"
/*
   NRRegister - Given a routine and two ids (an int and a string), 
                save that routine in the specified registry
 
   Input Parameters:
.      fl       - pointer registry
.      id       - integer (or enum) for routine
.      name     - string for routine
.      routine  - routine
*/
int NRRegister( NRList *fl, int id, char *name, int (*routine)(void*) )
{
  FuncList *entry;

  PetscFunctionBegin;
  entry          = (FuncList*) PetscMalloc(sizeof(FuncList));CHKPTRQ(entry);
  entry->id      = id;
  entry->name    = (char *)PetscMalloc( PetscStrlen(name) + 1 ); CHKPTRQ(entry->name);
  PetscStrcpy( entry->name, name );
  entry->routine = routine;
  entry->next = 0;
  if (fl->tail) fl->tail->next = entry;
  else          fl->head       = entry;
  fl->tail = entry;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "NRDestroy"
/*
    NRDestroy - Destroy a list of registered routines

    Input Parameter:
.   fl   - pointer to list
*/
int NRDestroy(NRList * fl )
{
  FuncList *entry = fl->head, *next;

  PetscFunctionBegin;
  while (entry) {
    next = entry->next;
    PetscFree( entry->name );
    PetscFree( entry );
    entry = next;
  }
  PetscFree( fl );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "NRFindRoutine"
/*
    NRFindRoutine - given an id or name, find the matching routine

    Input Parameters:
.   fl   - pointer to list
.   id   - id (-1 for ignore)
.   name - name string.  (Null for ignore)

    Returns:
    pointer to function.  Null otherwise.
*/
int (*NRFindRoutine(NRList   * fl, int id, char *name ))(void *)
{
  FuncList *entry = fl->head;
  
  PetscFunctionBegin;
  while (entry) {
    if (id >= 0 && entry->id == id) PetscFunctionReturn(entry->routine);
    if (name && PetscStrcmp(name,entry->name) == 0) PetscFunctionReturn(entry->routine);
    entry = entry->next;
  }
  PetscFunctionReturn((int (*)(void*))0);
}

#undef __FUNC__  
#define __FUNC__ "NRFindID"
/*
    NRFindID - Given a name, find the corresponding id

    Input Parameters:
.   fl   - pointer to list
.   name - name string

    Returns:
    id.  -1 on failure.
*/
int NRFindID( NRList *fl, char *name )
{
  FuncList *entry = fl->head;

  PetscFunctionBegin;
  while (entry) {
    if (name && PetscStrcmp(name,entry->name) == 0) PetscFunctionReturn(entry->id);
    entry = entry->next;
  }
  PetscFunctionReturn(-1);
}

#undef __FUNC__  
#define __FUNC__ "NRFindName"
/*
    NRFindName - Given an id, find the corresponding name

    Input Parameters:
.   fl   - pointer to list
.   id   - id 

    Returns:
    Pointer to name; null on failure.
*/
char *NRFindName( NRList *fl, int id )
{
  FuncList *entry = fl->head;

  PetscFunctionBegin;
  while (entry) {
    if (id == entry->id) PetscFunctionReturn(entry->name);
    entry = entry->next;
  }
  PetscFunctionReturn((char *)0);
}

#undef __FUNC__  
#define __FUNC__ "NRFindFreeId"
/*
   Return an unused index.
 */
int NRFindFreeId( NRList *fl )
{
  FuncList *entry = fl->head;
  int      id;

  PetscFunctionBegin;
  id = -1;
  while (entry) {
    if (id < entry->id) id = entry->id;
    entry = entry->next;
  }
  PetscFunctionReturn(id + 1);
}

#undef __FUNC__  
#define __FUNC__ "NRPrintTypes"
/*
   NRPrintTypes - Prints the methods available.

   Input Parameters:
.  comm   - The communicator (usually MPI_COMM_WORLD)
.  fd     - file to print to, usually stdout
.  list   - list of types

*/
int NRPrintTypes(MPI_Comm comm,FILE *fd,char *prefix,char *name,NRList *list)
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
#define __FUNC__ "NRGetTypeFromOptions" 
/*
   NRGetTypeFromOptions

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
int NRGetTypeFromOptions(char *prefix,char *name,NRList *list,void *type,int *flag)
{
  char sbuf[50];
  int  ierr,itype;
  
  PetscFunctionBegin;
  ierr = OptionsGetString(prefix,name, sbuf, 50,flag); CHKERRQ(ierr);
  if (*flag) {
    itype = NRFindID( list, sbuf );
    if (itype == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Invalid type name given");
    *(int *)type = itype;
  }
  PetscFunctionReturn(0);
}

