#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mapcreate.c,v 1.1 1999/06/21 02:03:50 knepley Exp $";
#endif

#include "src/vec/vecimpl.h"      /*I "petscvec.h"  I*/


#undef __FUNCT__  
#define __FUNCT__ "PetscMapCreate"
/*@C
  PetscMapCreate - Creates an empty map object. The type can then be set with PetscMapSetType().

  Collective on MPI_Comm
 
  Input Parameter:
. comm - The MPI communicator for the map object 

  Output Parameter:
. map  - The map object

  Level: beginner

.keywords: PetscMap, create
.seealso: PetscMapDestroy(), PetscMapGetLocalSize(), PetscMapGetSize(), PetscMapGetGlobalRange(), PetscMapGetLocalRange()
@*/ 
int PetscMapCreate(MPI_Comm comm, PetscMap *map)
{
  PetscMap m;
  int      ierr;

  PetscFunctionBegin;
  PetscValidPointer(map);
  *map = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(PETSC_NULL);                                                                CHKERRQ(ierr);
#endif

  PetscHeaderCreate(m, _p_PetscMap, struct _PetscMapOps, MAP_COOKIE, -1, "PetscMap", comm, PetscMapDestroy, PETSC_NULL);
  PetscLogObjectCreate(m);
  PetscLogObjectMemory(m, sizeof(struct _p_PetscMap));
  ierr = PetscMemzero(m->ops, sizeof(struct _PetscMapOps));                                               CHKERRQ(ierr);
  m->bops->publish  = PETSC_NULL /* PetscMapPublish_Petsc */;
  m->type_name      = PETSC_NULL;
  m->serialize_name = PETSC_NULL;

  m->n      = -1;
  m->N      = -1;
  m->rstart = -1;
  m->rend   = -1;
  m->range  = PETSC_NULL;

  *map = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMapSerialize"
/*@ 
  PetscMapSerialize - This function stores or recreates a map using a viewer for a binary file.

  Collective on MPI_Comm

  Input Parameters:
+ comm   - The communicator for the map object
. viewer - The viewer context
- store  - This flag is PETSC_TRUE is data is being written, otherwise it will be read

  Output Parameter:
. map    - The map

  Level: beginner

.keywords: maptor, serialize
.seealso: GridSerialize()
@*/
int PetscMapSerialize(MPI_Comm comm, PetscMap *map, PetscViewer viewer, PetscTruth store)
{
  int      (*serialize)(MPI_Comm, PetscMap *, PetscViewer, PetscTruth);
  int        fd, len;
  char      *name;
  PetscTruth match;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidPointer(map);

  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &match);                             CHKERRQ(ierr);
  if (match == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Must be binary viewer");
  ierr = PetscViewerBinaryGetDescriptor(viewer, &fd);                                                     CHKERRQ(ierr);

  if (PetscMapSerializeRegisterAllCalled == PETSC_FALSE) {
    ierr = PetscMapSerializeRegisterAll(PETSC_NULL);                                                      CHKERRQ(ierr);
  }
  if (PetscMapSerializeList == PETSC_NULL) SETERRQ(PETSC_ERR_ARG_CORRUPT, "Could not find table of methods");

  if (store) {
    PetscValidHeaderSpecific(*map, MAP_COOKIE);
    ierr = PetscStrlen((*map)->class_name, &len);                                                         CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &len,                    1,   PETSC_INT,  0);                             CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  (*map)->class_name,     len, PETSC_CHAR, 0);                             CHKERRQ(ierr);
    ierr = PetscStrlen((*map)->serialize_name, &len);                                                     CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &len,                    1,   PETSC_INT,  0);                             CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  (*map)->serialize_name, len, PETSC_CHAR, 0);                             CHKERRQ(ierr);
    ierr = PetscFListFind(comm, PetscMapSerializeList, (*map)->serialize_name, (void (**)(void)) &serialize);  CHKERRQ(ierr);
    if (!serialize) SETERRQ(PETSC_ERR_ARG_WRONG, "Type cannot be serialized");
    ierr = (*serialize)(comm, map, viewer, store);                                                        CHKERRQ(ierr);
  } else {
    ierr = PetscBinaryRead(fd, &len,    1,   PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = PetscMalloc((len+1) * sizeof(char), &name);                                                    CHKERRQ(ierr);
    name[len] = 0;
    ierr = PetscBinaryRead(fd,  name,   len, PETSC_CHAR);                                                 CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "PetscMap", &match);                                                         CHKERRQ(ierr);
    ierr = PetscFree(name);                                                                               CHKERRQ(ierr);
    if (match == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Non-map object");
    /* Dispatch to the correct routine */
    ierr = PetscBinaryRead(fd, &len,    1,   PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = PetscMalloc((len+1) * sizeof(char), &name);                                                    CHKERRQ(ierr);
    name[len] = 0;
    ierr = PetscBinaryRead(fd,  name,   len, PETSC_CHAR);                                                 CHKERRQ(ierr);
    ierr = PetscFListFind(comm, PetscMapSerializeList, name, (void (**)(void)) &serialize);               CHKERRQ(ierr);
    if (!serialize) SETERRQ(PETSC_ERR_ARG_WRONG, "Type cannot be serialized");
    ierr = (*serialize)(comm, map, viewer, store);                                                        CHKERRQ(ierr);
    ierr = PetscStrfree((*map)->serialize_name);                                                          CHKERRQ(ierr);
    (*map)->serialize_name = name;
  }
  
  PetscFunctionReturn(0);
}
