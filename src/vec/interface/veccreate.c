#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vecserialize.c,v 1.10 2000/01/10 03:18:14 knepley Exp $";
#endif

#include "src/vec/vecimpl.h"      /*I "petscvec.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "VecCreate"
/*@C
  VecCreate - Creates an empty vector object. The type can then be set with VecSetType(),
  or VecSetFromOptions().

   If you never  call VecSetType() or VecSetFromOptions() it will generate an 
   error when you try to use the vector.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the vector object

  Output Parameter:
. vec  - The vector object

  Level: beginner

.keywords: vector, create
.seealso: VecSetType(), VecSetSizes(), VecCreateMPIWithArray(), VecCreateMPI(), VecDuplicate(),
          VecDuplicateVecs(), VecCreateGhost(), VecCreateSeq(), VecPlaceArray()
@*/
int VecCreate(MPI_Comm comm, Vec *vec)
{
  Vec v;
  int ierr;

  PetscFunctionBegin;
  PetscValidPointer(vec);
  *vec = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(PETSC_NULL);                                                                CHKERRQ(ierr);
#endif

  PetscHeaderCreate(v, _p_Vec, struct _VecOps, VEC_COOKIE, -1, "Vec", comm, VecDestroy, VecView);
  PetscLogObjectCreate(v);
  PetscLogObjectMemory(v, sizeof(struct _p_Vec));
  ierr = PetscMemzero(v->ops, sizeof(struct _VecOps));                                                    CHKERRQ(ierr);
  v->bops->publish  = PETSC_NULL /* VecPublish_Petsc */;
  v->type_name      = PETSC_NULL;
  v->serialize_name = PETSC_NULL;

  v->map          = PETSC_NULL;
  v->data         = PETSC_NULL;
  v->n            = -1;
  v->N            = -1;
  v->bs           = -1;
  v->mapping      = PETSC_NULL;
  v->bmapping     = PETSC_NULL;
  v->array_gotten = PETSC_FALSE;
  v->petscnative  = PETSC_FALSE;
  v->esivec       = PETSC_NULL;

  *vec = v; 
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "VecSerialize"
/*@ 
  VecSerialize - This function stores or recreates a vector using a viewer for a binary file.

  Collective on MPI_Comm

  Input Parameters:
+ comm   - The communicator for the vector object
. viewer - The viewer context
- store  - This flag is PETSC_TRUE is data is being written, otherwise it will be read

  Output Parameter:
. v      - The vector

  Level: beginner

.keywords: vector, serialize
.seealso: GridSerialize()
@*/
int VecSerialize(MPI_Comm comm, Vec *v, PetscViewer viewer, PetscTruth store)
{
  int      (*serialize)(MPI_Comm, Vec *, PetscViewer, PetscTruth);
  int        fd, len;
  char      *name;
  PetscTruth match;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidPointer(v);

  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &match);                             CHKERRQ(ierr);
  if (match == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Must be binary viewer");
  ierr = PetscViewerBinaryGetDescriptor(viewer, &fd);                                                     CHKERRQ(ierr);

  if (VecSerializeRegisterAllCalled == PETSC_FALSE) {
    ierr = VecSerializeRegisterAll(PETSC_NULL);                                                           CHKERRQ(ierr);
  }
  if (VecSerializeList == PETSC_NULL) SETERRQ(PETSC_ERR_ARG_CORRUPT, "Could not find table of methods");

  if (store) {
    PetscValidHeaderSpecific(*v, VEC_COOKIE);
    ierr = PetscStrlen((*v)->class_name, &len);                                                           CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &len,                  1,   PETSC_INT,  0);                               CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  (*v)->class_name,     len, PETSC_CHAR, 0);                               CHKERRQ(ierr);
    ierr = PetscStrlen((*v)->serialize_name, &len);                                                       CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &len,                  1,   PETSC_INT,  0);                               CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  (*v)->serialize_name, len, PETSC_CHAR, 0);                               CHKERRQ(ierr);
    ierr = PetscFListFind(comm, VecSerializeList, (*v)->serialize_name, (void (**)(void)) &serialize);    CHKERRQ(ierr);
    if (!serialize) SETERRQ(PETSC_ERR_ARG_WRONG, "Type cannot be serialized");
    ierr = (*serialize)(comm, v, viewer, store);                                                          CHKERRQ(ierr);
  } else {
    ierr = PetscBinaryRead(fd, &len,    1,   PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = PetscMalloc((len+1) * sizeof(char), &name);                                                    CHKERRQ(ierr);
    name[len] = 0;
    ierr = PetscBinaryRead(fd,  name,   len, PETSC_CHAR);                                                 CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "Vec", &match);                                                              CHKERRQ(ierr);
    ierr = PetscFree(name);                                                                               CHKERRQ(ierr);
    if (match == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Non-vector object");
    /* Dispatch to the correct routine */
    ierr = PetscBinaryRead(fd, &len,    1,   PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = PetscMalloc((len+1) * sizeof(char), &name);                                                    CHKERRQ(ierr);
    name[len] = 0;
    ierr = PetscBinaryRead(fd,  name,   len, PETSC_CHAR);                                                 CHKERRQ(ierr);
    ierr = PetscFListFind(comm, VecSerializeList, name, (void (**)(void)) &serialize);                    CHKERRQ(ierr);
    if (!serialize) SETERRQ(PETSC_ERR_ARG_WRONG, "Type cannot be serialized");
    ierr = (*serialize)(comm, v, viewer, store);                                                          CHKERRQ(ierr);
    ierr = PetscStrfree((*v)->serialize_name);                                                            CHKERRQ(ierr);
    (*v)->serialize_name = name;
  }
  
  PetscFunctionReturn(0);
}
