#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bilinearserialize.c,v 1.4 2000/01/10 03:12:48 knepley Exp $";
#endif

#include "src/bilinear/bilinearimpl.h"      /*I "bilinear.h"  I*/

#undef  __FUNC__
#define __FUNC__ "BilinearSerialize"
/*@ 
  BilinearSerialize - This function stores or recreates a bilinear operator
  using a viewer for a binary file.

  Collective on comm

  Input Parameters:
+ comm   - The communicator for the bilinear operator object
. viewer - The viewer context
- store  - This flag is PETSC_TRUE is data is being written, otherwise it will be read

  Output Parameter:
. B      - The bilinear operator

  Level: beginner

.keywords: serialize, Bilinear
.seealso: GridSerialize()
@*/
int BilinearSerialize(MPI_Comm comm, Bilinear *B, PetscViewer viewer, PetscTruth store)
{
  int      (*serialize)(MPI_Comm, Bilinear *, PetscViewer, PetscTruth);
  int        fd, len;
  char      *name;
  PetscTruth match;
  int        ierr;

  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidPointer(B);

  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &match);                             CHKERRQ(ierr);
  if (match == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Must be binary viewer");
  ierr = PetscViewerBinaryGetDescriptor(viewer, &fd);                                                     CHKERRQ(ierr);

  if (store) {
    PetscValidHeaderSpecific(*B, BILINEAR_COOKIE);
    ierr = PetscStrlen((*B)->class_name, &len);                                                           CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &len,                  1,   PETSC_INT,  0);                               CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  (*B)->class_name,     len, PETSC_CHAR, 0);                               CHKERRQ(ierr);
    ierr = PetscStrlen((*B)->serialize_name, &len);                                                       CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd, &len,                  1,   PETSC_INT,  0);                               CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,  (*B)->serialize_name, len, PETSC_CHAR, 0);                               CHKERRQ(ierr);
    ierr = PetscFListFind(comm, BilinearSerializeList, (*B)->serialize_name, (void (**)(void)) &serialize);CHKERRQ(ierr);
    if (!serialize) SETERRQ(PETSC_ERR_ARG_WRONG, "Type cannot be serialized");
    ierr = (*serialize)(comm, B, viewer, store);                                                          CHKERRQ(ierr);
  } else {
    ierr = PetscBinaryRead(fd, &len,    1,   PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = PetscMalloc((len+1) * sizeof(char), &name);                                                    CHKERRQ(ierr);
    name[len] = 0;
    ierr = PetscBinaryRead(fd,  name,   len, PETSC_CHAR);                                                 CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "Bilinear", &match);                                                         CHKERRQ(ierr);
    ierr = PetscFree(name);                                                                               CHKERRQ(ierr);
    if (match == PETSC_FALSE) SETERRQ(PETSC_ERR_ARG_WRONG, "Non-bilinear object");
    /* Dispatch to the correct routine */
    if (!BilinearSerializeRegisterAllCalled) {
      ierr = BilinearSerializeRegisterAll(PETSC_NULL);                                                    CHKERRQ(ierr);
    }
    if (!BilinearSerializeList) SETERRQ(PETSC_ERR_ARG_CORRUPT, "Could not find table of methods");
    ierr = PetscBinaryRead(fd, &len,    1,   PETSC_INT);                                                  CHKERRQ(ierr);
    ierr = PetscMalloc((len+1) * sizeof(char), &name);                                                    CHKERRQ(ierr);
    name[len] = 0;
    ierr = PetscBinaryRead(fd,  name,   len, PETSC_CHAR);                                                 CHKERRQ(ierr);
    ierr = PetscFListFind(comm, BilinearSerializeList, name, (void (**)(void)) &serialize);               CHKERRQ(ierr);
    if (!serialize) SETERRQ(PETSC_ERR_ARG_WRONG, "Type cannot be serialized");
    ierr = (*serialize)(comm, B, viewer, store);                                                          CHKERRQ(ierr);
    ierr = PetscStrfree((*B)->serialize_name);                                                            CHKERRQ(ierr);
    (*B)->serialize_name = name;
  }

  PetscFunctionReturn(0);
}
