/*
     PetscViewers are objects where other objects can be looked at or stored.
*/

#ifndef PETSCVIEWERTYPES_H
#define PETSCVIEWERTYPES_H

/* SUBMANSEC = Viewer */

/*S
     PetscViewer - Abstract PETSc object that helps view (in ASCII, binary, graphically etc)
          PETSc objects

   Level: beginner

.seealso: `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerType`
S*/
typedef struct _p_PetscViewer *PetscViewer;

#endif
