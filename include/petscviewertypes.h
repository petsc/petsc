/*
     PetscViewers are objects where other objects can be looked at or stored.
*/

#if !defined(PETSCVIEWERTYPES_H)
#define PETSCVIEWERTYPES_H

/* SUBMANSEC = Viewer */

/*S
     PetscViewer - Abstract PETSc object that helps view (in ASCII, binary, graphically etc)
         other PETSc objects

   Level: beginner

.seealso: `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerType`
S*/
typedef struct _p_PetscViewer* PetscViewer;

#endif
