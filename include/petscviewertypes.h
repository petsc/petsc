/*
     PetscViewers are objects where other objects can be looked at or stored.
*/

#pragma once

/* MANSEC = Sys */
/* SUBMANSEC = Viewer */

/*S
  PetscViewer - Abstract PETSc object for displaying in ASCII, saving to a binary file, graphically displaying, etc.
                PETSc objects and their data

  Level: beginner

  Notes:
  Each PETSc class, for example `Vec`, has a viewer method associated with that class, for example `VecView()`, that can be used
  to view, display, store to a file information about that object, etc. Each class also has a method that uses
  the options database to view the object, for example `VecViewFromOptions()`.

  See `PetscViewerType` for a list of all `PetscViewer` types.

.seealso: [](sec_viewers), `PetscViewerType`, `PETSCVIEWERASCII`, `PetscViewerCreate()`, `PetscViewerSetType()`,
          `VecView()`, `VecViewFromOptions()`, `PetscObjectView()`
S*/
typedef struct _p_PetscViewer *PetscViewer;
