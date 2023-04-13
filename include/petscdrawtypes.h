#ifndef PETSCDRAWTYPES_H
#define PETSCDRAWTYPES_H

/* SUBMANSEC = Draw */

/*J
    PetscDrawType - String with the name of a `PetscDraw` implementation, for example `PETSC_DRAW_X` is for X Windows.

   Level: beginner

.seealso: `PetscDrawSetType()`, `PetscDraw`, `PetscViewer`, `PetscDrawCreate()`, `PetscDrawRegister()`
J*/
typedef const char *PetscDrawType;
#define PETSC_DRAW_X     "x"
#define PETSC_DRAW_NULL  "null"
#define PETSC_DRAW_WIN32 "win32"
#define PETSC_DRAW_TIKZ  "tikz"
#define PETSC_DRAW_IMAGE "image"

/*S
     PetscDraw - Abstract PETSc object for graphics, often represents a window on the screen

   Level: beginner

.seealso: `PetscDrawCreate()`, `PetscDrawSetType()`, `PetscDrawType`
S*/
typedef struct _p_PetscDraw *PetscDraw;

/*S
   PetscDrawAxis - An object that manages X-Y axis for a `PetscDraw`

   Level: advanced

.seealso: `PetscDraw`, `PetscDrawAxisCreate()`, `PetscDrawAxisSetLimits()`, `PetscDrawAxisSetColors()`, `PetscDrawAxisSetLabels()`
S*/
typedef struct _p_PetscDrawAxis *PetscDrawAxis;

/*S
     PetscDrawLG - An object that manages drawing simple x-y plots

   Level: advanced

.seealso: `PetscDrawAxis`, `PetscDraw`,  `PetscDrawBar`, `PetscDrawHG`, `PetscDrawSP`, `PetscDrawAxisCreate()`, `PetscDrawLGCreate()`, `PetscDrawLGAddPoint()`
S*/
typedef struct _p_PetscDrawLG *PetscDrawLG;

/*S
     PetscDrawSP - An object that manages drawing scatter plots

   Level: advanced

.seealso: `PetscDrawAxis`, `PetscDraw`, `PetscDrawLG`,  `PetscDrawBar`, `PetscDrawHG`, `PetscDrawSPCreate()`
S*/
typedef struct _p_PetscDrawSP *PetscDrawSP;

/*S
     PetscDrawHG - An object that manages drawing histograms

   Level: advanced

.seealso: `PetscDrawAxis`, `PetscDraw`, `PetscDrawLG`, `PetscDrawBar`, `PetscDrawSP`, `PetscDrawHGCreate()`
S*/
typedef struct _p_PetscDrawHG *PetscDrawHG;

/*S
     PetscDrawBar - An object that manages drawing bar graphs

   Level: advanced

.seealso: `PetscDrawAxis`, `PetscDraw`, `PetscDrawLG`, `PetscDrawHG`, `PetscDrawSP`, `PetscDrawBarCreate()`
S*/
typedef struct _p_PetscDrawBar *PetscDrawBar;

#endif
