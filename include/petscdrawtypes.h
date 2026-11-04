#pragma once

#include <petscsystypes.h>

/* MANSEC = Sys */
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

.seealso: `PetscDrawAxis`, `PetscDraw`, `PetscDrawBar`, `PetscDrawHG`, `PetscDrawSP`, `PetscDrawAxisCreate()`, `PetscDrawLGCreate()`, `PetscDrawLGAddPoint()`
S*/
typedef struct _p_PetscDrawLG *PetscDrawLG;

/*S
   PetscDrawSP - An object that manages drawing scatter plots

   Level: advanced

.seealso: `PetscDrawAxis`, `PetscDraw`, `PetscDrawLG`, `PetscDrawBar`, `PetscDrawHG`, `PetscDrawSPCreate()`
S*/
typedef struct _p_PetscDrawSP *PetscDrawSP;

/*S
   PetscDrawHG - An object that manages drawing histograms

   Level: advanced

   Note:
   Use a series of calls to `PetscDrawHGAddValue()` to create a standard histogram <https://en.wikipedia.org/wiki/Histogram>, where the bins have integer counts.  Use calls to `PetscDrawHGAddWeightedValue()` to create a histogram with non-integer bin heights, such as the following <https://mathematica.stackexchange.com/questions/103928/histogram-from-relative-frequency-data>

.seealso: `PetscDrawAxis`, `PetscDraw`, `PetscDrawLG`, `PetscDrawBar`, `PetscDrawSP`, `PetscDrawHGCreate()`, `PetscDrawHGAddValue()`, `PetscDrawHGAddWeightedValue()`
S*/
typedef struct _p_PetscDrawHG *PetscDrawHG;

/*S
   PetscDrawBar - An object that manages drawing bar graphs

   Level: advanced

.seealso: `PetscDrawAxis`, `PetscDraw`, `PetscDrawLG`, `PetscDrawHG`, `PetscDrawSP`, `PetscDrawBarCreate()`
S*/
typedef struct _p_PetscDrawBar *PetscDrawBar;

/*
   Number of basic colors in the draw routines, the others are used
   for a uniform colormap.
*/
#define PETSC_DRAW_BASIC_COLORS 33

#define PETSC_DRAW_ROTATE      -1 /* will rotate through the colors, start with 2 */
#define PETSC_DRAW_WHITE       0
#define PETSC_DRAW_BLACK       1
#define PETSC_DRAW_RED         2
#define PETSC_DRAW_GREEN       3
#define PETSC_DRAW_CYAN        4
#define PETSC_DRAW_BLUE        5
#define PETSC_DRAW_MAGENTA     6
#define PETSC_DRAW_AQUAMARINE  7
#define PETSC_DRAW_FORESTGREEN 8
#define PETSC_DRAW_ORANGE      9
#define PETSC_DRAW_VIOLET      10
#define PETSC_DRAW_BROWN       11
#define PETSC_DRAW_PINK        12
#define PETSC_DRAW_CORAL       13
#define PETSC_DRAW_GRAY        14
#define PETSC_DRAW_YELLOW      15

#define PETSC_DRAW_GOLD            16
#define PETSC_DRAW_LIGHTPINK       17
#define PETSC_DRAW_MEDIUMTURQUOISE 18
#define PETSC_DRAW_KHAKI           19
#define PETSC_DRAW_DIMGRAY         20
#define PETSC_DRAW_YELLOWGREEN     21
#define PETSC_DRAW_SKYBLUE         22
#define PETSC_DRAW_DARKGREEN       23
#define PETSC_DRAW_NAVYBLUE        24
#define PETSC_DRAW_SANDYBROWN      25
#define PETSC_DRAW_CADETBLUE       26
#define PETSC_DRAW_POWDERBLUE      27
#define PETSC_DRAW_DEEPPINK        28
#define PETSC_DRAW_THISTLE         29
#define PETSC_DRAW_LIMEGREEN       30
#define PETSC_DRAW_LAVENDERBLUSH   31
#define PETSC_DRAW_PLUM            32
#define PETSC_DRAW_MAXCOLOR        256

#define PETSC_DRAW_FULL_SIZE    -3
#define PETSC_DRAW_HALF_SIZE    -4
#define PETSC_DRAW_THIRD_SIZE   -5
#define PETSC_DRAW_QUARTER_SIZE -6

/*E
   PetscDrawMarkerType - How a "mark" is indicate in a figure

   Values:
+  `PETSC_MARKER_CROSS`  - a small pixel based x symbol or the character x if that is not available
.  `PETSC_MARKER_PLUS`   - a small pixel based + symbol or the character + if that is not available
.  `PETSC_MARKER_CIRCLE` - a small pixel based circle symbol or the character o if that is not available
-  `PETSC_MARKER_POINT`  - the make obtained with `PetscDrawPoint()`

   Level: intermediate

.seealso: `PetscDraw`, `PetscDrawMarker()`, `PetscDrawSetMarkerType()`
E*/
typedef enum {
  PETSC_DRAW_MARKER_CROSS,
  PETSC_DRAW_MARKER_POINT,
  PETSC_DRAW_MARKER_PLUS,
  PETSC_DRAW_MARKER_CIRCLE
} PetscDrawMarkerType;
PETSC_EXTERN const char *const PetscDrawMarkerTypes[];

/*E
   PetscDrawButton - Used to determine which button was pressed

   Values:
+  `PETSC_BUTTON_NONE`        - no button was pressed
.  `PETSC_BUTTON_LEFT`        - the left button
.  `PETSC_BUTTON_CENTER`      - the center button
.  `PETSC_BUTTON_RIGHT`       - the right button
.  `PETSC_BUTTON_WHEEL_UP`    - the wheel was moved up
.  `PETSC_BUTTON_WHEEL_DOWN`  - the wheel was moved down
.  `PETSC_BUTTON_LEFT_SHIFT`  - the left button and the shift key
.  `PETSC_BUTTON_CENTER_SHIFT`- the center button and the shift key
-  `PETSC_BUTTON_RIGHT_SHIFT` - the right button and the shift key

   Level: intermediate

.seealso: `PetscDraw`, `PetscDrawGetMouseButton()`
E*/
typedef enum {
  PETSC_BUTTON_NONE,
  PETSC_BUTTON_LEFT,
  PETSC_BUTTON_CENTER,
  PETSC_BUTTON_RIGHT,
  PETSC_BUTTON_WHEEL_UP,
  PETSC_BUTTON_WHEEL_DOWN,
  PETSC_BUTTON_LEFT_SHIFT,
  PETSC_BUTTON_CENTER_SHIFT,
  PETSC_BUTTON_RIGHT_SHIFT
} PetscDrawButton;

/*S
   PetscDrawViewPorts - Object representing subwindows in a `PetscDraw` object

   Level: intermediate

.seealso: `PetscDraw`, `PetscDrawViewPortsCreate()`, `PetscDrawViewPortsSet()`
S*/
typedef struct {
  PetscInt   nports;
  PetscReal *xl;
  PetscReal *xr;
  PetscReal *yl;
  PetscReal *yr;
  PetscDraw  draw;
  PetscReal  port_xl, port_yl, port_xr, port_yr; /* original port of parent PetscDraw */
} PetscDrawViewPorts;
