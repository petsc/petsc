/*
       Abstract data structure and functions for graphics.
*/

#ifndef PETSCDRAWIMPL_H
#define PETSCDRAWIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdraw.h>

PETSC_EXTERN PetscBool      PetscDrawRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscDrawRegisterAll(void);

struct _PetscDrawOps {
  PetscErrorCode (*setdoublebuffer)(PetscDraw);
  PetscErrorCode (*flush)(PetscDraw);
  PetscErrorCode (*line)(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int);
  PetscErrorCode (*linesetwidth)(PetscDraw, PetscReal);
  PetscErrorCode (*linegetwidth)(PetscDraw, PetscReal *);
  PetscErrorCode (*point)(PetscDraw, PetscReal, PetscReal, int);
  PetscErrorCode (*pointsetsize)(PetscDraw, PetscReal);
  PetscErrorCode (*string)(PetscDraw, PetscReal, PetscReal, int, const char[]);
  PetscErrorCode (*stringvertical)(PetscDraw, PetscReal, PetscReal, int, const char[]);
  PetscErrorCode (*stringsetsize)(PetscDraw, PetscReal, PetscReal);
  PetscErrorCode (*stringgetsize)(PetscDraw, PetscReal *, PetscReal *);
  PetscErrorCode (*setviewport)(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal);
  PetscErrorCode (*clear)(PetscDraw);
  PetscErrorCode (*rectangle)(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int, int, int, int);
  PetscErrorCode (*triangle)(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, int, int, int);
  PetscErrorCode (*ellipse)(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int);
  PetscErrorCode (*getmousebutton)(PetscDraw, PetscDrawButton *, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
  PetscErrorCode (*pause)(PetscDraw);
  PetscErrorCode (*beginpage)(PetscDraw);
  PetscErrorCode (*endpage)(PetscDraw);
  PetscErrorCode (*getpopup)(PetscDraw, PetscDraw *);
  PetscErrorCode (*settitle)(PetscDraw, const char[]);
  PetscErrorCode (*checkresizedwindow)(PetscDraw);
  PetscErrorCode (*resizewindow)(PetscDraw, int, int);
  PetscErrorCode (*destroy)(PetscDraw);
  PetscErrorCode (*view)(PetscDraw, PetscViewer);
  PetscErrorCode (*getsingleton)(PetscDraw, PetscDraw *);
  PetscErrorCode (*restoresingleton)(PetscDraw, PetscDraw *);
  PetscErrorCode (*save)(PetscDraw);
  PetscErrorCode (*getimage)(PetscDraw, unsigned char[][3], unsigned int *, unsigned int *, unsigned char *[]);
  PetscErrorCode (*setcoordinates)(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal);
  PetscErrorCode (*arrow)(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int);
  PetscErrorCode (*coordinatetopixel)(PetscDraw, PetscReal, PetscReal, int *, int *);
  PetscErrorCode (*pixeltocoordinate)(PetscDraw, int, int, PetscReal *, PetscReal *);
  PetscErrorCode (*pointpixel)(PetscDraw, int, int, int);
  PetscErrorCode (*boxedstring)(PetscDraw, PetscReal, PetscReal, int, int, const char[], PetscReal *, PetscReal *);
  PetscErrorCode (*setvisible)(PetscDraw, PetscBool);
};

struct _p_PetscDraw {
  PETSCHEADER(struct _PetscDrawOps);
  PetscReal           pause; /* sleep time after a synchronized flush */
  PetscReal           port_xl, port_yl, port_xr, port_yr;
  PetscReal           coor_xl, coor_yl, coor_xr, coor_yr;
  PetscReal           currentpoint_x[20], currentpoint_y[20];
  PetscReal           boundbox_xl, boundbox_yl, boundbox_xr, boundbox_yr; /* need to have this for each current point? */
  PetscInt            currentpoint;
  PetscDrawMarkerType markertype;
  char               *title;
  char               *display;
  PetscDraw           popup;
  int                 x, y, h, w;
  char               *savefilename;
  char               *saveimageext;
  char               *savemovieext;
  PetscInt            savefilecount;
  PetscBool           savesinglefile;
  PetscInt            savemoviefps;
  char               *savefinalfilename;
  PetscBool           saveonclear; /* save a new image for every PetscDrawClear() called */
  PetscBool           saveonflush; /* save a new image for every PetscDrawFlush() called */
  void               *data;
};

/* Contains the data structure for plotting several line
 * graphs in a window with an axis. This is intended for line
 * graphs that change dynamically by adding more points onto
 * the end of the X axis.
 */
struct _p_PetscDrawLG {
  PETSCHEADER(int);
  PetscErrorCode (*destroy)(PetscDrawLG);
  PetscErrorCode (*view)(PetscDrawLG, PetscViewer);
  int           len, loc;
  PetscDraw     win;
  PetscDrawAxis axis;
  PetscReal     xmin, xmax, ymin, ymax, *x, *y;
  int           nopts, dim, *colors;
  PetscBool     use_markers;
  char        **legend;
};
#define PETSC_DRAW_LG_CHUNK_SIZE 100

struct _p_PetscDrawAxis {
  PETSCHEADER(int);
  PetscReal xlow, ylow, xhigh, yhigh;                         /* User - coord limits */
  PetscErrorCode (*ylabelstr)(PetscReal, PetscReal, char **); /* routines to generate labels */
  PetscErrorCode (*xlabelstr)(PetscReal, PetscReal, char **);
  PetscErrorCode (*xticks)(PetscReal, PetscReal, int, int *, PetscReal *, int);
  PetscErrorCode (*yticks)(PetscReal, PetscReal, int, int *, PetscReal *, int);
  /* location and size of ticks */
  PetscDraw win;
  int       ac, tc, cc; /* axis,tick, character color */
  char     *xlabel, *ylabel, *toplabel;
  PetscBool hold;
};

PETSC_INTERN PetscErrorCode PetscADefTicks(PetscReal, PetscReal, int, int *, PetscReal *, int);
PETSC_INTERN PetscErrorCode PetscADefLabel(PetscReal, PetscReal, char **);
PETSC_INTERN PetscErrorCode PetscAGetNice(PetscReal, PetscReal, int, PetscReal *);
PETSC_INTERN PetscErrorCode PetscAGetBase(PetscReal, PetscReal, int, PetscReal *, int *);

PETSC_INTERN PetscErrorCode PetscStripe0(char *);
PETSC_INTERN PetscErrorCode PetscStripAllZeros(char *);
PETSC_INTERN PetscErrorCode PetscStripTrailingZeros(char *);
PETSC_INTERN PetscErrorCode PetscStripInitialZero(char *);
PETSC_INTERN PetscErrorCode PetscStripZeros(char *);
PETSC_INTERN PetscErrorCode PetscStripZerosPlus(char *);

struct _p_PetscDrawBar {
  PETSCHEADER(int);
  PetscErrorCode (*destroy)(PetscDrawSP);
  PetscErrorCode (*view)(PetscDrawSP, PetscViewer);
  PetscDraw     win;
  PetscDrawAxis axis;
  PetscReal     ymin, ymax;
  int           numBins;
  PetscReal    *values;
  int           color;
  char        **labels;
  PetscBool     sort;
  PetscReal     sorttolerance;
};

struct _p_PetscDrawSP {
  PETSCHEADER(int);
  PetscErrorCode (*destroy)(PetscDrawSP);
  PetscErrorCode (*view)(PetscDrawSP, PetscViewer);
  int           len, loc;
  PetscDraw     win;
  PetscDrawAxis axis;
  PetscReal     xmin, xmax, ymin, ymax, *x, *y;
  PetscReal     zmax, zmin, *z;
  int           nopts, dim;
  PetscBool     colorized;
};
#define PETSC_DRAW_SP_CHUNK_SIZE 100

#endif /* PETSCDRAWIMPL_H */
