
/*
   This include file defines an application-specific data structure for
   use in the potential flow code.
  */

#include "snes.h"
#include "draw.h"
#include "da.h"
#include "dfvec.h"
#include "color.h"
#include <math.h>
#include <stdio.h>
#define Min(a,b) ( ((a)<=(b)) ? a : b )
#define Max(a,b) ( ((a)>=(b)) ? a : b )

#define pi       3.14159265
#define epsfcn   0.000000031622776
#define height   0.1
#define pr_infty 1
#define Pinf     1
#define gama     1.6666666667

/* Application-specific structure */
typedef struct {
      Vec         X, F;           /* solution, function vectors */
      Vec         Fcopy;          /* copy of function vector */
      Vec         localX, localF; /* ghosted local vectors */
      Vec         localMach;      /* ghosted local Mach number */
      Vec         globalMach;     /* global Mach number */
      Vec         globalPressure; /* global pressure distribution along airfoil */
      Vec         localDensity;   /* local densities */
      Vec         x2, localXbak;  /* work vectors for computing Jacobian */
      Vec         jj2, localFbak; /* work vectors for computing Jacobian */
      Mat         J, Jmf;         /* Jacobian matrix, matrix-free context */
      DA          da;             /* distributed array */
      double      Qinf;           /* velocity at infinity */
      double      M;              /* Mach number at infinity */
      int         machflag;       /* flag - 1 to compute local Mach number */
      int         matrix_free;    /* flag - 1 to use matrix-free method */
      int         jfreq;          /* frequency of evaluating Jacobian */
      MPI_Comm    comm;           /* communicator */
      int         rank, size;     /* rank, size */

  /* Grid data */
      int         mx, my;         /* fine grid size in x- and y-directions */
      double      hx, hy;         /* grid spacing in x- and y-directions */
      double      xx0, xx1;       /* physical domain (xx0,yy0), (xx1,yy0) */
      double      yy0, yy1;       /*                 (xx1,yy1), (xx0,yy1) */
      int         nc;             /* number of degrees of freedom per node */
      int         xs, ys;         /* local starting grid points (no ghost points) */
      int         xe, ye;         /* local ending grid points (no ghost points) */
      int         xm, ym;         /* local widths (no ghost points) */
      int         Xs, Ys;         /* local starting grid points (including ghost points) */
      int         Xe, Ye;         /* local ending grid points (including ghost points) */
      int         Xm, Ym;         /* local widths (including ghost points) */
      int         ldim, gdim;     /* local and global vector dimensions */
      Coloring    *color;         /* coloring context */

  /* These parameters are NOT currently used, intended for multigrid version */
      int         jacobi_count;   /* count for fine Jacobian */
      int         coarse_count;   /* count for coarse Jacobian */
      int         coarse;         /* coarse grid flag */
      int         cmx;            /* coarse grid size in in x-direction */
      int         cmy;            /* coarse grid size in y-direction */
      Vec         CX;             /* coarse grid points */
      Vec         CF;             /* coarse function */
} AppCtx;

/* User-defined routines */
int Jacobian_PotentialFlow(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int InitialGuess_PotentialFlow(AppCtx*,Vec);
int Function_PotentialFlow(SNES,Vec,Vec,void*);
int UserSetJacobian(SNES,AppCtx*);
int UserMonitor(SNES,int,double,void*);
int EvaluateFunction(AppCtx*,double*,int,int,double*,double*);
int MySparseFunction(Vec,int,AppCtx*,Vec);
int InnerSparseFunction(int,int,Vec,int,AppCtx*,Vec);

