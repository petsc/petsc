
#if !defined(__PFULL_H)
#define __PFULL_H

/*
   Defines the application-specific data structure for
   use in the potential flow code.
*/

#include "snes.h"
#include "draw.h"
#include "da.h"
#include "dfvec.h"
#include <math.h>
#include <stdio.h>

#define pi       3.14159265
#define epsfcn   0.000000031622776
#define height   0.1
#define pr_infty 1
#define Pinf     1
#define gama     1.6666666667

/*
       This information is specific to a particular grid level 
*/
typedef struct {
  Vec           globalX;        /* solution */
  Vec           localX;         /* ghosted local solution vector */
  Vec           globalF;        /* where nonlinear function is evaluated */
  Vec           localF;         /* ghosted current nonlinear function value */
  Vec           globalMach;     /* global Mach number */
  Vec           localMach;      /* ghosted local Mach number */
  Vec           globalPressure; /* global pressure distribution along airfoil */
  Vec           localDensity;   /* local densities */

  Vec           x2, localXbak;  /* work vectors for computing Jacobian */
  Vec           jj2, localFbak; /* work vectors for computing Jacobian */

  Vec           *vec_g,*vec_l;  /* array that holds the work vectors */

  Mat           J;              /* Jacobian/Jacobian preconditioner matrix */
  MatFDColoring fdcoloring;     /* context used to compute Jacobian via finite differences. */

  DA            da;             /* distributed array */
  int           mx, my;         /* grid size in x- and y-directions */
  double        hx, hy;         /* grid spacing in x- and y-directions */
  int           xs, ys;         /* local starting grid points (no ghost points) */
  int           xe, ye;         /* local ending grid points (no ghost points) */
  int           xm, ym;         /* local widths (no ghost points) */
  int           Xs, Ys;         /* local starting grid points (including ghost points) */
  int           Xe, Ye;         /* local ending grid points (including ghost points) */
  int           Xm, Ym;         /* local widths (including ghost points) */
  int           ldim, gdim;     /* local and global vector dimensions */
} GridCtx;

/* 
    Application information, specific to all levels
*/
typedef struct {
  double      Qinf;           /* velocity at infinity */
  double      mach;           /* Mach number at infinity */
  int         machflag;       /* flag - 1 to compute local Mach number */
  int         jfreq;          /* frequency of evaluating Jacobian */
  MPI_Comm    comm;           /* communicator */
  double      xx0, xx1;       /* physical domain (xx0,yy0), (xx1,yy0) */
  double      yy0, yy1;       /*                 (xx1,yy1), (xx0,yy1) */
  int         nc;

  int         Nlevels;        /* current number of grids (now always 1) */
  GridCtx     grids[5];
} AppCtx;

/* -------------------User-defined routines---------------------------- */
extern int InitialGuess_PotentialFlow(AppCtx*,Vec);
extern int Function_PotentialFlow(SNES,Vec,Vec,void*);
extern int UserMonitor(SNES,int,double,void*);
extern int EvaluateFunction(AppCtx*,double*,int,int,double*,double*);

#endif
