
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdmatrix.c,v 1.51 1999/10/01 21:21:44 bsmith Exp bsmith $";
#endif

/*
   This is where the abstract matrix operations are defined that are
  used for finite difference computations of Jacobians using coloring.
*/

#include "petsc.h"
#include "src/mat/matimpl.h"        /*I "mat.h" I*/
#include "src/vec/vecimpl.h"  

#undef __FUNC__  
#define __FUNC__ "MatFDColoringView_Draw"
static int MatFDColoringView_Draw(MatFDColoring fd,Viewer viewer)
{
  int         ierr,i,j,pause;
  PetscTruth  isnull;
  Draw        draw;
  double      xr,yr,xl,yl,h,w,x,y,xc,yc,scale = 0.0;
  DrawButton  button;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);

  xr  = fd->N; yr = fd->M; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);

  /* loop over colors  */
  for (i=0; i<fd->ncolors; i++ ) {
    for ( j=0; j<fd->nrows[i]; j++ ) {
      y = fd->M - fd->rows[i][j] - fd->rstart;
      x = fd->columnsforrow[i][j];
      ierr = DrawRectangle(draw,x,y,x+1,y+1,i+1,i+1,i+1,i+1);CHKERRQ(ierr);
    }
  }
  ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr); 
  ierr = DrawGetPause(draw,&pause);CHKERRQ(ierr);
  if (pause >= 0) { PetscSleep(pause); PetscFunctionReturn(0);}
  ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0);CHKERRQ(ierr);
  while (button != BUTTON_RIGHT) {
    ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);
    if (button == BUTTON_LEFT) scale = .5;
    else if (button == BUTTON_CENTER) scale = 2.;
    xl = scale*(xl + w - xc) + xc - w*scale;
    xr = scale*(xr - w - xc) + xc + w*scale;
    yl = scale*(yl + h - yc) + yc - h*scale;
    yr = scale*(yr - h - yc) + yc + h*scale;
    w *= scale; h *= scale;
    ierr = DrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
    /* loop over colors  */
    for (i=0; i<fd->ncolors; i++ ) {
      for ( j=0; j<fd->nrows[i]; j++ ) {
        y = fd->M - fd->rows[i][j] - fd->rstart;
        x = fd->columnsforrow[i][j];
        ierr = DrawRectangle(draw,x,y,x+1,y+1,i+1,i+1,i+1,i+1);CHKERRQ(ierr);
      }
    }
    ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
    ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringView"
/*@C
   MatFDColoringView - Views a finite difference coloring context.

   Collective on MatFDColoring unless Viewer is VIEWER_STDOUT_SELF

   Input  Parameters:
+  c - the coloring context
-  viewer - visualization context

   Level: intermediate

   Notes:
   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
.     VIEWER_STDOUT_WORLD - synchronized standard
        output where only the first processor opens
        the file.  All other processors send their 
        data to the first processor to print. 
-     VIEWER_DRAW_WORLD - graphical display of nonzero structure

.seealso: MatFDColoringCreate()

.keywords: Mat, finite differences, coloring, view
@*/
int MatFDColoringView(MatFDColoring c,Viewer viewer)
{
  int        i,j,format,ierr;
  int        isdraw,isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c,MAT_FDCOLORING_COOKIE);
  if (!viewer) viewer = VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE); 

  isdraw  = PetscTypeCompare(viewer,DRAW_VIEWER);
  isascii = PetscTypeCompare(viewer,ASCII_VIEWER);
  if (isdraw) { 
    ierr = MatFDColoringView_Draw(c,viewer);CHKERRQ(ierr);
  } else if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"MatFDColoring Object:\n");CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"  Error tolerance=%g\n",c->error_rel);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"  Umin=%g\n",c->umin);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"  Number of colors=%d\n",c->ncolors);CHKERRQ(ierr);

    ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format != VIEWER_FORMAT_ASCII_INFO) {
      for ( i=0; i<c->ncolors; i++ ) {
        ierr = ViewerASCIIPrintf(viewer,"  Information for color %d\n",i);CHKERRQ(ierr);
        ierr = ViewerASCIIPrintf(viewer,"    Number of columns %d\n",c->ncolumns[i]);CHKERRQ(ierr);
        for ( j=0; j<c->ncolumns[i]; j++ ) {
          ierr = ViewerASCIIPrintf(viewer,"      %d\n",c->columns[i][j]);CHKERRQ(ierr);
        }
        ierr = ViewerASCIIPrintf(viewer,"    Number of rows %d\n",c->nrows[i]);CHKERRQ(ierr);
        for ( j=0; j<c->nrows[i]; j++ ) {
          ierr = ViewerASCIIPrintf(viewer,"      %d %d \n",c->rows[i][j],c->columnsforrow[i][j]);CHKERRQ(ierr);
        }
      }
    }
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for MatFDColoring",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringSetParameters"
/*@
   MatFDColoringSetParameters - Sets the parameters for the sparse approximation of
   a Jacobian matrix using finite differences.

   Collective on MatFDColoring

   The Jacobian is estimated with the differencing approximation
.vb
       F'(u)_{:,i} = [F(u+h*dx_{i}) - F(u)]/h where
       h = error_rel*u[i]                 if  abs(u[i]) > umin
         = +/- error_rel*umin             otherwise, with +/- determined by the sign of u[i]
       dx_{i} = (0, ... 1, .... 0)
.ve

   Input Parameters:
+  coloring - the coloring context
.  error_rel - relative error
-  umin - minimum allowable u-value magnitude

   Level: advanced

.keywords: Mat, finite differences, coloring, set, parameters

.seealso: MatFDColoringCreate()
@*/
int MatFDColoringSetParameters(MatFDColoring matfd,double error,double umin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  if (error != PETSC_DEFAULT) matfd->error_rel = error;
  if (umin != PETSC_DEFAULT)  matfd->umin      = umin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringSetFrequency"
/*@
   MatFDColoringSetFrequency - Sets the frequency for computing new Jacobian
   matrices. 

   Collective on MatFDColoring

   Input Parameters:
+  coloring - the coloring context
-  freq - frequency (default is 1)

   Options Database Keys:
.  -mat_fd_coloring_freq <freq>  - Sets coloring frequency

   Level: advanced

   Notes:
   Using a modified Newton strategy, where the Jacobian remains fixed for several
   iterations, can be cost effective in terms of overall nonlinear solution 
   efficiency.  This parameter indicates that a new Jacobian will be computed every
   <freq> nonlinear iterations.  

.keywords: Mat, finite differences, coloring, set, frequency

.seealso: MatFDColoringCreate(), MatFDColoringGetFrequency()
@*/
int MatFDColoringSetFrequency(MatFDColoring matfd,int freq)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  matfd->freq = freq;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringGetFrequency"
/*@
   MatFDColoringGetFrequency - Gets the frequency for computing new Jacobian
   matrices. 

   Not Collective

   Input Parameters:
.  coloring - the coloring context

   Output Parameters:
.  freq - frequency (default is 1)

   Options Database Keys:
.  -mat_fd_coloring_freq <freq> - Sets coloring frequency

   Level: advanced

   Notes:
   Using a modified Newton strategy, where the Jacobian remains fixed for several
   iterations, can be cost effective in terms of overall nonlinear solution 
   efficiency.  This parameter indicates that a new Jacobian will be computed every
   <freq> nonlinear iterations.  

.keywords: Mat, finite differences, coloring, get, frequency

.seealso: MatFDColoringSetFrequency()
@*/
int MatFDColoringGetFrequency(MatFDColoring matfd,int *freq)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  *freq = matfd->freq;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringSetFunction"
/*@C
   MatFDColoringSetFunction - Sets the function to use for computing the Jacobian.

   Collective on MatFDColoring

   Input Parameters:
+  coloring - the coloring context
.  f - the function
-  fctx - the optional user-defined function context

   Level: intermediate

   Notes:
    In Fortran you must call MatFDColoringSetFunctionSNES() for a coloring object to 
  be used with the SNES solvers and MatFDColoringSetFunctionTS() if it is to be used
  with the TS solvers.

.keywords: Mat, Jacobian, finite differences, set, function
@*/
int MatFDColoringSetFunction(MatFDColoring matfd,int (*f)(void),void *fctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  matfd->f    = f;
  matfd->fctx = fctx;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringSetFromOptions"
/*@
   MatFDColoringSetFromOptions - Sets coloring finite difference parameters from 
   the options database.

   Collective on MatFDColoring

   The Jacobian, F'(u), is estimated with the differencing approximation
.vb
       F'(u)_{:,i} = [F(u+h*dx_{i}) - F(u)]/h where
       h = error_rel*u[i]                 if  abs(u[i]) > umin
         = +/- error_rel*umin             otherwise, with +/- determined by the sign of u[i]
       dx_{i} = (0, ... 1, .... 0)
.ve

   Input Parameter:
.  coloring - the coloring context

   Options Database Keys:
+  -mat_fd_coloring_error <err> - Sets <err> (square root
           of relative error in the function)
.  -mat_fd_coloring_umin <umin> - Sets umin, the minimum allowable u-value magnitude
.  -mat_fd_coloring_freq <freq> - Sets frequency of computing a new Jacobian
.  -mat_fd_coloring_view - Activates basic viewing
.  -mat_fd_coloring_view_info - Activates viewing info
-  -mat_fd_coloring_view_draw - Activates drawing

    Level: intermediate

.keywords: Mat, finite differences, parameters
@*/
int MatFDColoringSetFromOptions(MatFDColoring matfd)
{
  int    ierr,flag,freq = 1;
  double error = PETSC_DEFAULT,umin = PETSC_DEFAULT;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  ierr = OptionsGetDouble(matfd->prefix,"-mat_fd_coloring_err",&error,&flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(matfd->prefix,"-mat_fd_coloring_umin",&umin,&flag);CHKERRQ(ierr);
  ierr = MatFDColoringSetParameters(matfd,error,umin);CHKERRQ(ierr);
  ierr = OptionsGetInt(matfd->prefix,"-mat_fd_coloring_freq",&freq,&flag);CHKERRQ(ierr);
  ierr = MatFDColoringSetFrequency(matfd,freq);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatFDColoringPrintHelp(matfd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringPrintHelp"
/*@
    MatFDColoringPrintHelp - Prints help message for matrix finite difference calculations 
    using coloring.

    Collective on MatFDColoring

    Input Parameter:
.   fdcoloring - the MatFDColoring context

    Level: intermediate

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringSetFromOptions()
@*/
int MatFDColoringPrintHelp(MatFDColoring fd)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd,MAT_FDCOLORING_COOKIE);

  ierr = (*PetscHelpPrintf)(fd->comm,"-mat_fd_coloring_err <err>: set sqrt rel tol in function, defaults to %g\n",fd->error_rel);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(fd->comm,"-mat_fd_coloring_umin <umin>: see users manual, defaults to %d\n",fd->umin);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(fd->comm,"-mat_fd_coloring_freq <freq>: frequency that Jacobian is recomputed, defaults to %d\n",fd->freq);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(fd->comm,"-mat_fd_coloring_view\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(fd->comm,"-mat_fd_coloring_view_draw\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(fd->comm,"-mat_fd_coloring_view_info\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringView_Private"
int MatFDColoringView_Private(MatFDColoring fd)
{
  int ierr,flg;

  PetscFunctionBegin;
  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatFDColoringView(fd,VIEWER_STDOUT_(fd->comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_view_info",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(fd->comm),VIEWER_FORMAT_ASCII_INFO,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatFDColoringView(fd,VIEWER_STDOUT_(fd->comm));CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(fd->comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_view_draw",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatFDColoringView(fd,VIEWER_DRAW_(fd->comm));CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_DRAW_(fd->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringCreate" 
/*@C
   MatFDColoringCreate - Creates a matrix coloring context for finite difference 
   computation of Jacobians.

   Collective on Mat

   Input Parameters:
+  mat - the matrix containing the nonzero structure of the Jacobian
-  iscoloring - the coloring of the matrix

    Output Parameter:
.   color - the new coloring context
   
    Options Database Keys:
+    -mat_fd_coloring_view - Activates basic viewing or coloring
.    -mat_fd_coloring_view_draw - Activates drawing of coloring
-    -mat_fd_coloring_view_info - Activates viewing of coloring info

    Level: intermediate

.seealso: MatFDColoringDestroy()
@*/
int MatFDColoringCreate(Mat mat,ISColoring iscoloring,MatFDColoring *color)
{
  MatFDColoring c;
  MPI_Comm      comm;
  int           ierr,M,N;

  PetscFunctionBegin;
  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
  if (M != N) SETERRQ(PETSC_ERR_SUP,0,"Only for square matrices");

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  PetscHeaderCreate(c,_p_MatFDColoring,int,MAT_FDCOLORING_COOKIE,0,"MatFDColoring",comm,
                    MatFDColoringDestroy,MatFDColoringView);
  PLogObjectCreate(c);

  if (mat->ops->fdcoloringcreate) {
    ierr = (*mat->ops->fdcoloringcreate)(mat,iscoloring,c);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"Code not yet written for this matrix type");
  }

  c->error_rel = 1.e-8;
  c->umin      = 1.e-6;
  c->freq      = 1;

  ierr = MatFDColoringView_Private(c);CHKERRQ(ierr);

  *color = c;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringDestroy"
/*@C
    MatFDColoringDestroy - Destroys a matrix coloring context that was created
    via MatFDColoringCreate().

    Collective on MatFDColoring

    Input Parameter:
.   c - coloring context

    Level: intermediate

.seealso: MatFDColoringCreate()
@*/
int MatFDColoringDestroy(MatFDColoring c)
{
  int i,ierr;

  PetscFunctionBegin;
  if (--c->refct > 0) PetscFunctionReturn(0);


  for ( i=0; i<c->ncolors; i++ ) {
    if (c->columns[i])       {ierr = PetscFree(c->columns[i]);CHKERRQ(ierr);}
    if (c->rows[i])          {ierr = PetscFree(c->rows[i]);CHKERRQ(ierr);}
    if (c->columnsforrow[i]) {ierr = PetscFree(c->columnsforrow[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(c->ncolumns);CHKERRQ(ierr);
  ierr = PetscFree(c->columns);CHKERRQ(ierr);
  ierr = PetscFree(c->nrows);CHKERRQ(ierr);
  ierr = PetscFree(c->rows);CHKERRQ(ierr);
  ierr = PetscFree(c->columnsforrow);CHKERRQ(ierr);
  ierr = PetscFree(c->scale);CHKERRQ(ierr);
  if (c->w1) {
    ierr = VecDestroy(c->w1);CHKERRQ(ierr);
    ierr = VecDestroy(c->w2);CHKERRQ(ierr);
    ierr = VecDestroy(c->w3);CHKERRQ(ierr);
  }
  PLogObjectDestroy(c);
  PetscHeaderDestroy(c);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatFDColoringApply"
/*@
    MatFDColoringApply - Given a matrix for which a MatFDColoring context 
    has been created, computes the Jacobian for a function via finite differences.

    Collective on MatFDColoring

    Input Parameters:
+   mat - location to store Jacobian
.   coloring - coloring context created with MatFDColoringCreate()
.   x1 - location at which Jacobian is to be computed
-   sctx - optional context required by function (actually a SNES context)

   Options Database Keys:
.  -mat_fd_coloring_freq <freq> - Sets coloring frequency

   Level: intermediate

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringView()

.keywords: coloring, Jacobian, finite differences
@*/
int MatFDColoringApply(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  int           k,fg,ierr,N,start,end,l,row,col,srow;
  Scalar        dx, mone = -1.0,*y,*scale = coloring->scale,*xx,*wscale = coloring->wscale;
  double        epsilon = coloring->error_rel, umin = coloring->umin; 
  MPI_Comm      comm = coloring->comm;
  Vec           w1,w2,w3;
  int           (*f)(void *,Vec,Vec,void *) = ( int (*)(void *,Vec,Vec,void *))coloring->f;
  void          *fctx = coloring->fctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_COOKIE);
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_COOKIE);
  PetscValidHeaderSpecific(x1,VEC_COOKIE);


  if (!coloring->w1) {
    ierr = VecDuplicate(x1,&coloring->w1);CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w1);
    ierr = VecDuplicate(x1,&coloring->w2);CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w2);
    ierr = VecDuplicate(x1,&coloring->w3);CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w3);
  }
  w1 = coloring->w1; w2 = coloring->w2; w3 = coloring->w3;

  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_dont_rezero",&fg);CHKERRQ(ierr);
  if (fg) {
    PLogInfo(coloring,"MatFDColoringApply: Not calling MatZeroEntries()\n");
  } else {
    ierr = MatZeroEntries(J);CHKERRQ(ierr);
  }

  ierr = VecGetOwnershipRange(x1,&start,&end);CHKERRQ(ierr);
  ierr = VecGetSize(x1,&N);CHKERRQ(ierr);
  ierr = (*f)(sctx,x1,w1,fctx);CHKERRQ(ierr);

  ierr = PetscMemzero(wscale,N*sizeof(Scalar));CHKERRQ(ierr);
  /*
      Loop over each color
  */

  ierr = VecGetArray(x1,&xx);CHKERRQ(ierr);
  for (k=0; k<coloring->ncolors; k++) { 
    ierr = VecCopy(x1,w3);CHKERRQ(ierr);
    /*
       Loop over each column associated with color adding the 
       perturbation to the vector w3.
    */
    for (l=0; l<coloring->ncolumns[k]; l++) {
      col = coloring->columns[k][l];    /* column of the matrix we are probing for */
      dx  = xx[col-start];
      if (dx == 0.0) dx = 1.0;
#if !defined(PETSC_USE_COMPLEX)
      if (dx < umin && dx >= 0.0)      dx = umin;
      else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
      if (PetscAbsScalar(dx) < umin && PetscReal(dx) >= 0.0)     dx = umin;
      else if (PetscReal(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
#endif
      dx          *= epsilon;
      wscale[col] = 1.0/dx;
      ierr = VecSetValues(w3,1,&col,&dx,ADD_VALUES);CHKERRQ(ierr); 
    } 

    /*
       Evaluate function at x1 + dx (here dx is a vector of perturbations)
    */
    ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr);
    ierr = VecAXPY(&mone,w1,w2);CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(PETSC_USE_COMPLEX)
    ierr = MPI_Allreduce(wscale,scale,N,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#else
    ierr = MPI_Allreduce(wscale,scale,2*N,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#endif
    /*
       Loop over rows of vector, putting results into Jacobian matrix
    */
    ierr = VecGetArray(w2,&y);CHKERRQ(ierr);
    for (l=0; l<coloring->nrows[k]; l++) {
      row    = coloring->rows[k][l];
      col    = coloring->columnsforrow[k][l];
      y[row] *= scale[col];
      srow   = row + start;
      ierr   = MatSetValues(J,1,&srow,1,&col,y+row,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(w2,&y);CHKERRQ(ierr);
  }
  ierr  = VecRestoreArray(x1,&xx);CHKERRQ(ierr);
  ierr  = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringApplyTS"
/*@
    MatFDColoringApplyTS - Given a matrix for which a MatFDColoring context 
    has been created, computes the Jacobian for a function via finite differences.

   Collective on Mat, MatFDColoring, and Vec

    Input Parameters:
+   mat - location to store Jacobian
.   coloring - coloring context created with MatFDColoringCreate()
.   x1 - location at which Jacobian is to be computed
-   sctx - optional context required by function (actually a SNES context)

   Options Database Keys:
.  -mat_fd_coloring_freq <freq> - Sets coloring frequency

   Level: intermediate

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringView()

.keywords: coloring, Jacobian, finite differences
@*/
int MatFDColoringApplyTS(Mat J,MatFDColoring coloring,double t,Vec x1,MatStructure *flag,void *sctx)
{
  int           k,fg,ierr,N,start,end,l,row,col,srow;
  Scalar        dx, mone = -1.0,*y,*scale = coloring->scale,*xx,*wscale = coloring->wscale;
  double        epsilon = coloring->error_rel, umin = coloring->umin; 
  MPI_Comm      comm = coloring->comm;
  Vec           w1,w2,w3;
  int           (*f)(void *,double,Vec,Vec,void *) = ( int (*)(void *,double,Vec,Vec,void *))coloring->f;
  void          *fctx = coloring->fctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_COOKIE);
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_COOKIE);
  PetscValidHeaderSpecific(x1,VEC_COOKIE);

  if (!coloring->w1) {
    ierr = VecDuplicate(x1,&coloring->w1);CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w1);
    ierr = VecDuplicate(x1,&coloring->w2);CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w2);
    ierr = VecDuplicate(x1,&coloring->w3);CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w3);
  }
  w1 = coloring->w1; w2 = coloring->w2; w3 = coloring->w3;

  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_dont_rezero",&fg);CHKERRQ(ierr);
  if (fg) {
    PLogInfo(coloring,"MatFDColoringApplyTS: Not calling MatZeroEntries()\n");
  } else {
    ierr = MatZeroEntries(J);CHKERRQ(ierr);
  }

  ierr = VecGetOwnershipRange(x1,&start,&end);CHKERRQ(ierr);
  ierr = VecGetSize(x1,&N);CHKERRQ(ierr);
  ierr = (*f)(sctx,t,x1,w1,fctx);CHKERRQ(ierr);

  ierr = PetscMemzero(wscale,N*sizeof(Scalar));CHKERRQ(ierr);
  /*
      Loop over each color
  */

  ierr = VecGetArray(x1,&xx);CHKERRQ(ierr);
  for (k=0; k<coloring->ncolors; k++) { 
    ierr = VecCopy(x1,w3);CHKERRQ(ierr);
    /*
       Loop over each column associated with color adding the 
       perturbation to the vector w3.
    */
    for (l=0; l<coloring->ncolumns[k]; l++) {
      col = coloring->columns[k][l];    /* column of the matrix we are probing for */
      dx  = xx[col-start];
      if (dx == 0.0) dx = 1.0;
#if !defined(PETSC_USE_COMPLEX)
      if (dx < umin && dx >= 0.0)      dx = umin;
      else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
      if (PetscAbsScalar(dx) < umin && PetscReal(dx) >= 0.0)     dx = umin;
      else if (PetscReal(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
#endif
      dx          *= epsilon;
      wscale[col] = 1.0/dx;
      ierr = VecSetValues(w3,1,&col,&dx,ADD_VALUES);CHKERRQ(ierr);
    } 
    /*
       Evaluate function at x1 + dx (here dx is a vector of perturbations)
    */
    ierr = (*f)(sctx,t,w3,w2,fctx);CHKERRQ(ierr);
    ierr = VecAXPY(&mone,w1,w2);CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(PETSC_USE_COMPLEX)
    ierr = MPI_Allreduce(wscale,scale,N,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#else
    ierr = MPI_Allreduce(wscale,scale,2*N,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
#endif
    /*
       Loop over rows of vector, putting results into Jacobian matrix
    */
    ierr = VecGetArray(w2,&y);CHKERRQ(ierr);
    for (l=0; l<coloring->nrows[k]; l++) {
      row    = coloring->rows[k][l];
      col    = coloring->columnsforrow[k][l];
      y[row] *= scale[col];
      srow   = row + start;
      ierr   = MatSetValues(J,1,&srow,1,&col,y+row,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(w2,&y);CHKERRQ(ierr);
  }
  ierr  = VecRestoreArray(x1,&xx);CHKERRQ(ierr);
  ierr  = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



