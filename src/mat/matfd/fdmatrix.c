#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdmatrix.c,v 1.18 1997/10/01 22:25:32 bsmith Exp bsmith $";
#endif

/*
   This is where the abstract matrix operations are defined that are
  used for finite difference computations of Jacobians using coloring.
*/

#include "petsc.h"
#include "src/mat/matimpl.h"        /*I "mat.h" I*/
#include "src/vec/vecimpl.h"  
#include "pinclude/pviewer.h"

#undef __FUNC__  
#define __FUNC__ "MatFDColoringView_Draw"
static int MatFDColoringView_Draw(MatFDColoring fd,Viewer viewer)
{
  int         ierr,i,j,pause;
  PetscTruth  isnull;
  Draw        draw;
  double      xr,yr,xl,yl,h,w,x,y,xc,yc,scale = 0.0;
  DrawButton  button;

  ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;
  ierr = DrawSyncClear(draw); CHKERRQ(ierr);

  xr  = fd->N; yr = fd->M; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);

  /* loop over colors  */
  for (i=0; i<fd->ncolors; i++ ) {
    for ( j=0; j<fd->nrows[i]; j++ ) {
      y = fd->M - fd->rows[i][j] - fd->rstart;
      x = fd->columnsforrow[i][j];
      DrawRectangle(draw,x,y,x+1,y+1,i+1,i+1,i+1,i+1);
    }
  }
  ierr = DrawSyncFlush(draw); CHKERRQ(ierr); 
  ierr = DrawGetPause(draw,&pause); CHKERRQ(ierr);
  if (pause >= 0) { PetscSleep(pause); return 0;}
  ierr = DrawCheckResizedWindow(draw);
  ierr = DrawSyncGetMouseButton(draw,&button,&xc,&yc,0,0); 
  while (button != BUTTON_RIGHT) {
    ierr = DrawSyncClear(draw); CHKERRQ(ierr);
    if (button == BUTTON_LEFT) scale = .5;
    else if (button == BUTTON_CENTER) scale = 2.;
    xl = scale*(xl + w - xc) + xc - w*scale;
    xr = scale*(xr - w - xc) + xc + w*scale;
    yl = scale*(yl + h - yc) + yc - h*scale;
    yr = scale*(yr - h - yc) + yc + h*scale;
    w *= scale; h *= scale;
    ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
    /* loop over colors  */
    for (i=0; i<fd->ncolors; i++ ) {
      for ( j=0; j<fd->nrows[i]; j++ ) {
        y = fd->M - fd->rows[i][j] - fd->rstart;
        x = fd->columnsforrow[i][j];
        DrawRectangle(draw,x,y,x+1,y+1,i+1,i+1,i+1,i+1);
      }
    }
    ierr = DrawCheckResizedWindow(draw);
    ierr = DrawSyncGetMouseButton(draw,&button,&xc,&yc,0,0); 
  }

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringView"
/*@C
   MatFDColoringView - Views a finite difference coloring context.

   Input  Parameters:
.  c - the coloring context
.  viewer - visualization context

   Notes:
   The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 
$     VIEWER_DRAWX_WORLD - graphical display of nonzero structure

.seealso: MatFDColoringCreate()

.keywords: Mat, finite differences, coloring, view
@*/
int MatFDColoringView(MatFDColoring c,Viewer viewer)
{
  ViewerType vtype;
  int        i,j,format,ierr;
  FILE       *fd;

  PetscValidHeaderSpecific(c,MAT_FDCOLORING_COOKIE);
  if (viewer) {PetscValidHeader(viewer);} 
  else {viewer = VIEWER_STDOUT_SELF;}

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == DRAW_VIEWER) { 
    ierr = MatFDColoringView_Draw(c,viewer); CHKERRQ(ierr);
    return 0;
  }
  else if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(c->comm,fd,"MatFDColoring Object:\n");
    PetscFPrintf(c->comm,fd,"  Error tolerance=%g\n",c->error_rel);
    PetscFPrintf(c->comm,fd,"  Umin=%g\n",c->umin);
    PetscFPrintf(c->comm,fd,"  Number of colors=%d\n",c->ncolors);

    ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);
    if (format != VIEWER_FORMAT_ASCII_INFO) {
      for ( i=0; i<c->ncolors; i++ ) {
        PetscFPrintf(c->comm,fd,"  Information for color %d\n",i);
        PetscFPrintf(c->comm,fd,"    Number of columns %d\n",c->ncolumns[i]);
        for ( j=0; j<c->ncolumns[i]; j++ ) {
          PetscFPrintf(c->comm,fd,"      %d\n",c->columns[i][j]);
        }
        PetscFPrintf(c->comm,fd,"    Number of rows %d\n",c->nrows[i]);
        for ( j=0; j<c->nrows[i]; j++ ) {
          PetscFPrintf(c->comm,fd,"      %d %d \n",c->rows[i][j],c->columnsforrow[i][j]);
        }
      }
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringSetParameters"
/*@
   MatFDColoringSetParameters - Sets the parameters for the sparse approximation of
   a Jacobian matrix using finite differences.

$       J(u)_{:,i} = [J(u+h*dx_{i}) - J(u)]/h where
$        h = error_rel*u[i]                    if  u[i] > umin
$          = error_rel*umin                    else
$
$   dx_{i} = (0, ... 1, .... 0)

   Input Parameters:
.  coloring - the coloring context
.  error_rel - relative error
.  umin - minimum allowable u-value

.keywords: Mat, finite differences, coloring, set, parameters

.seealso: MatFDColoringCreate()
@*/
int MatFDColoringSetParameters(MatFDColoring matfd,double error,double umin)
{
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  if (error != PETSC_DEFAULT) matfd->error_rel = error;
  if (umin != PETSC_DEFAULT)  matfd->umin      = umin;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringSetFrequency"
/*@
   MatFDColoringSetFrequency - Sets the frequency for computing new Jacobian
   matrices. 

   Input Parameters:
.  coloring - the coloring context
.  freq - frequency (default is 1)

   Notes:
   Using a modified Newton strategy, where the Jacobian remains fixed for several
   iterations, can be cost effective in terms of overall nonlinear solution 
   efficiency.  This parameter indicates that a new Jacobian will be computed every
   <freq> nonlinear iterations.  

   Options Database Keys:
$  -mat_fd_coloring_freq <freq> 

.keywords: Mat, finite differences, coloring, set, frequency
@*/
int MatFDColoringSetFrequency(MatFDColoring matfd,int freq)
{
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  matfd->freq = freq;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringSetFunction"
/*@
   MatFDColoringSetFunction - Sets the function to use for computing the Jacobian.

   Input Parameters:
.  coloring - the coloring context
.  f - the function
.  fctx - the function context

.keywords: Mat, Jacobian, finite differences, set, function
@*/
int MatFDColoringSetFunction(MatFDColoring matfd,int (*f)(void *,Vec,Vec,void *),void *fctx)
{
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  matfd->f    = f;
  matfd->fctx = fctx;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringSetFromOptions"
/*@
   MatFDColoringSetFromOptions - Sets coloring finite difference parameters from 
   the options database.

   The Jacobian is estimated with the differencing approximation
$       J(u)_{:,i} = [J(u+h*dx_{i}) - J(u)]/h where
$        h = error_rel*u[i]                    if  u[i] > umin
$          = error_rel*umin                      else
$
$   dx_{i} = (0, ... 1, .... 0)

   Input Parameters:
.  coloring - the coloring context

   Options Database Keys:
$  -mat_fd_coloring_error <err>, where <err> is the square root
$           of relative error in the function
$  -mat_fd_coloring_umin  <umin>, where umin is described above
$  -mat_fd_coloring_freq <freq> where <freq> is the frequency of
$           computing a new Jacobian
$  -mat_fd_coloring_view
$  -mat_fd_coloring_view_info
$  -mat_fd_coloring_view_draw

.keywords: Mat, finite differences, parameters
@*/
int MatFDColoringSetFromOptions(MatFDColoring matfd)
{
  int    ierr,flag,freq = 1;
  double error = PETSC_DEFAULT,umin = PETSC_DEFAULT;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  ierr = OptionsGetDouble(matfd->prefix,"-mat_fd_coloring_err",&error,&flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(matfd->prefix,"-mat_fd_coloring_umin",&umin,&flag);CHKERRQ(ierr);
  ierr = MatFDColoringSetParameters(matfd,error,umin); CHKERRQ(ierr);
  ierr = OptionsGetInt(matfd->prefix,"-mat_fd_coloring_freq",&freq,&flag);CHKERRQ(ierr);
  ierr = MatFDColoringSetFrequency(matfd,freq);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flag); CHKERRQ(ierr);
  if (flag) {
    ierr = MatFDColoringPrintHelp(matfd); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringPrintHelp"
/*@
    MatFDColoringPrintHelp - Prints help message for matrix finite difference calculations 
    using coloring.

    Input Parameter:
.   fdcoloring - the MatFDColoring context

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringSetFromOptions()
@*/
int MatFDColoringPrintHelp(MatFDColoring fd)
{
  PetscValidHeaderSpecific(fd,MAT_FDCOLORING_COOKIE);

  PetscPrintf(fd->comm,"-mat_fd_coloring_err <err>: set sqrt rel tol in function, defaults to %g\n",fd->error_rel);
  PetscPrintf(fd->comm,"-mat_fd_coloring_umin <umin>: see users manual, defaults to %d\n",fd->umin);
  PetscPrintf(fd->comm,"-mat_fd_coloring_freq <freq>: frequency that Jacobian is recomputed, defaults to %d\n",fd->freq);
  PetscPrintf(fd->comm,"-mat_fd_coloring_view\n");
  PetscPrintf(fd->comm,"-mat_fd_coloring_view_draw\n");
  PetscPrintf(fd->comm,"-mat_fd_coloring_view_info\n");
  return 0;
}

int MatFDColoringView_Private(MatFDColoring fd)
{
  int ierr,flg;

  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_view",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatFDColoringView(fd,VIEWER_STDOUT_(fd->comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_view_info",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(fd->comm),VIEWER_FORMAT_ASCII_INFO,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatFDColoringView(fd,VIEWER_STDOUT_(fd->comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(fd->comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_view_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatFDColoringView(fd,VIEWER_DRAWX_(fd->comm)); CHKERRQ(ierr);
    ierr = ViewerFlush(VIEWER_DRAWX_(fd->comm)); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringCreate" 
/*@C
   MatFDColoringCreate - Creates a matrix coloring context for finite difference 
   computation of Jacobians.

   Input Parameters:
.  mat - the matrix containing the nonzero structure of the Jacobian
.  iscoloring - the coloring of the matrix

    Output Parameter:
.   color - the new coloring context
   
    Options Database Keys:
$    -mat_fd_coloring_view 
$    -mat_fd_coloring_view_draw
$    -mat_fd_coloring_view_info

.seealso: MatFDColoringDestroy()
@*/
int MatFDColoringCreate(Mat mat,ISColoring iscoloring,MatFDColoring *color)
{
  MatFDColoring c;
  MPI_Comm      comm;
  int           ierr,M,N;

  ierr = MatGetSize(mat,&M,&N); CHKERRQ(ierr);
  if (M != N) SETERRQ(PETSC_ERR_SUP,0,"Only for square matrices");

  PetscObjectGetComm((PetscObject)mat,&comm);
  PetscHeaderCreate(c,_p_MatFDColoring,MAT_FDCOLORING_COOKIE,0,comm,MatFDColoringDestroy,MatFDColoringView);
  PLogObjectCreate(c);

  if (mat->ops.fdcoloringcreate) {
    ierr = (*mat->ops.fdcoloringcreate)(mat,iscoloring,c); CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"Code not yet written for this matrix type");
  }

  c->error_rel = 1.e-8;
  c->umin      = 1.e-6;
  c->freq      = 1;

  ierr = MatFDColoringView_Private(c); CHKERRQ(ierr);

  *color = c;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "MatFDColoringDestroy"
/*@C
    MatFDColoringDestroy - Destroys a matrix coloring context that was created
    via MatFDColoringCreate().

    Input Parameter:
.   c - coloring context

.seealso: MatFDColoringCreate()
@*/
int MatFDColoringDestroy(MatFDColoring c)
{
  int i,ierr,flag;

  if (--c->refct > 0) return 0;

  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_view",&flag);
  if (flag) {
    ierr = MatFDColoringView(c,VIEWER_STDOUT_(c->comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_view_info",&flag);
  if (flag) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(c->comm),VIEWER_FORMAT_ASCII_INFO,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatFDColoringView(c,VIEWER_STDOUT_(c->comm));CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(c->comm));
  }

  for ( i=0; i<c->ncolors; i++ ) {
    if (c->columns[i])       PetscFree(c->columns[i]);
    if (c->rows[i])          PetscFree(c->rows[i]);
    if (c->columnsforrow[i]) PetscFree(c->columnsforrow[i]);
  }
  PetscFree(c->ncolumns);
  PetscFree(c->columns);
  PetscFree(c->nrows);
  PetscFree(c->rows);
  PetscFree(c->columnsforrow);
  PetscFree(c->scale);
  if (c->w1) {
    ierr = VecDestroy(c->w1); CHKERRQ(ierr);
    ierr = VecDestroy(c->w2); CHKERRQ(ierr);
    ierr = VecDestroy(c->w3); CHKERRQ(ierr);
  }
  PLogObjectDestroy(c);
  PetscHeaderDestroy(c);
  return 0;
}

#include "snes.h"

#undef __FUNC__  
#define __FUNC__ "MatFDColoringApply"
/*@
    MatFDColoringApply - Given a matrix for which a MatFDColoring context 
    has been created, computes the Jacobian for a function via finite differences.

    Input Parameters:
.   mat - location to store Jacobian
.   coloring - coloring context created with MatFDColoringCreate()
.   x1 - location at which Jacobian is to be computed
.   sctx - optional context required by function (actually a SNES context)

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
  int           (*f)(void *,Vec,Vec,void *) = coloring->f;
  void          *fctx = coloring->fctx;

  PetscValidHeaderSpecific(J,MAT_COOKIE);
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_COOKIE);
  PetscValidHeaderSpecific(x1,VEC_COOKIE);

  /*
  ierr = SNESGetIterationNumber((SNES)sctx,&it); CHKERRQ(ierr);
  if ((freq > 1) && ((it % freq) != 1)) {
    PLogInfo(coloring,"MatFDColoringApply:Skipping Jacobian, iteration %d, freq %d\n",it,freq);
    *flag = SAME_PRECONDITIONER;
    return 0;
  } else {
    PLogInfo(coloring,"MatFDColoringApply:Computing Jacobian, iteration %d, freq %d\n",it,freq);
    *flag = SAME_NONZERO_PATTERN;
  }*/

  if (!coloring->w1) {
    ierr = VecDuplicate(x1,&coloring->w1); CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w1);
    ierr = VecDuplicate(x1,&coloring->w2); CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w2);
    ierr = VecDuplicate(x1,&coloring->w3); CHKERRQ(ierr);
    PLogObjectParent(coloring,coloring->w3);
  }
  w1 = coloring->w1; w2 = coloring->w2; w3 = coloring->w3;

  ierr = OptionsHasName(PETSC_NULL,"-mat_fd_coloring_dont_rezero",&fg); CHKERRQ(ierr);
  if (fg) {
    PLogInfo(coloring,"MatFDColoringApply: Not calling MatZeroEntries()\n");
  } else {
    ierr = MatZeroEntries(J); CHKERRQ(ierr);
  }

  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERRQ(ierr);
  ierr = VecGetSize(x1,&N); CHKERRQ(ierr);
  ierr = VecGetArray(x1,&xx); CHKERRQ(ierr);
  ierr = (*f)(sctx,x1,w1,fctx); CHKERRQ(ierr);

  PetscMemzero(wscale,N*sizeof(Scalar));
  /*
      Loop over each color
  */

  for (k=0; k<coloring->ncolors; k++) { 
    ierr = VecCopy(x1,w3); CHKERRQ(ierr);
    /*
       Loop over each column associated with color adding the 
       perturbation to the vector w3.
    */
    for (l=0; l<coloring->ncolumns[k]; l++) {
      col = coloring->columns[k][l];    /* column of the matrix we are probing for */
      dx  = xx[col-start];
      if (dx == 0.0) dx = 1.0;
#if !defined(PETSC_COMPLEX)
      if (dx < umin && dx >= 0.0)      dx = umin;
      else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
      if (abs(dx) < umin && real(dx) >= 0.0)     dx = umin;
      else if (real(dx) < 0.0 && abs(dx) < umin) dx = -umin;
#endif
      dx          *= epsilon;
      wscale[col] = 1.0/dx;
      VecSetValues(w3,1,&col,&dx,ADD_VALUES); 
    } 
    VecRestoreArray(x1,&xx);
    /*
       Evaluate function at x1 + dx (here dx is a vector of perturbations)
    */
    ierr = (*f)(sctx,w3,w2,fctx); CHKERRQ(ierr);
    ierr = VecAXPY(&mone,w1,w2); CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(PETSC_COMPLEX)
    MPI_Allreduce(wscale,scale,N,MPI_DOUBLE,MPI_SUM,comm);
#else
    MPI_Allreduce(wscale,scale,2*N,MPI_DOUBLE,MPI_SUM,comm);
#endif
    /*
       Loop over rows of vector, putting results into Jacobian matrix
    */
    VecGetArray(w2,&y);
    for (l=0; l<coloring->nrows[k]; l++) {
      row    = coloring->rows[k][l];
      col    = coloring->columnsforrow[k][l];
      y[row] *= scale[col];
      srow   = row + start;
      ierr   = MatSetValues(J,1,&srow,1,&col,y+row,INSERT_VALUES);CHKERRQ(ierr);
    }
    VecRestoreArray(w2,&y);
  }
  ierr  = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
