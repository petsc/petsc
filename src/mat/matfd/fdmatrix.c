
#ifndef lint
static char vcid[] = "$Id: fdmatrix.c,v 1.5 1996/12/18 22:03:34 balay Exp bsmith $";
#endif

/*
   This is where the abstract matrix operations are defined that are
  used for finite difference computations of Jacobians using coloring.
*/

#include "petsc.h"
#include "src/mat/matimpl.h"        /*I "mat.h" I*/
#include "src/vec/vecimpl.h"  
#include "pinclude/pviewer.h"

#undef __FUNCTION__  
#define __FUNCTION__ "MatFDColoringView"
/*@C
   MatFDColoringView - Views a finite difference coloring context.

   Input  Parameter:
.   color - the coloring context
   

.seealso: MatFDColoringCreate()
@*/
int MatFDColoringView(MatFDColoring color,Viewer viewer)
{
  int i,j,format,ierr;

  if (!viewer) viewer = VIEWER_STDOUT_WORLD;

  ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);

  if (format == VIEWER_FORMAT_ASCII_INFO) {
    printf("MatFDColoring Object:\n");
    printf("  Error tolerance %g\n",color->error_rel);
    printf("  umin %g\n",color->umin);
  } else {
    printf("MatFDColoring Object:\n");
    printf("  Error tolerance %g\n",color->error_rel);
    printf("  umin %g\n",color->umin);
    printf("Number of colors %d\n",color->ncolors);
    for ( i=0; i<color->ncolors; i++ ) {
      printf("Information for color %d\n",i);
      printf("Number of columns %d\n",color->ncolumns[i]);
      for ( j=0; j<color->ncolumns[i]; j++ ) {
        printf("  %d\n",color->columns[i][j]);
      }
      printf("Number of rows %d\n",color->nrows[i]);
      for ( j=0; j<color->nrows[i]; j++ ) {
        printf("  %d %d \n",color->rows[i][j],color->columnsforrow[i][j]);
      }
    }
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatFDColoringSetParameters"
/*@
   MatFDColoringSetParameters - Sets the parameters for the approximation of
   Jacobian using finite differences.

$       J(u)_{:,i} = [J(u+h*dx_{i}) - J(u)]/h where
$        h = error_rel*u[i]                    if  u[i] > umin
$          = error_rel*umin                    else
$
$   dx_{i} = (0, ... 1, .... 0)

   Input Parameters:
.  coloring - the jacobian coloring context
.  error_rel - relative error
.  umin - minimum allowable u-value

.keywords: SNES, Jacobian, finite differences, parameters
@*/
int MatFDColoringSetParameters(MatFDColoring matfd,double error,double umin)
{
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  if (error != PETSC_DEFAULT) matfd->error_rel = error;
  if (umin != PETSC_DEFAULT)  matfd->umin      = umin;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatFDColoringSetFromOptions"
/*@
   MatFDColoringSetFromOptions - Set coloring finite difference parameters from 
         the options database.

$       J(u)_{:,i} = [J(u+h*dx_{i}) - J(u)]/h where
$        h = error_rel*u[i]                    if  u[i] > umin
$          = error_rel*.1                      else
$
$   dx_{i} = (0, ... 1, .... 0)

   Input Parameters:
.  coloring - the jacobian coloring context

   Options Database:
.  -mat_fd_coloring_error square root of relative error in function
.  -mat_fd_coloring_umin  see above

.keywords: SNES, Jacobian, finite differences, parameters
@*/
int MatFDColoringSetFromOptions(MatFDColoring matfd)
{
  int    ierr,flag;
  double error = PETSC_DEFAULT,umin = PETSC_DEFAULT;
  PetscValidHeaderSpecific(matfd,MAT_FDCOLORING_COOKIE);

  ierr = OptionsGetDouble(matfd->prefix,"-mat_fd_coloring_err",&error,&flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(matfd->prefix,"-mat_fd_coloring_umin",&umin,&flag);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flag); CHKERRQ(ierr);
  ierr = MatFDColoringSetParameters(matfd,error,umin); CHKERRQ(ierr);
  if (flag) {
    ierr = MatFDColoringPrintHelp(matfd); CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatFDColoringPrintHelp"
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

  PetscPrintf(fd->comm,"-mat_fd_coloring_err <err> set sqrt rel tol in function. Default 1.e-8\n");
  PetscPrintf(fd->comm,"-mat_fd_coloring_umin <umin> see users manual. Default 1.e-8\n");
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatFDColoringCreate"
/*@C
   MatFDColoringCreate - Creates a matrix coloring context for finite difference 
        computation of Jacobians.

   Input Parameters:
.    mat - the matrix containing the nonzero structure of the Jacobian
.    iscoloring - the coloring of the matrix

   Output Parameter:
.   color - the new coloring context
   

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
  PetscHeaderCreate(c,_MatFDColoring,MAT_FDCOLORING_COOKIE,0,comm);
  PLogObjectCreate(c);

  if (mat->ops.fdcoloringcreate) {
    ierr = (*mat->ops.fdcoloringcreate)(mat,iscoloring,c); CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"Code not yet written for this matrix type");
  }

  c->error_rel = 1.e-8;
  c->umin      = 1.e-5;

  *color = c;

  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatFDColoringDestroy"
/*@C
    MatFDColoringDestroy - Destroys a matrix coloring context that was created
         via MatFDColoringCreate().

    Input Paramter:
.   c - coloring context

.seealso: MatFDColoringCreate()
@*/
int MatFDColoringDestroy(MatFDColoring c)
{
  int i,ierr,flag;

  ierr = OptionsHasName(PETSC_NULL,"-matfdcoloring_view",&flag);
  if (flag) {
    Viewer viewer;
    ierr = ViewerFileOpenASCII(c->comm,"stdout",&viewer);CHKERRQ(ierr);
    ierr = MatFDColoringView(c,viewer);CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-matfdcoloring_view_info",&flag);
  if (flag) {
    Viewer viewer;
    ierr = ViewerFileOpenASCII(c->comm,"stdout",&viewer);CHKERRQ(ierr);
    ierr = ViewerPushFormat(viewer,VIEWER_FORMAT_ASCII_INFO, PETSC_NULL); CHKERRQ(ierr);
    ierr = MatFDColoringView(c,viewer);CHKERRQ(ierr);
    ierr = ViewerPopFormat(viewer);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
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
  PLogObjectDestroy(c);
  PetscHeaderDestroy(c);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "MatFDColoringApply"
/*@
     MatFDColoringApply - Given a matrix for which a MatFDColoring has been created,
         computes the Jacobian for a function via finite differences.

  Input Parameters:
.   mat - location to store Jacobian
.   coloring - coloring context created with MatFDColoringCreate()
.   x1 - location at which Jacobian is to be computed
.   w1,w2,w3 - three work vectors
.   f - function for which Jacobian is required
.   fctx - optional context required by function

.seealso: MatFDColoringCreate(), MatFDColoringDestroy(), MatFDColoringView()

.keywords: coloring, Jacobian, finite differences
@*/
int MatFDColoringApply(Mat J,MatFDColoring coloring,Vec x1,Vec w1,Vec w2,Vec w3,
                       int (*f)(void *,Vec,Vec,void*),void *sctx,void *fctx)
{
  int           k, ierr,N,start,end,l,row,col,srow;
  Scalar        dx, mone = -1.0,*y,*scale = coloring->scale,*xx,*wscale = coloring->wscale;
  double        epsilon = coloring->error_rel, umin = coloring->umin; 
  MPI_Comm      comm = coloring->comm;

  ierr = MatZeroEntries(J); CHKERRQ(ierr);

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
#if !defined(PETSC_COMPLEX)
      if (dx < umin && dx >= 0.0) dx = .1;
      else if (dx < 0.0 && dx > -umin) dx = -.1;
#else
      if (abs(dx) < umin && real(dx) >= 0.0) dx = .1;
      else if (real(dx) < 0.0 && abs(dx) < umin) dx = -.1;
#endif
      dx          *= epsilon;
      wscale[col] = 1.0/dx;
      VecSetValues(w3,1,&col,&dx,ADD_VALUES); 
    } 
    VecRestoreArray(x1,&xx);
    /*
       Evaluate function at x1 + dx (here dx is a vector, of perturbations)
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
       Loop over rows of vector putting results into Jacobian matrix
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
