
#ifndef lint
static char vcid[] = "$Id: snesj2.c,v 1.1 1996/10/09 19:38:25 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"      /*I  "mat.h"  I*/
#include "src/snes/snesimpl.h"    /*I  "snes.h"  I*/


/*@C
     SNESDefaultComputeJacobianWithColoring
  
   Input Parameters:
.    snes - nonlinear solver object
.    x1 - location at which to evaluate Jacobian
.    ctx - MatFDColoring contex

   Output Parameters:
.    J - Jacobian matrix
.    B - Jacobian preconditioner
.    flag - flag indicating if the matrix nonzero structure has changed

.keywords: SNES, finite differences, Jacobian

.seealso: SNESSetJacobian(), SNESTestJacobian()
@*/
int SNESDefaultComputeJacobianWithColoring(SNES snes,Vec x1,Mat *JJ,Mat *B,MatStructure *flag,void *ctx)
{
  MatFDColoring color = (MatFDColoring) ctx;
  Mat           J = *B;
  Vec           jj1,jj2,x2;
  int           k, ierr,N,start,end,l,row,col,srow;
  Scalar        dx, mone = -1.0,*y,*scale = color->scale,*xx,*wscale = color->wscale;
  double        epsilon = color->error_rel, umin = color->umin; 
  MPI_Comm      comm = color->comm;

  ierr = MatZeroEntries(J); CHKERRQ(ierr);
  if (!snes->nvwork) {
    ierr = VecDuplicateVecs(x1,3,&snes->vwork); CHKERRQ(ierr);
    snes->nvwork = 3;
    PLogObjectParents(snes,3,snes->vwork);
  }
  jj1 = snes->vwork[0]; jj2 = snes->vwork[1]; x2 = snes->vwork[2];

  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERRQ(ierr);
  ierr = VecGetSize(x1,&N); CHKERRQ(ierr);
  ierr = VecGetArray(x1,&xx); CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,x1,jj1); CHKERRQ(ierr);

  PetscMemzero(wscale,N*sizeof(Scalar));
  /*
      Loop over each color
  */

  for (k=0; k<color->ncolors; k++) { 
    ierr = VecCopy(x1,x2); CHKERRQ(ierr);
    /*
       Loop over each column associated with color adding the 
       perturbation to the vector x2.
    */
    for (l=0; l<color->ncolumns[k]; l++) {
      col = color->columns[k][l];    /* column of the matrix we are probing for */
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
      VecSetValues(x2,1,&col,&dx,ADD_VALUES); 
    } 
    VecRestoreArray(x1,&xx);
    /*
       Evaluate function at x1 + dx (here dx is a vector, of perturbations)
    */
    ierr = SNESComputeFunction(snes,x2,jj2); CHKERRQ(ierr);
    ierr = VecAXPY(&mone,jj1,jj2); CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(PETSC_COMPLEX)
    MPI_Allreduce(wscale,scale,N,MPI_DOUBLE,MPI_SUM,comm);
#else
    MPI_Allreduce(wscale,scale,2*N,MPI_DOUBLE,MPI_SUM,comm);
#endif
    /*
       Loop over rows of vector putting results into Jacobian matrix
    */
    VecGetArray(jj2,&y);
    for (l=0; l<color->nrows[k]; l++) {
      row    = color->rows[k][l];
      col    = color->columnsforrow[k][l];
      y[row] *= scale[col];
      srow   = row + start;
      ierr   = MatSetValues(J,1,&srow,1,&col,y+row,INSERT_VALUES);CHKERRQ(ierr);
    }
    VecRestoreArray(jj2,&y);
  }
  ierr  = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}



