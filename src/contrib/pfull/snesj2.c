
#ifndef lint
static char vcid[] = "$Id: snesj2.c,v 1.1 1996/08/06 21:19:43 curfman Exp curfman $";
#endif

#include "draw.h"    /*I  "draw.h"  I*/
#include "snesimpl.h"    /*I  "snes.h"  I*/

#include "puser.h"

int MatCreateColoring(int dim,int nis,IS *isa,Coloring **color)
{
  Coloring *c;
  c = PetscNew(Coloring); CHKPTRQ(c);
  PetscMemzero(c,sizeof(Coloring));
  c->wscale = (Scalar *)PetscMalloc(2*dim*sizeof(Scalar)); CHKPTRQ(c->wscale);
  c->scale  = c->wscale + dim;
  c->nis    = nis;
  c->isa    = isa;
  *color = c;
  return 0;
}

int MatDestroyColoring(Coloring *c)
{
  PetscFree(c->wscale);
  return 0;
}

/*@C
   SNESSparseComputeJacobian - Computes the Jacobian using finite 
   differences - sparse variant using coloring.

   Input Parameters:
.  x1 - compute Jacobian at this point
.  ctx - application's function context, as set with SNESSetFunction()

   Output Parameters:
.  J - Jacobian
.  B - preconditioner, same as Jacobian
.  flag - matrix flag

.keywords: SNES, finite differences, Jacobian

.seealso: SNESSetJacobian(), SNESTestJacobian()
@*/
int SNESSparseComputeJacobian(SNES snes,Vec x1,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec      jj1,jj2,x2;
  int      *is, k, i,ierr,N,start,end,j, isize, count, l;
  Scalar   dx, mone = -1.0,*y,*scale,*xx,*wscale;
  double   amax, epsilon = 1.e-8; /* assumes double precision */
  MPI_Comm comm;
  int      (*eval_fct)(SNES,Vec,Vec);
  Coloring *color = (Coloring *)ctx;
  /*  int      Xm, Ys, Xs; */

  wscale = color->wscale;
  scale  = color->scale;
  /*  Xm     = color->Xm;
      Xs     = color->Xs;
      Ys     = color->Ys; */
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS)
    eval_fct = SNESComputeFunction;
  else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION)
    eval_fct = SNESComputeGradient;
  else SETERRQ(1,"SNESDefaultComputeJacobian: Invalid method class");

  PetscObjectGetComm((PetscObject)x1,&comm);
  MatZeroEntries(*J);
  if (!snes->nvwork) {
    ierr = VecDuplicateVecs(x1,3,&snes->vwork); CHKERRQ(ierr);
    snes->nvwork = 3;
    PLogObjectParents(snes,3,snes->vwork);
  }
  jj1 = snes->vwork[0]; jj2 = snes->vwork[1]; x2 = snes->vwork[2];

  ierr = VecGetSize(x1,&N); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x1,&start,&end); CHKERRQ(ierr);
  VecGetArray(x1,&xx);
  ierr = eval_fct(snes,x1,jj1); CHKERRQ(ierr);

  /* Compute Jacobian approximation, 1 column at a time. 
      x1 = current iterate, jj1 = F(x1)
      x2 = perturbed iterate, jj2 = F(x2)
   */
  for (k=0; k<color->nis; k++) { /* for ( i=0; i<N; i++ ) { */
    ierr = VecCopy(x1,x2); CHKERRQ(ierr);
    ierr = ISGetIndices(color->isa[k],&is); CHKERRQ(ierr);
    ierr = ISGetSize(color->isa[k],&isize); CHKERRQ(ierr);
    for (l=0; l<isize; l++) {
      i = is[l];
      wscale[l] = 0.0;
      scale[l]  = 0.0;
      if ( i>= start && i<end) {
        dx = xx[i-start];
#if !defined(PETSC_COMPLEX)
        if (dx < 1.e-16 && dx >= 0.0) dx = 1.e-1;
        else if (dx < 0.0 && dx > -1.e-16) dx = -1.e-1;
#else
        if (abs(dx) < 1.e-16 && real(dx) >= 0.0) dx = 1.e-1;
        else if (real(dx) < 0.0 && abs(dx) < 1.e-16) dx = -1.e-1;
#endif
        dx *= epsilon;
        wscale[l] = 1.0/dx;
        VecSetValues(x2,1,&i,&dx,ADD_VALUES); 
      } 
      else {
        wscale[l] = 0.0;
      }
    }
    ierr = eval_fct(snes,x2,jj2); CHKERRQ(ierr);
    ierr = VecAXPY(&mone,jj1,jj2); CHKERRQ(ierr);
    /* Communicate scale to all processors */
#if !defined(PETSC_COMPLEX)
    count = isize;
    MPI_Allreduce(&wscale,&scale,count,MPI_DOUBLE,MPI_SUM,comm);
#else
    count = 2*isize;
    MPI_Allreduce(&wscale,&scale,count,MPI_DOUBLE,MPI_SUM,comm);
#endif
    /* VecScale(&scale,jj2); */
    VecGetArray(jj2,&y);
    for (l=0; l<isize; l++) {
      i = is[l];
      for ( j=start; j<end; j++ ) {
        if (i>= start && i<end) y[j-start] *= scale[l];
      }
    }
    VecNorm(jj2,NORM_INFINITY,&amax); amax *= 1.e-14;
    for ( j=start; j<end; j++ ) {
      for (l=0; l<isize; l++) {
        i = is[l];
        if (PetscAbsScalar(y[j-start]) > amax) {
          ierr = MatSetValues(*J,1,&j,1,&i,y+j-start,INSERT_VALUES); CHKERRQ(ierr);
        }
      }
    }
    VecRestoreArray(jj2,&y);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatView(*J,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  return 0;
}

