#ifndef lint
static char vcid[] = "$Id: snesregi.c,v 1.6 1995/06/29 23:54:14 bsmith Exp curfman $";
#endif

static char help[] = 
"This example uses Newton-like methods to solve u`` + u^{2} = f\n\
 in parallel.\n";

#include "draw.h"
#include "da.h"
#include "snes.h"
#include "petsc.h"
#include <math.h>

int  FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(SNES,Vec,Vec,void*),
     FormInitialGuess(SNES,Vec,void*);

typedef struct {
   DA     da;
   Vec    F,xl;
   int    mytid,numtid;
   double h;
} ApplicationCtx;

int main( int argc, char **argv )
{
  SNES           snes;               /* SNES context */
  SNESMethod     method = SNES_NLS;  /* nonlinear solution method */
  Vec            x,r,U,F;
  Mat            J;                  /* Jacobian matrix */
  int            ierr, its, N = 5,i,start,end,n;
  Scalar         xp,v,*FF,*UU;
  ApplicationCtx ctx;

  PetscInitialize( &argc, &argv, 0,0 );
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  OptionsGetInt(0,"-n",&N);
  ctx.h = 1.0/(N-1);

  MPI_Comm_rank(MPI_COMM_WORLD,&ctx.mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&ctx.numtid);

  /* Set up data structures */
  ierr = DACreate1d(MPI_COMM_WORLD,N,1,1,DA_NONPERIODIC,&ctx.da);
  CHKERRA(ierr);
  ierr = DAGetDistributedVector(ctx.da,&x); CHKERRA(ierr);
  ierr = DAGetLocalVector(ctx.da,&ctx.xl); CHKERRQ(ierr);

  PetscObjectSetName((PetscObject)x,"Approximate Solution");
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr); ctx.F = F;
  ierr = VecDuplicate(x,&U); CHKERRA(ierr); 
  PetscObjectSetName((PetscObject)U,"Exact Solution");
  ierr=MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,3,0,
                       0,0,&J); CHKERRA(ierr);

  /* Store right-hand-side of PDE and exact solution */
  ierr = VecGetOwnershipRange(x,&start,&end); CHKERRQ(ierr);
  ierr = VecGetArray(F,&FF); CHKERRQ(ierr);
  ierr = VecGetArray(U,&UU); CHKERRQ(ierr);
  xp = ctx.h*start; n = end - start;
  for ( i=0; i<n; i++ ) {
    FF[i] = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    UU[i] = xp*xp*xp;
    xp += ctx.h;
  }
  VecAssemblyBegin(F);
  VecAssemblyEnd(F);
  VecAssemblyBegin(U);
  VecAssemblyEnd(U);

  /* Create nonlinear solver */  
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);
  CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetSolution(snes,x,FormInitialGuess,0); CHKERRA(ierr);
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&ctx,1); CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&ctx); CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);
  ierr = SNESSetUp(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,&its); CHKERRA(ierr);
  MPIU_printf(MPI_COMM_WORLD,"Number of Newton iterations = %d\n\n", its );

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(U); CHKERRA(ierr);
  ierr = VecDestroy(F); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

int FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
   ApplicationCtx *ctx = (ApplicationCtx*) dummy;
   DA             da = (DA) ctx->da;
   Scalar         *xx, *ff,*FF,d;
   int            i, ierr, n,mytid = ctx->mytid,numtid = ctx->numtid,s;
   Vec            xl;

   xl = ctx->xl; 
   ierr = DAGlobalToLocalBegin(da,x,INSERTVALUES,xl); CHKERRQ(ierr);
   ierr = DAGlobalToLocalEnd(da,x,INSERTVALUES,xl); CHKERRQ(ierr);

   ierr = VecGetArray(xl,&xx); CHKERRQ(ierr);
   ierr = VecGetArray(f,&ff); CHKERRQ(ierr);
   ierr = VecGetArray(ctx->F,&FF); CHKERRQ(ierr);
   ierr = VecGetLocalSize(xl,&n); CHKERRQ(ierr);
   d = ctx->h*ctx->h;
   if (mytid == 0) {s = 0; ff[0] = -xx[0];} else s = 1;
   for ( i=1; i<n-1; i++ ) {
     ff[i-s] = -d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) - xx[i]*xx[i] + FF[i-s];
   }
   if (mytid == numtid-1) ff[n-1-s] = -xx[n-1] + 1.0;
   return 0;
}
/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess(SNES snes,Vec x,void *dummy)
{
   int    ierr;
   Scalar pfive = .50;
   ierr = VecSet(&pfive,x); CHKERRQ(ierr);
   return 0;
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure*flag,void *dummy)
{
  ApplicationCtx *ctx = (ApplicationCtx*) dummy;
  DA             da = (DA) ctx->da;
  Scalar         *xx, *ff,*FF,d, A, s;
  int            i, j, ierr, n,mytid = ctx->mytid,numtid = ctx->numtid;
  Vec            xl;
  int            start,end,N,ii,istart,iend;

  xl = ctx->xl; 
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  ierr =  VecGetOwnershipRange(x,&start,&end); CHKERRQ(ierr);
  n = end - start; 
  d = ctx->h*ctx->h;
  ierr = VecGetSize(x,&N); CHKERRQ(ierr);
  if (mytid == 0) {
    A = 1.0; 
    ierr = MatSetValues(*jac,1,&start,1,&start,&A,INSERTVALUES); CHKERRQ(ierr);
    istart = 1;
  }
  else {istart = 0; }
  if (mytid == numtid-1) {
    i = N-1; A = 1.0; 
    ierr = MatSetValues(*jac,1,&i,1,&i,&A,INSERTVALUES); CHKERRQ(ierr);
    iend = n-1;
  }
  else iend = n;
  for ( i=istart; i<iend; i++ ) {
    ii = i + start;
    j = start + i - 1; 
    ierr = MatSetValues(*jac,1,&ii,1,&j,&d,INSERTVALUES); CHKERRQ(ierr);
    j = start + i + 1; 
    ierr = MatSetValues(*jac,1,&ii,1,&j,&d,INSERTVALUES); CHKERRQ(ierr);
    A = -2.0*d + 2.0*xx[i];
    ierr = MatSetValues(*jac,1,&ii,1,&ii,&A,INSERTVALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*jac,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}


