#ifndef lint
static char vcid[] = "$Id: ex8.c,v 1.32 1996/03/05 16:27:35 balay Exp bsmith $";
#endif

static char help[] = "Uses Newton-like methods to solve u`` + u^{2} = f\n\
 in parallel.\n";

#include "draw.h"
#include "da.h"
#include "snes.h"
#include <math.h>

int  FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(SNES,Vec,Vec,void*),
     FormInitialGuess(SNES,Vec);

typedef struct {
   DA     da;
   Vec    F,xl;
   int    rank,size;
   double h;
} ApplicationCtx;

int main( int argc, char **argv )
{
  SNES           snes;                  /* SNES context */
  SNESType       method = SNES_EQ_NLS;  /* nonlinear solution method */
  Mat            J;                     /* Jacobian matrix */
  ApplicationCtx ctx;                   /* user-defined context */
  Vec            x, r, U, F;
  Scalar         xp, *FF, *UU;
  int            ierr, its, N = 5, i, start, end, n, set,flg;
  MatType        mtype=MATMPIAIJ;

  PetscInitialize( &argc, &argv,(char *)0,help );
  ierr = OptionsGetInt(PETSC_NULL,"-n",&N,&flg); CHKERRA(ierr);
  ctx.h = 1.0/(N-1);

  MPI_Comm_rank(MPI_COMM_WORLD,&ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD,&ctx.size);

  /* Set up data structures */
  ierr = DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,N,1,1,&ctx.da);CHKERRA(ierr);
  ierr = DAGetDistributedVector(ctx.da,&x); CHKERRA(ierr);
  ierr = DAGetLocalVector(ctx.da,&ctx.xl); CHKERRQ(ierr);

  PetscObjectSetName((PetscObject)x,"Approximate Solution");
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr); ctx.F = F;
  ierr = VecDuplicate(x,&U); CHKERRA(ierr); 
  PetscObjectSetName((PetscObject)U,"Exact Solution");
  ierr = MatGetTypeFromOptions(MPI_COMM_WORLD,0,&mtype,&set); CHKERRA(ierr);
  if (mtype == MATMPIBDIAG) {
    int diag[3]; diag[0] = -1; diag[1] = 0; diag[2] = 1;
    ierr = MatCreateMPIBDiag(MPI_COMM_WORLD,PETSC_DECIDE,N,N,3,1,diag,
           PETSC_NULL,&J); CHKERRA(ierr);
  } else if (mtype == MATSEQAIJ) {
    ierr = MatCreateSeqAIJ(MPI_COMM_WORLD,N,N,3,PETSC_NULL,&J);CHKERRA(ierr);
  } else {
    ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,3,
           PETSC_NULL,0,PETSC_NULL,&J); CHKERRA(ierr);
  }

  /* Store right-hand-side of PDE and exact solution */
  ierr = VecGetOwnershipRange(x,&start,&end); CHKERRA(ierr);
  ierr = VecGetArray(F,&FF); CHKERRA(ierr);
  ierr = VecGetArray(U,&UU); CHKERRA(ierr);
  xp = ctx.h*start; n = end - start;
  for ( i=0; i<n; i++ ) {
    FF[i] = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    UU[i] = xp*xp*xp;
    xp += ctx.h;
  }
  ierr = VecRestoreArray(F,&FF); CHKERRA(ierr);
  ierr = VecRestoreArray(U,&UU); CHKERRA(ierr);

  /* Create nonlinear solver */  
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);
  ierr = SNESSetType(snes,method); CHKERRA(ierr);

  /* Set various routines and options */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&ctx);CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&ctx); CHKERRA(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* Solve nonlinear system */
  ierr = FormInitialGuess(snes,x); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  PetscPrintf(MPI_COMM_WORLD,"Number of Newton iterations = %d\n\n", its );

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(ctx.xl); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(U); CHKERRA(ierr);
  ierr = VecDestroy(F); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = DADestroy(ctx.da); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

int FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
   ApplicationCtx *ctx = (ApplicationCtx*) dummy;
   DA             da = (DA) ctx->da;
   Scalar         *xx, *ff,*FF,d;
   int            i, ierr, n,rank = ctx->rank,size = ctx->size,s;
   Vec            xl;

   xl = ctx->xl; 
   ierr = DAGlobalToLocalBegin(da,x,INSERT_VALUES,xl); CHKERRQ(ierr);
   ierr = DAGlobalToLocalEnd(da,x,INSERT_VALUES,xl); CHKERRQ(ierr);
   ierr = VecGetArray(xl,&xx); CHKERRQ(ierr);
   ierr = VecGetArray(f,&ff); CHKERRQ(ierr);
   ierr = VecGetArray(ctx->F,&FF); CHKERRQ(ierr);
   ierr = VecGetLocalSize(xl,&n); CHKERRQ(ierr);
   d = ctx->h*ctx->h;
   if (rank == 0) {s = 0; ff[0] = xx[0];} else s = 1;
   for ( i=1; i<n-1; i++ ) {
     ff[i-s] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i-s];
   }
   if (rank == size-1) ff[n-1-s] = xx[n-1] - 1.0;
   ierr = VecRestoreArray(xl,&xx); CHKERRQ(ierr);
   ierr = VecRestoreArray(f,&ff); CHKERRQ(ierr);
   ierr = VecRestoreArray(ctx->F,&FF); CHKERRQ(ierr);
   return 0;
}
/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess(SNES snes,Vec x)
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
  Scalar         *xx, d, A;
  int            i, j, ierr, n, rank = ctx->rank, size = ctx->size;
  int            start, end, N, ii, istart, iend;
  Vec            xl;

  xl = ctx->xl; 
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  ierr =  VecGetOwnershipRange(x,&start,&end); CHKERRQ(ierr);
  n = end - start; 
  d = ctx->h*ctx->h;
  ierr = VecGetSize(x,&N); CHKERRQ(ierr);
  if (rank == 0) {
    A = 1.0; 
    ierr = MatSetValues(*jac,1,&start,1,&start,&A,INSERT_VALUES); CHKERRQ(ierr);
    istart = 1;
  }
  else {istart = 0; }
  if (rank == size-1) {
    i = N-1; A = 1.0; 
    ierr = MatSetValues(*jac,1,&i,1,&i,&A,INSERT_VALUES); CHKERRQ(ierr);
    iend = n-1;
  }
  else iend = n;
  for ( i=istart; i<iend; i++ ) {
    ii = i + start;
    j = start + i - 1; 
    ierr = MatSetValues(*jac,1,&ii,1,&j,&d,INSERT_VALUES); CHKERRQ(ierr);
    j = start + i + 1; 
    ierr = MatSetValues(*jac,1,&ii,1,&j,&d,INSERT_VALUES); CHKERRQ(ierr);
    A = -2.0*d + 2.0*xx[i];
    ierr = MatSetValues(*jac,1,&ii,1,&ii,&A,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*jac,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,FINAL_ASSEMBLY); CHKERRQ(ierr);
  /* MatView(*jac,STDOUT_VIEWER_SELF); */
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
  return 0;
}


