#include "puser.h"

/*
------------------------------------------------------------------------
   This file contains routines for computing function values for the 2D
   potential flow problem in parallel. A finite difference approximation
   with the standard 9-point stencil is used to discretize the boundary
   value problem to obtain a nonlinear system of equations.
--------------------------------------------------------------------------
*/

/* 
   InitialGuess_PotentialFlow - Computes the initial guess for the 2D
   potential flow problem in parallel.

   Input Parameter:
.  user - user-defined application context

   Output Parameter:
.  X - newly computed initial guess

   Notes:
   This routine should be called before SNESSolve().
 */
int InitialGuess_PotentialFlow(AppCtx *user,Vec X)
{
  int    i, j, ierr, row, xs, xe, ys, ye, Xs, Xm, Ys;
  double Qinf, hx, xx0, *x;
  Vec    localX = user->localX;

  hx   = user->hx;
  Xs   = user->Xs;
  Xm   = user->Xm;
  Ys   = user->Ys;
  xs   = user->xs;
  xe   = user->xe; 
  ys   = user->ys; 
  ye   = user->ye;
  xx0  = user->xx0;
  Qinf = user->Qinf;

  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  for (j=ys; j<ye; j++) {
    for (i=xs; i<xe; i++) {
      row = i - Xs + (j - Ys)*Xm; 
      x[row]=Qinf*(i*hx+xx0);
     }
  }
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);

  /* Place local values in global vector */
  ierr = DALocalToGlobal(user->da,localX,INSERT_VALUES,X); CHKERRQ(ierr);
  return 0;
}       

static int ComputeDensity(double*,AppCtx*);
/* --------------------------------------------------------------- */
/* 
   Function_PotentialFlow - Evaluates the function for the 2D potential
   flow problem in parallel.

   Input Parameters:
.  snes - SNES context
.  user - user-defined application context
.  X - current iterate vector

   Output Parameter:
.  F - newly computed function vector

   Notes:
   This routine should be set for use by the SNES solvers by calling
   SNESSetFunction() before SNESSolve().
 */
int Function_PotentialFlow(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx  *user = (AppCtx *) ptr;
  int     i, j, row, mx, my, ierr;
  double  hx, hy, xx0, xx1, yy0, yy1, Qinf, M;
  double  *x, *f, *Mach;
  Vec     localX = user->localX, localF = user->localF;
  Vec     localMach = user->localMach;
  double  Ux, Uy, d, Q20, Q2inf, a00;
  int     xs, xe, ys, ye, Xs, Xm, Ys;

  hx    = user->hx;
  hy    = user->hy;
  Xs    = user->Xs;
  Xm    = user->Xm;
  Ys    = user->Ys;
  xs    = user->xs;
  xe    = user->xe; 
  ys    = user->ys; 
  ye    = user->ye;
  mx    = user->mx; 
  my    = user->my;
  xx0   = user->xx0;
  xx1   = user->xx1;
  yy0   = user->yy0;
  yy1   = user->yy1;
  M     = user->M;
  Qinf  = user->Qinf;
  Q2inf = Qinf*Qinf;
  user->machflag = 1;

  /* Set ghost points for current iterate in local work vector */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);

  /* Get pointers to local vector data */
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f); CHKERRQ(ierr);
  ierr = VecGetArray(localMach,&Mach); CHKERRQ(ierr);

  /* Compute density */
  ierr = ComputeDensity(x,user); CHKERRQ(ierr);

  /* Evaluate function on each processor's local grid section */
  for (j=ys; j<ye; j++) {
    for (i=xs; i<xe; i++) {
      row = i - Xs + (j - Ys)*Xm; 

      /* freestream */
      if (j == 0 && i != mx-1 && i != 0) {
        /* Compute pressure for transpiration model */
        if ( user->machflag ) {
           Ux = (x[row+1] - x[row])/hx;
           Uy = (x[row+Xm] - x[row])/hy;
           Q20=Ux*Ux+Uy*Uy;
           a00=1+(gama-1)*M*M/2*(1-Q20/Q2inf); 
           d = gama/(gama-1);
           a00 = pr_infty*pow(a00,d);
           ierr = VecSetValues(user->globalPressure,1,&i,&a00,INSERT_VALUES); CHKERRQ(ierr);
         }
      }

      /* Evaluate function at grid point (i,j) */
      ierr = EvaluateFunction(user,x,i,j,Mach,f); CHKERRQ(ierr);
    }
  }

  /* Place newly computed function values in global vector */
  ierr = DALocalToGlobal(user->da,localF,INSERT_VALUES,F); CHKERRQ(ierr);

  /* Save a copy of the function for later use in Jacobian evaluation */
  ierr = VecCopy(F,user->Fcopy); CHKERRQ(ierr);
 
  /* Compute local Mach number at boundaries; interior points are handled by the
     routine EvaluateFunction(). */
  for (j=ys; j<ye; j++) {
    for (i=xs; i<xe; i++) {
       row = i - Xs + (j - Ys)*Xm; 
       /* upstream */
       if (i == 0) {
              Mach[row] = Mach[row+1];
              continue;     
       }
       /* freestream */
       if (j == my-1) {
              Mach[row] = Mach[row-Xm];
              continue;
       }
       /* downstream */
       if (i == mx-1) { 
             Mach[row] = Mach[row-1]; 
             continue;
       }
       /* freestream */
       if (j == 0) { 
          /* airfoil and freestream */
          Mach[row] = Mach[row+Xm];
          continue;
       }
    }
  }
  /* Restore vectors so that they are ready for later use */
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f); CHKERRQ(ierr);
  ierr = VecRestoreArray(localMach,&Mach); CHKERRQ(ierr);

  /* Place newly computed local Mach number vectors in global vector */
  ierr = DALocalToGlobal(user->da,localMach,INSERT_VALUES,user->globalMach); CHKERRQ(ierr);
  user->machflag = 0;

  return 0;
}
/* --------------------------------------------------------------- */
/*
   EvaluateFunction - Evaluates the function f(x) at the grid point (i,j)
   for the 2D potential flow problem.  This lower-level routine 
   handles the entire grid (including boundaries) and assumes that
   all necessary communication of ghost points for x is handled prior
   to calling this routine. 

   Input Parameters:
.  user - user-defined application context
.  x - current iterate (or perturbed iterate for Jacobian approx)
.  i,j - global grid point numbers in the x- and y-directions
.  Mach - Mach number
.  f - f(x)

   Notes:
   The standard, 9-point finite difference stencil for 2D discretizes
   the boundary problem to obtain a nonlinear system of equations.

   EvaluateFunction() is called by 3 routines in this application code:
    - Function_PotentialFlow() - basic function evaluation
    - Jacobian_PotentialFlow() - finite difference approximation of Jacobian
       - InnerSparseFunction() - interior grid points for each processor 
       - MySparseFunction() - edge points for each processor
*/
int EvaluateFunction(AppCtx *user,double *x,int i,int j,double *Mach,double *f)
{
   int      ierr, row, mx, my;
   double   hx, hy;
   double   U, Ut, Ub, Ul, Ur, Utr, Utl, Ubr, Ubl;
   double   Ux, Uy;
   double   Q2n, Q2s, Q2e, Q2w, Q2inf, an, as, aw, ae, a0, a00;
   double   xx0, xx1, yy0, yy1, Qinf, M, b, d;
   double   anhalf, ashalf, awhalf, aehalf, M2c, Unorm, Q20, mu;
   double   loca1, loca2, mach_number;
   double   *density;
   int      xs, xe, ys, ye, Xs, Xm, Ys;

   hx    = user->hx;
   hy    = user->hy;
   Xs    = user->Xs;
   Xm    = user->Xm;
   Ys    = user->Ys;
   xs    = user->xs;
   xe    = user->xe; 
   ys    = user->ys; 
   ye    = user->ye;
   mx    = user->mx; 
   my    = user->my;
   xx0   = user->xx0;
   xx1   = user->xx1;
   yy0   = user->yy0;
   yy1   = user->yy1;
   M     = user->M;
   Qinf  = user->Qinf;

   M2c   = 0.95;
   Q2inf = Qinf*Qinf;

   row  = i - Xs + (j - Ys)*Xm;

  /* -------- This section of code applies boundary conditions --------- */

   /* freestream */
   if (j == my-1 && i != mx-1) {
             f[row] = (x[row]-Qinf*(hx*i+xx0));
   return 0;
   }

   /* freestream */
   if (j == 0 && i != mx-1 && i != 0) {
              /* airfoil */
              if ( (hx*i+xx0) >= -1 && (hx*i+xx0) <= 1 )
                    f[row] = ( (x[row+Xm]-x[row])/hy + height
                       *pi*Qinf*cos(pi/2*(hx*i+xx0))*sin(pi/2*(hx*i+xx0)) );
              else     
                    /* other positions */
                    f[row] = (x[row+Xm]-x[row])/hy;
              return 0;
   }

   /* downstream */
   if (i == mx-1 && j !=0 && j != my-1) {
              /* f[row] = ( (3*x[row]-4*x[row-1]+x[row-2])/(2*hx) -Qinf ); */
              /* f[row] = ( (x[row]-2*x[row-1]+x[row-2])/(hx*hx) ); */
              /* f[row] = (x[row]-Qinf*(hx*i+xx0)); */

              if ( M<=1.0 ) f[row] = ( (x[row] - x[row-1])/hx - Qinf ); 
              else          f[row] = ( (x[row] - x[row-1]) );
              return 0;
   }
   if (i == mx-1 && j == my-1) {                /* upper right  corner */
              /* f[row] = ( (x[row] - x[row-1])/hx - Qinf ); */
              /* f[row] = (x[row]-Qinf*(hx*i+xx0)); */
              /* f[row] = ( f[row] - (x[row]-Qinf*(hx*i+xx0)) )/2; */

              if ( M<=1.0 ) f[row] = ( x[row] - (x[row-1]+x[row-Xm])/2 );
              else          f[row] = ( (x[row] - x[row-1]) );
              return 0;
   }
   if (i == mx-1 && j == 0) {                   /* lower right corner */
              /* f[row] = ( (x[row] - x[row-1])/hx - Qinf ); */
              /* f[row] = (x[row]-Qinf*(hx*i+xx0)); */
              /* f[row] = ( f[row] -(x[row+Xm]-x[row])/hy )/2; */

              if ( M<=1.0 ) f[row] = ( x[row] - (x[row-1]+x[row+Xm])/2 );
              else          f[row] = ( (x[row] - x[row-1]) );
              return 0;
   }

   /* upstream */
   if (i == 0  && j != 0) { 
              f[row] = (x[row]-Qinf*xx0);
              return 0;
   }
   if (i == 0 && j == 0) {                /* lower left corner */
              /* f[row] = (x[row]-Qinf*xx0); */
              /* f[row] = ( f[row] - (x[row+Xm]-x[row])/hy )/2; */

              f[row] = ( x[row] - (x[row+1]+x[row+Xm])/2 );  
              return 0;              
   }

   /* ------- This section of code is for interior grid points only! ------- */
   /* Evaluate the full potential residual at point (i,j) using the standard
      nine-point FD stencil */

   /* Get density from saved results in user structure */  
   ierr = VecGetArray(user->localDensity,&density); CHKERRQ(ierr);

         U   = x[row];                  /* current point */
         Ub  = x[row - Xm];             /* bottom point */
         Ul  = x[row - 1];              /* left point */
         Ut  = x[row + Xm];             /* top point */
         Ur  = x[row + 1];              /* right point */
         Utr = x[row + Xm + 1];         /* top-right point */
         Utl = x[row + Xm - 1];         /* top-left point */
         Ubr = x[row - Xm + 1];         /* bottom-right point */
         Ubl = x[row - Xm - 1];         /* bottom-left point */

         /* Q-squared north (i,j+1) */
         Ux=(Utr-Utl)/(2*hx);
         Uy=(Ut-U)/hy;
         Q2n=Ux*Ux+Uy*Uy;
 
         /* Q-squared south (i,j-1) */
         Ux=(Ubr-Ubl)/(2*hx);
         Uy=(U-Ub)/hy;
         Q2s=Ux*Ux+Uy*Uy;
 
         /* Q-squared, staggered east (i+1,j) */
         Ux=(Ur-U)/hx;
         Uy=(Utr-Ubr)/(2*hy);
         Q2e=Ux*Ux+Uy*Uy;
 
         /* Q-squared, staggered west (i-1,j) */
         Ux=(U-Ul)/hx;
         Uy=(Utl-Ubl)/(2*hy);
         Q2w=Ux*Ux+Uy*Uy;
 
         /* Velocity at centerpoint (i,j)  */
         Ux=(Ur-Ul)/(2*hx);
         Uy=(Ut-Ub)/(2*hy);

Q20=Ux*Ux+Uy*Uy;

         /* Multiplier of q-squared term */
         b=(gama-1)*M*M/2;
 
         /* Exponent of density term */
         d=1/(gama-1);
 
if (!user->machflag) {
  a00=1+b*(1-Q20/Q2inf);
  a0=Pinf*pow(a00,d);
}
else 
  a0 = density[row]; 

/*
if (user->machflag && a00 != a0) printf("bad, a0=%5.10g, density[%d]=%5.10g\n", a0, row, a00 ); if (!user->machflag && a00 != a0) printf("BAD, a0=%5.10g, density[%d]=%5.10g\n", a0, row, a00 );
*/

         /* Densities, at the four corner points and middle */
         an=1+b*(1-Q2n/Q2inf);
         an=Pinf*pow(an,d);
         as=1+b*(1-Q2s/Q2inf);
         as=Pinf*pow(as,d);
         ae=1+b*(1-Q2e/Q2inf);
         ae=Pinf*pow(ae,d);
         aw=1+b*(1-Q2w/Q2inf);
         aw=Pinf*pow(aw,d);

         /* compute local pressure for other model */
         /*
         if ( user->machflag && ((hx*i+xx0) >= -1 && (hx*i+xx0) <= 1) && 
          fabs(hy*j-height*cos(pi/2*(hx*i+xx0))*cos(pi/2*(hx*i+xx0))) <= ((double)hy)/2. ) {
            d = gama/(gama-1);
            a00 = pr_infty*pow(a00,d);
	    VecSetValues(user->globalPressure,1,&i,&a00,INSERT_VALUES);
         }
         */

         /* compute local Mach number */
         loca1 = pow(a0/Pinf, gama-1) - 1;
         loca2 = sqrt(1 - 2*loca1/(M*M*(gama-1)));
         mach_number  = M*pow(Pinf/a0, (gama-1)/2)*loca2;

         /* store local Mach number into user context */
         if (user->machflag) Mach[row] = mach_number;

         /* Normalized velocity */
         Unorm=fabs(Ux)+fabs(Uy);
         Ux=Ux/Unorm;
         Uy=Uy/Unorm;

         /* upwinding switch */
         mu=Max(0,1-M2c/(mach_number*mach_number));
         
         /* upwinding */
         if (mu == 0) {
            awhalf = 0.5*(a0+aw);
            aehalf = 0.5*(a0+ae);
            anhalf = 0.5*(a0+an);
            ashalf = 0.5*(a0+as);
          }
          else{
            if (Ux > 0) {
              awhalf=mu*Ux*aw+(1-mu*Ux)*a0;  
              aehalf=mu*Ux*a0+(1-mu*Ux)*ae;
            }
            else {
              aehalf=-mu*Ux*ae+(1+mu*Ux)*a0;  
              awhalf=-mu*Ux*a0+(1+mu*Ux)*aw;  
            }
            if (Uy > 0) {
              ashalf=mu*Uy*as+(1-mu*Uy)*a0;  
              anhalf=mu*Uy*a0+(1-mu*Uy)*an;  
            }
            else {
              anhalf=-mu*Uy*an+(1+mu*Uy)*a0;  
              ashalf=-mu*Uy*a0+(1+mu*Uy)*as;  
            }
         }

         /* Formation of second-differences */
         f[row]=(
                        (anhalf*(Ut-U)-ashalf*(U-Ub))/(hy*hy)
                   +
                        (aehalf*(Ur-U)-awhalf*(U-Ul))/(hx*hx)
                 );

   ierr = VecRestoreArray(user->localDensity,&density); CHKERRQ(ierr);

   return 0;
}
/* --------------------------------------------------------------- */
/*
   ComputeDensity - Computes central density and stores in a vector
   vector within application context.

   Input Parameters:
.  user - user-defined application context
.  x - current iterate

   Notes:
   ComputeDensity() is called within the routine Function_PotentialFlow().
*/
int ComputeDensity(double *x,AppCtx *user)
{
  Vec    localDensity = user->localDensity; 
  double Ux, Uy, Ut, Ub, Ul, Ur, hx, hy;
  double Qinf, M, b, d, Q20, Q2inf, a0, *density;
  int    i, j, ierr, row, mx, my;
  int    xs, xe, ys, ye, Xs, Xm, Ys;

  hx    = user->hx;
  hy    = user->hy;
  Xs    = user->Xs;
  Xm    = user->Xm;
  Ys    = user->Ys;
  xs    = user->xs;
  xe    = user->xe; 
  ys    = user->ys; 
  ye    = user->ye;
  mx    = user->mx; 
  my    = user->my;
  M     = user->M;
  Qinf  = user->Qinf;
  Q2inf = Qinf*Qinf;

  ierr = VecGetArray(localDensity,&density); CHKERRQ(ierr);
  for (j=ys; j<ye; j++) {
    for (i=xs; i<xe; i++) {
       row = i - Xs + (j - Ys)*Xm; 
       if ( i == 0 || i == mx-1 || j == 0 || j == my-1 ) {
         density[row] = 0.0;
       }
       else { /* Evaluate the full potential density at point (i,j)
                 using a four-point FD stencil */

         Ub  = x[row - Xm];  /* bottom point */
         Ul  = x[row - 1];   /* left point */
         Ut  = x[row + Xm];  /* top point */
         Ur  = x[row + 1];   /* right point */

         /* Velocity at centerpoint (i,j)  */
         Ux=(Ur-Ul)/(2*hx);
         Uy=(Ut-Ub)/(2*hy);
         Q20=Ux*Ux+Uy*Uy;

         /* Multiplier of q-squared term */
         b=(gama-1)*M*M/2;
             
         /* Exponent of density term */
         d=1/(gama-1); 

         a0=1+b*(1-Q20/Q2inf);
         density[row]=Pinf*pow(a0,d);  
      }  
    }
  }
  ierr = VecRestoreArray(localDensity,&density); CHKERRQ(ierr);
  return 0;
}
