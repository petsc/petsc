
#include "mgimpl.h"

/*@
      MGCycle - Runs either an additive, multiplicative or full cycle of 
                multigrid. 

  Input Parameters:
.   mg - the multigrid context 
.   am - either Multiplicative, Additive or FullMultigrid 


  Note: a simple  wrapper which calls MGMCycle(),MGACycle(), or MGFCycle(). 
@*/ 
int MGCycle(MG *mg,int am)
{
   if (am == Multiplicative)      return MGMCycle(mg); 
   else if (am == Additive)       return MGACycle(mg);
   else                           return MGFCycle(mg);
   return 0;
}

/*@
      MGGetCoarseSolve - Gets the solver contex to be used on the 
                         coarse grid.

  Input Parameters:
.   mg - the multigrid context 

  Output Parameters:
.   sles - the coarse grid solver context 

@*/ 
int MGGetCoarseSolve(MG *mg,SLES *sles)  
{ 
  *sles =  mg[mg[0]->level]->csles;  
  return 0;
}

/*@
     MGDefaultResidual - Default routine to calculate the residual.

  Input Parameters:
.  mat - the matrix
.  b - the right hand side
.  x - the approximate solution
 
  Output Parameters:
.  r - location to store the residual
int MGDefaultResidual(Mat mat,Vec b,Vec x,Vec r)
{
  int    ierr;
  Scalar mone = -1.0;
  ierr = MatMult(mat,x,r); CHKERR(ierr);
  ierr = VecAYPX(&mone,b,r); CHKERR(ierr);
  return 0;
}

/*@
      MGSetResidual - Sets the function to be used to calculate the 
                      residual on the lth level. 

  Input Paramters:
.  mg - the multigrid context
.  l - the level to supply
.  mat - matrix associated with residual
.  residual - function used to form residual (usually MGDefaultResidual)
@*/
int MGSetResidual(MG *mg,int l,int (*residual)(Mat,Vec,Vec,Vec),Mat mat) 
{
  mg[mg[0]->level - l]->residual = residual;  
  mg[mg[0]->level - l]->A        = mat;
  return 0;
}

/*@
      MGSetInterpolate - Sets the function to be used to calculate the 
                      interpolation on the lth level. 

  Input Parameters:
.   mg - the multigrid context
.   mat - the interpolation operator
.  l - the level to supply

@*/
int MGSetInterpolate(MG *mg,int l,Mat mat)
{ 
  mg[mg[0]->level - l]->interpolate = mat;  
  return 0;
}


/*@
      MGSetRestriction - Sets the function to be used to restrict vector
                        from lth level to l-1. 

  Input Parameters:
.   mg - the multigrid context 
.   sles - the smoother
.  l - the level to supply

@*/
int MGSetRestriction(MG *mg,int l,Mat mat)  
{
  mg[mg[0]->level - l]->restrict  = mat;  
  return 0;
}

/*@
      MGGetSmoother - Gets the SLES context to be used as smoother for 
                      both pre and post smoothing. If you want a different
                      function for each call both MGGetSmootherUp() and 
                      MGGetSmootherDown().

  Input Parameters:
.   mg - the multigrid context 
.  l - the level to supply

  Ouput Parameters:
.   sles - the smoother
@*/
int MGGetSmoother(MG *mg,int l,SLES *sles)
{
  *sles = mg[mg[0]->level - l]->smoothu;  
  return 0;
}

/*@
      MGGetSmootherUp - Gets the SLES context to be used as smoother after 
                        coarse grid correction (post-smoother). 

  Input Parameters:
.   mg - the multigrid context 
.  l - the level to supply

  Ouput Parameters:
.   sles - the smoother
@*/
int MGGetSmootherUp(MG *mg,int l,SLES *sles)
{
  *sles = mg[mg[0]->level - l]->smoothu;  
  return 0;
}

/*@
      MGGetSmootherDown - Gets the SLES context to be used as smoother before 
                        coarse grid correction (pre-smoother). 

  Input Parameters:
.   mg - the multigrid context 
.   l - the level to supply

  Ouput Parameters:
.   sles - the smoother
@*/
int MGGetSmootherDown(MG *mg,int l,SLES *sles)
{
  int ierr;
  /*
     This is called only if user wants a different pre-smoother from post.
     Thus we check if a different one has already been allocated, 
     if not we allocate it.
  */
  if (mg[mg[0]->level - 1]->smoothd == mg[mg[0]->level -1]->smoothu) {
    ierr = SLESCreate(&mg[mg[0]->level - 1]->smoothd); CHKERR(ierr);
  }
  *sles = mg[mg[0]->level - l]->smoothd;
  return 0;
}

/*@
      MGSetCyclesOnLevel - Sets the number of cycles to run on this level. 

  Input Parameters:
.   mg - the multigrid context 
.   l - the level this is to be used for
.   n - the number of cycles

@*/
int MGSetCyclesOnLevel(MG *mg,int l,int c) 
{
  mg[mg[0]->level - l]->cycles  = c;
  return 0;
}

/*@
      MGSetRhs - Sets the vector space to be used to store right hand 
                 side on a particular level. User should free this 
                 space at conclusion of multigrid use. 

  Input Parameters:
.   mg - the multigrid context 
.   l - the level this is to be used for
.   c - the space

@*/
int MGSetRhs(MG *mg,int l,Vec c)  
{ 
  mg[mg[0]->level - l]->b  = c;
  return 0;
}

/*@
      MGSetX - Sets the vector space to be used to store solution 
                 on a particular level.User should free this 
                 space at conclusion of multigrid use.

  Input Parameters:
.   mg - the multigrid context 
.   l - the level this is to be used for
.   c - the space

@*/
int MGSetX(MG *mg,int l,Vec c)  
{ 
  mg[mg[0]->level - l]->x  = c;
  return 0;
}

/*@
      MGSetR - Sets the vector space to be used to store residual 
                 on a particular level. User should free this 
                 space at conclusion of multigrid use.

  Input Parameters:
.   mg - the multigrid context 
.   l - the level this is to be used for
.   c - the space

@*/
int MGSetR(MG *mg,int l,Vec c)
{ 
  mg[mg[0]->level - l]->r  = c;
  return 0;
}


