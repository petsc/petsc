#ifndef lint
static char vcid[] = "$Id: $";
#endif

#include "mgimpl.h"


/*@
      MGGetCoarseSolve - Gets the solver contex to be used on the 
                         coarse grid.

  Input Parameters:
.   mg - the multigrid context 

  Output Parameters:
.   sles - the coarse grid solver context 

@*/ 
int MGGetCoarseSolve(PC pc,SLES *sles)  
{ 
  MG *mg = (MG*) pc->data;
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
@*/
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
int MGSetResidual(PC pc,int l,int (*residual)(Mat,Vec,Vec,Vec),Mat mat) 
{
  MG *mg = (MG*) pc->data;
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
int MGSetInterpolate(PC pc,int l,Mat mat)
{ 
  MG *mg = (MG*) pc->data;
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
int MGSetRestriction(PC pc,int l,Mat mat)  
{
  MG *mg = (MG*) pc->data;
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
int MGGetSmoother(PC pc,int l,SLES *sles)
{
  MG *mg = (MG*) pc->data;
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
int MGGetSmootherUp(PC pc,int l,SLES *sles)
{
  MG *mg = (MG*) pc->data;
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
int MGGetSmootherDown(PC pc,int l,SLES *sles)
{
  MG *mg = (MG*) pc->data;
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
int MGSetCyclesOnLevel(PC pc,int l,int c) 
{
  MG *mg = (MG*) pc->data;
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
int MGSetRhs(PC pc,int l,Vec c)  
{ 
  MG *mg = (MG*) pc->data;
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
int MGSetX(PC pc,int l,Vec c)  
{ 
  MG *mg = (MG*) pc->data;
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
int MGSetR(PC pc,int l,Vec c)
{ 
  MG *mg = (MG*) pc->data;
  mg[mg[0]->level - l]->r  = c;
  return 0;
}


