
#include "appctx.h"

/*
         Performs the numerical integration and setup at the element level
*/

/* A -
     Returns the value of the shape function or its xi or eta derivative at 
   any point in the REFERENCE element. xi and eta are the coordinates in the reference
   element.
*/


#undef __FUNCT__
#define __FUNCT__ "InterpolatingFunctionsElement"
static int InterpolatingFunctionsElement(int partial,int node,double xi,double eta, double *value)
{
  /* 4 node bilinear interpolation functions */
  PetscFunctionBegin;
  switch (partial) {
  case 0:  /*  function itself  */
    switch (node) {
    case 0: *value = 0.25 * (1-xi) *          (1-eta)          ; break;
    case 1: *value = 0.25 *          (1+xi) * (1-eta)          ; break;
    case 2: *value = 0.25 *          (1+xi) *           (1+eta); break;
    case 3: *value = 0.25 * (1-xi) *                    (1+eta); break;
    } break;
  case 1:  /*  d() / d(xi)  */
    switch (node) {
    case 0: *value = 0.25 * (  -1) *          (1-eta)          ; break;
    case 1: *value = 0.25 *          (   1) * (1-eta)          ; break;
    case 2: *value = 0.25 *          (   1) *           (1+eta); break;
    case 3: *value = 0.25 * (  -1) *                    (1+eta); break;
    } break;
  case 2:  /*  d() / d(eta)  */
    switch (node) {
    case 0: *value = 0.25 * (1-xi) *          (   -1)          ; break;
    case 1: *value = 0.25 *          (1+xi) * (   -1)          ; break;
    case 2: *value = 0.25 *          (1+xi) *           (    1); break;
    case 3: *value = 0.25 * (1-xi) *                    (    1); break;
    } break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetReferenceElement"
/* 
   Computes the numerical integration (Gauss) points and evaluates the basis funtions at
   these points. This is done ONCE for the element, the information is stored in the AppElement
   data structure and then used repeatedly to compute each element load and element stiffness.
   Uses InterpolatingFunctionsElement(). 
*/
int SetReferenceElement(AppCtx* appctx)
{
  int        i,j;
  int        bn = 4; /* number of basis functions*/
  int        qn = 4; /* number of quadrature points */
  double     t;  /* for quadrature point */
  double     gx[4],gy[4]; /* Gauss points: */  
  AppElement *phi = &appctx->element;

  PetscFunctionBegin;
  t =  sqrt(3.0)/3.0;

  /* set Gauss points */
  gx[0] = -t;   gx[1] = t; 
  gx[2] = t;  gx[3] = -t; 

  gy[0] = -t; gy[1] = -t; 
  gy[2] = t;  gy[3] = t; 

  /* set quadrature weights */
  phi->weights[0] = 1; phi->weights[1] = 1; 
  phi->weights[2] = 1; phi->weights[3] = 1; 

  /* Set the reference values, 
     i.e., the values of the basis functions at the Gauss points  */
  for(i=0;i<bn;i++){  /* loop over functions*/
    for(j=0;j<qn;j++){/* loop over Gauss points */
      InterpolatingFunctionsElement(0,i,gx[j],gy[j],&(appctx->element.RefVal[i][j]));
      InterpolatingFunctionsElement(1,i,gx[j],gy[j],&(appctx->element.RefDx[i][j]));
      InterpolatingFunctionsElement(2,i,gx[j],gy[j],&(appctx->element.RefDy[i][j]));
    }
  }
  PetscFunctionReturn(0);
}
		  
/*------------------------------------------------------------------
    B - Computes derivative information for each real element (this is called once per
    element. This data is used in C) and D) to compute the element load and stiffness.
*/
#undef __FUNCT__
#define __FUNCT__ "SetLocalElement"
int SetLocalElement(AppElement *phi)
{
  /* the coordinates array consists of pairs (x[0],y[0],...,x[3],y[3]) representing 
     the images of the support points for the 4 basis functions */ 
  int    i,j;
  int    bn = 4,qn = 4; /* number of basis functions, number of quadrature points */
  double Dh[4][2][2],Dhinv[4][2][2]; /* the Jacobian and inverse of the Jacobian */
 
  PetscFunctionBegin;
 /* The function h takes the reference element to the local (true) element.
                  h(x,y) = sum(i) of alpha_i*phi_i(x,y),
   where alpha_i is the image of the support point of the ith basis function */

  /*Values, i.e., (x(xi, eta), y(xi, eta)), 
            the images of the Gauss points in the local element */
  for(i=0;i<qn;i++){ /* loop over the Gauss points */
    phi->xy[2*i] = 0; phi->xy[2*i+1] = 0; 
    for(j=0;j<bn;j++){/*loop over the basis functions, and support points */
      phi->xy[2*i]   += phi->coords[2*j]*phi->RefVal[j][i];
      phi->xy[2*i+1] += phi->coords[2*j+1]*phi->RefVal[j][i];
    }
  }

  /* Jacobian */
  for(i=0;i<qn;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(j=0; j<bn; j++){/* loop over functions */
      Dh[i][0][0] += phi->coords[2*j]*phi->RefDx[j][i];
      Dh[i][0][1] += phi->coords[2*j]*phi->RefDy[j][i];
      Dh[i][1][0] += phi->coords[2*j+1]*phi->RefDx[j][i];
      Dh[i][1][1] += phi->coords[2*j+1]*phi->RefDy[j][i];    
    }
  }

  /* Determinant of the Jacobian */
  for(i=0; i<qn; i++){   /* loop over Gauss points */
    phi->detDh[i] = Dh[i][0][0]*Dh[i][1][1] - Dh[i][0][1]*Dh[i][1][0];
  }

  /* Inverse of the Jacobian */
  for(i=0; i<qn; i++){   /* loop over Gauss points */
    Dhinv[i][0][0] = Dh[i][1][1]/phi->detDh[i];
    Dhinv[i][0][1] = -Dh[i][0][1]/phi->detDh[i];
    Dhinv[i][1][0] = -Dh[i][1][0]/phi->detDh[i];
    Dhinv[i][1][1] = Dh[i][0][0]/phi->detDh[i];
  }
    

  /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, (chain rule)
     so Dphi~ = Dphi*(Dh)inv    (multiply by (Dh)inv   */       
  /* partial of phi at h(gauss pt) times Dhinv */
  /* loop over Gauss, the basis functions, then d/dx or d/dy */
  for(i=0;i<qn;i++){  /* loop over Gauss points */
    for(j=0;j<bn;j++){ /* loop over basis functions */
      phi->dx[j][i] = phi->RefDx[j][i]*Dhinv[i][0][0] + phi->RefDy[j][i]*Dhinv[i][1][0];
      phi->dy[j][i] = phi->RefDx[j][i]*Dhinv[i][0][1] + phi->RefDy[j][i]*Dhinv[i][1][1];
    }
  }

  PetscFunctionReturn(0);
}
/*------------------------------------------------
        C - Computes an element load
*/
#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
int ComputeRHSElement(AppElement *phi)
{
  int    i,j,ierr; 
  int    bn = 4,qn = 4; /* number of basis functions, number of quadrature points */
  PetscScalar f;

  PetscFunctionBegin;

  for(i = 0; i < bn; i++){ /* loop over basis functions */
    phi->rhsresult[i] = 0.0; 
    for(j = 0; j < qn; j++){ /* loop over Gauss points */

      /* evaluate right hand side function */
      ierr = PFApply(phi->rhs,1,&phi->xy[2*j],&f);CHKERRQ(ierr);

      phi->rhsresult[i] +=  phi->weights[j]*f*(phi->RefVal[i][j])*PetscAbsReal(phi->detDh[j]); 
   }
 }
 PetscFunctionReturn(0);
}

/* ---------------------------------------------------

    D - ComputeStiffness: computes integrals of gradients of local phi_i and phi_j on the given quadrangle 
     by changing variables to the reference quadrangle and reference basis elements phi_i and phi_j.  
     The formula used is

     integral (given element) of <grad phi_j', grad phi_i'> =
                                        integral over (ref element) of 
                                      <(grad phi_j composed with h)*(grad h)^-1,
                                      (grad phi_i composed with h)*(grad h)^-1>*det(grad h).
      this is evaluated by quadrature:
      = sum over Gauss points, above evaluated at Gauss pts
*/
#undef __FUNCT__
#define __FUNCT__ "ComputeStiffness"
int ComputeStiffnessElement(AppElement *phi)
{
  int i,j,k;
  int bn = 4,qn = 4; /* number of basis functions, number of Gauss points */

  PetscFunctionBegin;
  /* could even do half as many by exploiting symmetry  */
  for(i=0;i<bn;i++){ /* loop over first basis function */
    for(j=0; j<bn; j++){ /* loop over second besis function */
      phi->stiffnessresult[i][j] = 0;
    }
  }

  /* Now Integral.  term is <DphiDhinv[i],DphiDhinv[j]>*abs(detDh) */
  for(i=0;i<bn;i++){ /* loop over first basis function */
    for(j=0; j<bn; j++){ /* loop over second basis function*/
      for(k=0;k<qn;k++){ /* loop over Gauss points */
        phi->stiffnessresult[i][j] += phi->weights[k]*
                  (phi->dx[i][k]*phi->dx[j][k] + phi->dy[i][k]*phi->dy[j][k])*PetscAbsReal(phi->detDh[k]);
      }
    }
  }
  PetscFunctionReturn(0);
}
