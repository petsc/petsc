#include "appctx.h"


/* The following functions set the reference element, and the local element for the quadrature.  Set reference element is called only once, at initialization, while set reference element must be called over each element.  */
int SetReferenceElement(AppCtx* appctx){
  int i,j;
  double gx[4], gy[4]; /* gauss points: */  
  int bn = 4; /* basis count*/
  int qn = 4; /*quadrature count */
  double t;  /* for quadrature point */
  AppElement *phi = &appctx->element;
  t =  sqrt(3.0)/3.0;

  /* set gauss points */
  gx[0] = -t;   gx[1] = t; 
  gx[2] = t;  gx[3] = -t; 

  gy[0] = -t; gy[1] = -t; 
  gy[2] = t;  gy[3] = t; 

  /* set quadrature weights */
  phi->weights[0] = 1; phi->weights[1] = 1; 
  phi->weights[2] = 1; phi->weights[3] = 1; 


  /* Set the reference values  */
  for(i=0;i<bn;i++){  /* loop over functions*/
    for(j=0;j<qn;j++){/* loop over gauss points */
      appctx->element.RefVal[i][j] = InterpolatingFunctions(0,i,gx[j], gy[j]);
      appctx->element.RefDx[i][j] =  InterpolatingFunctions(1,i,gx[j], gy[j]);
      appctx->element.RefDy[i][j] =  InterpolatingFunctions(2,i,gx[j], gy[j]);
    }
  }
  PetscFunctionReturn(0);
}
			  
			  

int SetLocalElement(AppElement *phi )
{
  /* the coords array consists of pairs (x[0],y[0],...,x[7],y[7]) representing the images of the
support points for the 8 basis functions */ 

  int i,j;
  double Dh[4][2][2], Dhinv[4][2][2];
  int bn = 4, qn = 4; /* basis count, quadrature count */
 
 /* The function h takes the reference element to the local element.
                  h(x,y) = sum(i) of alpha_i*phi_i(x,y),
   where alpha_i is the image of the support point of the ith basis fn */

  /*Values */
  for(i=0;i<qn;i++){ /* loop over the gauss points */
    phi->x[i] = 0; phi->y[i] = 0; 
    for(j=0;j<bn;j++){/*loop over the basis functions, and support points */
      phi->x[i] += phi->coords[2*j]*phi->RefVal[j][i];
      phi->y[i] += phi->coords[2*j+1]*phi->RefVal[j][i];
    }
  }

  /* Jacobian */
  for(i=0;i<qn;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(j=0; j<bn; j++ ){/* loop over functions */
      Dh[i][0][0] += phi->coords[2*j]*phi->RefDx[j][i];
      Dh[i][0][1] += phi->coords[2*j]*phi->RefDy[j][i];
      Dh[i][1][0] += phi->coords[2*j+1]*phi->RefDx[j][i];
      Dh[i][1][1] += phi->coords[2*j+1]*phi->RefDy[j][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( i=0; i<qn; i++){   /* loop over Gauss points */
    phi->detDh[i] = Dh[i][0][0]*Dh[i][1][1] - Dh[i][0][1]*Dh[i][1][0];
  }

  /* Inverse of the Jacobian */
    for( i=0; i<qn; i++){   /* loop over Gauss points */
      Dhinv[i][0][0] = Dh[i][1][1]/phi->detDh[i];
      Dhinv[i][0][1] = -Dh[i][0][1]/phi->detDh[i];
      Dhinv[i][1][0] = -Dh[i][1][0]/phi->detDh[i];
      Dhinv[i][1][1] = Dh[i][0][0]/phi->detDh[i];
    }
    

    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, (chain rule)
       so Dphi~ = Dphi*(Dh)inv    (multiply by (Dh)inv   */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for( i=0;i<qn;i++ ){  /* loop over Gauss points */
      for( j=0;j<bn;j++ ){ /* loop over basis functions */
	phi->dx[j][i] = phi->RefDx[j][i]*Dhinv[i][0][0] + phi->RefDy[j][i]*Dhinv[i][1][0];
	phi->dy[j][i] = phi->RefDx[j][i]*Dhinv[i][0][1] + phi->RefDy[j][i]*Dhinv[i][1][1];
      }
    }

 PetscFunctionReturn(0);
}





double InterpolatingFunctions(int partial, int node, double x, double y){

  /* 4 node bilinear interpolation functions */
  if (partial == 0){
    if( node == 0){return 0.25 *(1-x)*          (1-y)         ;}
    if( node == 1){return 0.25 *         (1+x)*(1-y)         ;}
    if( node == 2){return 0.25 *         (1+x)         *(1+y);}
    if( node == 3){return 0.25 *(1-x)*                   (1+y);}
  }  
  /*d/dx */
  if (partial == 1){
    if( node == 0){return 0.25 *(  -1)*          (1-y)         ;}
    if( node == 1){return 0.25 *                 1*(1-y)         ;}
    if( node == 2){return 0.25 *                 1         *(1+y);}
    if( node == 3){return 0.25 *(  -1)*                   (1+y);}
  }   
  /*d/dy*/
  if (partial == 2){
    if( node == 0){return 0.25 *(1-x)*          (-1)         ;}
    if( node == 1){return 0.25 *         (1+x)*(-1)         ;}
    if( node == 2){return 0.25 *         (1+x)         *(1);}
    if( node == 3){return 0.25 *(1-x)*                   (1);}
  }  
}

