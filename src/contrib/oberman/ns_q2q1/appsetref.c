#include "appctx.h"

int SetQuadrature(AppElement *element){
  /* need to set up the quadrature weights, and the basis functions */
  int i;
  for(i=0;i<9;i++){ element->BiquadWeights[i]= 4.0/9.0; }
  PetscFunctionReturn(0);
}

double Elements(int which, int partial, int node, double x, double y){
  if(which == 1){
    /* 4 node bilinear */
    if (partial == 0){
      if( node == 0){return 0.25 *(1-x)*          (1-y)         ;}
      if( node == 1){return 0.25 *         (1+x)*(1-y)         ;}
      if( node == 2){return 0.25 *         (1+x)         *(1+y);}
      if( node == 3){return 0.25 *(1-x)*                   (1+y);}
    }
  }

  if(which == 2){
    /* eight node biquadratic - (skipping the centre node)  */
    if (partial == 0){
      if( node == 0){return 0.25 *(1-x)*x*          (1-y)*y          ;}
      if( node == 2){return -0.25 *          x*(1+x)*(1-y)*y             ;}
      if( node == 4){return 0.25 *          x*(1+x)          *y*(1+y);}
      if( node == 6){return -0.25 *(1-x)*x                     *y*(1+y);}

      if( node == 1){return 0.25 *(1-x)*(1+x)*(1-y)*(1-y)    ;} 
      if( node == 3){return 0.25 *(1+x)*(1+x)*(1-y)*(1+y)    ;} 
      if( node == 5){return 0.25 *(1-x)*(1+x)*(1+y)*(1+y)    ;} 
      if( node == 7){return 0.25 *(1-x)*(1-x)*(1-y)*(1+y)    ;} 
    }
    /*d/dx */
   if (partial == 1){
      if( node == 0){return 0.25 *(1-2*x)          *(1-y)*y          ;}
      if( node == 2){return -0.25 *          (1+2*x)*(1-y)*y             ;}
      if( node == 4){return 0.25 *          (1+2*x)          *y*(1+y);}
      if( node == 6){return -0.25 *(1-2*x)                    *y*(1+y);}

      if( node == 1){return 0.25 *(-2*x)*(1-y)*(1-y)    ;} 
      if( node == 3){return 0.25 *(2+2*x)*(1-y)*(1+y)    ;} 
      if( node == 5){return 0.25 *(-2*x)*(1+y)*(1+y)    ;} 
      if( node == 7){return 0.25 *(-2+2*x)*(1-y)*(1+y)    ;} 
    }
   /*d/dy*/
   if (partial == 2){
      if( node == 0){return 0.25 *(1-x)*x          *(1-2*y)          ;}
      if( node == 2){return -0.25 *          x*(1+x)*(1-2*y)             ;}
      if( node == 4){return 0.25 *          x*(1+x)          *(1+2*y);}
      if( node == 6){return -0.25 *(1-x)*x                    *(1+2*y);}

      if( node == 1){return 0.25 *(1-x)*(1+x)*(-2+2*y)    ;} 
      if( node == 3){return 0.25 *(1+x)*(1+x)*(-2*y)    ;} 
      if( node == 5){return 0.25 *(1-x)*(1+x)*(2+2*y)    ;} 
      if( node == 7){return 0.25 *(1-x)*(1-x)*(-2*y)    ;} 
    }
  }

  if(which == 3){
    /* quadrilateral basis */
    if (partial == 0){
      if( node == 0){return 0.25 *(1-x)*x           *(1-y)*y           ;}
      if( node == 1){return  -0.5 *(1-x)   *(1+x)*(1-y)*y           ;}
      if( node == 2){return -0.25 *         x*(1+x)*(1-y)*y           ;}
      if( node == 3){return   0.5 *         x*(1+x)*(1-y)*   (1+y);}
      if( node == 4){return 0.25 *         x*(1+x)*         y*(1+y);}
      if( node == 5){return   0.5 *(1-x)   *(1+x)*         y*(1+y);}
      if( node == 6){return -0.25 *(1-x)*x*                    y*(1+y);}
      if( node == 7){return  -0.5 *(1-x)*x            *(1-y)  *(1+y);}
      if( node == 8){return      1 *(1-x)    *(1+x)*(1-y)  *(1+y);}
    }
      /* partials: d/dx */
   if (partial == 1){
     if( node == 0){return 0.25 *(1-2*x)          *(1-y)*y           ;}
     if( node == 1){return  -0.5 *(-2*x)        *(1-y)*y           ;}
     if( node == 2){return -0.25 * (1+2*x)        *(1-y)*y           ;}
     if( node == 3){return   0.5 *         (1+2*x)*(1-y)*   (1+y);}
     if( node == 4){return 0.25 *        (1+2*x) *         y*(1+y);}
     if( node == 5){return   0.5 *(-2*x)        *         y*(1+y);}
     if( node == 6){return -0.25 *(1-2*x)*                    y*(1+y);}
     if( node == 7){return  -0.5 *(1-2*x)            *(1-y)  *(1+y);}
     if( node == 8){return      1 *(-2*x)         *(1-y)  *(1+y);}
    }
       /* partials d/dy */
   if (partial == 2){
     if( node == 0){return 0.25 *(1-x)*x           *(1-2*y)           ;}
     if( node == 1){return  -0.5 *(1-x)   *(1+x)*(1-2*y)           ;}
     if( node == 2){return -0.25 *         x*(1+x)*(1-2*y)           ;}
     if( node == 3){return   0.5 *         x*(1+x)*(-2*y)        ;}
     if( node == 4){return 0.25 *         x*(1+x)*          (1+2*y);}
     if( node == 5){return   0.5 *(1-x)   *(1+x)*     (1+2*y)    ;}
     if( node == 6){return -0.25 *(1-x)*x*                 (1+2*y)   ;}
     if( node == 7){return  -0.5 *(1-x)*x            *(-2*y)       ;}
     if( node == 8){return      1 *(1-x)    *(1+x)*(-2*y)       ;}
   }
  }
  /* shouldn't ever get here */
  return 0.0;
}

/* The following functions set the reference element, and the local element for the quadrature.  Set reference element is called only once, at initialization, while set reference element must be called over each element.  */
int SetReferenceElement(AppCtx* appctx){
  int i,j;
  double gx[9], gy[9]; /* gauss points: */  
  int vbn, vqn, pbn, pqn; /* basis count, quadrature count */
  double t;
  AppElement *phi = &appctx->element;
  vbn = phi->vel_basis_count;  vqn = phi->vel_quad_count;
  pbn = phi->p_basis_count;    pqn = phi->p_quad_count;
  t = sqrt(.5);

  gx[0] = -t; gx[1] = 0; gx[2] = t;
  gx[3] = t; gx[4] = t; gx[5] = 0; 
  gx[6] = -t; gx[7] = -t; gx[8] = 0;

  gy[0] = -t; gy[1] = -t; gy[2] = -t;
  gy[3] = 0; gy[4] = t; gy[5] = t;
  gy[6] = t; gy[7] = 0; gy[8] = 0;

  /* set the bilinear for velocity */
  /* Later the 2 should be a parameter */
  for(i=0;i<vbn;i++){  /* loop over functions*/
    for(j=0;j<vqn;j++){/* loop over gauss points */
      appctx->element.RefVal[i][j] = Elements(3,0,i,gx[j], gy[j]);
      appctx->element.RefDx[i][j] =  Elements(3,1,i,gx[j], gy[j]);
      appctx->element.RefDy[i][j] =  Elements(3,2,i,gx[j], gy[j]);
    }
  }

  /* set the linear for pressure */
 for(i=0;i<pbn;i++){  /* loop over functions*/
    for(j=0;j<pqn;j++){/* loop over gauss points */
      appctx->element.PRefVal[i][j] = Elements(1,0,i,gx[j], gy[j]);
    }
  }


  PetscFunctionReturn(0);
}
			  

int SetLocalElement(AppElement *phi, double *coords)
{
  /* the coords array consists of pairs (x[0],y[0],...,x[7],y[7]) representing the images of the
support points for the 8 basis functions */ 

  int i,j;
  double Dh[9][2][2], Dhinv[9][2][2];
  int vbn, vqn; /* basis count, quadrature count */
  vbn = phi->vel_basis_count;  vqn = phi->vel_quad_count;

 /* The function h takes the reference element to the local element.
                  h(x,y) = sum(i) of alpha_i*phi_i(x,y),
   where alpha_i is the image of the support point of the ith basis fn */

  /*Values */
  for(i=0;i<vqn;i++){ /* loop over the gauss points */
    phi->x[i] = 0; phi->y[i] = 0; 
    for(j=0;j<vbn;j++){/*loop over the basis functions, and support points */
      phi->x[i] += coords[2*j]*phi->RefVal[j][i];
      phi->y[i] += coords[2*j+1]*phi->RefVal[j][i];
    }
  }

  /* Jacobian */
  for(i=0;i<vqn;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(j=0; j<vbn; j++ ){/* loop over functions */
      Dh[i][0][0] += coords[2*j]*phi->RefDx[j][i];
      Dh[i][0][1] += coords[2*j]*phi->RefDy[j][i];
      Dh[i][1][0] += coords[2*j+1]*phi->RefDx[j][i];
      Dh[i][1][1] += coords[2*j+1]*phi->RefDy[j][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( i=0; i<vqn; i++){   /* loop over Gauss points */
    phi->detDh[i] = Dh[i][0][0]*Dh[i][1][1] - Dh[i][0][1]*Dh[i][1][0];
  }

  /* Inverse of the Jacobian */
    for( i=0; i<vqn; i++){   /* loop over Gauss points */
      Dhinv[i][0][0] = Dh[i][1][1]/phi->detDh[i];
      Dhinv[i][0][1] = -Dh[i][0][1]/phi->detDh[i];
      Dhinv[i][1][0] = -Dh[i][1][0]/phi->detDh[i];
      Dhinv[i][1][1] = Dh[i][0][0]/phi->detDh[i];
    }
    

    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, (chain rule)
       so Dphi~ = Dphi*(Dh)inv    (multiply by (Dh)inv   */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for( i=0;i<vqn;i++ ){  /* loop over Gauss points */
      for( j=0;j<vbn;j++ ){ /* loop over basis functions */
	phi->dx[j][i] = phi->RefDx[j][i]*Dhinv[i][0][0] + phi->RefDy[j][i]*Dhinv[i][1][0];
	phi->dy[j][i] = phi->RefDx[j][i]*Dhinv[i][0][1] + phi->RefDy[j][i]*Dhinv[i][1][1];
      }
    }

 PetscFunctionReturn(0);
}

