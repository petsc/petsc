#include "appctx.h"

int SetQuadrature(AppCtx* appctx){
  /* need to set up the quadrature weights, and the basis functions */
  AppElement *element = &appctx->element;
  int i;
  for(i=0;i<9;i++){ element->BiQuadWeights[i]= 4/9; }
  PetscFunctionReturn(0);
}

Elements(int which, int partial, int node, double x, double y){
  if(which == 2){
    /* quadrilateral basis */
    /*
node              function 
  0:       0.25 * ( 1 - x )* ( 1 - y);
  2:      0.25*  (1 + x) * (1- y );
  4:       0.25*  (1 + x) *  (1 + y);
  6:       0.25*  (1 - x) * (1 + y);

  1:      0.5 * (1- x)*(1 + x)*(1-y);
  3:      0.5*  (1 + x)* (1- y)*(1+y);
  5:      0.5* (1-x)*(1+x)*(1+y);
  7:      0.5* (1+x) * (1-y)*(1+y); 
  */
    if (partial == 0){
      if( node == 0){return   0.25 * ( 1 - x )* ( 1 - y);           }
      if( node == 1){return 0.5 * (1- x)*(1 + x)*(1-y);        } 
      if( node == 2){return 0.25*  (1 + x) * (1- y );           }
      if( node == 3){return   0.5*  (1 + x)* (1- y)*(1+y);   }
      if( node == 4){return  0.25*  (1 + x) *  (1 + y);       }
      if( node == 5){return  0.5* (1-x)*(1+x)*(1+y);      }
      if( node == 6){return 0.25*  (1 - x) * (1 + y);       }
      if( node == 7){return  0.5* (1+x) * (1-y)*(1+y); }
    }
      /* partials: d/dx */
   if (partial == 1){
      if( node == 0){return   -0.25 *( 1 - y);           }
      if( node == 1){return 0.5 * (1- 2*x)*(1-y);        } 
      if( node == 2){return 0.25*(1- y );           }
      if( node == 3){return   0.5*(1- y)*(1+y);   }
      if( node == 4){return  0.25*(1 + y);       }
      if( node == 5){return  0.5* (1-2*x)*(1+y);      }
      if( node == 6){return -0.25*(1 + y);       }
      if( node == 7){return  0.5*(1-y)*(1+y); }
   }
      /* partials d/dy */
 if (partial == 2){
      if( node == 0){return   -0.25 * ( 1 - x );           }
      if( node == 1){return -0.5 * (1- x)*(1 + x);        } 
      if( node == 2){return -0.25*  (1 + x);           }
      if( node == 3){return   0.5*  (1 + x)* (1- 2*y);   }
      if( node == 4){return  0.25*  (1 + x);       }
      if( node == 5){return  0.5* (1-x)*(1+x);      }
      if( node == 6){return 0.25*  (1 - x);       }
      if( node == 7){return  0.5* (1+x) * (1-2*y); }
 }
  }

/* The following functions set the reference element, and the local element for the quadrature.  Set reference element is called only once, at initialization, while set reference element must be called over each element.  */
int SetBiQuadReferenceElement(AppCtx* appctx){
  /* gauss points: */
  int i;
  double dx[8][9], dy[8][9], val[8][9];
  double gx[9], gy[9];
  double t = sqrt(.5);
  gx[0] = -t; gx[1] = 0; gx[2] = t;
  gx[3] = t; gx[4] = t; gx[5] = 0; 
  gx[6] = -t; gx[7] = -t; gx[8] = 0;

  gy[0] = -t; gy[1] = -t; gy[2] = -t;
  gy[3] = 0; gy[4] = t; gy[5] = t;
  gy[6] = t; gy[7] = 0; gy[8[ = 0;

  for(i=0;i<8;i++){  
    for(j=0;j<9;j++){
      val[i][j] = Elements(2,0,i,gx[j], gy[j]);
      dx[i][j] = Elements(2,1,i,gx[j], gy[j]);
      dy[i][j] = Elements(2,2,i,gx[j], gy[j]);
    }
  }
  appctx->element.BiquadRefVal = val;
  appctx->element.BiquadRefDx = Dx;
  appctx->element.BiquadRefDy = Dy;
  PetscFunctionReturn(0);
  }
			  

int SetLocalBiQuadElement(AppElement *phi, double *coords)
{
  /* the coords array consists of pairs (x[0],y[0],...,x[7],y[7]) representing the images of the
support points for the 8 basis functions */ 

  int i,j;
  double Dh[9][2][2], Dhinv[9][2][2];

  /* will set these to phi */
  double dx[8][9];  
  double dy[8][9];
  double detDh[9];
  double x[9], y[9];


 /* The function h takes the reference element to the local element.
                  h(x,y) = sum(i) of alpha_i*phi_i(x,y),
   where alpha_i is the image of the support point of the ith basis fn */

  /*Values */
  for(i=0;i<9;i++){ /* loop over the gauss points */
    x[i] = 0; y[i] = 0; 
    for(j=0;j<8,j++){/*loop over the basis functions, and support points */
      x[i] += coords[2*j]*phi->Val[j][i];
      y[i] += coords[2*j]*phi->Val[j][i];
    }
  }

  /* Jacobian */
  for(i=0;i<9;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(j=0; j<8; j++ ){/* loop over functions */
      Dh[i][0][0] += coords[2*j]*phi->Dx[j][i];
      Dh[i][0][1] += coords[2*j]*phi->Dy[j][i];
      Dh[i][1][0] += coords[2*j+1]*phi->Dx[j][i];
      Dh[i][1][1] += coords[2*j+1]*phi->Dy[j][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( i=0; i<9; i++){   /* loop over Gauss points */
    detDh[i] = Dh[i][0][0]*Dh[i][1][1] - Dh[i][0][1]*Dh[i][1][0];
  }

  /* Inverse of the Jacobian */
    for( i=0; i<9; i++){   /* loop over Gauss points */
      Dhinv[i][0][0] = Dh[i][1][1]/detDh[i];
      Dhinv[i][0][1] = -Dh[i][0][1]/detDh[i];
      Dhinv[i][1][0] = -Dh[i][1][0]/detDh[i];
      Dhinv[i][1][1] = Dh[i][0][0]/detDh[i];
    }
    

    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, (chain rule)
       so Dphi~ = Dphi*(Dh)inv    (multiply by (Dh)inv   */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for( i=0;i<9;i++ ){  /* loop over Gauss points */
      for( j=0;j<8;j++ ){ /* loop over basis functions */
	dx[j][i] = phi->Dx[j][i]*Dhinv[i][0][0] + phi->Dy[j][i]*Dhinv[i][1][0];
	dy[j][i] = phi->Dx[j][i]*Dhinv[i][0][1] + phi->Dy[j][i]*Dhinv[i][1][1];
      }
    }

 /* set these to phi */
 phi->dx = dx;
 phi->dy = dy;
 phi->detDh  = detDh;
 phi->x = x; phi->y = y;

 PetscFunctionReturn(0);
}

int SetLocalBiLinElement(AppElement *phi, double *coords)
{
  int i,j,k,ii ;

  double Dh[4][2][2], Dhinv[4][2][2]; 
  
  /* will set these to phi */
  double bdx[4][4];  
  double bdy[4][4];
  double bdetDh[4];
  double bx[4], by[4];

  /* the image of the reference element is given by sum (coord i)*phi_i */

    for(j=0;j<4;j++){ /* loop over Gauss points */
      bx[j] = 0; by[j] = 0;
      for( k=0;k<4;k++ ){/* loop over functions */
	bx[j] += coords[2*k]*phi->Values[k][j];
	by[j] += coords[2*k+1]*phi->Values[k][j];
      }
    }

  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++ ){/* loop over functions */
      Dh[i][0][0] += coords[2*k]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[2*k]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[2*k+1]*phi->DxValues[k][i];
      Dh[i][1][1] += coords[2*k+1]*phi->DyValues[k][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    bdetDh[j] = Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0];
  }
  /* Inverse of the Jacobian */
    for( j=0; j<4; j++){   /* loop over Gauss points */
      Dhinv[j][0][0] = Dh[j][1][1]/bdetDh[j];
      Dhinv[j][0][1] = -Dh[j][0][1]/bdetDh[j];
      Dhinv[j][1][0] = -Dh[j][1][0]/bdetDh[j];
      Dhinv[j][1][1] = Dh[j][0][0]/bdetDh[j];
    }
    
    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, (chain rule)
       so Dphi~ = Dphi*(Dh)inv    (multiply by (Dh)inv   */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for( i=0;i<4;i++ ){  /* loop over Gauss points */
      for( j=0;j<4;j++ ){ /* loop over basis functions */
	bdx[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][0] +  phi->DyValues[j][i]*Dhinv[i][1][0];
	bdy[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][1] + phi->DyValues[j][i]*Dhinv[i][1][1];
      }
    }

 /* set these to phi */
 phi->bdx = bdx;
 phi->bdy = bdy;
 phi->bdetDh  = bdetDh;
 phi->bx = bx; phi->by = by;

PetscFunctionReturn(0);
}
