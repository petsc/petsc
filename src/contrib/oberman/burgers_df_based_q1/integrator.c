#include "appctx.h"

/* The following functions do the integration over one element to
 compute the  Jacobian, Stiffness, Rhs etc */

/*-----------------------------------------------------------------------*/

/*-------------------------------------------------------------*/
/* 6) Set function evaluation rountine and vector (non-linear parts), and
   7) Set Jacobian */

int SetLocalElement(AppElement *phi, double *coords)
{
  int i,j,k;
  double Dh[4][2][2], Dhinv[4][2][2]; 
  double *dx = phi->dx, *dy = phi->dy;
  double *detDh = phi->detDh;
  double *x = phi->x, *y = phi->y;  /* image of gauss point */

  /* Could put in a flag to skip computing this when it isn't needed */

  /* the image of the reference element is given by sum (coord i)*phi_i */
    for(j=0;j<4;j++){ /* loop over gauss points */
      x[j] = 0; y[j] = 0;
      for( k=0;k<4;k++ ){
	x[j] += coords[2*k]*phi->Values[k][j];
	y[j] += coords[2*k+1]*phi->Values[k][j];
      }
    }
  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++ ){
      Dh[i][0][0] += coords[2*k]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[2*k]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[2*k+1]*phi->DxValues[k][i];
      Dh[i][1][1] += coords[2*k+1]*phi->DyValues[k][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    detDh[j] = PetscAbsDouble(Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0]);
  }
  /* Inverse of the Jacobian */
    for( j=0; j<4; j++){   /* loop over Gauss points */
      Dhinv[j][0][0] = Dh[j][1][1]/detDh[j];
      Dhinv[j][0][1] = -Dh[j][0][1]/detDh[j];
      Dhinv[j][1][0] = -Dh[j][1][0]/detDh[j];
      Dhinv[j][1][1] = Dh[j][0][0]/detDh[j];
    }
    
    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, so Dphi~ = Dphi*(Dh)inv */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for( i=0;i<4;i++ ){  /* loop over Gauss points */
      for( j=0;j<4;j++ ){ /* loop over basis functions */
	dx[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][0] +  phi->DyValues[j][i]*Dhinv[i][1][0];
	dy[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][1] + phi->DyValues[j][i]*Dhinv[i][1][1];
      }
    }
PetscFunctionReturn(0);
}

