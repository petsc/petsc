
#include "appctx.h"

/* The following functions do the integration over one element to
 compute the  Jacobian,Stiffness,Rhs etc */

#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian"

/* input is x, output the nonlinear part into f for a particulat element */
/* Much of the code is dublicated from ComputeMatrix; the integral is different */
int ComputeJacobian(AppElement *phi,double *uv,double *result)
{
  /* How can I test this??  */
  int i,j,k,ii ;
  double u[4],v[4];
  double dxint[4][4][4]; /* This is integral of phi_dx[i]*phi[j]*phi[k] */
  double dyint[4][4][4]; /* This is integral of phi_dy[i]*phi[j]*phi[k] */

  /* copy array into more convenient form */
  for(i=0;i<4;i++){    u[i] = uv[2*i];     v[i] = uv[2*i+1];}
 
  /* INTEGRAL */ 
  /* The nonlinear map takes(u0,v0,u1,v1,u2,v2,u3,v3) to 
      (integral term1 *  phi0,integral term2 * phi0,..., integral term1*phi3, int term2*phi3)
   Loop first over the phi.  Then integrate two parts of the terms.
Term 1: (ui*uj*phi_i*dx_j + vi*uj*phi_i*dy_j)
Term 2: (ui*vj*phi_i*dx_j + vi*vj*phi_i*dy_j)
*/

  /* could  exploit symmetry to cut down on iterations tohere */
/* Make a database of integrals of phi_i*phi_j(dx or dy)*phi_k */
  for(j=0;j<4;j++){
    for(i=0;i<4;i++){
      for(k=0;k < 4;k++){
	 dxint[i][j][k] = 0; 
	 dyint[i][j][k] = 0;
	for(ii=0;ii<4;ii++){/* loop over basis points */
	  dxint[i][j][k] += phi->dx[4*i+ii]*phi->Values[j][ii]*phi->Values[k][ii]*phi->detDh[ii];
	  dyint[i][j][k] += phi->dy[4*i+ii]*phi->Values[j][ii]*phi->Values[k][ii]*phi->detDh[ii];
	}
      }
    }
  }

  /* now loop over the columns of the matrix */
  for(k=0;k<4;k++){ 
    /* the terms are u*ux + v*uy and u*vx+v*vy  */
    for(i = 0;i<4;i++){  
      result[16*k + 2*i] = 0;
      result[16*k + 2*i + 1] = 0;   /* Stuff from Term 1 */
      result[16*k + 8 + 2*i]=0; 
      result[16*k + 8 + 2*i + 1] = 0;  /* Stuff from Term 2 */
      for(j=0;j<4;j++){
	result[16*k + 2*i] +=   u[j]*dxint[i][j][k] + u[j]*dxint[j][i][k] + v[j]*dyint[j][i][k];
	result[16*k+2*i+1] +=   u[j]*dyint[j][i][k];
	result[16*k + 8 + 2*i] += v[j]*dxint[j][i][k];
	result[16*k+ 8 + 2*i+1] += u[j]*dxint[i][j][k] + v[j]*dyint[j][i][k] + v[j]*dyint[i][j][k];
      }     
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeNonlinear"
/* input is x, output the nonlinear part into f for a particular element */
int ComputeNonlinear(AppElement *phi,double *uvvals,double *result)
{ 
  int i,j,k,ii ;
  double u[4],v[4];

  /* copy array into more convenient form */
  for(i=0;i<4;i++){  u[i] = uvvals[2*i]; v[i] = uvvals[2*i+1];  }

  /* INTEGRAL */
 /* terms are u*du/dx + v*du/dy, u*dv/dx + v*dv/dy */
  /* Go element by element.  
Compute 
(u_i * phi_i * u_j * phi_j_x + v_i*phi_i*u_j*phi_j_y) * phi_k
and
(u_i * phi_i * v_j * phi_j_x + v_i*phi_i*v_j*phi_j_y) * phi_k.
Put the result in index k.  Add all possibilities up to get contribution to k, and loop over k.*/

/* Could exploit a little symetry to cut iterations from 4*4*4 to 2*4*4  */
   for(k=0;k<4;k++){ /* loop over first basis fn */
     result[2*k] = 0; result[2*k+1] = 0;
     for(i=0; i<4; i++){ /* loop over second */
       for(j=0; j<4; j++){/* loop over third */
	 for(ii=0;ii<4;ii++){ /* loop over gauss points */
	 result[2*k] += 
	   (u[i]*u[j]*phi->Values[i][ii]*phi->dx[4*j+ii] +
	    v[i]*u[j]*phi->Values[i][ii]*phi->dy[4*j+ii])*phi->Values[k][ii]*phi->detDh[ii]; 
	 result[2*k+1] +=
	   (u[i]*v[j]*phi->Values[i][ii]*phi->dx[4*j+ii] +
	    v[i]*v[j]*phi->Values[i][ii]*phi->dy[4*j+ii])*phi->Values[k][ii]*phi->detDh[ii];
	 }
       }
     }
   }

   PetscFunctionReturn(0);
}

int ComputeRHS(DFP f,DFP g,AppElement *phi,double *integrals){
  int i,j;
  /* need to go over each element, then each variable */
 for(i = 0; i < 4; i++){ /* loop over basis functions */
   integrals[2*i] = 0.0; 
   integrals[2*i+1] = 0.0; 
   for(j = 0; j < 4; j++){ /* loop over Gauss points */
     integrals[2*i] +=  f(phi->x[j],phi->y[j])*(phi->Values[i][j])*phi->detDh[j];
     integrals[2*i+1] +=  g(phi->x[j],phi->y[j])*(phi->Values[i][j])*phi->detDh[j];
   }
 }
PetscFunctionReturn(0);
}

/* ComputeMatrix: computes integrals of gradients of local phi_i and phi_j on the given quadrangle by changing variables to the reference quadrangle and reference basis elements phi_i and phi_j.  The formula used is

integral (given element) of <grad phi_j, grad phi_i> =
integral over (ref element) of 
    <(grad phi_j composed with h)*(grad h)^-1,
     (grad phi_i composed with h)*(grad h)^-1>*det(grad h).
this is evaluated by quadrature:
= sum over gauss points, above evaluated at gauss pts
*/
int ComputeMatrix(AppElement *phi,double *result){
   int i,j,k;
 
   /******* Messed the indexing up when I put in  the dx ***********/

  /* Stiffness Terms */
  /* Now Integral.  term is <DphiDhinv[i],DphiDhinv[j]>*abs(detDh) */
   for(i=0;i<4;i++){ /* loop over first basis fn */
     for(j=0; j<4; j++){ /* loop over second */
       /* keep in mind we are throwing in a 2x2 block for each 1x1 */
       result[16*i + 2*j] = 0;
       result[16*i + 2*j+1] = 0;
       result[16*i + 8 +2*j] = 0;
       result[16*i + 9 +2*j] = 0;

       /* funny ordering of 2x2 blocks in the 4x4 piece */
       for(k=0;k<4;k++){ /* loop over gauss points */
	 result[16*i + 2*j] +=  (phi->dx[4*i+k]*phi->dx[4*j+k] + 
                                                        phi->dy[4*i+k]*phi->dy[4*j+k])* phi->detDh[k];
       }
       /* the off-diagonals stay zero */
       for(k=0;k<4;k++){ /* loop over gauss points */
	 result[16*i +9 + 2*j] +=  (phi->dx[4*i+k]*phi->dx[4*j+k] + 
                                                              phi->dy[4*i+k]*phi->dy[4*j+k])*phi->detDh[k];
       }
     }
   }
PetscFunctionReturn(0);
}

/* The following functions set the reference element, and the local element for the quadrature.  Set reference element is called only once, at initialization, while set reference element must be called over each element.  */
int AppCtxSetReferenceElement(AppCtx* appctx){

  AppElement *phi = &appctx->element;
  double psi,psi_m,psi_p,psi_pp,psi_mp,psi_pm,psi_mm;

phi->dorhs = 0;

  psi = sqrt(3.0)/3.0;
  psi_p = 0.25*(1.0 + psi);   psi_m = 0.25*(1.0 - psi);
  psi_pp = 0.25*(1.0 + psi)*(1.0 + psi);  psi_pm = 0.25*(1.0 + psi)*(1.0 - psi); 
  psi_mp = 0.25*(1.0 - psi)*(1.0 + psi);  psi_mm = 0.25*(1.0 - psi)*(1.0 - psi);

phi->Values[0][0] = psi_pp; phi->Values[0][1] = psi_pm;phi->Values[0][2] = psi_mm;
phi->Values[0][3] = psi_mp;phi->Values[1][0] = psi_mp; phi->Values[1][1] = psi_pp;
phi->Values[1][2] = psi_pm;phi->Values[1][3] = psi_mm;phi->Values[2][0] = psi_mm; 
phi->Values[2][1] = psi_pm;phi->Values[2][2] = psi_pp;phi->Values[2][3] = psi_mp;
phi->Values[3][0] = psi_pm; phi->Values[3][1] = psi_mm;phi->Values[3][2] = psi_mp;
phi->Values[3][3] = psi_pp;

phi->DxValues[0][0] = -psi_p; phi->DxValues[0][1] = -psi_p;phi->DxValues[0][2] = -psi_m;
phi->DxValues[0][3] = -psi_m;phi->DxValues[1][0] = psi_p; phi->DxValues[1][1] = psi_p;
phi->DxValues[1][2] = psi_m;phi->DxValues[1][3] = psi_m;phi->DxValues[2][0] = psi_m; 
phi->DxValues[2][1] = psi_m;phi->DxValues[2][2] = psi_p;phi->DxValues[2][3] = psi_p;
phi->DxValues[3][0] = -psi_m; phi->DxValues[3][1] = -psi_m;phi->DxValues[3][2] = -psi_p;
phi->DxValues[3][3] = -psi_p;

phi->DyValues[0][0] = -psi_p; phi->DyValues[0][1] = -psi_m;phi->DyValues[0][2] = -psi_m;
phi->DyValues[0][3] = -psi_p;phi->DyValues[1][0] = -psi_m; phi->DyValues[1][1] = -psi_p;
phi->DyValues[1][2] = -psi_p;phi->DyValues[1][3] = -psi_m;phi->DyValues[2][0] = psi_m; 
phi->DyValues[2][1] = psi_p;phi->DyValues[2][2] = psi_p;phi->DyValues[2][3] = psi_m;
phi->DyValues[3][0] = psi_p; phi->DyValues[3][1] = psi_m;phi->DyValues[3][2] = psi_m;
phi->DyValues[3][3] = psi_p;
PetscFunctionReturn(0);
}


int SetLocalElement(AppElement *phi,double *coords)
{
  int i,j,k;
  double Dh[4][2][2],Dhinv[4][2][2]; 
  double *dx = phi->dx,*dy = phi->dy;
  double *detDh = phi->detDh;
  double *x = phi->x,*y = phi->y;  /* image of gauss point */

  /* Could put in a flag to skip computing this when it is not needed */

  /* the image of the reference element is given by sum (coord i)*phi_i */
    for(j=0;j<4;j++){ /* loop over points */
      x[j] = 0; y[j] = 0;
      for(k=0;k<4;k++){
	x[j] += coords[2*k]*phi->Values[k][j];
	y[j] += coords[2*k+1]*phi->Values[k][j];
      }
    }
  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++){
      Dh[i][0][0] += coords[2*k]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[2*k]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[2*k+1]*phi->DxValues[k][i];
      Dh[i][1][1] += coords[2*k+1]*phi->DyValues[k][i];    
    }
  }

  /* Determinant of the Jacobian */
  for(j=0; j<4; j++){   /* loop over Gauss points */
    detDh[j] = PetscAbsReal(Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0]);
  }
  /* Inverse of the Jacobian */
    for(j=0; j<4; j++){   /* loop over Gauss points */
      Dhinv[j][0][0] = Dh[j][1][1]/detDh[j];
      Dhinv[j][0][1] = -Dh[j][0][1]/detDh[j];
      Dhinv[j][1][0] = -Dh[j][1][0]/detDh[j];
      Dhinv[j][1][1] = Dh[j][0][0]/detDh[j];
    }
    
    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, so Dphi~ = Dphi*(Dh)inv */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for(i=0;i<4;i++){  /* loop over Gauss points */
      for(j=0;j<4;j++){ /* loop over basis functions */
	dx[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][0] +  phi->DyValues[j][i]*Dhinv[i][1][0];
	dy[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][1] + phi->DyValues[j][i]*Dhinv[i][1][1];
      }
    }
PetscFunctionReturn(0);
}

