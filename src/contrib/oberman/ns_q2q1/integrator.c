
#include "appctx.h"

/* REMEMBER quadrature weights  */

int ComputeRHS( AppElement *phi ){
  int i,j,k; 
  int bn, qn; /* basis count, quadrature count */
  bn = phi->vel_basis_count;  
  qn = phi->vel_quad_count;
  /* need to go over each element , then each variable */
 for( i = 0; i < bn; i++ ){ /* loop over basis functions */
   phi->rhsresult[2*i] = 0.0; 
   phi->rhsresult[2*i+1] = 0.0; 
   for( j = 0; j < qn; j++ ){ /* loop over Gauss points */
     phi->rhsresult[2*i] +=  phi->vweights[j] *f(phi->x[j], phi->y[j]) 
       *(phi->RefVal[i][j])*PetscAbsDouble(phi->detDh[j]); 
     phi->rhsresult[2*i+1] +=  phi->vweights[j]*g(phi->x[j], phi->y[j])
       *(phi->RefVal[i][j])*PetscAbsDouble(phi->detDh[j]); 
   }
 }
PetscFunctionReturn(0);
}

/*WILL NEED MASS Matrix */

/* ComputeMatrix: computes integrals of gradients of local phi_i and phi_j on the given quadrangle by changing variables to the reference quadrangle and reference basis elements phi_i and phi_j.  The formula used is

integral (given element) of <grad phi_j', grad phi_i'> =
integral over (ref element) of 
    <(grad phi_j composed with h)*(grad h)^-1,
     (grad phi_i composed with h)*(grad h)^-1>*det(grad h).
this is evaluated by quadrature:
= sum over gauss points, above evaluated at gauss pts
*/
int ComputeStiffness( AppElement *phi ){
   int i,j,k;
   int bn, qn; /* basis count, quadrature count */
bn = phi->vel_basis_count;  
qn = phi->vel_quad_count;
  /* Stiffness Terms *//* could even do half as many by exploiting symmetry  */
   for( i=0;i<bn;i++ ){ /* loop over first basis fn */
     for( j=0; j<bn; j++){ /* loop over second */
       phi->vstiff[i][j] = 0;
     }
   }

  /* Now Integral.  term is <DphiDhinv[i],DphiDhinv[j]>*abs(detDh) */
   for( i=0;i<bn;i++ ){ /* loop over first basis fn */
     for( j=0; j<bn; j++){ /* loop over second */
       for(k=0;k<qn;k++){ /* loop over gauss points */
	 phi->vstiff[i][j] +=
	  - phi->vweights[k]*
	   (phi->dx[i][k]*phi->dx[j][k] + 
	     phi->dy[i][k]*phi->dy[j][k])*
	   PetscAbsDouble(phi->detDh[k]);
       }
     }
   }
   PetscFunctionReturn(0);
}

#undef __FUNC__ 
#define __FUNC__ "ComputePressure"
ComputePressure( AppElement *phi) 
{ 
  int i,j,k;
  int vbn, vqn, pbn; /* basis count, quadrature count */
  vbn = phi->vel_basis_count;  
  vqn = phi->vel_quad_count;
  pbn = phi->p_basis_count;

/* computing thepressure terms,CAREFUL OF THE SIGN */ 
   for(i=0; i<pbn; i++){/* pressure basis fn loop */
     for( j=0; j<2*vbn; j++){ /* velocity basis fn loop  */
       phi->presult[i][j] = 0;
     }
   }

  /* now integral */
  for(i=0; i<pbn; i++){/* pressure basis fn loop */
     for( j=0; j<vbn; j++){ /* velocity basis fn loop  */
         for(k=0;k<vqn;k++){ /* gauss points */
	     phi->presult[i][2*j] +=
	       -phi->vweights[k]*PetscAbsDouble(phi->detDh[k])*
	       phi->PRefVal[i][k]*phi->dx[j][k];
     
	     phi->presult[i][2*j+1] +=
	       -phi->vweights[k]*PetscAbsDouble(phi->detDh[k])*
	      phi->PRefVal[i][k]*phi->dy[j][k];
	 }
     }
  }
  PetscFunctionReturn(0); 
}



#undef __FUNC__
#define __FUNC__ "ComputeNonlinear"
/* input is x, output the nonlinear part into f for a particular element */
int ComputeNonlinear(AppElement *phi )
{ 
  int i,j,k,ii ;
  int vbn, vqn;
 
  vbn = phi->vel_basis_count;  vqn = phi->vel_quad_count;

  /* INTEGRAL */
 /* terms are u*du/dx + v*du/dy, u*dv/dx + v*dv/dy */
  /* Go element by element.  
Compute 
( u_i * phi_i * u_j * phi_j_x + v_i*phi_i*u_j*phi_j_y) * phi_k
and
( u_i * phi_i * v_j * phi_j_x + v_i*phi_i*v_j*phi_j_y) * phi_k.
Put the result in index k.  Add all possibilities up to get contribution to k, and loop over k.*/

/* Could exploit a little symetry to cut iterations from 4*4*4 to 2*4*4  */
   for( k=0;k<vbn;k++ ){ /* loop over first basis fn */
     phi->nlresult[2*k] = 0; phi->nlresult[2*k+1] = 0;
     for( i=0; i<vbn; i++){ /* loop over second */
       for( j=0; j<vbn; j++){/* loop over third */
	 for(ii=0;ii<vqn;ii++){ /* loop over gauss points */
	 phi->nlresult[2*k] += 
	   (phi->u[i]*phi->u[j]*phi->RefVal[i][ii]*phi->dx[j][ii] +
	    phi->v[i]*phi->u[j]*phi->RefVal[i][ii]*phi->dy[j][ii])*phi->RefVal[k][ii]*
	 phi->vweights[ii]*PetscAbsDouble(phi->detDh[ii]); 
	 phi->nlresult[2*k+1] +=
	   (phi->u[i]*phi->v[j]*phi->RefVal[i][ii]*phi->dx[j][ii] +
	    phi->v[i]*phi->v[j]*phi->RefVal[i][ii]*phi->dy[j][ii])*phi->RefVal[k][ii]*
	  phi->vweights[ii]*PetscAbsDouble( phi->detDh[ii]);
	 }
       }
     }
   }
   PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "ComputeJacobian"
/* input is x, output the nonlinear part into f for a particulat element */
int ComputeJacobian(AppElement *phi, double *uv, double *result)
{
 
  int i,j,k,ii ;
  double u[9],v[9];
  double dxint[9][9][9]; /* This is integral of phi_dx[i]*phi[j]*phi[k] */
  double dyint[9][9][9]; /* This is integral of phi_dy[i]*phi[j]*phi[k] */
  int vbn, vqn;

  vbn = phi->vel_basis_count;  vqn = phi->vel_quad_count;
  /* copy array into more convenient form */
  for(i=0;i<9;i++){    u[i] = uv[2*i];     v[i] = uv[2*i+1];}
 
  /* INTEGRAL */ 
  /* The nonlinear map takes( u0,v0,u1,v1,u2,v2,u3,v3 ) to 
      ( integral term1 *  phi0, integral term2 * phi0, ..., integral term1*phi3, int term2*phi3)
   Loop first over the phi.  Then integrate two parts of the terms.
Term 1: (ui*uj*phi_i*dx_j + vi*uj*phi_i*dy_j)
Term 2: (ui*vj*phi_i*dx_j + vi*vj*phi_i*dy_j)
*/

  /* could  exploit symmetry to cut down on iterations tohere */
/* Make a database of integrals of phi_i*phi_j(dx or dy)*phi_k */
  for(j=0;j<vbn;j++){
    for(i=0;i<vbn;i++){
      for(k=0;k < vbn;k++){
	 dxint[i][j][k] = 0; 
	 dyint[i][j][k] = 0;
	for(ii=0;ii<vqn;ii++){/* loop over basis points */
	  dxint[i][j][k] += 
	    phi->dx[i][ii]*phi->RefVal[j][ii]*phi->RefVal[k][ii]*
	    phi->vweights[ii]*PetscAbsDouble(phi->detDh[ii]);
	  dyint[i][j][k] += 
	    phi->dy[i][ii]*phi->RefVal[j][ii]*phi->RefVal[k][ii]*
	     phi->vweights[ii]*PetscAbsDouble(phi->detDh[ii]);
	}
      }
    }
  }

  /* now loop over the columns of the matrix */
  for( k=0;k<vbn;k++ ){ 
    /* the terms are u*ux + v*uy and u*vx+v*vy  */
    for(i = 0;i<vbn;i++){  

      result[4*vbn*k + 2*i] = 0;
      result[4*vbn*k + 2*i + 1] = 0;   /* Stuff from Term 1 */
      result[4*vbn*k + 2*vbn + 2*i]=0; 
      result[4*vbn*k + 2*vbn + 2*i + 1] = 0;  /* Stuff from Term 2 */
      for(j=0;j<vbn;j++){
	result[4*vbn*k + 2*i] +=   u[j]*dxint[i][j][k] + u[j]*dxint[j][i][k] + v[j]*dyint[j][i][k];
	result[4*vbn*k+2*i+1] +=   u[j]*dyint[j][i][k];

	result[4*vbn*k + 2*vbn + 2*i] += v[j]*dxint[j][i][k];
	result[4*vbn*k+2*vbn+2*i+1] += u[j]*dxint[i][j][k] + v[j]*dyint[j][i][k] + v[j]*dyint[i][j][k];
      }     
    }
  }
  PetscFunctionReturn(0);
}




























