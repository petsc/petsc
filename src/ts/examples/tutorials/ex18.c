static const char help[] = "Isogeometric analysis of isothermal Navier-Stokes-Korteweg in 2D.";

#include <petscts.h>
#include <petscdmiga.h>

#define SQ(x) ((x)*(x))

typedef struct {
  DM iga;
  PetscScalar L0,h;
  PetscScalar Ca,alpha,theta,Re;

  // bubble centers
  PetscScalar C1x,C1y,C2x,C2y,C3x,C3y;
  PetscScalar R1,R2,R3;

} AppCtx;

typedef struct {
  PetscScalar rho,ux,uy;
} Field;

PetscErrorCode InterpolateSolution(double **basis2D,Field **x,Field **xdot,PetscInt px,PetscInt py,PetscInt boffsetX,PetscInt boffsetY,
				   PetscScalar *rho,PetscScalar *rho_t,PetscScalar *rho_x,PetscScalar *rho_y,
				   PetscScalar *rho_xx,PetscScalar *rho_yy,PetscScalar *rho_xy,
				   PetscScalar *ux,PetscScalar *ux_t,PetscScalar *ux_x,PetscScalar *ux_y,
				   PetscScalar *uy,PetscScalar *uy_t,PetscScalar *uy_x,PetscScalar *uy_y);
PetscErrorCode FormResidual(TS ts,PetscReal t,Vec U,Vec Udot,Vec R,void *ctx);
PetscErrorCode FormResidualLocal(DMDALocalInfo *info,PetscReal t,Field **h,Field **hdot,Field **r,AppCtx *user);
PetscErrorCode FormTangent(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal shift,Mat *A,Mat *B,MatStructure *flag,void *ctx);
PetscErrorCode FormTangentLocal(DMDALocalInfo *info,PetscReal t,Field **h,Field **hdot,PetscReal shift,Mat *A,AppCtx *user);
PetscErrorCode FormInitialCondition(AppCtx *user,Vec U);
PetscErrorCode WriteSolution(Vec U, const char pattern[],int number);
PetscErrorCode OutputMonitor(TS ts,PetscInt it_number,PetscReal c_time,Vec U,void *mctx);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  AppCtx          user;
  PetscInt        p=2,N=64,C=1;
  PetscInt ng = p+2; /* integration in each direction */
  PetscInt Nx,Ny;
  Vec            U; /* solution vector */
  Mat            J;
  TS             ts;
  PetscInt steps;
  PetscReal ftime;

  /* This code solve the dimensionless form of the isothermal
     Navier-Stokes-Korteweg equations as presented in:

     Gomez, Hughes, Nogueira, Calo
     Isogeometric analysis of the isothermal Navier-Stokes-Korteweg equations
     CMAME, 2010

     Equation/section numbers reflect this publication.
 */

  // Petsc Initialization rite of passage
  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  // Define simulation specific parameters
  user.L0 = 1.0; // length scale
  user.C1x = 0.75; user.C1y = 0.50; // bubble centers
  user.C2x = 0.25; user.C2y = 0.50;
  user.C3x = 0.40; user.C3y = 0.75;
  user.R1 = 0.10;  user.R2 = 0.15;  user.R3 = 0.08; // bubble radii

  user.alpha = 2.0; // (Eq. 41)
  user.theta = 0.85; // temperature parameter (just before section 5.1)

  // Set discretization options
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "NavierStokesKorteweg Options", "IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N", "number of elements (along one dimension)", __FILE__, N, &N, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Compute simulation parameters
  user.h = user.L0/N; // characteristic length scale of mesh (Eq. 43, simplified for uniform elements)
  user.Ca = user.h/user.L0; // capillarity number (Eq. 38)
  user.Re = user.alpha/user.Ca; // Reynolds number (Eq. 39)

  // Test C < p
  if(p <= C){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Discretization inconsistent: polynomial order must be greater than degree of continuity");
  }

  Nx=Ny=N;

  // Initialize B-spline space
  ierr = DMCreate(PETSC_COMM_WORLD,&user.iga);CHKERRQ(ierr);
  ierr = DMSetType(user.iga, DMIGA);CHKERRQ(ierr);
  ierr = DMIGAInitializeUniform2d(user.iga,PETSC_FALSE,2,3,
                                  p,Nx,C,0.0,1.0,PETSC_TRUE,ng,
                                  p,Ny,C,0.0,1.0,PETSC_TRUE,ng);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(&user,U);CHKERRQ(ierr);
  ierr = DMIGASetFieldName(user.iga, 0, "density");CHKERRQ(ierr);
  ierr = DMIGASetFieldName(user.iga, 1, "velocity-u");CHKERRQ(ierr);
  ierr = DMIGASetFieldName(user.iga, 2, "velocity-v");CHKERRQ(ierr);

  ierr = DMCreateMatrix(user.iga, MATAIJ, &J);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.iga);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,PETSC_NULL,FormResidual,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormTangent,&user);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,1000000,1000.0);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,0.001);CHKERRQ(ierr);
  ierr = TSAlphaSetAdapt(ts,TSAlphaAdaptDefault,PETSC_NULL);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,OutputMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,U,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

  // Cleanup
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&user.iga);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialCondition"
PetscErrorCode FormInitialCondition(AppCtx *user,Vec U)
{
  DMDALocalInfo  info;
  Field        **u;
  PetscScalar    x,y,d1,d2,d3;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMIGAGetLocalInfo(user->iga,&info);CHKERRQ(ierr);
  ierr = DMIGAVecGetArray(user->iga,U,&u);CHKERRQ(ierr);

  for(i=info.xs;i<info.xs+info.xm;i++){
    x = user->L0*( (PetscScalar)i/(PetscScalar)info.mx );
    for(j=info.ys;j<info.ys+info.ym;j++){
      y = user->L0*( (PetscScalar)j/(PetscScalar)info.my );

      d1 = PetscSqrtReal(SQ(x-user->C1x)+SQ(y-user->C1y));
      d2 = PetscSqrtReal(SQ(x-user->C2x)+SQ(y-user->C2y));
      d3 = PetscSqrtReal(SQ(x-user->C3x)+SQ(y-user->C3y));

      u[j][i].rho = -0.15+0.25*( tanh(0.5*(d1-user->R1)/user->Ca) +
				 tanh(0.5*(d2-user->R2)/user->Ca) +
				 tanh(0.5*(d3-user->R3)/user->Ca) );
      u[j][i].ux = 0.0;
      u[j][i].uy = 0.0;
    }
  }
  ierr = DMIGAVecRestoreArray(user->iga,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormResidual"
PetscErrorCode FormResidual(TS ts,PetscReal t,Vec U,Vec Udot,Vec R,void *user)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;

  DMDALocalInfo    info;
  DM               dm;
  Vec              localU,localUdot,localR; // local versions
  Field          **h,**hdot,**r;

  /* get the da from the snes */
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  /* handle the vec U */
  ierr = DMGetLocalVector(dm,&localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /* handle the vec Udot */
  ierr = DMGetLocalVector(dm,&localUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,Udot,INSERT_VALUES,localUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,Udot,INSERT_VALUES,localUdot);CHKERRQ(ierr);

  /* handle the vec R */
  ierr = DMGetLocalVector(dm,&localR);CHKERRQ(ierr);
  ierr = VecZeroEntries(localR);CHKERRQ(ierr);

  /* Get the arrays from the vectors */
  ierr = DMIGAVecGetArray(dm,localU,&h);CHKERRQ(ierr);
  ierr = DMIGAVecGetArray(dm,localUdot,&hdot);CHKERRQ(ierr);
  ierr = DMIGAVecGetArray(dm,localR,&r);CHKERRQ(ierr);

  /* Grab the local info and call the local residual routine */
  ierr = DMIGAGetLocalInfo(dm,&info);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = FormResidualLocal(&info,t,h,hdot,r,(AppCtx *) user);CHKERRQ(ierr);

  /* Restore the arrays */
  ierr = DMIGAVecRestoreArray(dm,localR,&r);CHKERRQ(ierr);
  ierr = DMIGAVecRestoreArray(dm,localUdot,&hdot);CHKERRQ(ierr);
  ierr = DMIGAVecRestoreArray(dm,localU,&h);CHKERRQ(ierr);

  /* Add contributions back to global R */
  ierr = VecZeroEntries(R);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,localR,ADD_VALUES,R);CHKERRQ(ierr); // this one adds the values
  ierr = DMLocalToGlobalEnd(dm,localR,ADD_VALUES,R);CHKERRQ(ierr);

  /* Restore the local vectors */
  ierr = DMRestoreLocalVector(dm,&localU);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localUdot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localR);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormResidualLocal"
PetscErrorCode FormResidualLocal(DMDALocalInfo *info,PetscReal t,Field **h,Field **hdot,Field **r,AppCtx *user)
{
  DM             iga = user->iga;
  BD             bdX, bdY;
  PetscInt       px, py, ngx, ngy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMIGAGetPolynomialOrder(iga, &px, &py, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMIGAGetNumQuadraturePoints(iga, &ngx, &ngy, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMIGAGetBasisData(iga, &bdX, &bdY, PETSC_NULL);CHKERRQ(ierr);

  // begin and end elements for this processor
  PetscInt bex = bdX->own_b, eex = bdX->own_e;
  PetscInt bey = bdY->own_b, eey = bdY->own_e;

  // Allocate space for the 3D basis to be formed
  PetscInt Nl = (px+1)*(py+1); // number of local basis functions
  int numD = 6; // [0] = N, [1] = dN/dx, [2] = dN/dy
  double **basis2D;
  ierr= PetscMalloc(numD*sizeof(double*), &basis2D);CHKERRQ(ierr);
  int i;
  for(i=0;i<numD;i++) {
    ierr = PetscMalloc(Nl*sizeof(double), &basis2D[i]);CHKERRQ(ierr);
  }

  PetscInt ind;   // temporary index variable
  PetscInt ie,je; // iterators for elements
  PetscInt boffsetX,boffsetY; // offsets for l2g mapping
  PetscInt ig,jg; // iterators for gauss points
  PetscScalar gx,gy; // gauss point locations
  PetscScalar wgtx,wgty,wgt; // gauss point weights
  PetscInt iba,jba; // iterators for local basis (a, matrix rows)
  PetscScalar Nx,dNx,dNxx,Ny,dNy,dNyy; // 1D basis functions and derivatives
  PetscInt Ax,Ay; // global matrix row/col index
  PetscScalar Na,Na_x,Na_y,Na_xx,Na_yy,Na_xy; // 2D basis for row loop (a)

  PetscScalar R_rho,R_ux,R_uy;
  PetscScalar rho,rho_t,rho_x,rho_y,rho_xx,rho_yy,rho_xy;
  PetscScalar ux,ux_t,ux_x,ux_y;
  PetscScalar uy,uy_t,uy_x,uy_y;
  PetscScalar tau_xx,tau_xy,tau_yx,tau_yy;
  PetscScalar p;

  PetscScalar Ca2 = user->Ca*user->Ca;
  PetscScalar rRe = 1.0/user->Re;

  for(ie=bex;ie<=eex;ie++) { // Loop over elements
    for(je=bey;je<=eey;je++) {

      // get basis offsets used in the local-->global mapping
      ierr = BDGetBasisOffset(bdX,ie,&boffsetX);CHKERRQ(ierr);
      ierr = BDGetBasisOffset(bdY,je,&boffsetY);CHKERRQ(ierr);

      for(ig=0;ig<ngx;ig++) { // Loop over gauss points
	for(jg=0;jg<ngy;jg++) {

	  // Get gauss point locations and weights
	  // NOTE: gauss point and weight already mapped to the parameter space
	  ierr = BDGetGaussPt(bdX,ie,ig,&gx);CHKERRQ(ierr);
	  ierr = BDGetGaussPt(bdY,je,jg,&gy);CHKERRQ(ierr);
	  ierr = BDGetGaussWt(bdX,ie,ig,&wgtx);CHKERRQ(ierr);
	  ierr = BDGetGaussWt(bdY,je,jg,&wgty);CHKERRQ(ierr);

	  wgt = wgtx*wgty;

	  for(jba=0;jba<(py+1);jba++) { // Assemble the 2D basis
	    for(iba=0;iba<(px+1);iba++) {

	      ierr = BDGetBasis(bdX,ie,ig,iba,0,&Nx);CHKERRQ(ierr);
	      ierr = BDGetBasis(bdX,ie,ig,iba,1,&dNx);CHKERRQ(ierr);
	      ierr = BDGetBasis(bdX,ie,ig,iba,2,&dNxx);CHKERRQ(ierr);

	      ierr = BDGetBasis(bdY,je,jg,jba,0,&Ny);CHKERRQ(ierr);
	      ierr = BDGetBasis(bdY,je,jg,jba,1,&dNy);CHKERRQ(ierr);
	      ierr = BDGetBasis(bdY,je,jg,jba,2,&dNyy);CHKERRQ(ierr);

	      // 2D basis is a tensor product
	      ind = jba*(px+1)+iba;
	      basis2D[0][ind] =   Nx *   Ny; // N
	      basis2D[1][ind] =  dNx *   Ny; // dN/dx
	      basis2D[2][ind] =   Nx *  dNy; // dN/dy
	      basis2D[3][ind] = dNxx *   Ny; // d^2N/dx^2
	      basis2D[4][ind] =   Nx * dNyy; // d^2N/dy^2
	      basis2D[5][ind] =  dNx *  dNy; // d^2N/dxdy

	    }
	  } // end 2D basis assembly

	  // Problem coefficient evaluation
	  InterpolateSolution(basis2D,h,hdot,px,py,boffsetX,boffsetY,
			      &rho,&rho_t,&rho_x,&rho_y,
			      &rho_xx,&rho_yy,&rho_xy,
			      &ux,&ux_t,&ux_x,&ux_y,
			      &uy,&uy_t,&uy_x,&uy_y);

	  // compute pressure (Eq. 34.3)
	  p = 8.0/27.0*user->theta*rho/(1.0-rho)-rho*rho;

	  // compute viscous stress tensor (Eq. 34.4)
	  tau_xx = 2.0*ux_x - 2.0/3.0*(ux_x+uy_y);
	  tau_xy = ux_y + uy_x ;
	  tau_yy = 2.0*uy_y - 2.0/3.0*(ux_x+uy_y);
	  tau_yx = tau_xy;

	  for(jba=0;jba<(py+1);jba++) { // loop over basis 1st time (a, matrix rows)
	    for(iba=0;iba<(px+1);iba++) {

	      Ax = boffsetX+iba; // local to global map
	      Ay = boffsetY+jba;

	      ind = jba*(px+1)+iba;
	      Na     = basis2D[0][ind];
	      Na_x   = basis2D[1][ind];
	      Na_y   = basis2D[2][ind];
	      Na_xx  = basis2D[3][ind];
	      Na_yy  = basis2D[4][ind];
	      Na_xy  = basis2D[5][ind];

	      // (Eq. 19, modified to be dimensionless)
	      R_rho = Na*rho_t;
	      R_rho += -rho*(Na_x*ux + Na_y*uy);

	      R_ux = Na*ux*rho_t;
	      R_ux += Na*rho*ux_t;
	      R_ux += -rho*(Na_x*ux*ux + Na_y*ux*uy);
	      R_ux += -Na_x*p;
	      R_ux += rRe*(Na_x*tau_xx + Na_y*tau_xy);
	      //R_ux += -Ca2*rho*rho_x*(Na_xx+Na_xy);
	      //R_ux += -Ca2*Na_x*(rho_x*rho_x+rho_y*rho_y);
	      //R_ux += -Ca2*(rho_xx*Na + rho_x*Na_x + rho_xy*Na + rho_y*Na_x)*rho_x;
	      // rewritten uses Victor's corrections
	      R_ux += -Ca2*(Na_xx*rho_x + Na_xy*rho_y);
	      R_ux += -Ca2*Na_x*(rho_x*rho_x+rho_y*rho_y);
	      R_ux += -Ca2*Na*(rho_xx*rho_x+rho_xy*rho_y);
	      R_ux += -Ca2*rho_x*(Na_x*rho_x+Na_y*rho_y);

	      R_uy = Na*uy*rho_t;
	      R_uy += Na*rho*uy_t;
	      R_uy += -rho*(Na_x*uy*ux + Na_y*uy*uy);
	      R_uy += -Na_y*p;
	      R_uy += rRe*(Na_x*tau_yx + Na_y*tau_yy);

	      //R_uy += -Ca2*rho*rho_y*(Na_xy+Na_yy);
	      //R_uy += -Ca2*Na_y*(rho_x*rho_x+rho_y*rho_y);
	      //R_uy += -Ca2*(rho_xy*Na + rho_x*Na_y + rho_yy*Na + rho_y*Na_y)*rho_y;

	      R_uy += -Ca2*(Na_xy*rho_x + Na_yy*rho_y);
	      R_uy += -Ca2*Na_y*(rho_x*rho_x+rho_y*rho_y);
	      R_uy += -Ca2*Na*(rho_xy*rho_x+rho_yy*rho_y);
	      R_uy += -Ca2*rho_y*(Na_x*rho_x+Na_y*rho_y);

	      r[Ay][Ax].rho += R_rho*wgt;
	      r[Ay][Ax].ux += R_ux*wgt;
	      r[Ay][Ax].uy += R_uy*wgt;

	    }
	  } // end basis a loop

	}
      } // end gauss point loop

    }
  } // end element loop

  for(i=0;i<numD;i++) {
    ierr = PetscFree(basis2D[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(basis2D); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormTangent"
PetscErrorCode FormTangent(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal shift,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "FormTangent not implemented, use -snes_mf");

  DMDALocalInfo    info;
  DM               da_dof;
  Vec              localU,localUdot; // local versions
  Field          **h,**hdot;

  /* get the da from the snes */
  ierr = TSGetDM(ts,(DM*)&da_dof);CHKERRQ(ierr);

  /* handle the vec U */
  ierr = DMGetLocalVector(da_dof,&localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da_dof,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da_dof,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /* handle the vec Udot */
  ierr = DMGetLocalVector(da_dof,&localUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da_dof,Udot,INSERT_VALUES,localUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da_dof,Udot,INSERT_VALUES,localUdot);CHKERRQ(ierr);

  /* Get the arrays from the vectors */
  ierr = DMDAVecGetArray(da_dof,localU,&h);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da_dof,localUdot,&hdot);CHKERRQ(ierr);

  /* Grab the local info and call the local tangent routine */
  ierr = DMDAGetLocalInfo(da_dof,&info);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = MatZeroEntries(*B);CHKERRQ(ierr); // pre-zero the matrix
  ierr = FormTangentLocal(&info,t,h,hdot,shift,B,(AppCtx *) ctx);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (*A != *B) { // then we could be matrix free
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN; /* the sparsity pattern does not change */

  ierr = DMDAVecRestoreArray(da_dof,localUdot,&hdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da_dof,localU,&h);CHKERRQ(ierr);

  /* Restore the arrays and local vectors */
  ierr = DMRestoreLocalVector(da_dof,&localU);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da_dof,&localUdot);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormTangentLocal"
PetscErrorCode FormTangentLocal(DMDALocalInfo *info,PetscReal t,Field **h,Field **hdot,PetscReal shift,Mat *A,AppCtx *user)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InterpolateSolution"
PetscErrorCode InterpolateSolution(double **basis2D,Field **x,Field **xdot,PetscInt px,PetscInt py,PetscInt boffsetX,PetscInt boffsetY,
				   PetscScalar *rho,PetscScalar *rho_t,PetscScalar *rho_x,PetscScalar *rho_y,
				   PetscScalar *rho_xx,PetscScalar *rho_yy,PetscScalar *rho_xy,
				   PetscScalar *ux,PetscScalar *ux_t,PetscScalar *ux_x,PetscScalar *ux_y,
				   PetscScalar *uy,PetscScalar *uy_t,PetscScalar *uy_x,PetscScalar *uy_y)
{
  PetscFunctionBegin;
  (*rho) = 0.0; (*rho_x) = 0.0; (*rho_y) = 0.0; (*rho_t) = 0.0;
  (*rho_xx) = 0.0; (*rho_yy) = 0.0; (*rho_xy) = 0.0;
  (*ux) = 0.0; (*ux_x) = 0.0; (*ux_y) = 0.0; (*ux_t) = 0.0;
  (*uy) = 0.0; (*uy_x) = 0.0; (*uy_y) = 0.0; (*uy_t) = 0.0;

  int ipa,jpa,ind;
  for(jpa=0;jpa<(py+1);jpa++) {
    for(ipa=0;ipa<(px+1);ipa++) {

      ind = jpa*(px+1)+ipa;
      (*rho) += basis2D[0][ind] * x[boffsetY+jpa][boffsetX+ipa].rho;
      (*ux) += basis2D[0][ind] * x[boffsetY+jpa][boffsetX+ipa].ux;
      (*uy) += basis2D[0][ind] * x[boffsetY+jpa][boffsetX+ipa].uy;

      (*rho_x) += basis2D[1][ind] * x[boffsetY+jpa][boffsetX+ipa].rho;
      (*ux_x) += basis2D[1][ind] * x[boffsetY+jpa][boffsetX+ipa].ux;
      (*uy_x) += basis2D[1][ind] * x[boffsetY+jpa][boffsetX+ipa].uy;

      (*rho_y) += basis2D[2][ind] * x[boffsetY+jpa][boffsetX+ipa].rho;
      (*ux_y) += basis2D[2][ind] * x[boffsetY+jpa][boffsetX+ipa].ux;
      (*uy_y) += basis2D[2][ind] * x[boffsetY+jpa][boffsetX+ipa].uy;

      (*rho_xx) += basis2D[3][ind] * x[boffsetY+jpa][boffsetX+ipa].rho;
      (*rho_yy) += basis2D[4][ind] * x[boffsetY+jpa][boffsetX+ipa].rho;
      (*rho_xy) += basis2D[5][ind] * x[boffsetY+jpa][boffsetX+ipa].rho;

      (*rho_t) += basis2D[0][ind] * xdot[boffsetY+jpa][boffsetX+ipa].rho;
      (*ux_t) += basis2D[0][ind] * xdot[boffsetY+jpa][boffsetX+ipa].ux;
      (*uy_t) += basis2D[0][ind] * xdot[boffsetY+jpa][boffsetX+ipa].uy;

    }
  }


  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "WriteSolution"
PetscErrorCode WriteSolution(Vec U, const char pattern[],int number)
{
  PetscFunctionBegin;
  PetscErrorCode  ierr;
  MPI_Comm        comm;
  char            filename[256];
  PetscViewer     viewer;

  PetscFunctionBegin;
  sprintf(filename,pattern,number);
  ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(U,viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputMonitor"
PetscErrorCode OutputMonitor(TS ts,PetscInt it_number,PetscReal c_time,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = WriteSolution(U,"./nsk%d.dat",it_number);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
