
static char help[] = "Large-deformation Elasticity Buckling Example";

/*F-----------------------------------------------------------------------

    This example solves the 3D large deformation elasticity problem

\begin{equation}
 \int_{\Omega}F \cdot S : \nabla v d\Omega + \int_{\Omega} (loading)\mathbf{e}_y \cdot v d\Omega = 0
\end{equation}

    F is the deformation gradient, and S is the second Piola-Kirchhoff tensor from the Saint Venant-Kirchhoff model of
    hyperelasticity.  \Omega is a (arc) angle subsection of a cylindrical shell of thickness (height), inner radius
    (rad) and width (width).  The problem is discretized using Q1 finite elements on a logically structured grid.
    Homogenous Dirichlet boundary conditions are applied at the centers of the ends of the sphere.

    This example is tunable with the following options:
    -rad : the radius of the circle
    -arc : set the angle of the arch represented
    -loading : set the bulk loading (the mass)
    -ploading : set the point loading at the top
    -height : set the height of the arch
    -width : set the width of the arch
    -view_line : print initial and final offsets of the centerline of the
                 beam along the x direction

    The material properties may be modified using either:
    -mu      : the first lame' parameter
    -lambda  : the second lame' parameter

    Or:
    -young   : the Young's modulus
    -poisson : the poisson ratio

    This example is meant to show the strain placed upon the nonlinear solvers when trying to "snap through" the arch
    using the loading.  Under certain parameter regimes, the arch will invert under the load, and the number of Newton
    steps will jump considerably.  Composed nonlinear solvers may be used to mitigate this difficulty.

    The initial setup follows the example in pg. 268 of "Nonlinear Finite Element Methods" by Peter Wriggers, but is a
    3D extension.

  ------------------------------------------------------------------------F*/

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

#define QP0 0.2113248654051871
#define QP1 0.7886751345948129
#define NQ  2
#define NB  2
#define NEB 8
#define NEQ 8
#define NPB 24

#define NVALS NEB*NEQ
const PetscReal pts[NQ] = {QP0,QP1};
const PetscReal wts[NQ] = {0.5,0.5};

PetscScalar vals[NVALS];
PetscScalar grad[3*NVALS];

typedef PetscScalar Field[3];
typedef PetscScalar CoordField[3];

typedef PetscScalar JacField[9];

PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field***,Field***,void*);
PetscErrorCode FormJacobianLocal(DMDALocalInfo *,Field ***,Mat,Mat,void *);
PetscErrorCode DisplayLine(SNES,Vec);
PetscErrorCode FormElements();

typedef struct {
  PetscReal loading;
  PetscReal mu;
  PetscReal lambda;
  PetscReal rad;
  PetscReal height;
  PetscReal width;
  PetscReal arc;
  PetscReal ploading;
} AppCtx;

PetscErrorCode InitialGuess(DM,AppCtx *,Vec);
PetscErrorCode FormRHS(DM,AppCtx *,Vec);
PetscErrorCode FormCoordinates(DM,AppCtx *);
extern PetscErrorCode NonlinearGS(SNES,Vec,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  MPI_Comm       comm;
  SNES           snes;
  DM             da;
  Vec            x,X,b;
  PetscBool      youngflg,poissonflg,muflg,lambdaflg,view=PETSC_FALSE,viewline=PETSC_FALSE;
  PetscReal      poisson=0.2,young=4e4;
  char           filename[PETSC_MAX_PATH_LEN] = "ex16.vts";
  char           filename_def[PETSC_MAX_PATH_LEN] = "ex16_def.vts";

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(FormElements());
  comm = PETSC_COMM_WORLD;
  PetscCall(SNESCreate(comm,&snes));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,11,2,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(SNESSetDM(snes,(DM)da));

  PetscCall(SNESSetNGS(snes,NonlinearGS,&user));

  PetscCall(DMDAGetInfo(da,0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
  user.loading     = 0.0;
  user.arc         = PETSC_PI/3.;
  user.mu          = 4.0;
  user.lambda      = 1.0;
  user.rad         = 100.0;
  user.height      = 3.;
  user.width       = 1.;
  user.ploading    = -5e3;

  PetscCall(PetscOptionsGetReal(NULL,NULL,"-arc",&user.arc,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,&muflg));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-lambda",&user.lambda,&lambdaflg));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-rad",&user.rad,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-height",&user.height,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-width",&user.width,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-loading",&user.loading,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-ploading",&user.ploading,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-poisson",&poisson,&poissonflg));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-young",&young,&youngflg));
  if ((youngflg || poissonflg) || !(muflg || lambdaflg)) {
    /* set the lame' parameters based upon the poisson ratio and young's modulus */
    user.lambda = poisson*young / ((1. + poisson)*(1. - 2.*poisson));
    user.mu     = young/(2.*(1. + poisson));
  }
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view",&view,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view_line",&viewline,NULL));

  PetscCall(DMDASetFieldName(da,0,"x_disp"));
  PetscCall(DMDASetFieldName(da,1,"y_disp"));
  PetscCall(DMDASetFieldName(da,2,"z_disp"));

  PetscCall(DMSetApplicationContext(da,&user));
  PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user));
  PetscCall(DMDASNESSetJacobianLocal(da,(DMDASNESJacobian)FormJacobianLocal,&user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(FormCoordinates(da,&user));

  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(DMCreateGlobalVector(da,&b));
  PetscCall(InitialGuess(da,&user,x));
  PetscCall(FormRHS(da,&user,b));

  PetscCall(PetscPrintf(comm,"lambda: %f mu: %f\n",(double)user.lambda,(double)user.mu));

  /* show a cross-section of the initial state */
  if (viewline) PetscCall(DisplayLine(snes,x));

  /* get the loaded configuration */
  PetscCall(SNESSolve(snes,b,x));

  PetscCall(SNESGetIterationNumber(snes,&its));
  PetscCall(PetscPrintf(comm,"Number of SNES iterations = %" PetscInt_FMT "\n", its));
  PetscCall(SNESGetSolution(snes,&X));
  /* show a cross-section of the final state */
  if (viewline) PetscCall(DisplayLine(snes,X));

  if (view) {
    PetscViewer viewer;
    Vec         coords;
    PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(x,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(DMGetCoordinates(da,&coords));
    PetscCall(VecAXPY(coords,1.0,x));
    PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD,filename_def,FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(x,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&da));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

PetscInt OnBoundary(PetscInt i,PetscInt j,PetscInt k,PetscInt mx,PetscInt my,PetscInt mz)
{
  if ((i == 0 || i == mx-1) && j == my-1) return 1;
  return 0;
}

void BoundaryValue(PetscInt i,PetscInt j,PetscInt k,PetscInt mx,PetscInt my,PetscInt mz,PetscScalar *val,AppCtx *user)
{
  /* reference coordinates */
  PetscReal p_x = ((PetscReal)i) / (((PetscReal)(mx-1)));
  PetscReal p_y = ((PetscReal)j) / (((PetscReal)(my-1)));
  PetscReal p_z = ((PetscReal)k) / (((PetscReal)(mz-1)));
  PetscReal o_x = p_x;
  PetscReal o_y = p_y;
  PetscReal o_z = p_z;
  val[0] = o_x - p_x;
  val[1] = o_y - p_y;
  val[2] = o_z - p_z;
}

void InvertTensor(PetscScalar *t, PetscScalar *ti,PetscReal *dett)
{
  const PetscScalar a = t[0];
  const PetscScalar b = t[1];
  const PetscScalar c = t[2];
  const PetscScalar d = t[3];
  const PetscScalar e = t[4];
  const PetscScalar f = t[5];
  const PetscScalar g = t[6];
  const PetscScalar h = t[7];
  const PetscScalar i = t[8];
  const PetscReal   det = PetscRealPart(a*(e*i - f*h) - b*(i*d - f*g) + c*(d*h - e*g));
  const PetscReal   di = 1. / det;
  if (ti) {
    const PetscScalar A = (e*i - f*h);
    const PetscScalar B = -(d*i - f*g);
    const PetscScalar C = (d*h - e*g);
    const PetscScalar D = -(b*i - c*h);
    const PetscScalar E = (a*i - c*g);
    const PetscScalar F = -(a*h - b*g);
    const PetscScalar G = (b*f - c*e);
    const PetscScalar H = -(a*f - c*d);
    const PetscScalar II = (a*e - b*d);
    ti[0] = di*A;
    ti[1] = di*D;
    ti[2] = di*G;
    ti[3] = di*B;
    ti[4] = di*E;
    ti[5] = di*H;
    ti[6] = di*C;
    ti[7] = di*F;
    ti[8] = di*II;
  }
  if (dett) *dett = det;
}

void TensorTensor(PetscScalar *a,PetscScalar *b,PetscScalar *c)
{
  PetscInt i,j,m;
  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      c[i+3*j] = 0;
      for (m=0;m<3;m++)
        c[i+3*j] += a[m+3*j]*b[i+3*m];
    }
  }
}

void TensorTransposeTensor(PetscScalar *a,PetscScalar *b,PetscScalar *c)
{
  PetscInt i,j,m;
  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      c[i+3*j] = 0;
      for (m=0;m<3;m++)
        c[i+3*j] += a[3*m+j]*b[i+3*m];
    }
  }
}

void TensorVector(PetscScalar *rot, PetscScalar *vec, PetscScalar *tvec)
{
  tvec[0] = rot[0]*vec[0] + rot[1]*vec[1] + rot[2]*vec[2];
  tvec[1] = rot[3]*vec[0] + rot[4]*vec[1] + rot[5]*vec[2];
  tvec[2] = rot[6]*vec[0] + rot[7]*vec[1] + rot[8]*vec[2];
}

void DeformationGradient(Field *ex,PetscInt qi,PetscInt qj,PetscInt qk,PetscScalar *invJ,PetscScalar *F)
{
  PetscInt ii,jj,kk,l;
  for (l = 0; l < 9; l++) {
    F[l] = 0.;
  }
  F[0] = 1.;
  F[4] = 1.;
  F[8] = 1.;
  /* form the deformation gradient at this basis function -- loop over element unknowns */
  for (kk=0;kk<NB;kk++) {
    for (jj=0;jj<NB;jj++) {
      for (ii=0;ii<NB;ii++) {
        PetscInt    idx = ii + jj*NB + kk*NB*NB;
        PetscInt    bidx = NEB*idx + qi + NQ*qj + NQ*NQ*qk;
        PetscScalar lgrad[3];
        TensorVector(invJ,&grad[3*bidx],lgrad);
        F[0] += lgrad[0]*ex[idx][0]; F[1] += lgrad[1]*ex[idx][0]; F[2] += lgrad[2]*ex[idx][0];
        F[3] += lgrad[0]*ex[idx][1]; F[4] += lgrad[1]*ex[idx][1]; F[5] += lgrad[2]*ex[idx][1];
        F[6] += lgrad[0]*ex[idx][2]; F[7] += lgrad[1]*ex[idx][2]; F[8] += lgrad[2]*ex[idx][2];
      }
    }
  }
}

void DeformationGradientJacobian(PetscInt qi,PetscInt qj,PetscInt qk,PetscInt ii,PetscInt jj,PetscInt kk,PetscInt fld,PetscScalar *invJ,PetscScalar *dF)
{
  PetscInt         l;
  PetscScalar lgrad[3];
  PetscInt    idx = ii + jj*NB + kk*NB*NB;
  PetscInt    bidx = NEB*idx + qi + NQ*qj + NQ*NQ*qk;
  for (l = 0; l < 9; l++) {
    dF[l] = 0.;
  }
  /* form the deformation gradient at this basis function -- loop over element unknowns */
  TensorVector(invJ,&grad[3*bidx],lgrad);
  dF[3*fld] = lgrad[0]; dF[3*fld + 1] = lgrad[1]; dF[3*fld + 2] = lgrad[2];
}

void LagrangeGreenStrain(PetscScalar *F,PetscScalar *E)
{
  PetscInt i,j,m;
  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      E[i+3*j] = 0;
      for (m=0;m<3;m++)
        E[i+3*j] += 0.5*F[3*m+j]*F[i+3*m];
    }
  }
  for (i=0;i<3;i++) {
    E[i+3*i] -= 0.5;
  }
}

void SaintVenantKirchoff(PetscReal lambda,PetscReal mu,PetscScalar *F,PetscScalar *S)
{
  PetscInt    i,j;
  PetscScalar E[9];
  PetscScalar trE=0;
  LagrangeGreenStrain(F,E);
  for (i=0;i<3;i++) {
    trE += E[i+3*i];
  }
  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      S[i+3*j] = 2.*mu*E[i+3*j];
      if (i == j) {
        S[i+3*j] += trE*lambda;
      }
    }
  }
}

void SaintVenantKirchoffJacobian(PetscReal lambda,PetscReal mu,PetscScalar *F,PetscScalar *dF,PetscScalar *dS)
{
  PetscScalar FtdF[9],dE[9];
  PetscInt    i,j;
  PetscScalar dtrE=0.;
  TensorTransposeTensor(dF,F,dE);
  TensorTransposeTensor(F,dF,FtdF);
  for (i=0;i<9;i++) dE[i] += FtdF[i];
  for (i=0;i<9;i++) dE[i] *= 0.5;
  for (i=0;i<3;i++) {
    dtrE += dE[i+3*i];
  }
  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      dS[i+3*j] = 2.*mu*dE[i+3*j];
      if (i == j) {
        dS[i+3*j] += dtrE*lambda;
      }
    }
  }
}

PetscErrorCode FormElements()
{
  PetscInt i,j,k,ii,jj,kk;
  PetscReal bx,by,bz,dbx,dby,dbz;

  PetscFunctionBegin;
  /* construct the basis function values and derivatives */
  for (k = 0; k < NB; k++) {
    for (j = 0; j < NB; j++) {
      for (i = 0; i < NB; i++) {
        /* loop over the quadrature points */
        for (kk = 0; kk < NQ; kk++) {
          for (jj = 0; jj < NQ; jj++) {
            for (ii = 0; ii < NQ; ii++) {
              PetscInt idx = ii + NQ*jj + NQ*NQ*kk + NEQ*i + NEQ*NB*j + NEQ*NB*NB*k;
              bx = pts[ii];
              by = pts[jj];
              bz = pts[kk];
              dbx = 1.;
              dby = 1.;
              dbz = 1.;
              if (i == 0) {bx = 1. - bx; dbx = -1;}
              if (j == 0) {by = 1. - by; dby = -1;}
              if (k == 0) {bz = 1. - bz; dbz = -1;}
              vals[idx] = bx*by*bz;
              grad[3*idx + 0] = dbx*by*bz;
              grad[3*idx + 1] = dby*bx*bz;
              grad[3*idx + 2] = dbz*bx*by;
            }
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

void GatherElementData(PetscInt mx,PetscInt my,PetscInt mz,Field ***x,CoordField ***c,PetscInt i,PetscInt j,PetscInt k,Field *ex,CoordField *ec,AppCtx *user)
{
  PetscInt m;
  PetscInt ii,jj,kk;
  /* gather the data -- loop over element unknowns */
  for (kk=0;kk<NB;kk++) {
    for (jj=0;jj<NB;jj++) {
      for (ii=0;ii<NB;ii++) {
        PetscInt idx = ii + jj*NB + kk*NB*NB;
        /* decouple the boundary nodes for the displacement variables */
        if (OnBoundary(i+ii,j+jj,k+kk,mx,my,mz)) {
          BoundaryValue(i+ii,j+jj,k+kk,mx,my,mz,ex[idx],user);
        } else {
          for (m=0;m<3;m++) {
            ex[idx][m] = x[k+kk][j+jj][i+ii][m];
          }
        }
        for (m=0;m<3;m++) {
          ec[idx][m] = c[k+kk][j+jj][i+ii][m];
        }
      }
    }
  }
}

void QuadraturePointGeometricJacobian(CoordField *ec,PetscInt qi,PetscInt qj,PetscInt qk, PetscScalar *J)
{
  PetscInt ii,jj,kk;
  /* construct the gradient at the given quadrature point named by i,j,k */
  for (ii = 0; ii < 9; ii++) {
    J[ii] = 0;
  }
  for (kk = 0; kk < NB; kk++) {
    for (jj = 0; jj < NB; jj++) {
      for (ii = 0; ii < NB; ii++) {
        PetscInt idx = ii + jj*NB + kk*NB*NB;
        PetscInt bidx = NEB*idx + qi + NQ*qj + NQ*NQ*qk;
        J[0] += grad[3*bidx + 0]*ec[idx][0]; J[1] += grad[3*bidx + 1]*ec[idx][0]; J[2] += grad[3*bidx + 2]*ec[idx][0];
        J[3] += grad[3*bidx + 0]*ec[idx][1]; J[4] += grad[3*bidx + 1]*ec[idx][1]; J[5] += grad[3*bidx + 2]*ec[idx][1];
        J[6] += grad[3*bidx + 0]*ec[idx][2]; J[7] += grad[3*bidx + 1]*ec[idx][2]; J[8] += grad[3*bidx + 2]*ec[idx][2];
      }
    }
  }
}

void FormElementJacobian(Field *ex,CoordField *ec,Field *ef,PetscScalar *ej,AppCtx *user)
{
  PetscReal   vol;
  PetscScalar J[9];
  PetscScalar invJ[9];
  PetscScalar F[9],S[9],dF[9],dS[9],dFS[9],FdS[9],FS[9];
  PetscReal   scl;
  PetscInt    i,j,k,l,ii,jj,kk,ll,qi,qj,qk,m;

  if (ej) for (i = 0; i < NPB*NPB; i++) ej[i] = 0.;
  if (ef) for (i = 0; i < NEB; i++) {ef[i][0] = 0.;ef[i][1] = 0.;ef[i][2] = 0.;}
  /* loop over quadrature */
  for (qk = 0; qk < NQ; qk++) {
    for (qj = 0; qj < NQ; qj++) {
      for (qi = 0; qi < NQ; qi++) {
        QuadraturePointGeometricJacobian(ec,qi,qj,qk,J);
        InvertTensor(J,invJ,&vol);
        scl = vol*wts[qi]*wts[qj]*wts[qk];
        DeformationGradient(ex,qi,qj,qk,invJ,F);
        SaintVenantKirchoff(user->lambda,user->mu,F,S);
        /* form the function */
        if (ef) {
          TensorTensor(F,S,FS);
          for (kk=0;kk<NB;kk++) {
            for (jj=0;jj<NB;jj++) {
              for (ii=0;ii<NB;ii++) {
                PetscInt idx = ii + jj*NB + kk*NB*NB;
                PetscInt bidx = NEB*idx + qi + NQ*qj + NQ*NQ*qk;
                PetscScalar lgrad[3];
                TensorVector(invJ,&grad[3*bidx],lgrad);
                /* mu*F : grad phi_{u,v,w} */
                for (m=0;m<3;m++) {
                  ef[idx][m] += scl*
                    (lgrad[0]*FS[3*m + 0] + lgrad[1]*FS[3*m + 1] + lgrad[2]*FS[3*m + 2]);
                }
                ef[idx][1] -= scl*user->loading*vals[bidx];
              }
            }
          }
        }
        /* form the jacobian */
        if (ej) {
          /* loop over trialfunctions */
          for (k=0;k<NB;k++) {
            for (j=0;j<NB;j++) {
              for (i=0;i<NB;i++) {
                for (l=0;l<3;l++) {
                  PetscInt tridx = l + 3*(i + j*NB + k*NB*NB);
                  DeformationGradientJacobian(qi,qj,qk,i,j,k,l,invJ,dF);
                  SaintVenantKirchoffJacobian(user->lambda,user->mu,F,dF,dS);
                  TensorTensor(dF,S,dFS);
                  TensorTensor(F,dS,FdS);
                  for (m=0;m<9;m++) dFS[m] += FdS[m];
                  /* loop over testfunctions */
                  for (kk=0;kk<NB;kk++) {
                    for (jj=0;jj<NB;jj++) {
                      for (ii=0;ii<NB;ii++) {
                        PetscInt idx = ii + jj*NB + kk*NB*NB;
                        PetscInt bidx = 8*idx + qi + NQ*qj + NQ*NQ*qk;
                        PetscScalar lgrad[3];
                        TensorVector(invJ,&grad[3*bidx],lgrad);
                        for (ll=0; ll<3;ll++) {
                          PetscInt teidx = ll + 3*(ii + jj*NB + kk*NB*NB);
                          ej[teidx + NPB*tridx] += scl*
                            (lgrad[0]*dFS[3*ll + 0] + lgrad[1]*dFS[3*ll + 1] + lgrad[2]*dFS[3*ll+2]);
                        }
                      }
                    }
                  } /* end of testfunctions */
                }
              }
            }
          } /* end of trialfunctions */
        }
      }
    }
  } /* end of quadrature points */
}

void FormPBJacobian(PetscInt i,PetscInt j,PetscInt k,Field *ex,CoordField *ec,Field *ef,PetscScalar *ej,AppCtx *user)
{
  PetscReal   vol;
  PetscScalar J[9];
  PetscScalar invJ[9];
  PetscScalar F[9],S[9],dF[9],dS[9],dFS[9],FdS[9],FS[9];
  PetscReal   scl;
  PetscInt    l,ll,qi,qj,qk,m;
  PetscInt idx = i + j*NB + k*NB*NB;
  PetscScalar lgrad[3];

  if (ej) for (l = 0; l < 9; l++) ej[l] = 0.;
  if (ef) for (l = 0; l < 1; l++) {ef[l][0] = 0.;ef[l][1] = 0.;ef[l][2] = 0.;}
  /* loop over quadrature */
  for (qk = 0; qk < NQ; qk++) {
    for (qj = 0; qj < NQ; qj++) {
      for (qi = 0; qi < NQ; qi++) {
        PetscInt bidx = NEB*idx + qi + NQ*qj + NQ*NQ*qk;
        QuadraturePointGeometricJacobian(ec,qi,qj,qk,J);
        InvertTensor(J,invJ,&vol);
        TensorVector(invJ,&grad[3*bidx],lgrad);
        scl = vol*wts[qi]*wts[qj]*wts[qk];
        DeformationGradient(ex,qi,qj,qk,invJ,F);
        SaintVenantKirchoff(user->lambda,user->mu,F,S);
        /* form the function */
        if (ef) {
          TensorTensor(F,S,FS);
          for (m=0;m<3;m++) {
            ef[0][m] += scl*
              (lgrad[0]*FS[3*m + 0] + lgrad[1]*FS[3*m + 1] + lgrad[2]*FS[3*m + 2]);
          }
          ef[0][1] -= scl*user->loading*vals[bidx];
        }
        /* form the jacobian */
        if (ej) {
          for (l=0;l<3;l++) {
            DeformationGradientJacobian(qi,qj,qk,i,j,k,l,invJ,dF);
            SaintVenantKirchoffJacobian(user->lambda,user->mu,F,dF,dS);
            TensorTensor(dF,S,dFS);
            TensorTensor(F,dS,FdS);
            for (m=0;m<9;m++) dFS[m] += FdS[m];
            for (ll=0; ll<3;ll++) {
              ej[ll + 3*l] += scl*
                (lgrad[0]*dFS[3*ll + 0] + lgrad[1]*dFS[3*ll + 1] + lgrad[2]*dFS[3*ll+2]);
            }
          }
        }
      }
    } /* end of quadrature points */
  }
}

void ApplyBCsElement(PetscInt mx,PetscInt my, PetscInt mz, PetscInt i, PetscInt j, PetscInt k,PetscScalar *jacobian)
{
  PetscInt ii,jj,kk,ll,ei,ej,ek,el;
  for (kk=0;kk<NB;kk++) {
    for (jj=0;jj<NB;jj++) {
      for (ii=0;ii<NB;ii++) {
        for (ll = 0;ll<3;ll++) {
          PetscInt tridx = ll + 3*(ii + jj*NB + kk*NB*NB);
          for (ek=0;ek<NB;ek++) {
            for (ej=0;ej<NB;ej++) {
              for (ei=0;ei<NB;ei++) {
                for (el=0;el<3;el++) {
                  if (OnBoundary(i+ii,j+jj,k+kk,mx,my,mz) || OnBoundary(i+ei,j+ej,k+ek,mx,my,mz)) {
                    PetscInt teidx = el + 3*(ei + ej*NB + ek*NB*NB);
                    if (teidx == tridx) {
                      jacobian[tridx + NPB*teidx] = 1.;
                    } else {
                      jacobian[tridx + NPB*teidx] = 0.;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,Field ***x,Mat jacpre,Mat jac,void *ptr)
{
  /* values for each basis function at each quadrature point */
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       i,j,k,m,l;
  PetscInt       ii,jj,kk;
  PetscScalar    ej[NPB*NPB];
  PetscScalar    vals[NPB*NPB];
  Field          ex[NEB];
  CoordField     ec[NEB];

  PetscInt       xs=info->xs,ys=info->ys,zs=info->zs;
  PetscInt       xm=info->xm,ym=info->ym,zm=info->zm;
  PetscInt       xes,yes,zes,xee,yee,zee;
  PetscInt       mx=info->mx,my=info->my,mz=info->mz;
  DM             cda;
  CoordField     ***c;
  Vec            C;
  PetscInt       nrows;
  MatStencil     col[NPB],row[NPB];
  PetscScalar    v[9];

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(info->da,&cda));
  PetscCall(DMGetCoordinatesLocal(info->da,&C));
  PetscCall(DMDAVecGetArray(cda,C,&c));
  PetscCall(MatScale(jac,0.0));

  xes = xs;
  yes = ys;
  zes = zs;
  xee = xs+xm;
  yee = ys+ym;
  zee = zs+zm;
  if (xs > 0) xes = xs-1;
  if (ys > 0) yes = ys-1;
  if (zs > 0) zes = zs-1;
  if (xs+xm == mx) xee = xs+xm-1;
  if (ys+ym == my) yee = ys+ym-1;
  if (zs+zm == mz) zee = zs+zm-1;
  for (k=zes; k<zee; k++) {
    for (j=yes; j<yee; j++) {
      for (i=xes; i<xee; i++) {
        GatherElementData(mx,my,mz,x,c,i,j,k,ex,ec,user);
        FormElementJacobian(ex,ec,NULL,ej,user);
        ApplyBCsElement(mx,my,mz,i,j,k,ej);
        nrows = 0.;
        for (kk=0;kk<NB;kk++) {
          for (jj=0;jj<NB;jj++) {
            for (ii=0;ii<NB;ii++) {
              PetscInt idx = ii + jj*2 + kk*4;
              for (m=0;m<3;m++) {
                col[3*idx+m].i = i+ii;
                col[3*idx+m].j = j+jj;
                col[3*idx+m].k = k+kk;
                col[3*idx+m].c = m;
                if (i+ii >= xs && i+ii < xm+xs && j+jj >= ys && j+jj < ys+ym && k+kk >= zs && k+kk < zs+zm) {
                  row[nrows].i = i+ii;
                  row[nrows].j = j+jj;
                  row[nrows].k = k+kk;
                  row[nrows].c = m;
                  for (l=0;l<NPB;l++) vals[NPB*nrows + l] = ej[NPB*(3*idx+m) + l];
                  nrows++;
                }
              }
            }
          }
        }
        PetscCall(MatSetValuesStencil(jac,nrows,row,NPB,col,vals,ADD_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(jac,MAT_FLUSH_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FLUSH_ASSEMBLY));

  /* set the diagonal */
  v[0] = 1.;v[1] = 0.;v[2] = 0.;v[3] = 0.;v[4] = 1.;v[5] = 0.;v[6] = 0.;v[7] = 0.;v[8] = 1.;
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (OnBoundary(i,j,k,mx,my,mz)) {
          for (m=0; m<3;m++) {
            col[m].i = i;
            col[m].j = j;
            col[m].k = k;
            col[m].c = m;
          }
          PetscCall(MatSetValuesStencil(jac,3,col,3,col,v,INSERT_VALUES));
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));

  PetscCall(DMDAVecRestoreArray(cda,C,&c));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field ***x,Field ***f,void *ptr)
{
  /* values for each basis function at each quadrature point */
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       i,j,k,l;
  PetscInt       ii,jj,kk;

  Field          ef[NEB];
  Field          ex[NEB];
  CoordField     ec[NEB];

  PetscInt       xs=info->xs,ys=info->ys,zs=info->zs;
  PetscInt       xm=info->xm,ym=info->ym,zm=info->zm;
  PetscInt       xes,yes,zes,xee,yee,zee;
  PetscInt       mx=info->mx,my=info->my,mz=info->mz;
  DM             cda;
  CoordField     ***c;
  Vec            C;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(info->da,&cda));
  PetscCall(DMGetCoordinatesLocal(info->da,&C));
  PetscCall(DMDAVecGetArray(cda,C,&c));
  PetscCall(DMDAGetInfo(info->da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  PetscCall(DMDAGetCorners(info->da,&xs,&ys,&zs,&xm,&ym,&zm));

  /* loop over elements */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        for (l=0;l<3;l++) {
          f[k][j][i][l] = 0.;
        }
      }
    }
  }
  /* element starts and ends */
  xes = xs;
  yes = ys;
  zes = zs;
  xee = xs+xm;
  yee = ys+ym;
  zee = zs+zm;
  if (xs > 0) xes = xs - 1;
  if (ys > 0) yes = ys - 1;
  if (zs > 0) zes = zs - 1;
  if (xs+xm == mx) xee = xs+xm-1;
  if (ys+ym == my) yee = ys+ym-1;
  if (zs+zm == mz) zee = zs+zm-1;
  for (k=zes; k<zee; k++) {
    for (j=yes; j<yee; j++) {
      for (i=xes; i<xee; i++) {
        GatherElementData(mx,my,mz,x,c,i,j,k,ex,ec,user);
        FormElementJacobian(ex,ec,ef,NULL,user);
        /* put this element's additions into the residuals */
        for (kk=0;kk<NB;kk++) {
          for (jj=0;jj<NB;jj++) {
            for (ii=0;ii<NB;ii++) {
              PetscInt idx = ii + jj*NB + kk*NB*NB;
              if (k+kk >= zs && j+jj >= ys && i+ii >= xs && k+kk < zs+zm && j+jj < ys+ym && i+ii < xs+xm) {
                if (OnBoundary(i+ii,j+jj,k+kk,mx,my,mz)) {
                  for (l=0;l<3;l++)
                    f[k+kk][j+jj][i+ii][l] = x[k+kk][j+jj][i+ii][l] - ex[idx][l];
                } else {
                  for (l=0;l<3;l++)
                    f[k+kk][j+jj][i+ii][l] += ef[idx][l];
                }
              }
            }
          }
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(cda,C,&c));
  PetscFunctionReturn(0);
}

PetscErrorCode NonlinearGS(SNES snes,Vec X,Vec B,void *ptr)
{
  /* values for each basis function at each quadrature point */
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       i,j,k,l,m,n,s;
  PetscInt       pi,pj,pk;
  Field          ef[1];
  Field          ex[8];
  PetscScalar    ej[9];
  CoordField     ec[8];
  PetscScalar    pjac[9],pjinv[9];
  PetscScalar    pf[3],py[3];
  PetscInt       xs,ys,zs;
  PetscInt       xm,ym,zm;
  PetscInt       mx,my,mz;
  DM             cda;
  CoordField     ***c;
  Vec            C;
  DM             da;
  Vec            Xl,Bl;
  Field          ***x,***b;
  PetscInt       sweeps,its;
  PetscReal      atol,rtol,stol;
  PetscReal      fnorm0 = 0.0,fnorm,ynorm,xnorm = 0.0;

  PetscFunctionBegin;
  PetscCall(SNESNGSGetSweeps(snes,&sweeps));
  PetscCall(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&its));

  PetscCall(SNESGetDM(snes,&da));
  PetscCall(DMGetLocalVector(da,&Xl));
  if (B) {
    PetscCall(DMGetLocalVector(da,&Bl));
  }
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xl));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xl));
  if (B) {
    PetscCall(DMGlobalToLocalBegin(da,B,INSERT_VALUES,Bl));
    PetscCall(DMGlobalToLocalEnd(da,B,INSERT_VALUES,Bl));
  }
  PetscCall(DMDAVecGetArray(da,Xl,&x));
  if (B) PetscCall(DMDAVecGetArray(da,Bl,&b));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&C));
  PetscCall(DMDAVecGetArray(cda,C,&c));
  PetscCall(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));

  for (s=0;s<sweeps;s++) {
    for (k=zs; k<zs+zm; k++) {
      for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
          if (OnBoundary(i,j,k,mx,my,mz)) {
            BoundaryValue(i,j,k,mx,my,mz,x[k][j][i],user);
          } else {
            for (n=0;n<its;n++) {
              for (m=0;m<9;m++) pjac[m] = 0.;
              for (m=0;m<3;m++) pf[m] = 0.;
              /* gather the elements for this point */
              for (pk=-1; pk<1; pk++) {
                for (pj=-1; pj<1; pj++) {
                  for (pi=-1; pi<1; pi++) {
                    /* check that this element exists */
                    if (i+pi >= 0 && i+pi < mx-1 && j+pj >= 0 && j+pj < my-1 && k+pk >= 0 && k+pk < mz-1) {
                      /* create the element function and jacobian */
                      GatherElementData(mx,my,mz,x,c,i+pi,j+pj,k+pk,ex,ec,user);
                      FormPBJacobian(-pi,-pj,-pk,ex,ec,ef,ej,user);
                      /* extract the point named by i,j,k from the whole element jacobian and function */
                      for (l=0;l<3;l++) {
                        pf[l] += ef[0][l];
                        for (m=0;m<3;m++) {
                          pjac[3*m+l] += ej[3*m+l];
                        }
                      }
                    }
                  }
                }
              }
              /* invert */
              InvertTensor(pjac,pjinv,NULL);
              /* apply */
              if (B) for (m=0;m<3;m++) {
                  pf[m] -= b[k][j][i][m];
                }
              TensorVector(pjinv,pf,py);
              xnorm=0.;
              for (m=0;m<3;m++) {
                x[k][j][i][m] -= py[m];
                xnorm += PetscRealPart(x[k][j][i][m]*x[k][j][i][m]);
              }
              fnorm = PetscRealPart(pf[0]*pf[0]+pf[1]*pf[1]+pf[2]*pf[2]);
              if (n==0) fnorm0 = fnorm;
              ynorm = PetscRealPart(py[0]*py[0]+py[1]*py[1]+py[2]*py[2]);
              if (fnorm < atol*atol || fnorm < rtol*rtol*fnorm0 || ynorm < stol*stol*xnorm) break;
            }
          }
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,Xl,&x));
  PetscCall(DMLocalToGlobalBegin(da,Xl,INSERT_VALUES,X));
  PetscCall(DMLocalToGlobalEnd(da,Xl,INSERT_VALUES,X));
  PetscCall(DMRestoreLocalVector(da,&Xl));
  if (B) {
    PetscCall(DMDAVecRestoreArray(da,Bl,&b));
    PetscCall(DMRestoreLocalVector(da,&Bl));
  }
  PetscCall(DMDAVecRestoreArray(cda,C,&c));
  PetscFunctionReturn(0);
}

PetscErrorCode FormCoordinates(DM da,AppCtx *user)
{
  Vec            coords;
  DM             cda;
  PetscInt       mx,my,mz;
  PetscInt       i,j,k,xs,ys,zs,xm,ym,zm;
  CoordField     ***x;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMCreateGlobalVector(cda,&coords));
  PetscCall(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));
  PetscCall(DMDAVecGetArray(da,coords,&x));
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        PetscReal cx = ((PetscReal)i) / (((PetscReal)(mx-1)));
        PetscReal cy = ((PetscReal)j) / (((PetscReal)(my-1)));
        PetscReal cz = ((PetscReal)k) / (((PetscReal)(mz-1)));
        PetscReal rad = user->rad + cy*user->height;
        PetscReal ang = (cx - 0.5)*user->arc;
        x[k][j][i][0] = rad*PetscSinReal(ang);
        x[k][j][i][1] = rad*PetscCosReal(ang) - (user->rad + 0.5*user->height)*PetscCosReal(-0.5*user->arc);
        x[k][j][i][2] = user->width*(cz - 0.5);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,coords,&x));
  PetscCall(DMSetCoordinates(da,coords));
  PetscCall(VecDestroy(&coords));
  PetscFunctionReturn(0);
}

PetscErrorCode InitialGuess(DM da,AppCtx *user,Vec X)
{
  PetscInt       i,j,k,xs,ys,zs,xm,ym,zm;
  PetscInt       mx,my,mz;
  Field          ***x;

  PetscFunctionBegin;
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));
  PetscCall(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  PetscCall(DMDAVecGetArray(da,X,&x));

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        /* reference coordinates */
        PetscReal p_x = ((PetscReal)i) / (((PetscReal)(mx-1)));
        PetscReal p_y = ((PetscReal)j) / (((PetscReal)(my-1)));
        PetscReal p_z = ((PetscReal)k) / (((PetscReal)(mz-1)));
        PetscReal o_x = p_x;
        PetscReal o_y = p_y;
        PetscReal o_z = p_z;
        x[k][j][i][0] = o_x - p_x;
        x[k][j][i][1] = o_y - p_y;
        x[k][j][i][2] = o_z - p_z;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode FormRHS(DM da,AppCtx *user,Vec X)
{
  PetscInt       i,j,k,xs,ys,zs,xm,ym,zm;
  PetscInt       mx,my,mz;
  Field          ***x;

  PetscFunctionBegin;
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));
  PetscCall(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  PetscCall(DMDAVecGetArray(da,X,&x));

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        x[k][j][i][0] = 0.;
        x[k][j][i][1] = 0.;
        x[k][j][i][2] = 0.;
        if (i == (mx-1)/2 && j == (my-1)) x[k][j][i][1] = user->ploading/(mz-1);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode DisplayLine(SNES snes,Vec X)
{
  PetscInt       r,i,j=0,k=0,xs,xm,ys,ym,zs,zm,mx,my,mz;
  Field          ***x;
  CoordField     ***c;
  DM             da,cda;
  Vec            C;
  PetscMPIInt    size,rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(SNESGetDM(snes,&da));
  PetscCall(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinates(da,&C));
  j = my / 2;
  k = mz / 2;
  PetscCall(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));
  PetscCall(DMDAVecGetArray(da,X,&x));
  PetscCall(DMDAVecGetArray(cda,C,&c));
  for (r=0;r<size;r++) {
    if (rank == r) {
      if (j >= ys && j < ys+ym && k >= zs && k < zs+zm) {
        for (i=xs; i<xs+xm; i++) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " %d %d: %f %f %f\n",i,0,0,(double)PetscRealPart(c[k][j][i][0] + x[k][j][i][0]),(double)PetscRealPart(c[k][j][i][1] + x[k][j][i][1]),(double)PetscRealPart(c[k][j][i][2] + x[k][j][i][2])));
        }
      }
    }
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }
  PetscCall(DMDAVecRestoreArray(da,X,&x));
  PetscCall(DMDAVecRestoreArray(cda,C,&c));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 2
      args: -da_refine 2 -pc_type mg -rad 10.0 -young 10. -ploading 0.0 -loading -1. -mg_levels_ksp_max_it 2 -snes_monitor_short -ksp_monitor_short -snes_max_it 7
      requires: !single
      timeoutfactor: 3

   test:
      suffix: 2
      args: -da_refine 2 -pc_type mg -rad 10.0 -young 10. -ploading 0.0 -loading -1. -mg_levels_ksp_max_it 2 -snes_monitor_short -ksp_monitor_short -npc_snes_type fas -npc_fas_levels_snes_type ncg -npc_fas_levels_snes_max_it 3 -npc_snes_monitor_short -snes_max_it 2
      requires: !single

   test:
      suffix: 3
      args: -da_refine 1 -da_overlap 3 -da_local_subdomains 4 -snes_type aspin -rad 10.0 -young 10. -ploading 0.0 -loading -0.5 -snes_monitor_short -ksp_monitor_short -npc_sub_snes_rtol 1e-2 -ksp_rtol 1e-2 -ksp_max_it 14 -snes_converged_reason -snes_max_linear_solve_fail 100 -snes_max_it 4
      requires: !single

TEST*/
