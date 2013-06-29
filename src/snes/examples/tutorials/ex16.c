
static char help[] = "Large-deformation Elasticity Buckling Example";
/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DMDA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/*F-----------------------------------------------------------------------

 ------------------------------------------------------------------------F*/

#include <petscsnes.h>
#include <petscdmda.h>

#define QP0 0.2113248654051871
#define QP1 0.7886751345948129
#define NQP 2
#define NB  2
#define NVALS NQP*NQP*NQP*NB*NB*NB
const PetscReal pts[NQP] = {QP0,QP1};
const PetscReal wts[NQP] = {0.5,0.5};

PetscReal vals[NVALS];
PetscReal grad[3*NVALS];

typedef PetscScalar Field[3];
typedef PetscScalar CoordField[3];



typedef PetscScalar JacField[9];

PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field***,Field***,void*);
PetscErrorCode FormJacobianLocal(DMDALocalInfo *,Field ***,Mat,Mat,MatStructure *,void *);
PetscErrorCode DisplayLine(DM,Vec);
PetscErrorCode FormElements();

typedef struct {
  PetscReal loading;
  PetscReal mu;
  PetscReal lambda;
  PetscReal squeeze;
  PetscReal len;
  PetscReal arch;
} AppCtx;

PetscErrorCode InitialGuess(DM,AppCtx *,Vec);
PetscErrorCode FormCoordinates(DM,AppCtx *);
extern PetscErrorCode NonlinearGS(SNES,Vec,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da;
  Vec            x;
  PetscBool      youngflg,poissonflg,view=PETSC_FALSE,viewline=PETSC_FALSE;
  PetscReal      poisson=1.0,young=1.0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return(1);

  PetscFunctionBeginUser;
  ierr = FormElements();CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,-4,-4,-4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,(DM)da);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  user.loading     = -0.1;
  user.arch        = 0.5;
  user.mu          = 4.0;
  user.lambda      = 1.0;
  user.squeeze     = 0.0;
  user.len         = 5.0;

  ierr = PetscOptionsGetReal(NULL,"-loading",&user.loading,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-arch",&user.arch,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-lambda",&user.lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-squeeze",&user.squeeze,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-len",&user.len,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(NULL,"-poisson",&poisson,&poissonflg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-young",&young,&youngflg);CHKERRQ(ierr);
  if (youngflg || poissonflg) {
    /* set the lame' parameters based upon the poisson ratio and young's modulus */
    user.lambda = poisson*young / ((1. + poisson)*(1. - 2.*poisson));
    user.mu     = young/(2.*(1. + poisson));
  }
  ierr = PetscOptionsGetBool(NULL,"-view",&view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-view_line",&viewline,NULL);CHKERRQ(ierr);

  ierr = DMDASetFieldName(da,0,"x_disp");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"y_disp");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"y_disp");CHKERRQ(ierr);

  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user);CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,(DMDASNESJacobian)FormJacobianLocal,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = FormCoordinates(da,&user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = InitialGuess(da,&user,x);CHKERRQ(ierr);

  /* show a cross-section of the initial state */
  if (viewline) {
    ierr = DisplayLine(da,x);CHKERRQ(ierr);
  }

  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Number of SNES iterations = %D\n", its);CHKERRQ(ierr);

  /* show a cross-section of the final state */
  if (viewline) {
    ierr = DisplayLine(da,x);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
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
  const PetscReal   det = a*(e*i - f*h) - b*(i*d - f*g) + c*(d*h - e*g);
  if (dett) *dett = det;
  const PetscReal   di = 1. / det;
  const PetscScalar A = (e*i - f*h);
  const PetscScalar B = -(d*i - f*g);
  const PetscScalar C = (d*h - e*g);
  const PetscScalar D = -(b*i - c*h);
  const PetscScalar E = (a*i - c*g);
  const PetscScalar F = -(a*h - b*g);
  const PetscScalar G = (b*f - c*e);
  const PetscScalar H = -(a*f - c*d);
  const PetscScalar I = (a*e - b*d);
  ti[0] = di*A;
  ti[1] = di*D;
  ti[2] = di*G;
  ti[3] = di*B;
  ti[4] = di*E;
  ti[5] = di*H;
  ti[6] = di*C;
  ti[7] = di*F;
  ti[8] = di*I;
}

void TensorTensor(PetscScalar *a,PetscScalar *b,PetscScalar *c)
{
  int i,j,m;
  for(i=0;i<3;i++) {
    for(j=0;j<3;j++) {
      c[i+3*j] = 0;
      for(m=0;m<3;m++)
        c[i+3*j] += a[m+3*j]*b[i+3*m];
    }
  }
}

void TensorTransposeTensor(PetscScalar *a,PetscScalar *b,PetscScalar *c)
{
  int i,j,m;
  for(i=0;i<3;i++) {
    for(j=0;j<3;j++) {
      c[i+3*j] = 0;
      for(m=0;m<3;m++)
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

void DeformationGradient(Field *ex,PetscInt qi,PetscInt qj,PetscInt qk,PetscScalar *invJ,PetscScalar *F) {
  int ii,jj,kk,l;
  for (l = 0; l < 9; l++) {
    F[l] = 0.;
  }
  F[0] = 1.;
  F[4] = 1.;
  F[8] = 1.;
  /* form the deformation gradient at this basis function -- loop over element unknowns */
  for (kk=0;kk<2;kk++){
    for (jj=0;jj<2;jj++) {
      for (ii=0;ii<2;ii++) {
        PetscInt idx = ii + jj*2 + kk*4;
        PetscInt bidx = 8*idx + qi + 2*qj + 4*qk;
        PetscScalar lgrad[3];
        TensorVector(invJ,&grad[3*bidx],lgrad);
        F[0] += lgrad[0]*ex[idx][0]; F[1] += lgrad[1]*ex[idx][0]; F[2] += lgrad[2]*ex[idx][0];
        F[3] += lgrad[0]*ex[idx][1]; F[4] += lgrad[1]*ex[idx][1]; F[5] += lgrad[2]*ex[idx][1];
        F[6] += lgrad[0]*ex[idx][2]; F[7] += lgrad[1]*ex[idx][2]; F[8] += lgrad[2]*ex[idx][2];
      }
    }
  }
}

void DeformationGradientJacobian(PetscInt qi,PetscInt qj,PetscInt qk,PetscInt ii,PetscInt jj,PetscInt kk,PetscInt fld,PetscScalar *invJ,PetscScalar *dF) {
  int l;
  for (l = 0; l < 9; l++) {
    dF[l] = 0.;
  }
  /* form the deformation gradient at this basis function -- loop over element unknowns */
  PetscScalar lgrad[3];
  PetscInt idx = ii + jj*2 + kk*4;
  PetscInt bidx = 8*idx + qi + 2*qj + 4*qk;
  TensorVector(invJ,&grad[3*bidx],lgrad);
  dF[3*fld] = lgrad[0]; dF[3*fld + 1] = lgrad[1]; dF[3*fld + 2] = lgrad[2];
}

void LagrangeGreenStrain(PetscScalar *F,PetscScalar *E)
{
  int i,j,m;
  for(i=0;i<3;i++) {
    for(j=0;j<3;j++) {
      E[i+3*j] = 0;
      for(m=0;m<3;m++)
        E[i+3*j] += 0.5*F[3*m+j]*F[i+3*m];
    }
  }
  for(i=0;i<3;i++) {
    E[i+3*i] -= 0.5;
  }
}

void SaintVenantKirchoff(PetscReal lambda,PetscReal mu,PetscScalar *F,PetscScalar *S) {
  int i,j;
  PetscScalar E[9];
  LagrangeGreenStrain(F,E);
  PetscReal trE=0;
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

void SaintVenantKirchoffJacobian(PetscReal lambda,PetscReal mu,PetscScalar *F,PetscScalar *dF,PetscScalar *dS) {
  PetscScalar FtdF[9],dE[9];
  PetscInt i,j;
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

#undef __FUNCT__
#define __FUNCT__ "FormElements"
PetscErrorCode FormElements()
{
  PetscInt i,j,k,ii,jj,kk;
  PetscReal bx,by,bz,dbx,dby,dbz;
  PetscFunctionBegin;
  /* construct the basis function values and derivatives */
  for (k = 0; k < 2; k++) {
    for (j = 0; j < 2; j++) {
      for (i = 0; i < 2; i++) {
        /* loop over the quadrature points */
        for (kk = 0; kk < 2; kk++) {
          for (jj = 0; jj < 2; jj++) {
            for (ii = 0; ii < 2; ii++) {
              PetscInt idx = ii + 2*jj + 4*kk + 8*i + 16*j + 32*k;
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


void QuadraturePointGeometricJacobian(CoordField *ec,PetscInt qi,PetscInt qj,PetscInt qk, PetscScalar *J) {
  PetscInt ii,jj,kk;
  /* construct the gradient at the given quadrature point named by i,j,k */
  for (ii = 0; ii < 9; ii++) {
    J[ii] = 0;
  }
  for (kk = 0; kk < NB; kk++) {
    for (jj = 0; jj < NB; jj++) {
      for (ii = 0; ii < NB; ii++) {
        PetscInt idx = ii + jj*2 + kk*4;
        PetscInt bidx = 8*idx + qi + 2*qj + 4*qk;
        J[0] += grad[3*bidx + 0]*ec[idx][0]; J[1] += grad[3*bidx + 1]*ec[idx][0]; J[2] += grad[3*bidx + 2]*ec[idx][0];
        J[3] += grad[3*bidx + 0]*ec[idx][1]; J[4] += grad[3*bidx + 1]*ec[idx][1]; J[5] += grad[3*bidx + 2]*ec[idx][1];
        J[6] += grad[3*bidx + 0]*ec[idx][2]; J[7] += grad[3*bidx + 1]*ec[idx][2]; J[8] += grad[3*bidx + 2]*ec[idx][2];
      }
    }
  }
}

void FormElementFunction(Field *ex, CoordField *ec, Field *ef,AppCtx *user) {
  PetscScalar vol;
  PetscScalar J[9];
  PetscScalar invJ[9];
  PetscScalar F[9];
  PetscScalar S[9];
  PetscScalar FS[9];
  PetscReal   scl;
  PetscInt    ii,jj,kk,qi,qj,qk,m;
  /* form the residuals -- this loop is over test functions*/
  for (qk = 0; qk < NQP; qk++) {
    for (qj = 0; qj < NQP; qj++) {
      for (qi = 0; qi < NQP; qi++) {
        QuadraturePointGeometricJacobian(ec,qi,qj,qk,J);
        InvertTensor(J,invJ,&vol);
        scl = vol*wts[qi]*wts[qj]*wts[qk];
        DeformationGradient(ex,qi,qj,qk,invJ,F);
        SaintVenantKirchoff(user->lambda,user->mu,F,S);
        TensorTensor(F,S,FS);
        for (kk=0;kk<2;kk++){
          for (jj=0;jj<2;jj++) {
            for (ii=0;ii<2;ii++) {
              PetscInt idx = ii + jj*2 + kk*4;
              PetscInt bidx = 8*idx + qi + 2*qj + 4*qk;
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
    }
  }
}

void FormElementJacobian(Field *ex,CoordField *ec,PetscScalar *ej,AppCtx *user) {
  PetscScalar vol;
  PetscScalar J[9];
  PetscScalar invJ[9];
  PetscScalar F[9],S[9],dF[9],dS[9],dFS[9],FdS[9];
  PetscReal   scl;
  PetscInt    i,j,k,l,ii,jj,kk,ll,qi,qj,qk,m;
  for (i = 0; i < 8*8*3*3; i++) ej[i] = 0.;
  /* loop over quadrature */
  for (qk = 0; qk < NQP; qk++) {
    for (qj = 0; qj < NQP; qj++) {
      for (qi = 0; qi < NQP; qi++) {
        QuadraturePointGeometricJacobian(ec,qi,qj,qk,J);
        InvertTensor(J,invJ,&vol);
        scl = vol*wts[qi]*wts[qj]*wts[qk];
        DeformationGradient(ex,qi,qj,qk,invJ,F);
        SaintVenantKirchoff(user->lambda,user->mu,F,S);
        /* loop over trialfunctions */
        for (k=0;k<2;k++){
          for (j=0;j<2;j++) {
            for (i=0;i<2;i++) {
              for (l=0;l<3;l++) {
                PetscInt tridx = l + 3*(i + j*2 + k*4);
                DeformationGradientJacobian(qi,qj,qk,i,j,k,l,invJ,dF);
                SaintVenantKirchoffJacobian(user->lambda,user->mu,F,dF,dS);
                TensorTensor(dF,S,dFS);
                TensorTensor(F,dS,FdS);
                for (m=0;m<9;m++) dFS[m] += FdS[m];
                /* loop over testfunctions */
                for (kk=0;kk<2;kk++){
                  for (jj=0;jj<2;jj++) {
                    for (ii=0;ii<2;ii++) {
                      PetscInt idx = ii + jj*2 + kk*4;
                      PetscInt bidx = 8*idx + qi + 2*qj + 4*qk;
                      PetscScalar lgrad[3];
                      TensorVector(invJ,&grad[3*bidx],lgrad);
                      for (ll=0; ll<3;ll++) {
                        PetscInt teidx = ll + 3*(ii + jj*2 + kk*4);
                        ej[teidx + 24*tridx] += scl*
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
  } /* end of quadrature points */
}

void ApplyBCsElement(PetscInt mx,PetscInt i, PetscInt j, PetscInt k,PetscScalar *jacobian) {
  PetscInt ii,jj,kk,ll,ei,ej,ek,el;
  for (kk=0;kk<2;kk++){
    for (jj=0;jj<2;jj++) {
      for (ii=0;ii<2;ii++) {
        for(ll = 0;ll<3;ll++) {
          PetscInt tridx = ll + 3*(ii + jj*2 + kk*4);
          for (ek=0;ek<2;ek++){
            for (ej=0;ej<2;ej++) {
              for (ei=0;ei<2;ei++) {
                for (el=0;el<3;el++) {
                  if ((i + ii == 0) || (i + ii == mx - 1) || (i + ei == 0) || (i + ei == mx - 1)) {
                    PetscInt teidx = el + 3*(ei + ej*2 + ek*4);
                    if (teidx == tridx) {
                      jacobian[tridx + 24*teidx] = 1.;
                    } else {
                      jacobian[tridx + 24*teidx] = 0.;

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

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,Field ***x,Mat jacpre,Mat jac,MatStructure *flg,void *ptr)
{
  PetscFunctionBegin;
  /* values for each basis function at each quadrature point */
  AppCtx      *user = (AppCtx*)ptr;

  PetscInt i,j,k,m;
  PetscInt ii,jj,kk;
  PetscInt mx,my,mz;

  PetscScalar ej[8*3*8*3];
  Field ex[8];
  CoordField ec[8];

  PetscErrorCode ierr;
  PetscInt xs,ys,zs,xm,ym,zm;
  DM          cda;
  CoordField  ***c;
  Vec         C;
  MatStencil  col[24];

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(info->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(info->da,&C);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,C,&c);CHKERRQ(ierr);
  ierr = DMDAGetInfo(info->da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(info->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = MatScale(jac,0.0);CHKERRQ(ierr);

  for (k=zs; k<zs+zm-1; k++) {
    for (j=ys; j<ys+ym-1; j++) {
      for (i=xs; i<xs+xm-1; i++) {
        /* gather the data -- loop over element unknowns */
        for (kk=0;kk<2;kk++){
          for (jj=0;jj<2;jj++) {
            for (ii=0;ii<2;ii++) {
              PetscInt idx = ii + jj*2 + kk*4;

              /* decouple the boundary nodes for the displacement variables */
              if ((i == 0 && ii == 0)) {
                ex[idx][0] = user->squeeze/2;
                ex[idx][1] = 0.;
                ex[idx][2] = 0.;
              } else if ((i == mx-2 && ii == 1)) {
                ex[idx][0] = -user->squeeze/2;
                ex[idx][1] = 0.;
                ex[idx][2] = 0.;
              } else {
                for (m=0;m<3;m++) {
                  ex[idx][m] = x[k+kk][j+jj][i+ii][m];
                }
              }
              for (m=0;m<3;m++) {
                ec[idx][m] = c[k+kk][j+jj][i+ii][m];
              }
              for (m=0;m<3;m++) {
                col[3*idx+m].i = i+ii;
                col[3*idx+m].j = j+jj;
                col[3*idx+m].k = k+kk;
                col[3*idx+m].c = m;
              }
            }
          }
        }
        FormElementJacobian(ex,ec,ej,user);
        ApplyBCsElement(mx,i,j,k,ej);
        ierr = MatSetValuesStencil(jac,24,col,24,col,ej,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cda,C,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field ***x,Field ***f,void *ptr)
{
  /* values for each basis function at each quadrature point */
  AppCtx      *user = (AppCtx*)ptr;

  PetscInt i,j,k,l;
  PetscInt ii,jj,kk;
  PetscInt mx,my,mz;

  Field ef[8];
  Field ex[8];
  CoordField ec[8];

  PetscErrorCode ierr;
  PetscInt xs,ys,zs,xm,ym,zm;
  DM          cda;
  CoordField  ***c;
  Vec         C;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(info->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(info->da,&C);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,C,&c);CHKERRQ(ierr);
  ierr = DMDAGetInfo(info->da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(info->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

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
  for (k=zs; k<zs+zm-1; k++) {
    for (j=ys; j<ys+ym-1; j++) {
      for (i=xs; i<xs+xm-1; i++) {
        /* gather the data -- loop over element unknowns */
        for (kk=0;kk<2;kk++){
          for (jj=0;jj<2;jj++) {
            for (ii=0;ii<2;ii++) {
              PetscInt idx = ii + jj*2 + kk*4;
              /* decouple the boundary nodes for the displacement variables */
              if ((i == 0 && ii == 0)) {
                ex[idx][0] = user->squeeze/2;
                ex[idx][1] = 0.;
                ex[idx][2] = 0.;
              } else if ((i == mx-2 && ii == 1)) {
                ex[idx][0] = -user->squeeze/2;
                ex[idx][1] = 0.;
                ex[idx][2] = 0.;
              } else {
                for (l=0;l<3;l++) {
                  ex[idx][l] = x[k+kk][j+jj][i+ii][l];
                }
              }
              for (l=0;l<3;l++) {
                ec[idx][l] = c[k+kk][j+jj][i+ii][l];
                ef[idx][l] = 0.;
              }
            }
          }
        }
        FormElementFunction(ex,ec,ef,user);
        /* put this element's additions into the residuals */
        for (kk=0;kk<2;kk++){
          for (jj=0;jj<2;jj++) {
            for (ii=0;ii<2;ii++) {
              PetscInt idx = ii + jj*2 + kk*4;
              if ((i == 0 && ii == 0) || (i == mx-2 && ii == 1)) {
              } else {
                for (l=0;l<3;l++) {
                  f[k+kk][j+jj][i+ii][l] += ef[idx][l];
                }
              }
            }
          }
        }
      }
    }
  }
ierr = DMDAVecRestoreArray(cda,C,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormCoordinates"
PetscErrorCode FormCoordinates(DM da,AppCtx *user) {
  PetscErrorCode ierr;
  Vec coords,lcoords;
  DM cda;
  PetscInt       mx,my,mz;
  PetscInt       i,j,k,xs,ys,zs,xm,ym,zm;
  CoordField ***x;
  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(da,&cda);
  ierr = DMCreateGlobalVector(cda,&coords);
  ierr = DMCreateLocalVector(cda,&lcoords);
  ierr = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,coords,&x);CHKERRQ(ierr);
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        PetscReal cx = ((PetscReal)i) / (((PetscReal)(mx-1)));
        x[k][j][i][0] = user->len*((PetscReal)i) / (((PetscReal)(mx-1)));
        x[k][j][i][1] = ((PetscReal)j) / (((PetscReal)(my-1))) + 4.*cx*(1.-cx)*user->arch;
        x[k][j][i][2] = ((PetscReal)k) / (((PetscReal)(mz-1)));
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,coords,&x);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(cda,coords,INSERT_VALUES,lcoords);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(cda,coords,INSERT_VALUES,lcoords);CHKERRQ(ierr);
  ierr = DMSetCoordinates(da,coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(da,lcoords);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  ierr = VecDestroy(&lcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitialGuess"
PetscErrorCode InitialGuess(DM da,AppCtx *user,Vec X)
{
  PetscInt       i,j,k,xs,ys,zs,xm,ym,zm;
  PetscInt       mx,my,mz;
  PetscErrorCode ierr;
  Field          ***x;
  PetscFunctionBegin;

  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        /* reference coordinates */
        PetscReal p_x = user->len*((PetscReal)i) / (((PetscReal)(mx-1)));
        PetscReal p_y = ((PetscReal)j) / (((PetscReal)(my-1)));
        PetscReal p_z = ((PetscReal)k) / (((PetscReal)(mz-1)));
        PetscReal sqz = -((PetscReal)i - ((PetscReal)(mx-1))/2)*(user->squeeze/2) / (((PetscReal)(mx-1))/2);
        PetscReal o_x = p_x + sqz;
        PetscReal o_y = p_y;
        PetscReal o_z = p_z;
        x[k][j][i][0] = o_x - p_x;
        x[k][j][i][1] = o_y - p_y;
        x[k][j][i][2] = o_z - p_z;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DisplayLine"
PetscErrorCode DisplayLine(DM da,Vec X)
{
  PetscInt       i,j=0,k=0,xs,xm,mx,my,mz;
  PetscErrorCode ierr;
  Field          ***x;
  CoordField     ***c;
  DM             cda;
  Vec            C;
  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&C);CHKERRQ(ierr);
  j = my / 2;
  k = mz / 2;
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,C,&c);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    PetscPrintf(PETSC_COMM_SELF,"%d %d %d: %f %f %f\n",i,0,0,c[k][j][i][0] + x[k][j][i][0],c[k][j][i][1] + x[k][j][i][1],c[k][j][i][2] + x[k][j][i][2]);
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cda,C,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
