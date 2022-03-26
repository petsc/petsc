
static char help[] = "Linear elastiticty with dimensions using 20 node serendipity elements.\n\
This also demonstrates use of  block\n\
diagonal data structure.  Input arguments are:\n\
  -m : problem size\n\n";

#include <petscksp.h>

/* This code is not intended as an efficient implementation, it is only
   here to produce an interesting sparse matrix quickly.

   PLEASE DO NOT BASE ANY OF YOUR CODES ON CODE LIKE THIS, THERE ARE MUCH
   BETTER WAYS TO DO THIS. */

extern PetscErrorCode GetElasticityMatrix(PetscInt,Mat*);
extern PetscErrorCode Elastic20Stiff(PetscReal**);
extern PetscErrorCode AddElement(Mat,PetscInt,PetscInt,PetscReal**,PetscInt,PetscInt);
extern PetscErrorCode paulsetup20(void);
extern PetscErrorCode paulintegrate20(PetscReal K[60][60]);

int main(int argc,char **args)
{
  Mat            mat;
  PetscInt       i,its,m = 3,rdim,cdim,rstart,rend;
  PetscMPIInt    rank,size;
  PetscScalar    v,neg1 = -1.0;
  Vec            u,x,b;
  KSP            ksp;
  PetscReal      norm;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Form matrix */
  PetscCall(GetElasticityMatrix(m,&mat));

  /* Generate vectors */
  PetscCall(MatGetSize(mat,&rdim,&cdim));
  PetscCall(MatGetOwnershipRange(mat,&rstart,&rend));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,rdim));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecDuplicate(b,&x));
  for (i=rstart; i<rend; i++) {
    v    = (PetscScalar)(i-rstart + 100*rank);
    PetscCall(VecSetValues(u,1,&i,&v,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));

  /* Compute right-hand-side */
  PetscCall(MatMult(mat,u,b));

  /* Solve linear system */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,mat,mat));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  /* Check error */
  PetscCall(VecAXPY(x,neg1,u));
  PetscCall(VecNorm(x,NORM_2,&norm));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of residual %g Number of iterations %D\n",(double)norm,its));

  /* Free work space */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&mat));

  PetscCall(PetscFinalize());
  return 0;
}
/* -------------------------------------------------------------------- */
/*
  GetElasticityMatrix - Forms 3D linear elasticity matrix.
 */
PetscErrorCode GetElasticityMatrix(PetscInt m,Mat *newmat)
{
  PetscInt       i,j,k,i1,i2,j_1,j2,k1,k2,h1,h2,shiftx,shifty,shiftz;
  PetscInt       ict,nz,base,r1,r2,N,*rowkeep,nstart;
  IS             iskeep;
  PetscReal      **K,norm;
  Mat            mat,submat = 0,*submatb;
  MatType        type = MATSEQBAIJ;

  m   /= 2; /* This is done just to be consistent with the old example */
  N    = 3*(2*m+1)*(2*m+1)*(2*m+1);
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"m = %D, N=%D\n",m,N));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,80,NULL,&mat));

  /* Form stiffness for element */
  PetscCall(PetscMalloc1(81,&K));
  for (i=0; i<81; i++) {
    PetscCall(PetscMalloc1(81,&K[i]));
  }
  PetscCall(Elastic20Stiff(K));

  /* Loop over elements and add contribution to stiffness */
  shiftx = 3; shifty = 3*(2*m+1); shiftz = 3*(2*m+1)*(2*m+1);
  for (k=0; k<m; k++) {
    for (j=0; j<m; j++) {
      for (i=0; i<m; i++) {
        h1   = 0;
        base = 2*k*shiftz + 2*j*shifty + 2*i*shiftx;
        for (k1=0; k1<3; k1++) {
          for (j_1=0; j_1<3; j_1++) {
            for (i1=0; i1<3; i1++) {
              h2 = 0;
              r1 = base + i1*shiftx + j_1*shifty + k1*shiftz;
              for (k2=0; k2<3; k2++) {
                for (j2=0; j2<3; j2++) {
                  for (i2=0; i2<3; i2++) {
                    r2   = base + i2*shiftx + j2*shifty + k2*shiftz;
                    PetscCall(AddElement(mat,r1,r2,K,h1,h2));
                    h2  += 3;
                  }
                }
              }
              h1 += 3;
            }
          }
        }
      }
    }
  }

  for (i=0; i<81; i++) {
    PetscCall(PetscFree(K[i]));
  }
  PetscCall(PetscFree(K));

  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* Exclude any superfluous rows and columns */
  nstart = 3*(2*m+1)*(2*m+1);
  ict    = 0;
  PetscCall(PetscMalloc1(N-nstart,&rowkeep));
  for (i=nstart; i<N; i++) {
    PetscCall(MatGetRow(mat,i,&nz,0,0));
    if (nz) rowkeep[ict++] = i;
    PetscCall(MatRestoreRow(mat,i,&nz,0,0));
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,ict,rowkeep,PETSC_COPY_VALUES,&iskeep));
  PetscCall(MatCreateSubMatrices(mat,1,&iskeep,&iskeep,MAT_INITIAL_MATRIX,&submatb));
  submat = *submatb;
  PetscCall(PetscFree(submatb));
  PetscCall(PetscFree(rowkeep));
  PetscCall(ISDestroy(&iskeep));
  PetscCall(MatDestroy(&mat));

  /* Convert storage formats -- just to demonstrate conversion to various
     formats (in particular, block diagonal storage).  This is NOT the
     recommended means to solve such a problem.  */
  PetscCall(MatConvert(submat,type,MAT_INITIAL_MATRIX,newmat));
  PetscCall(MatDestroy(&submat));

  PetscCall(MatNorm(*newmat,NORM_1,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"matrix 1 norm = %g\n",(double)norm));

  return 0;
}
/* -------------------------------------------------------------------- */
PetscErrorCode AddElement(Mat mat,PetscInt r1,PetscInt r2,PetscReal **K,PetscInt h1,PetscInt h2)
{
  PetscScalar    val;
  PetscInt       l1,l2,row,col;

  for (l1=0; l1<3; l1++) {
    for (l2=0; l2<3; l2++) {
/*
   NOTE you should never do this! Inserting values 1 at a time is
   just too expensive!
*/
      if (K[h1+l1][h2+l2] != 0.0) {
        row  = r1+l1; col = r2+l2; val = K[h1+l1][h2+l2];
        PetscCall(MatSetValues(mat,1,&row,1,&col,&val,ADD_VALUES));
        row  = r2+l2; col = r1+l1;
        PetscCall(MatSetValues(mat,1,&row,1,&col,&val,ADD_VALUES));
      }
    }
  }
  return 0;
}
/* -------------------------------------------------------------------- */
PetscReal N[20][64];                  /* Interpolation function. */
PetscReal part_N[3][20][64];          /* Partials of interpolation function. */
PetscReal rst[3][64];                 /* Location of integration pts in (r,s,t) */
PetscReal weight[64];                 /* Gaussian quadrature weights. */
PetscReal xyz[20][3];                 /* (x,y,z) coordinates of nodes  */
PetscReal E,nu;                       /* Physcial constants. */
PetscInt  n_int,N_int;                /* N_int = n_int^3, number of int. pts. */
/* Ordering of the vertices, (r,s,t) coordinates, of the canonical cell. */
PetscReal r2[20] = {-1.0,0.0,1.0,-1.0,1.0,-1.0,0.0,1.0,
                    -1.0,1.0,-1.0,1.0,
                    -1.0,0.0,1.0,-1.0,1.0,-1.0,0.0,1.0};
PetscReal s2[20] = {-1.0,-1.0, -1.0,0.0,0.0,1.0, 1.0, 1.0,
                    -1.0,-1.0,1.0,1.0,
                    -1.0,-1.0, -1.0,0.0,0.0,1.0, 1.0, 1.0};
PetscReal t2[20] = {-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
                    0.0,0.0,0.0,0.0,
                    1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
PetscInt  rmap[20] = {0,1,2,3,5,6,7,8,9,11,15,17,18,19,20,21,23,24,25,26};
/* -------------------------------------------------------------------- */
/*
  Elastic20Stiff - Forms 20 node elastic stiffness for element.
 */
PetscErrorCode Elastic20Stiff(PetscReal **Ke)
{
  PetscReal K[60][60],x,y,z,dx,dy,dz;
  PetscInt  i,j,k,l,Ii,J;

  paulsetup20();

  x          = -1.0;  y = -1.0; z = -1.0; dx = 2.0; dy = 2.0; dz = 2.0;
  xyz[0][0]  = x;          xyz[0][1] = y;          xyz[0][2] = z;
  xyz[1][0]  = x + dx;     xyz[1][1] = y;          xyz[1][2] = z;
  xyz[2][0]  = x + 2.*dx;  xyz[2][1] = y;          xyz[2][2] = z;
  xyz[3][0]  = x;          xyz[3][1] = y + dy;     xyz[3][2] = z;
  xyz[4][0]  = x + 2.*dx;  xyz[4][1] = y + dy;     xyz[4][2] = z;
  xyz[5][0]  = x;          xyz[5][1] = y + 2.*dy;  xyz[5][2] = z;
  xyz[6][0]  = x + dx;     xyz[6][1] = y + 2.*dy;  xyz[6][2] = z;
  xyz[7][0]  = x + 2.*dx;  xyz[7][1] = y + 2.*dy;  xyz[7][2] = z;
  xyz[8][0]  = x;          xyz[8][1] = y;          xyz[8][2] = z + dz;
  xyz[9][0]  = x + 2.*dx;  xyz[9][1] = y;          xyz[9][2] = z + dz;
  xyz[10][0] = x;         xyz[10][1] = y + 2.*dy; xyz[10][2] = z + dz;
  xyz[11][0] = x + 2.*dx; xyz[11][1] = y + 2.*dy; xyz[11][2] = z + dz;
  xyz[12][0] = x;         xyz[12][1] = y;         xyz[12][2] = z + 2.*dz;
  xyz[13][0] = x + dx;    xyz[13][1] = y;         xyz[13][2] = z + 2.*dz;
  xyz[14][0] = x + 2.*dx; xyz[14][1] = y;         xyz[14][2] = z + 2.*dz;
  xyz[15][0] = x;         xyz[15][1] = y + dy;    xyz[15][2] = z + 2.*dz;
  xyz[16][0] = x + 2.*dx; xyz[16][1] = y + dy;    xyz[16][2] = z + 2.*dz;
  xyz[17][0] = x;         xyz[17][1] = y + 2.*dy; xyz[17][2] = z + 2.*dz;
  xyz[18][0] = x + dx;    xyz[18][1] = y + 2.*dy; xyz[18][2] = z + 2.*dz;
  xyz[19][0] = x + 2.*dx; xyz[19][1] = y + 2.*dy; xyz[19][2] = z + 2.*dz;
  paulintegrate20(K);

  /* copy the stiffness from K into format used by Ke */
  for (i=0; i<81; i++) {
    for (j=0; j<81; j++) {
      Ke[i][j] = 0.0;
    }
  }
  Ii = 0;
  for (i=0; i<20; i++) {
    J = 0;
    for (j=0; j<20; j++) {
      for (k=0; k<3; k++) {
        for (l=0; l<3; l++) {
          Ke[3*rmap[i]+k][3*rmap[j]+l] = K[Ii+k][J+l];
        }
      }
      J += 3;
    }
    Ii += 3;
  }

  /* force the matrix to be exactly symmetric */
  for (i=0; i<81; i++) {
    for (j=0; j<i; j++) {
      Ke[i][j] = (Ke[i][j] + Ke[j][i])/2.0;
    }
  }
  return 0;
}
/* -------------------------------------------------------------------- */
/*
  paulsetup20 - Sets up data structure for forming local elastic stiffness.
 */
PetscErrorCode paulsetup20(void)
{
  PetscInt  i,j,k,cnt;
  PetscReal x[4],w[4];
  PetscReal c;

  n_int = 3;
  nu    = 0.3;
  E     = 1.0;

  /* Assign integration points and weights for
       Gaussian quadrature formulae. */
  if (n_int == 2) {
    x[0] = (-0.577350269189626);
    x[1] = (0.577350269189626);
    w[0] = 1.0000000;
    w[1] = 1.0000000;
  } else if (n_int == 3) {
    x[0] = (-0.774596669241483);
    x[1] = 0.0000000;
    x[2] = 0.774596669241483;
    w[0] = 0.555555555555555;
    w[1] = 0.888888888888888;
    w[2] = 0.555555555555555;
  } else if (n_int == 4) {
    x[0] = (-0.861136311594053);
    x[1] = (-0.339981043584856);
    x[2] = 0.339981043584856;
    x[3] = 0.861136311594053;
    w[0] = 0.347854845137454;
    w[1] = 0.652145154862546;
    w[2] = 0.652145154862546;
    w[3] = 0.347854845137454;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Value for n_int not supported");

  /* rst[][i] contains the location of the i-th integration point
      in the canonical (r,s,t) coordinate system.  weight[i] contains
      the Gaussian weighting factor. */

  cnt = 0;
  for (i=0; i<n_int; i++) {
    for (j=0; j<n_int; j++) {
      for (k=0; k<n_int; k++) {
        rst[0][cnt] =x[i];
        rst[1][cnt] =x[j];
        rst[2][cnt] =x[k];
        weight[cnt] = w[i]*w[j]*w[k];
        ++cnt;
      }
    }
  }
  N_int = cnt;

  /* N[][j] is the interpolation vector, N[][j] .* xyz[] */
  /* yields the (x,y,z)  locations of the integration point. */
  /*  part_N[][][j] is the partials of the N function */
  /*  w.r.t. (r,s,t). */

  c = 1.0/8.0;
  for (j=0; j<N_int; j++) {
    for (i=0; i<20; i++) {
      if (i==0 || i==2 || i==5 || i==7 || i==12 || i==14 || i== 17 || i==19) {
        N[i][j] = c*(1.0 + r2[i]*rst[0][j])*
                  (1.0 + s2[i]*rst[1][j])*(1.0 + t2[i]*rst[2][j])*
                  (-2.0 + r2[i]*rst[0][j] + s2[i]*rst[1][j] + t2[i]*rst[2][j]);
        part_N[0][i][j] = c*r2[i]*(1 + s2[i]*rst[1][j])*(1 + t2[i]*rst[2][j])*
                          (-1.0 + 2.0*r2[i]*rst[0][j] + s2[i]*rst[1][j] +
                           t2[i]*rst[2][j]);
        part_N[1][i][j] = c*s2[i]*(1 + r2[i]*rst[0][j])*(1 + t2[i]*rst[2][j])*
                          (-1.0 + r2[i]*rst[0][j] + 2.0*s2[i]*rst[1][j] +
                           t2[i]*rst[2][j]);
        part_N[2][i][j] = c*t2[i]*(1 + r2[i]*rst[0][j])*(1 + s2[i]*rst[1][j])*
                          (-1.0 + r2[i]*rst[0][j] + s2[i]*rst[1][j] +
                           2.0*t2[i]*rst[2][j]);
      } else if (i==1 || i==6 || i==13 || i==18) {
        N[i][j] = .25*(1.0 - rst[0][j]*rst[0][j])*
                  (1.0 + s2[i]*rst[1][j])*(1.0 + t2[i]*rst[2][j]);
        part_N[0][i][j] = -.5*rst[0][j]*(1 + s2[i]*rst[1][j])*
                          (1 + t2[i]*rst[2][j]);
        part_N[1][i][j] = .25*s2[i]*(1 + t2[i]*rst[2][j])*
                          (1.0 - rst[0][j]*rst[0][j]);
        part_N[2][i][j] = .25*t2[i]*(1.0 - rst[0][j]*rst[0][j])*
                          (1 + s2[i]*rst[1][j]);
      } else if (i==3 || i==4 || i==15 || i==16) {
        N[i][j] = .25*(1.0 - rst[1][j]*rst[1][j])*
                  (1.0 + r2[i]*rst[0][j])*(1.0 + t2[i]*rst[2][j]);
        part_N[0][i][j] = .25*r2[i]*(1 + t2[i]*rst[2][j])*
                          (1.0 - rst[1][j]*rst[1][j]);
        part_N[1][i][j] = -.5*rst[1][j]*(1 + r2[i]*rst[0][j])*
                          (1 + t2[i]*rst[2][j]);
        part_N[2][i][j] = .25*t2[i]*(1.0 - rst[1][j]*rst[1][j])*
                          (1 + r2[i]*rst[0][j]);
      } else if (i==8 || i==9 || i==10 || i==11) {
        N[i][j] = .25*(1.0 - rst[2][j]*rst[2][j])*
                  (1.0 + r2[i]*rst[0][j])*(1.0 + s2[i]*rst[1][j]);
        part_N[0][i][j] = .25*r2[i]*(1 + s2[i]*rst[1][j])*
                          (1.0 - rst[2][j]*rst[2][j]);
        part_N[1][i][j] = .25*s2[i]*(1.0 - rst[2][j]*rst[2][j])*
                          (1 + r2[i]*rst[0][j]);
        part_N[2][i][j] = -.5*rst[2][j]*(1 + r2[i]*rst[0][j])*
                          (1 + s2[i]*rst[1][j]);
      }
    }
  }
  return 0;
}
/* -------------------------------------------------------------------- */
/*
   paulintegrate20 - Does actual numerical integration on 20 node element.
 */
PetscErrorCode paulintegrate20(PetscReal K[60][60])
{
  PetscReal det_jac,jac[3][3],inv_jac[3][3];
  PetscReal B[6][60],B_temp[6][60],C[6][6];
  PetscReal temp;
  PetscInt  i,j,k,step;

  /* Zero out K, since we will accumulate the result here */
  for (i=0; i<60; i++) {
    for (j=0; j<60; j++) {
      K[i][j] = 0.0;
    }
  }

  /* Loop over integration points ... */
  for (step=0; step<N_int; step++) {

    /* Compute the Jacobian, its determinant, and inverse. */
    for (i=0; i<3; i++) {
      for (j=0; j<3; j++) {
        jac[i][j] = 0;
        for (k=0; k<20; k++) {
          jac[i][j] += part_N[i][k][step]*xyz[k][j];
        }
      }
    }
    det_jac = jac[0][0]*(jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])
              + jac[0][1]*(jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])
              + jac[0][2]*(jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0]);
    inv_jac[0][0] = (jac[1][1]*jac[2][2]-jac[1][2]*jac[2][1])/det_jac;
    inv_jac[0][1] = (jac[0][2]*jac[2][1]-jac[0][1]*jac[2][2])/det_jac;
    inv_jac[0][2] = (jac[0][1]*jac[1][2]-jac[1][1]*jac[0][2])/det_jac;
    inv_jac[1][0] = (jac[1][2]*jac[2][0]-jac[1][0]*jac[2][2])/det_jac;
    inv_jac[1][1] = (jac[0][0]*jac[2][2]-jac[2][0]*jac[0][2])/det_jac;
    inv_jac[1][2] = (jac[0][2]*jac[1][0]-jac[0][0]*jac[1][2])/det_jac;
    inv_jac[2][0] = (jac[1][0]*jac[2][1]-jac[1][1]*jac[2][0])/det_jac;
    inv_jac[2][1] = (jac[0][1]*jac[2][0]-jac[0][0]*jac[2][1])/det_jac;
    inv_jac[2][2] = (jac[0][0]*jac[1][1]-jac[1][0]*jac[0][1])/det_jac;

    /* Compute the B matrix. */
    for (i=0; i<3; i++) {
      for (j=0; j<20; j++) {
        B_temp[i][j] = 0.0;
        for (k=0; k<3; k++) {
          B_temp[i][j] += inv_jac[i][k]*part_N[k][j][step];
        }
      }
    }
    for (i=0; i<6; i++) {
      for (j=0; j<60; j++) {
        B[i][j] = 0.0;
      }
    }

    /* Put values in correct places in B. */
    for (k=0; k<20; k++) {
      B[0][3*k]   = B_temp[0][k];
      B[1][3*k+1] = B_temp[1][k];
      B[2][3*k+2] = B_temp[2][k];
      B[3][3*k]   = B_temp[1][k];
      B[3][3*k+1] = B_temp[0][k];
      B[4][3*k+1] = B_temp[2][k];
      B[4][3*k+2] = B_temp[1][k];
      B[5][3*k]   = B_temp[2][k];
      B[5][3*k+2] = B_temp[0][k];
    }

    /* Construct the C matrix, uses the constants "nu" and "E". */
    for (i=0; i<6; i++) {
      for (j=0; j<6; j++) {
        C[i][j] = 0.0;
      }
    }
    temp = (1.0 + nu)*(1.0 - 2.0*nu);
    temp = E/temp;
    C[0][0] = temp*(1.0 - nu);
    C[1][1] = C[0][0];
    C[2][2] = C[0][0];
    C[3][3] = temp*(0.5 - nu);
    C[4][4] = C[3][3];
    C[5][5] = C[3][3];
    C[0][1] = temp*nu;
    C[0][2] = C[0][1];
    C[1][0] = C[0][1];
    C[1][2] = C[0][1];
    C[2][0] = C[0][1];
    C[2][1] = C[0][1];

    for (i=0; i<6; i++) {
      for (j=0; j<60; j++) {
        B_temp[i][j] = 0.0;
        for (k=0; k<6; k++) {
          B_temp[i][j] += C[i][k]*B[k][j];
        }
        B_temp[i][j] *= det_jac;
      }
    }

    /* Accumulate B'*C*B*det(J)*weight, as a function of (r,s,t), in K. */
    for (i=0; i<60; i++) {
      for (j=0; j<60; j++) {
        temp = 0.0;
        for (k=0; k<6; k++) {
          temp += B[k][i]*B_temp[k][j];
        }
        K[i][j] += temp*weight[step];
      }
    }
  }  /* end of loop over integration points */
  return 0;
}

/*TEST

    test:
      args: -matconvert_type seqaij -ksp_monitor_short -ksp_rtol 1.e-2  -pc_type jacobi
      requires: x

TEST*/
