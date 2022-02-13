/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
This example was derived from src/ksp/ksp/tutorials ex29.c

Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-x^2/\nu} e^{-y^2/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions.
*/

static char help[] = "Solves 2D inhomogeneous Laplacian. Demonstates using PCTelescopeSetCoarseDM functionality of PCTelescope via a DMShell\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscksp.h>

PetscErrorCode ComputeMatrix_ShellDA(KSP,Mat,Mat,void*);
PetscErrorCode ComputeMatrix_DMDA(DM,Mat,Mat,void*);
PetscErrorCode ComputeRHS_DMDA(DM,Vec,void*);
PetscErrorCode DMShellCreate_ShellDA(DM,DM*);
PetscErrorCode DMFieldScatter_ShellDA(DM,Vec,ScatterMode,DM,Vec);
PetscErrorCode DMStateScatter_ShellDA(DM,ScatterMode,DM);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscReal rho;
  PetscReal nu;
  BCType    bcType;
  MPI_Comm  comm;
} UserContext;

PetscErrorCode UserContextCreate(MPI_Comm comm,UserContext **ctx)
{
  UserContext    *user;
  const char     *bcTypes[2] = {"dirichlet","neumann"};
  PetscInt       bc;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscCalloc1(1,&user);CHKERRQ(ierr);
  user->comm = comm;
  ierr = PetscOptionsBegin(comm, "", "Options for the inhomogeneous Poisson equation", "DMqq");CHKERRQ(ierr);
  user->rho = 1.0;
  ierr = PetscOptionsReal("-rho", "The conductivity", "ex29.c", user->rho, &user->rho, NULL);CHKERRQ(ierr);
  user->nu = 0.1;
  ierr = PetscOptionsReal("-nu", "The width of the Gaussian source", "ex29.c", user->nu, &user->nu, NULL);CHKERRQ(ierr);
  bc = (PetscInt)DIRICHLET;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex29.c",bcTypes,2,bcTypes[0],&bc,NULL);CHKERRQ(ierr);
  user->bcType = (BCType)bc;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  *ctx = user;
  PetscFunctionReturn(0);
}

PetscErrorCode CommCoarsen(MPI_Comm comm,PetscInt number,PetscSubcomm *p)
{
  PetscSubcomm   psubcomm;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscSubcommCreate(comm,&psubcomm);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(psubcomm,number);CHKERRQ(ierr);
  ierr = PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
  *p = psubcomm;
  PetscFunctionReturn(0);
}

PetscErrorCode CommHierarchyCreate(MPI_Comm comm,PetscInt n,PetscInt number[],PetscSubcomm pscommlist[])
{
  PetscInt       k;
  PetscBool      view_hierarchy = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (k=0; k<n; k++) {
    pscommlist[k] = NULL;
  }

  if (n < 1) PetscFunctionReturn(0);

  ierr = CommCoarsen(comm,number[n-1],&pscommlist[n-1]);CHKERRQ(ierr);
  for (k=n-2; k>=0; k--) {
    MPI_Comm comm_k = PetscSubcommChild(pscommlist[k+1]);
    if (pscommlist[k+1]->color == 0) {
      ierr = CommCoarsen(comm_k,number[k],&pscommlist[k]);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsGetBool(NULL,NULL,"-view_hierarchy",&view_hierarchy,NULL);CHKERRQ(ierr);
  if (view_hierarchy) {
    PetscMPIInt size;

    ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
    ierr = PetscPrintf(comm,"level[%D] size %d\n",n,(int)size);CHKERRQ(ierr);
    for (k=n-1; k>=0; k--) {
      if (pscommlist[k]) {
        MPI_Comm comm_k = PetscSubcommChild(pscommlist[k]);

        if (pscommlist[k]->color == 0) {
          ierr = MPI_Comm_size(comm_k,&size);CHKERRMPI(ierr);
          ierr = PetscPrintf(comm_k,"level[%D] size %d\n",k,(int)size);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* taken from src/ksp/pc/impls/telescope/telescope_dmda.c */
static PetscErrorCode _DMDADetermineRankFromGlobalIJ_2d(PetscInt i,PetscInt j,PetscInt Mp,PetscInt Np,
                                                        PetscInt start_i[],PetscInt start_j[],
                                                        PetscInt span_i[],PetscInt span_j[],
                                                        PetscMPIInt *_pi,PetscMPIInt *_pj,PetscMPIInt *rank_re)
{
  PetscInt pi,pj,n;

  PetscFunctionBeginUser;
  *rank_re = -1;
  pi = pj = -1;
  if (_pi) {
    for (n=0; n<Mp; n++) {
      if ((i >= start_i[n]) && (i < start_i[n]+span_i[n])) {
        pi = n;
        break;
      }
    }
    PetscCheckFalse(pi == -1,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmda-ij] pi cannot be determined : range %D, val %D",Mp,i);
    *_pi = (PetscMPIInt)pi;
  }

  if (_pj) {
    for (n=0; n<Np; n++) {
      if ((j >= start_j[n]) && (j < start_j[n]+span_j[n])) {
        pj = n;
        break;
      }
    }
    PetscCheckFalse(pj == -1,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmda-ij] pj cannot be determined : range %D, val %D",Np,j);
    *_pj = (PetscMPIInt)pj;
  }

  *rank_re = (PetscMPIInt)(pi + pj * Mp);
  PetscFunctionReturn(0);
}

/* taken from src/ksp/pc/impls/telescope/telescope_dmda.c */
static PetscErrorCode _DMDADetermineGlobalS0_2d(PetscMPIInt rank_re,PetscInt Mp_re,PetscInt Np_re,
                                                PetscInt range_i_re[],PetscInt range_j_re[],PetscInt *s0)
{
  PetscInt    i,j,start_IJ = 0;
  PetscMPIInt rank_ij;

  PetscFunctionBeginUser;
  *s0 = -1;
  for (j=0; j<Np_re; j++) {
    for (i=0; i<Mp_re; i++) {
      rank_ij = (PetscMPIInt)(i + j*Mp_re);
      if (rank_ij < rank_re) {
        start_IJ += range_i_re[i]*range_j_re[j];
      }
    }
  }
  *s0 = start_IJ;
  PetscFunctionReturn(0);
}

/* adapted from src/ksp/pc/impls/telescope/telescope_dmda.c */
static PetscErrorCode DMDACreatePermutation_2d(DM dmrepart,DM dmf,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       k,sum,Mp_re = 0,Np_re = 0;
  PetscInt       nx,ny,sr,er,Mr,ndof;
  PetscInt       i,j,location,startI[2],endI[2],lenI[2];
  const PetscInt *_range_i_re = NULL,*_range_j_re = NULL;
  PetscInt       *range_i_re,*range_j_re;
  PetscInt       *start_i_re,*start_j_re;
  MPI_Comm       comm;
  Vec            V;
  Mat            Pscalar;

  PetscFunctionBeginUser;
  ierr = PetscInfo(dmf,"setting up the permutation matrix (DMDA-2D)\n");CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dmf,&comm);CHKERRQ(ierr);

  _range_i_re = _range_j_re = NULL;
  /* Create DMDA on the child communicator */
  if (dmrepart) {
    ierr = DMDAGetInfo(dmrepart,NULL,NULL,NULL,NULL,&Mp_re,&Np_re,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DMDAGetOwnershipRanges(dmrepart,&_range_i_re,&_range_j_re,NULL);CHKERRQ(ierr);
  }

  /* note - assume rank 0 always participates */
  ierr = MPI_Bcast(&Mp_re,1,MPIU_INT,0,comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(&Np_re,1,MPIU_INT,0,comm);CHKERRMPI(ierr);

  ierr = PetscCalloc1(Mp_re,&range_i_re);CHKERRQ(ierr);
  ierr = PetscCalloc1(Np_re,&range_j_re);CHKERRQ(ierr);

  if (_range_i_re) {ierr = PetscArraycpy(range_i_re,_range_i_re,Mp_re);CHKERRQ(ierr);}
  if (_range_j_re) {ierr = PetscArraycpy(range_j_re,_range_j_re,Np_re);CHKERRQ(ierr);}

  ierr = MPI_Bcast(range_i_re,Mp_re,MPIU_INT,0,comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(range_j_re,Np_re,MPIU_INT,0,comm);CHKERRMPI(ierr);

  ierr = PetscMalloc1(Mp_re,&start_i_re);CHKERRQ(ierr);
  ierr = PetscMalloc1(Np_re,&start_j_re);CHKERRQ(ierr);

  sum = 0;
  for (k=0; k<Mp_re; k++) {
    start_i_re[k] = sum;
    sum += range_i_re[k];
  }

  sum = 0;
  for (k=0; k<Np_re; k++) {
    start_j_re[k] = sum;
    sum += range_j_re[k];
  }

  /* Create permutation */
  ierr = DMDAGetInfo(dmf,NULL,&nx,&ny,NULL,NULL,NULL,NULL,&ndof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmf,&V);CHKERRQ(ierr);
  ierr = VecGetSize(V,&Mr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(V,&sr,&er);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmf,&V);CHKERRQ(ierr);
  sr = sr / ndof;
  er = er / ndof;
  Mr = Mr / ndof;

  ierr = MatCreate(comm,&Pscalar);CHKERRQ(ierr);
  ierr = MatSetSizes(Pscalar,(er-sr),(er-sr),Mr,Mr);CHKERRQ(ierr);
  ierr = MatSetType(Pscalar,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Pscalar,1,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Pscalar,1,NULL,1,NULL);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dmf,NULL,NULL,NULL,&lenI[0],&lenI[1],NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dmf,&startI[0],&startI[1],NULL,&endI[0],&endI[1],NULL);CHKERRQ(ierr);
  endI[0] += startI[0];
  endI[1] += startI[1];

  for (j=startI[1]; j<endI[1]; j++) {
    for (i=startI[0]; i<endI[0]; i++) {
      PetscMPIInt rank_ijk_re,rank_reI[] = {0,0};
      PetscInt    s0_re;
      PetscInt    ii,jj,local_ijk_re,mapped_ijk;
      PetscInt    lenI_re[] = {0,0};

      location = (i - startI[0]) + (j - startI[1])*lenI[0];
      ierr = _DMDADetermineRankFromGlobalIJ_2d(i,j,Mp_re,Np_re,
                                             start_i_re,start_j_re,
                                             range_i_re,range_j_re,
                                             &rank_reI[0],&rank_reI[1],&rank_ijk_re);CHKERRQ(ierr);

      ierr = _DMDADetermineGlobalS0_2d(rank_ijk_re,Mp_re,Np_re,range_i_re,range_j_re,&s0_re);CHKERRQ(ierr);

      ii = i - start_i_re[ rank_reI[0] ];
      PetscCheckFalse(ii < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmdarepart-perm2d] index error ii");
      jj = j - start_j_re[ rank_reI[1] ];
      PetscCheckFalse(jj < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmdarepart-perm2d] index error jj");

      lenI_re[0] = range_i_re[ rank_reI[0] ];
      lenI_re[1] = range_j_re[ rank_reI[1] ];
      local_ijk_re = ii + jj * lenI_re[0];
      mapped_ijk = s0_re + local_ijk_re;
      ierr = MatSetValue(Pscalar,sr+location,mapped_ijk,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Pscalar,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pscalar,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  *mat = Pscalar;

  ierr = PetscFree(range_i_re);CHKERRQ(ierr);
  ierr = PetscFree(range_j_re);CHKERRQ(ierr);
  ierr = PetscFree(start_i_re);CHKERRQ(ierr);
  ierr = PetscFree(start_j_re);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* adapted from src/ksp/pc/impls/telescope/telescope_dmda.c */
static PetscErrorCode PCTelescopeSetUp_dmda_scatters(DM dmf,DM dmc)
{
  PetscErrorCode ierr;
  Vec            xred,yred,xtmp,x,xp;
  VecScatter     scatter;
  IS             isin;
  PetscInt       m,bs,st,ed;
  MPI_Comm       comm;
  VecType        vectype;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)dmf,&comm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmf,&x);CHKERRQ(ierr);
  ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);
  ierr = VecGetType(x,&vectype);CHKERRQ(ierr);

  /* cannot use VecDuplicate as xp is already composed with dmf */
  /*ierr = VecDuplicate(x,&xp);CHKERRQ(ierr);*/
  {
    PetscInt m,M;

    ierr = VecGetSize(x,&M);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x,&m);CHKERRQ(ierr);
    ierr = VecCreate(comm,&xp);CHKERRQ(ierr);
    ierr = VecSetSizes(xp,m,M);CHKERRQ(ierr);
    ierr = VecSetBlockSize(xp,bs);CHKERRQ(ierr);
    ierr = VecSetType(xp,vectype);CHKERRQ(ierr);
  }

  m = 0;
  xred = NULL;
  yred = NULL;
  if (dmc) {
    ierr = DMCreateGlobalVector(dmc,&xred);CHKERRQ(ierr);
    ierr = VecDuplicate(xred,&yred);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(xred,&st,&ed);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,ed-st,st,1,&isin);CHKERRQ(ierr);
    ierr = VecGetLocalSize(xred,&m);CHKERRQ(ierr);
  } else {
    ierr = VecGetOwnershipRange(x,&st,&ed);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,0,st,1,&isin);CHKERRQ(ierr);
  }
  ierr = ISSetBlockSize(isin,bs);CHKERRQ(ierr);
  ierr = VecCreate(comm,&xtmp);CHKERRQ(ierr);
  ierr = VecSetSizes(xtmp,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(xtmp,bs);CHKERRQ(ierr);
  ierr = VecSetType(xtmp,vectype);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,isin,xtmp,NULL,&scatter);CHKERRQ(ierr);

  ierr = PetscObjectCompose((PetscObject)dmf,"isin",(PetscObject)isin);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dmf,"scatter",(PetscObject)scatter);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dmf,"xtmp",(PetscObject)xtmp);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dmf,"xp",(PetscObject)xp);CHKERRQ(ierr);

  ierr = VecDestroy(&xred);CHKERRQ(ierr);
  ierr = VecDestroy(&yred);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_ShellDA(DM dm,Mat *A)
{
  DM             da;
  MPI_Comm       comm;
  PetscMPIInt    size;
  UserContext    *ctx = NULL;
  PetscInt       M,N;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMShellGetContext(dm,&da);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = DMCreateMatrix(da,A);CHKERRQ(ierr);
  ierr = MatGetSize(*A,&M,&N);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"[size %D] DMCreateMatrix_ShellDA (%D x %D)\n",(PetscInt)size,M,N);CHKERRQ(ierr);

  ierr = DMGetApplicationContext(dm,&ctx);CHKERRQ(ierr);
  if (ctx->bcType == NEUMANN) {
    MatNullSpace nullspace = NULL;
    ierr = PetscPrintf(comm,"[size %D] DMCreateMatrix_ShellDA: using neumann bcs\n",(PetscInt)size);CHKERRQ(ierr);

    ierr = MatGetNullSpace(*A,&nullspace);CHKERRQ(ierr);
    if (!nullspace) {
      ierr = PetscPrintf(comm,"[size %D] DMCreateMatrix_ShellDA: operator does not have nullspace - attaching\n",(PetscInt)size);CHKERRQ(ierr);
      ierr = MatNullSpaceCreate(comm,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
      ierr = MatSetNullSpace(*A,nullspace);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(comm,"[size %D] DMCreateMatrix_ShellDA: operator already has a nullspace\n",(PetscInt)size);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateGlobalVector_ShellDA(DM dm,Vec *x)
{
  DM             da;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = DMShellGetContext(dm,&da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,x);CHKERRQ(ierr);
  ierr = VecSetDM(*x,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateLocalVector_ShellDA(DM dm,Vec *x)
{
  DM             da;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = DMShellGetContext(dm,&da);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,x);CHKERRQ(ierr);
  ierr = VecSetDM(*x,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCoarsen_ShellDA(DM dm,MPI_Comm comm,DM *dmc)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  *dmc = NULL;
  ierr = DMGetCoarseDM(dm,dmc);CHKERRQ(ierr);
  if (!*dmc) {
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"The coarse DM should never be NULL. The DM hierarchy should have already been defined");
  } else {
    ierr = PetscObjectReference((PetscObject)(*dmc));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInterpolation_ShellDA(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  DM             da1,da2;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = DMShellGetContext(dm1,&da1);CHKERRQ(ierr);
  ierr = DMShellGetContext(dm2,&da2);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(da1,da2,mat,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMShellDASetUp_TelescopeDMScatter(DM dmf_shell,DM dmc_shell)
{
  PetscErrorCode ierr;
  Mat            P = NULL;
  DM             dmf = NULL,dmc = NULL;

  PetscFunctionBeginUser;
  ierr = DMShellGetContext(dmf_shell,&dmf);CHKERRQ(ierr);
  if (dmc_shell) {
    ierr = DMShellGetContext(dmc_shell,&dmc);CHKERRQ(ierr);
  }
  ierr = DMDACreatePermutation_2d(dmc,dmf,&P);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dmf,"P",(PetscObject)P);CHKERRQ(ierr);
  ierr = PCTelescopeSetUp_dmda_scatters(dmf,dmc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dmf_shell,"PCTelescopeFieldScatter",DMFieldScatter_ShellDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dmf_shell,"PCTelescopeStateScatter",DMStateScatter_ShellDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMShellDAFieldScatter_Forward(DM dmf,Vec x,DM dmc,Vec xc)
{
  PetscErrorCode    ierr;
  Mat               P = NULL;
  Vec               xp = NULL,xtmp = NULL;
  VecScatter        scatter = NULL;
  const PetscScalar *x_array;
  PetscInt          i,st,ed;

  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject)dmf,"P",(PetscObject*)&P);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dmf,"xp",(PetscObject*)&xp);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dmf,"scatter",(PetscObject*)&scatter);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dmf,"xtmp",(PetscObject*)&xtmp);CHKERRQ(ierr);
  PetscCheckFalse(!P,PETSC_COMM_SELF,PETSC_ERR_USER,"Require a permutation matrix (\"P\")to be composed with the parent (fine) DM");
  PetscCheckFalse(!xp,PETSC_COMM_SELF,PETSC_ERR_USER,"Require \"xp\" to be composed with the parent (fine) DM");
  PetscCheckFalse(!scatter,PETSC_COMM_SELF,PETSC_ERR_USER,"Require \"scatter\" to be composed with the parent (fine) DM");
  PetscCheckFalse(!xtmp,PETSC_COMM_SELF,PETSC_ERR_USER,"Require \"xtmp\" to be composed with the parent (fine) DM");

  ierr = MatMultTranspose(P,x,xp);CHKERRQ(ierr);

  /* pull in vector x->xtmp */
  ierr = VecScatterBegin(scatter,xp,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,xp,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* copy vector entries into xred */
  ierr = VecGetArrayRead(xtmp,&x_array);CHKERRQ(ierr);
  if (xc) {
    PetscScalar *LA_xred;
    ierr = VecGetOwnershipRange(xc,&st,&ed);CHKERRQ(ierr);

    ierr = VecGetArray(xc,&LA_xred);CHKERRQ(ierr);
    for (i=0; i<ed-st; i++) {
      LA_xred[i] = x_array[i];
    }
    ierr = VecRestoreArray(xc,&LA_xred);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xtmp,&x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMShellDAFieldScatter_Reverse(DM dmf,Vec y,DM dmc,Vec yc)
{
  PetscErrorCode ierr;
  Mat            P = NULL;
  Vec            xp = NULL,xtmp = NULL;
  VecScatter     scatter = NULL;
  PetscScalar    *array;
  PetscInt       i,st,ed;

  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject)dmf,"P",(PetscObject*)&P);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dmf,"xp",(PetscObject*)&xp);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dmf,"scatter",(PetscObject*)&scatter);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dmf,"xtmp",(PetscObject*)&xtmp);CHKERRQ(ierr);

  PetscCheckFalse(!P,PETSC_COMM_SELF,PETSC_ERR_USER,"Require a permutation matrix (\"P\")to be composed with the parent (fine) DM");
  PetscCheckFalse(!xp,PETSC_COMM_SELF,PETSC_ERR_USER,"Require \"xp\" to be composed with the parent (fine) DM");
  PetscCheckFalse(!scatter,PETSC_COMM_SELF,PETSC_ERR_USER,"Require \"scatter\" to be composed with the parent (fine) DM");
  PetscCheckFalse(!xtmp,PETSC_COMM_SELF,PETSC_ERR_USER,"Require \"xtmp\" to be composed with the parent (fine) DM");

  /* return vector */
  ierr = VecGetArray(xtmp,&array);CHKERRQ(ierr);
  if (yc) {
    const PetscScalar *LA_yred;
    ierr = VecGetOwnershipRange(yc,&st,&ed);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yc,&LA_yred);CHKERRQ(ierr);
    for (i=0; i<ed-st; i++) {
      array[i] = LA_yred[i];
    }
    ierr = VecRestoreArrayRead(yc,&LA_yred);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xtmp,&array);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,xtmp,xp,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,xtmp,xp,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMult(P,xp,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldScatter_ShellDA(DM dmf_shell,Vec x,ScatterMode mode,DM dmc_shell,Vec xc)
{
  PetscErrorCode ierr;
  DM             dmf = NULL,dmc = NULL;

  PetscFunctionBeginUser;
  ierr = DMShellGetContext(dmf_shell,&dmf);CHKERRQ(ierr);
  if (dmc_shell) {
    ierr = DMShellGetContext(dmc_shell,&dmc);CHKERRQ(ierr);
  }
  if (mode == SCATTER_FORWARD) {
    ierr = DMShellDAFieldScatter_Forward(dmf,x,dmc,xc);CHKERRQ(ierr);
  } else if (mode == SCATTER_REVERSE) {
    ierr = DMShellDAFieldScatter_Reverse(dmf,x,dmc,xc);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)dmf_shell),PETSC_ERR_SUP,"Only mode = SCATTER_FORWARD, SCATTER_REVERSE supported");
  PetscFunctionReturn(0);
}

PetscErrorCode DMStateScatter_ShellDA(DM dmf_shell,ScatterMode mode,DM dmc_shell)
{
  PetscMPIInt    size_f = 0,size_c = 0;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dmf_shell),&size_f);CHKERRMPI(ierr);
  if (dmc_shell) {
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dmc_shell),&size_c);CHKERRMPI(ierr);
  }
  if (mode == SCATTER_FORWARD) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dmf_shell),"User supplied state scatter (fine [size %d]-> coarse [size %d])\n",(int)size_f,(int)size_c);CHKERRQ(ierr);
  } else if (mode == SCATTER_REVERSE) {
  } else SETERRQ(PetscObjectComm((PetscObject)dmf_shell),PETSC_ERR_SUP,"Only mode = SCATTER_FORWARD, SCATTER_REVERSE supported");
  PetscFunctionReturn(0);
}

PetscErrorCode DMShellCreate_ShellDA(DM da,DM *dms)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if (da) {
    ierr = DMShellCreate(PetscObjectComm((PetscObject)da),dms);CHKERRQ(ierr);
    ierr = DMShellSetContext(*dms,da);CHKERRQ(ierr);
    ierr = DMShellSetCreateGlobalVector(*dms,DMCreateGlobalVector_ShellDA);CHKERRQ(ierr);
    ierr = DMShellSetCreateLocalVector(*dms,DMCreateLocalVector_ShellDA);CHKERRQ(ierr);
    ierr = DMShellSetCreateMatrix(*dms,DMCreateMatrix_ShellDA);CHKERRQ(ierr);
    ierr = DMShellSetCoarsen(*dms,DMCoarsen_ShellDA);CHKERRQ(ierr);
    ierr = DMShellSetCreateInterpolation(*dms,DMCreateInterpolation_ShellDA);CHKERRQ(ierr);
  } else {
    *dms = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroyShellDMDA(DM *_dm)
{
  PetscErrorCode ierr;
  DM             dm,da = NULL;

  PetscFunctionBeginUser;
  if (!_dm) PetscFunctionReturn(0);
  dm = *_dm;
  if (!dm) PetscFunctionReturn(0);

  ierr = DMShellGetContext(dm,&da);CHKERRQ(ierr);
  if (da) {
    Vec        vec;
    VecScatter scatter = NULL;
    IS         is = NULL;
    Mat        P = NULL;

    ierr = PetscObjectQuery((PetscObject)da,"P",(PetscObject*)&P);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);

    vec = NULL;
    ierr = PetscObjectQuery((PetscObject)da,"xp",(PetscObject*)&vec);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);

    ierr = PetscObjectQuery((PetscObject)da,"scatter",(PetscObject*)&scatter);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);

    vec = NULL;
    ierr = PetscObjectQuery((PetscObject)da,"xtmp",(PetscObject*)&vec);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);

    ierr = PetscObjectQuery((PetscObject)da,"isin",(PetscObject*)&is);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);

    ierr = DMDestroy(&da);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  *_dm = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode HierarchyCreate_Basic(DM *dm_f,DM *dm_c,UserContext *ctx)
{
  PetscErrorCode ierr;
  DM             dm,dmc,dm_shell,dmc_shell;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,17,17,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dm,0,1,0,1,0,0);CHKERRQ(ierr);
  ierr = DMDASetFieldName(dm,0,"Pressure");CHKERRQ(ierr);
  ierr = DMShellCreate_ShellDA(dm,&dm_shell);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm_shell,ctx);CHKERRQ(ierr);

  dmc = NULL;
  dmc_shell = NULL;
  if (rank == 0) {
    ierr = DMDACreate2d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,17,17,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&dmc);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmc);CHKERRQ(ierr);
    ierr = DMSetUp(dmc);CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(dmc,0,1,0,1,0,0);CHKERRQ(ierr);
    ierr = DMDASetFieldName(dmc,0,"Pressure");CHKERRQ(ierr);
    ierr = DMShellCreate_ShellDA(dmc,&dmc_shell);CHKERRQ(ierr);
    ierr = DMSetApplicationContext(dmc_shell,ctx);CHKERRQ(ierr);
  }

  ierr = DMSetCoarseDM(dm_shell,dmc_shell);CHKERRQ(ierr);
  ierr = DMShellDASetUp_TelescopeDMScatter(dm_shell,dmc_shell);CHKERRQ(ierr);

  *dm_f = dm_shell;
  *dm_c = dmc_shell;
  PetscFunctionReturn(0);
}

PetscErrorCode HierarchyCreate(PetscInt *_nd,PetscInt *_nref,MPI_Comm **_cl,DM **_dl)
{
  PetscInt       d,k,ndecomps,ncoarsen,found,nx;
  PetscInt       levelrefs,*number;
  PetscSubcomm   *pscommlist;
  MPI_Comm       *commlist;
  DM             *dalist,*dmlist;
  PetscBool      set;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ndecomps = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-ndecomps",&ndecomps,NULL);CHKERRQ(ierr);
  ncoarsen = ndecomps - 1;
  PetscCheckFalse(ncoarsen < 0,PETSC_COMM_WORLD,PETSC_ERR_USER,"-ndecomps must be >= 1");

  levelrefs = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-level_nrefs",&levelrefs,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncoarsen+1,&number);CHKERRQ(ierr);
  for (k=0; k<ncoarsen+1; k++) {
    number[k] = 2;
  }
  found = ncoarsen;
  set = PETSC_FALSE;
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-level_comm_red_factor",number,&found,&set);CHKERRQ(ierr);
  if (set) {
    PetscCheckFalse(found != ncoarsen,PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected %D values for -level_comm_red_factor. Found %D",ncoarsen,found);
  }

  ierr = PetscMalloc1(ncoarsen+1,&pscommlist);CHKERRQ(ierr);
  for (k=0; k<ncoarsen+1; k++) {
    pscommlist[k] = NULL;
  }

  ierr = PetscMalloc1(ndecomps,&commlist);CHKERRQ(ierr);
  for (k=0; k<ndecomps; k++) {
    commlist[k] = MPI_COMM_NULL;
  }
  ierr = PetscMalloc1(ndecomps*levelrefs,&dalist);CHKERRQ(ierr);
  ierr = PetscMalloc1(ndecomps*levelrefs,&dmlist);CHKERRQ(ierr);
  for (k=0; k<ndecomps*levelrefs; k++) {
    dalist[k] = NULL;
    dmlist[k] = NULL;
  }

  ierr = CommHierarchyCreate(PETSC_COMM_WORLD,ncoarsen,number,pscommlist);CHKERRQ(ierr);
  for (k=0; k<ncoarsen; k++) {
    if (pscommlist[k]) {
      MPI_Comm comm_k = PetscSubcommChild(pscommlist[k]);
      if (pscommlist[k]->color == 0) {
        ierr = PetscCommDuplicate(comm_k,&commlist[k],NULL);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&commlist[ndecomps-1],NULL);CHKERRQ(ierr);

  for (k=0; k<ncoarsen; k++) {
    if (pscommlist[k]) {
      ierr = PetscSubcommDestroy(&pscommlist[k]);CHKERRQ(ierr);
    }
  }

  nx = 17;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&nx,NULL);CHKERRQ(ierr);
  for (d=0; d<ndecomps; d++) {
    DM   dmroot = NULL;
    char name[PETSC_MAX_PATH_LEN];

    if (commlist[d] != MPI_COMM_NULL) {
      ierr = DMDACreate2d(commlist[d],DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,nx,nx,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&dmroot);CHKERRQ(ierr);
      ierr = DMSetUp(dmroot);CHKERRQ(ierr);
      ierr = DMDASetUniformCoordinates(dmroot,0,1,0,1,0,0);CHKERRQ(ierr);
      ierr = DMDASetFieldName(dmroot,0,"Pressure");CHKERRQ(ierr);
      ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"root-decomp-%D",d);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)dmroot,name);CHKERRQ(ierr);
      /*ierr = DMView(dmroot,PETSC_VIEWER_STDOUT_(commlist[d]));CHKERRQ(ierr);*/
    }

    dalist[d*levelrefs + 0] = dmroot;
    for (k=1; k<levelrefs; k++) {
      DM dmref = NULL;

      if (commlist[d] != MPI_COMM_NULL) {
        ierr = DMRefine(dalist[d*levelrefs + (k-1)],MPI_COMM_NULL,&dmref);CHKERRQ(ierr);
        ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"ref%D-decomp-%D",k,d);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)dmref,name);CHKERRQ(ierr);
        ierr = DMDAGetInfo(dmref,NULL,&nx,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
        /*ierr = DMView(dmref,PETSC_VIEWER_STDOUT_(commlist[d]));CHKERRQ(ierr);*/
      }
      dalist[d*levelrefs + k] = dmref;
    }
    ierr = MPI_Allreduce(MPI_IN_PLACE,&nx,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  }

  /* create the hierarchy of DMShell's */
  for (d=0; d<ndecomps; d++) {
    char name[PETSC_MAX_PATH_LEN];

    UserContext *ctx = NULL;
    if (commlist[d] != MPI_COMM_NULL) {
      ierr = UserContextCreate(commlist[d],&ctx);CHKERRQ(ierr);
      for (k=0; k<levelrefs; k++) {
        ierr = DMShellCreate_ShellDA(dalist[d*levelrefs + k],&dmlist[d*levelrefs + k]);CHKERRQ(ierr);
        ierr = DMSetApplicationContext(dmlist[d*levelrefs + k],ctx);CHKERRQ(ierr);
        ierr = PetscSNPrintf(name,PETSC_MAX_PATH_LEN-1,"level%D-decomp-%D",k,d);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)dmlist[d*levelrefs + k],name);CHKERRQ(ierr);
      }
    }
  }

  /* set all the coarse DMs */
  for (k=1; k<ndecomps*levelrefs; k++) { /* skip first DM as it doesn't have a coarse representation */
    DM dmfine = NULL,dmcoarse = NULL;

    dmfine = dmlist[k];
    dmcoarse = dmlist[k-1];
    if (dmfine) {
      ierr = DMSetCoarseDM(dmfine,dmcoarse);CHKERRQ(ierr);
    }
  }

  /* do special setup on the fine DM coupling different decompositions */
  for (d=1; d<ndecomps; d++) { /* skip first decomposition as it doesn't have a coarse representation */
    DM dmfine = NULL,dmcoarse = NULL;

    dmfine = dmlist[d*levelrefs + 0];
    dmcoarse = dmlist[(d-1)*levelrefs + (levelrefs-1)];
    if (dmfine) {
      ierr = DMShellDASetUp_TelescopeDMScatter(dmfine,dmcoarse);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree(number);CHKERRQ(ierr);
  for (k=0; k<ncoarsen; k++) {
    ierr = PetscSubcommDestroy(&pscommlist[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pscommlist);CHKERRQ(ierr);

  if (_nd) {
    *_nd = ndecomps;
  }
  if (_nref) {
    *_nref = levelrefs;
  }
  if (_cl) {
    *_cl = commlist;
  } else {
    for (k=0; k<ndecomps; k++) {
      if (commlist[k] != MPI_COMM_NULL) {
        ierr = PetscCommDestroy(&commlist[k]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(commlist);CHKERRQ(ierr);
  }
  if (_dl) {
    *_dl = dmlist;
    ierr = PetscFree(dalist);CHKERRQ(ierr);
  } else {
    for (k=0; k<ndecomps*levelrefs; k++) {
      ierr = DMDestroy(&dmlist[k]);CHKERRQ(ierr);
      ierr = DMDestroy(&dalist[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dmlist);CHKERRQ(ierr);
    ierr = PetscFree(dalist);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode test_hierarchy(void)
{
  PetscErrorCode ierr;
  PetscInt       d,k,nd,nref;
  MPI_Comm       *comms;
  DM             *dms;

  PetscFunctionBeginUser;
  ierr = HierarchyCreate(&nd,&nref,&comms,&dms);CHKERRQ(ierr);

  /* destroy user context */
  for (d=0; d<nd; d++) {
    DM first = dms[d*nref+0];

    if (first) {
      UserContext *ctx = NULL;

      ierr = DMGetApplicationContext(first,&ctx);CHKERRQ(ierr);
      if (ctx) { ierr = PetscFree(ctx);CHKERRQ(ierr); }
      ierr = DMSetApplicationContext(first,NULL);CHKERRQ(ierr);
    }
    for (k=1; k<nref; k++) {
      DM dm = dms[d*nref+k];
      if (dm) {
        ierr = DMSetApplicationContext(dm,NULL);CHKERRQ(ierr);
      }
    }
  }

  /* destroy DMs */
  for (k=0; k<nd*nref; k++) {
    if (dms[k]) {
      ierr = DMDestroyShellDMDA(&dms[k]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(dms);CHKERRQ(ierr);

  /* destroy communicators */
  for (k=0; k<nd; k++) {
    if (comms[k] != MPI_COMM_NULL) {
      ierr = PetscCommDestroy(&comms[k]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(comms);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode test_basic(void)
{
  PetscErrorCode ierr;
  DM             dmF,dmdaF = NULL,dmC = NULL;
  Mat            A;
  Vec            x,b;
  KSP            ksp;
  PC             pc;
  UserContext    *user = NULL;

  PetscFunctionBeginUser;
  ierr = UserContextCreate(PETSC_COMM_WORLD,&user);CHKERRQ(ierr);
  ierr = HierarchyCreate_Basic(&dmF,&dmC,user);CHKERRQ(ierr);
  ierr = DMShellGetContext(dmF,&dmdaF);CHKERRQ(ierr);

  ierr = DMCreateMatrix(dmF,&A);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmF,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmF,&b);CHKERRQ(ierr);
  ierr = ComputeRHS_DMDA(dmdaF,b,user);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix_ShellDA,user);CHKERRQ(ierr);
  /*ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);*/
  ierr = KSPSetDM(ksp,dmF);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCTelescopeSetUseCoarseDM(pc,PETSC_TRUE);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  if (dmC) {
    ierr = DMDestroyShellDMDA(&dmC);CHKERRQ(ierr);
  }
  ierr = DMDestroyShellDMDA(&dmF);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode test_mg(void)
{
  PetscErrorCode ierr;
  DM             dmF,dmdaF = NULL,*dms = NULL;
  Mat            A;
  Vec            x,b;
  KSP            ksp;
  PetscInt       k,d,nd,nref;
  MPI_Comm       *comms = NULL;
  UserContext    *user = NULL;

  PetscFunctionBeginUser;
  ierr = HierarchyCreate(&nd,&nref,&comms,&dms);CHKERRQ(ierr);
  dmF = dms[nd*nref-1];

  ierr = DMShellGetContext(dmF,&dmdaF);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dmF,&user);CHKERRQ(ierr);

  ierr = DMCreateMatrix(dmF,&A);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmF,&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmF,&b);CHKERRQ(ierr);
  ierr = ComputeRHS_DMDA(dmdaF,b,user);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix_ShellDA,user);CHKERRQ(ierr);
  /*ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);*/
  ierr = KSPSetDM(ksp,dmF);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  for (d=0; d<nd; d++) {
    DM first = dms[d*nref+0];

    if (first) {
      UserContext *ctx = NULL;

      ierr = DMGetApplicationContext(first,&ctx);CHKERRQ(ierr);
      if (ctx) { ierr = PetscFree(ctx);CHKERRQ(ierr); }
      ierr = DMSetApplicationContext(first,NULL);CHKERRQ(ierr);
    }
    for (k=1; k<nref; k++) {
      DM dm = dms[d*nref+k];
      if (dm) {
        ierr = DMSetApplicationContext(dm,NULL);CHKERRQ(ierr);
      }
    }
  }

  for (k=0; k<nd*nref; k++) {
    if (dms[k]) {
      ierr = DMDestroyShellDMDA(&dms[k]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(dms);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  for (k=0; k<nd; k++) {
    if (comms[k] != MPI_COMM_NULL) {
      ierr = PetscCommDestroy(&comms[k]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(comms);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       test_id = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-tid",&test_id,NULL);CHKERRQ(ierr);
  switch (test_id) {
  case 0:
    ierr = test_basic();CHKERRQ(ierr);
      break;
  case 1:
    ierr = test_hierarchy();CHKERRQ(ierr);
      break;
  case 2:
    ierr = test_mg();CHKERRQ(ierr);
      break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"-tid must be 0,1,2");
  }
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS_DMDA(DM da,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;
  PetscBool      isda = PETSC_FALSE;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject)da,DMDA,&isda);CHKERRQ(ierr);
  PetscCheckFalse(!isda,PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"DM provided must be a DMDA");
  ierr = DMDAGetInfo(da,NULL,&mx,&my,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,b,&array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)*PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRho(PetscInt i, PetscInt j, PetscInt mx, PetscInt my, PetscReal centerRho, PetscReal *rho)
{
  PetscFunctionBeginUser;
  if ((i > mx/3.0) && (i < 2.0*mx/3.0) && (j > my/3.0) && (j < 2.0*my/3.0)) {
    *rho = centerRho;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix_DMDA(DM da,Mat J,Mat jac,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscReal      centerRho;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hx,Hy,HydHx,HxdHy,rho;
  MatStencil     row, col[5];
  PetscBool      isda = PETSC_FALSE;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject)da,DMDA,&isda);CHKERRQ(ierr);
  PetscCheckFalse(!isda,PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"DM provided must be a DMDA");
  ierr = MatZeroEntries(jac);CHKERRQ(ierr);
  centerRho = user->rho;
  ierr      = DMDAGetInfo(da,NULL,&mx,&my,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx-1);
  Hy        = 1.0 / (PetscReal)(my-1);
  HxdHy     = Hx/Hy;
  HydHx     = Hy/Hx;
  ierr      = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      ierr  = ComputeRho(i, j, mx, my, centerRho, &rho);CHKERRQ(ierr);
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0*rho*(HxdHy + HydHx);
          ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else if (user->bcType == NEUMANN) {
          PetscInt numx = 0, numy = 0, num = 0;
          if (j!=0) {
            v[num] = -rho*HxdHy;  col[num].i = i;    col[num].j = j-1;
            numy++; num++;
          }
          if (i!=0) {
            v[num] = -rho*HydHx;  col[num].i = i-1;  col[num].j = j;
            numx++; num++;
          }
          if (i!=mx-1) {
            v[num] = -rho*HydHx;  col[num].i = i+1;  col[num].j = j;
            numx++; num++;
          }
          if (j!=my-1) {
            v[num] = -rho*HxdHy;  col[num].i = i;    col[num].j = j+1;
            numy++; num++;
          }
          v[num] = numx*rho*HydHx + numy*rho*HxdHy;  col[num].i = i; col[num].j = j;
          num++;
          ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix_ShellDA(KSP ksp,Mat J,Mat jac,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm,da;
  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMShellGetContext(dm,&da);CHKERRQ(ierr);
  ierr = ComputeMatrix_DMDA(da,J,jac,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

  test:
    suffix: basic_dirichlet
    nsize: 4
    args: -tid 0 -ksp_monitor_short -pc_type telescope -telescope_ksp_max_it 100000 -telescope_pc_type lu -telescope_ksp_type fgmres -telescope_ksp_monitor_short -ksp_type gcr

  test:
    suffix: basic_neumann
    nsize: 4
    requires: !single
    args: -tid 0 -ksp_monitor_short -pc_type telescope -telescope_ksp_max_it 100000 -telescope_pc_type jacobi -telescope_ksp_type fgmres -telescope_ksp_monitor_short -ksp_type gcr -bc_type neumann

  test:
    suffix: mg_2lv_2mg
    nsize: 6
    args: -tid 2 -ksp_monitor_short -ksp_type gcr -pc_type mg -pc_mg_levels 3 -level_nrefs 3 -ndecomps 2 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_use_coarse_dm -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_levels 3 -level_comm_red_factor 6 -mg_coarse_telescope_mg_coarse_pc_type lu

  test:
    suffix: mg_3lv_2mg
    nsize: 4
    args: -tid 2 -ksp_monitor_short -ksp_type gcr -pc_type mg -pc_mg_levels 3 -level_nrefs 3 -ndecomps 3 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_use_coarse_dm -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_coarse_pc_type telescope -mg_coarse_telescope_mg_coarse_pc_telescope_use_coarse_dm -mg_coarse_telescope_mg_coarse_telescope_pc_type lu -m 5

  test:
    suffix: mg_3lv_2mg_customcommsize
    nsize: 12
    args: -tid 2 -ksp_monitor_short -ksp_type gcr -pc_type mg -pc_mg_levels 3 -level_nrefs 3 -ndecomps 3 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_use_coarse_dm  -m 5 -level_comm_red_factor 2,6 -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_coarse_pc_type telescope -mg_coarse_telescope_mg_coarse_pc_telescope_use_coarse_dm -mg_coarse_telescope_mg_coarse_telescope_pc_type lu

 TEST*/
