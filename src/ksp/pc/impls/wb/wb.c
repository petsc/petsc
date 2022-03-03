
#include <petscdmda.h>   /*I "petscdmda.h" I*/
#include <petsc/private/pcmgimpl.h>   /*I "petscksp.h" I*/
#include <petscctable.h>

typedef struct {
  PCExoticType type;
  Mat          P;            /* the constructed interpolation matrix */
  PetscBool    directSolve;  /* use direct LU factorization to construct interpolation */
  KSP          ksp;
} PC_Exotic;

const char *const PCExoticTypes[] = {"face","wirebasket","PCExoticType","PC_Exotic",NULL};

/*
      DMDAGetWireBasketInterpolation - Gets the interpolation for a wirebasket based coarse space

*/
PetscErrorCode DMDAGetWireBasketInterpolation(PC pc,DM da,PC_Exotic *exotic,Mat Aglobal,MatReuse reuse,Mat *P)
{
  PetscInt               dim,i,j,k,m,n,p,dof,Nint,Nface,Nwire,Nsurf,*Iint,*Isurf,cint = 0,csurf = 0,istart,jstart,kstart,*II,N,c = 0;
  PetscInt               mwidth,nwidth,pwidth,cnt,mp,np,pp,Ntotal,gl[26],*globals,Ng,*IIint,*IIsurf,Nt;
  Mat                    Xint, Xsurf,Xint_tmp;
  IS                     isint,issurf,is,row,col;
  ISLocalToGlobalMapping ltg;
  MPI_Comm               comm;
  Mat                    A,Aii,Ais,Asi,*Aholder,iAii;
  MatFactorInfo          info;
  PetscScalar            *xsurf,*xint;
  const PetscScalar      *rxint;
#if defined(PETSC_USE_DEBUG_foo)
  PetscScalar            tmp;
#endif
  PetscTable             ht;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,&dim,NULL,NULL,NULL,&mp,&np,&pp,&dof,NULL,NULL,NULL,NULL,NULL));
  PetscCheckFalse(dof != 1,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Only for single field problems");
  PetscCheckFalse(dim != 3,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Only coded for 3d problems");
  CHKERRQ(DMDAGetCorners(da,NULL,NULL,NULL,&m,&n,&p));
  CHKERRQ(DMDAGetGhostCorners(da,&istart,&jstart,&kstart,&mwidth,&nwidth,&pwidth));
  istart = istart ? -1 : 0;
  jstart = jstart ? -1 : 0;
  kstart = kstart ? -1 : 0;

  /*
    the columns of P are the interpolation of each coarse grid point (one for each vertex and edge)
    to all the local degrees of freedom (this includes the vertices, edges and faces).

    Xint are the subset of the interpolation into the interior

    Xface are the interpolation onto faces but not into the interior

    Xsurf are the interpolation onto the vertices and edges (the surfbasket)
                                      Xint
    Symbolically one could write P = (Xface) after interchanging the rows to match the natural ordering on the domain
                                      Xsurf
  */
  N     = (m - istart)*(n - jstart)*(p - kstart);
  Nint  = (m-2-istart)*(n-2-jstart)*(p-2-kstart);
  Nface = 2*((m-2-istart)*(n-2-jstart) + (m-2-istart)*(p-2-kstart) + (n-2-jstart)*(p-2-kstart));
  Nwire = 4*((m-2-istart) + (n-2-jstart) + (p-2-kstart)) + 8;
  Nsurf = Nface + Nwire;
  CHKERRQ(MatCreateSeqDense(MPI_COMM_SELF,Nint,26,NULL,&Xint));
  CHKERRQ(MatCreateSeqDense(MPI_COMM_SELF,Nsurf,26,NULL,&Xsurf));
  CHKERRQ(MatDenseGetArray(Xsurf,&xsurf));

  /*
     Require that all 12 edges and 6 faces have at least one grid point. Otherwise some of the columns of
     Xsurf will be all zero (thus making the coarse matrix singular).
  */
  PetscCheckFalse(m-istart < 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in X direction must be at least 3");
  PetscCheckFalse(n-jstart < 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in Y direction must be at least 3");
  PetscCheckFalse(p-kstart < 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in Z direction must be at least 3");

  cnt = 0;

  xsurf[cnt++] = 1;
  for (i=1; i<m-istart-1; i++) xsurf[cnt++ + Nsurf] = 1;
  xsurf[cnt++ + 2*Nsurf] = 1;

  for (j=1; j<n-1-jstart; j++) {
    xsurf[cnt++ + 3*Nsurf] = 1;
    for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 4*Nsurf] = 1;
    xsurf[cnt++ + 5*Nsurf] = 1;
  }

  xsurf[cnt++ + 6*Nsurf] = 1;
  for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 7*Nsurf] = 1;
  xsurf[cnt++ + 8*Nsurf] = 1;

  for (k=1; k<p-1-kstart; k++) {
    xsurf[cnt++ + 9*Nsurf] = 1;
    for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 10*Nsurf] = 1;
    xsurf[cnt++ + 11*Nsurf] = 1;

    for (j=1; j<n-1-jstart; j++) {
      xsurf[cnt++ + 12*Nsurf] = 1;
      /* these are the interior nodes */
      xsurf[cnt++ + 13*Nsurf] = 1;
    }

    xsurf[cnt++ + 14*Nsurf] = 1;
    for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 15*Nsurf] = 1;
    xsurf[cnt++ + 16*Nsurf] = 1;
  }

  xsurf[cnt++ + 17*Nsurf] = 1;
  for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 18*Nsurf] = 1;
  xsurf[cnt++ + 19*Nsurf] = 1;

  for (j=1;j<n-1-jstart;j++) {
    xsurf[cnt++ + 20*Nsurf] = 1;
    for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 21*Nsurf] = 1;
    xsurf[cnt++ + 22*Nsurf] = 1;
  }

  xsurf[cnt++ + 23*Nsurf] = 1;
  for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 24*Nsurf] = 1;
  xsurf[cnt++ + 25*Nsurf] = 1;

  /* interpolations only sum to 1 when using direct solver */
#if defined(PETSC_USE_DEBUG_foo)
  for (i=0; i<Nsurf; i++) {
    tmp = 0.0;
    for (j=0; j<26; j++) tmp += xsurf[i+j*Nsurf];
    PetscCheckFalse(PetscAbsScalar(tmp-1.0) > 1.e-10,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong Xsurf interpolation at i %D value %g",i,(double)PetscAbsScalar(tmp));
  }
#endif
  CHKERRQ(MatDenseRestoreArray(Xsurf,&xsurf));
  /* CHKERRQ(MatView(Xsurf,PETSC_VIEWER_STDOUT_WORLD));*/

  /*
       I are the indices for all the needed vertices (in global numbering)
       Iint are the indices for the interior values, I surf for the surface values
            (This is just for the part of the global matrix obtained with MatCreateSubMatrix(), it
             is NOT the local DMDA ordering.)
       IIint and IIsurf are the same as the Iint, Isurf except they are in the global numbering
  */
#define Endpoint(a,start,b) (a == 0 || a == (b-1-start))
  CHKERRQ(PetscMalloc3(N,&II,Nint,&Iint,Nsurf,&Isurf));
  CHKERRQ(PetscMalloc2(Nint,&IIint,Nsurf,&IIsurf));
  for (k=0; k<p-kstart; k++) {
    for (j=0; j<n-jstart; j++) {
      for (i=0; i<m-istart; i++) {
        II[c++] = i + j*mwidth + k*mwidth*nwidth;

        if (!Endpoint(i,istart,m) && !Endpoint(j,jstart,n) && !Endpoint(k,kstart,p)) {
          IIint[cint]  = i + j*mwidth + k*mwidth*nwidth;
          Iint[cint++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        } else {
          IIsurf[csurf]  = i + j*mwidth + k*mwidth*nwidth;
          Isurf[csurf++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        }
      }
    }
  }
  PetscCheckFalse(c != N,PETSC_COMM_SELF,PETSC_ERR_PLIB,"c != N");
  PetscCheckFalse(cint != Nint,PETSC_COMM_SELF,PETSC_ERR_PLIB,"cint != Nint");
  PetscCheckFalse(csurf != Nsurf,PETSC_COMM_SELF,PETSC_ERR_PLIB,"csurf != Nsurf");
  CHKERRQ(DMGetLocalToGlobalMapping(da,&ltg));
  CHKERRQ(ISLocalToGlobalMappingApply(ltg,N,II,II));
  CHKERRQ(ISLocalToGlobalMappingApply(ltg,Nint,IIint,IIint));
  CHKERRQ(ISLocalToGlobalMappingApply(ltg,Nsurf,IIsurf,IIsurf));
  CHKERRQ(PetscObjectGetComm((PetscObject)da,&comm));
  CHKERRQ(ISCreateGeneral(comm,N,II,PETSC_COPY_VALUES,&is));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,Nint,Iint,PETSC_COPY_VALUES,&isint));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,Nsurf,Isurf,PETSC_COPY_VALUES,&issurf));
  CHKERRQ(PetscFree3(II,Iint,Isurf));

  CHKERRQ(MatCreateSubMatrices(Aglobal,1,&is,&is,MAT_INITIAL_MATRIX,&Aholder));
  A    = *Aholder;
  CHKERRQ(PetscFree(Aholder));

  CHKERRQ(MatCreateSubMatrix(A,isint,isint,MAT_INITIAL_MATRIX,&Aii));
  CHKERRQ(MatCreateSubMatrix(A,isint,issurf,MAT_INITIAL_MATRIX,&Ais));
  CHKERRQ(MatCreateSubMatrix(A,issurf,isint,MAT_INITIAL_MATRIX,&Asi));

  /*
     Solve for the interpolation onto the interior Xint
  */
  CHKERRQ(MatMatMult(Ais,Xsurf,MAT_INITIAL_MATRIX,PETSC_DETERMINE,&Xint_tmp));
  CHKERRQ(MatScale(Xint_tmp,-1.0));
  if (exotic->directSolve) {
    CHKERRQ(MatGetFactor(Aii,MATSOLVERPETSC,MAT_FACTOR_LU,&iAii));
    CHKERRQ(MatFactorInfoInitialize(&info));
    CHKERRQ(MatGetOrdering(Aii,MATORDERINGND,&row,&col));
    CHKERRQ(MatLUFactorSymbolic(iAii,Aii,row,col,&info));
    CHKERRQ(ISDestroy(&row));
    CHKERRQ(ISDestroy(&col));
    CHKERRQ(MatLUFactorNumeric(iAii,Aii,&info));
    CHKERRQ(MatMatSolve(iAii,Xint_tmp,Xint));
    CHKERRQ(MatDestroy(&iAii));
  } else {
    Vec         b,x;
    PetscScalar *xint_tmp;

    CHKERRQ(MatDenseGetArray(Xint,&xint));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,Nint,NULL,&x));
    CHKERRQ(MatDenseGetArray(Xint_tmp,&xint_tmp));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,Nint,NULL,&b));
    CHKERRQ(KSPSetOperators(exotic->ksp,Aii,Aii));
    for (i=0; i<26; i++) {
      CHKERRQ(VecPlaceArray(x,xint+i*Nint));
      CHKERRQ(VecPlaceArray(b,xint_tmp+i*Nint));
      CHKERRQ(KSPSolve(exotic->ksp,b,x));
      CHKERRQ(KSPCheckSolve(exotic->ksp,pc,x));
      CHKERRQ(VecResetArray(x));
      CHKERRQ(VecResetArray(b));
    }
    CHKERRQ(MatDenseRestoreArray(Xint,&xint));
    CHKERRQ(MatDenseRestoreArray(Xint_tmp,&xint_tmp));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&b));
  }
  CHKERRQ(MatDestroy(&Xint_tmp));

#if defined(PETSC_USE_DEBUG_foo)
  CHKERRQ(MatDenseGetArrayRead(Xint,&rxint));
  for (i=0; i<Nint; i++) {
    tmp = 0.0;
    for (j=0; j<26; j++) tmp += rxint[i+j*Nint];

    PetscCheckFalse(PetscAbsScalar(tmp-1.0) > 1.e-10,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong Xint interpolation at i %D value %g",i,(double)PetscAbsScalar(tmp));
  }
  CHKERRQ(MatDenseRestoreArrayRead(Xint,&rxint));
  /* CHKERRQ(MatView(Xint,PETSC_VIEWER_STDOUT_WORLD)); */
#endif

  /*         total vertices             total faces                                  total edges */
  Ntotal = (mp + 1)*(np + 1)*(pp + 1) + mp*np*(pp+1) + mp*pp*(np+1) + np*pp*(mp+1) + mp*(np+1)*(pp+1) + np*(mp+1)*(pp+1) +  pp*(mp+1)*(np+1);

  /*
      For each vertex, edge, face on process (in the same orderings as used above) determine its local number including ghost points
  */
  cnt = 0;

  gl[cnt++] = 0;  { gl[cnt++] = 1;} gl[cnt++] = m-istart-1;
  { gl[cnt++] = mwidth;  { gl[cnt++] = mwidth+1;} gl[cnt++] = mwidth + m-istart-1;}
  gl[cnt++] = mwidth*(n-jstart-1);  { gl[cnt++] = mwidth*(n-jstart-1)+1;} gl[cnt++] = mwidth*(n-jstart-1) + m-istart-1;
  {
    gl[cnt++] = mwidth*nwidth;  { gl[cnt++] = mwidth*nwidth+1;}  gl[cnt++] = mwidth*nwidth+ m-istart-1;
    { gl[cnt++] = mwidth*nwidth + mwidth; /* these are the interior nodes */ gl[cnt++] = mwidth*nwidth + mwidth+m-istart-1;}
    gl[cnt++] = mwidth*nwidth+ mwidth*(n-jstart-1);   { gl[cnt++] = mwidth*nwidth+mwidth*(n-jstart-1)+1;} gl[cnt++] = mwidth*nwidth+mwidth*(n-jstart-1) + m-istart-1;
  }
  gl[cnt++] = mwidth*nwidth*(p-kstart-1); { gl[cnt++] = mwidth*nwidth*(p-kstart-1)+1;} gl[cnt++] = mwidth*nwidth*(p-kstart-1) +  m-istart-1;
  { gl[cnt++] = mwidth*nwidth*(p-kstart-1) + mwidth;   { gl[cnt++] = mwidth*nwidth*(p-kstart-1) + mwidth+1;} gl[cnt++] = mwidth*nwidth*(p-kstart-1)+mwidth+m-istart-1;}
  gl[cnt++] = mwidth*nwidth*(p-kstart-1) +  mwidth*(n-jstart-1);  { gl[cnt++] = mwidth*nwidth*(p-kstart-1)+ mwidth*(n-jstart-1)+1;} gl[cnt++] = mwidth*nwidth*(p-kstart-1) + mwidth*(n-jstart-1) + m-istart-1;

  /* PetscIntView(26,gl,PETSC_VIEWER_STDOUT_WORLD); */
  /* convert that to global numbering and get them on all processes */
  CHKERRQ(ISLocalToGlobalMappingApply(ltg,26,gl,gl));
  /* PetscIntView(26,gl,PETSC_VIEWER_STDOUT_WORLD); */
  CHKERRQ(PetscMalloc1(26*mp*np*pp,&globals));
  CHKERRMPI(MPI_Allgather(gl,26,MPIU_INT,globals,26,MPIU_INT,PetscObjectComm((PetscObject)da)));

  /* Number the coarse grid points from 0 to Ntotal */
  CHKERRQ(MatGetSize(Aglobal,&Nt,NULL));
  CHKERRQ(PetscTableCreate(Ntotal/3,Nt+1,&ht));
  for (i=0; i<26*mp*np*pp; i++) {
    CHKERRQ(PetscTableAddCount(ht,globals[i]+1));
  }
  CHKERRQ(PetscTableGetCount(ht,&cnt));
  PetscCheckFalse(cnt != Ntotal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hash table size %D not equal to total number coarse grid points %D",cnt,Ntotal);
  CHKERRQ(PetscFree(globals));
  for (i=0; i<26; i++) {
    CHKERRQ(PetscTableFind(ht,gl[i]+1,&gl[i]));
    gl[i]--;
  }
  CHKERRQ(PetscTableDestroy(&ht));
  /* PetscIntView(26,gl,PETSC_VIEWER_STDOUT_WORLD); */

  /* construct global interpolation matrix */
  CHKERRQ(MatGetLocalSize(Aglobal,&Ng,NULL));
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatCreateAIJ(PetscObjectComm((PetscObject)da),Ng,PETSC_DECIDE,PETSC_DECIDE,Ntotal,Nint+Nsurf,NULL,Nint+Nsurf,NULL,P));
  } else {
    CHKERRQ(MatZeroEntries(*P));
  }
  CHKERRQ(MatSetOption(*P,MAT_ROW_ORIENTED,PETSC_FALSE));
  CHKERRQ(MatDenseGetArrayRead(Xint,&rxint));
  CHKERRQ(MatSetValues(*P,Nint,IIint,26,gl,rxint,INSERT_VALUES));
  CHKERRQ(MatDenseRestoreArrayRead(Xint,&rxint));
  CHKERRQ(MatDenseGetArrayRead(Xsurf,&rxint));
  CHKERRQ(MatSetValues(*P,Nsurf,IIsurf,26,gl,rxint,INSERT_VALUES));
  CHKERRQ(MatDenseRestoreArrayRead(Xsurf,&rxint));
  CHKERRQ(MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscFree2(IIint,IIsurf));

#if defined(PETSC_USE_DEBUG_foo)
  {
    Vec         x,y;
    PetscScalar *yy;
    CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)da),Ng,PETSC_DETERMINE,&y));
    CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)da),PETSC_DETERMINE,Ntotal,&x));
    CHKERRQ(VecSet(x,1.0));
    CHKERRQ(MatMult(*P,x,y));
    CHKERRQ(VecGetArray(y,&yy));
    for (i=0; i<Ng; i++) {
      PetscCheckFalse(PetscAbsScalar(yy[i]-1.0) > 1.e-10,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong p interpolation at i %D value %g",i,(double)PetscAbsScalar(yy[i]));
    }
    CHKERRQ(VecRestoreArray(y,&yy));
    CHKERRQ(VecDestroy(x));
    CHKERRQ(VecDestroy(y));
  }
#endif

  CHKERRQ(MatDestroy(&Aii));
  CHKERRQ(MatDestroy(&Ais));
  CHKERRQ(MatDestroy(&Asi));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISDestroy(&isint));
  CHKERRQ(ISDestroy(&issurf));
  CHKERRQ(MatDestroy(&Xint));
  CHKERRQ(MatDestroy(&Xsurf));
  PetscFunctionReturn(0);
}

/*
      DMDAGetFaceInterpolation - Gets the interpolation for a face based coarse space

*/
PetscErrorCode DMDAGetFaceInterpolation(PC pc,DM da,PC_Exotic *exotic,Mat Aglobal,MatReuse reuse,Mat *P)
{
  PetscInt               dim,i,j,k,m,n,p,dof,Nint,Nface,Nwire,Nsurf,*Iint,*Isurf,cint = 0,csurf = 0,istart,jstart,kstart,*II,N,c = 0;
  PetscInt               mwidth,nwidth,pwidth,cnt,mp,np,pp,Ntotal,gl[6],*globals,Ng,*IIint,*IIsurf,Nt;
  Mat                    Xint, Xsurf,Xint_tmp;
  IS                     isint,issurf,is,row,col;
  ISLocalToGlobalMapping ltg;
  MPI_Comm               comm;
  Mat                    A,Aii,Ais,Asi,*Aholder,iAii;
  MatFactorInfo          info;
  PetscScalar            *xsurf,*xint;
  const PetscScalar      *rxint;
#if defined(PETSC_USE_DEBUG_foo)
  PetscScalar            tmp;
#endif
  PetscTable             ht;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(da,&dim,NULL,NULL,NULL,&mp,&np,&pp,&dof,NULL,NULL,NULL,NULL,NULL));
  PetscCheckFalse(dof != 1,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Only for single field problems");
  PetscCheckFalse(dim != 3,PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Only coded for 3d problems");
  CHKERRQ(DMDAGetCorners(da,NULL,NULL,NULL,&m,&n,&p));
  CHKERRQ(DMDAGetGhostCorners(da,&istart,&jstart,&kstart,&mwidth,&nwidth,&pwidth));
  istart = istart ? -1 : 0;
  jstart = jstart ? -1 : 0;
  kstart = kstart ? -1 : 0;

  /*
    the columns of P are the interpolation of each coarse grid point (one for each vertex and edge)
    to all the local degrees of freedom (this includes the vertices, edges and faces).

    Xint are the subset of the interpolation into the interior

    Xface are the interpolation onto faces but not into the interior

    Xsurf are the interpolation onto the vertices and edges (the surfbasket)
                                      Xint
    Symbolically one could write P = (Xface) after interchanging the rows to match the natural ordering on the domain
                                      Xsurf
  */
  N     = (m - istart)*(n - jstart)*(p - kstart);
  Nint  = (m-2-istart)*(n-2-jstart)*(p-2-kstart);
  Nface = 2*((m-2-istart)*(n-2-jstart) + (m-2-istart)*(p-2-kstart) + (n-2-jstart)*(p-2-kstart));
  Nwire = 4*((m-2-istart) + (n-2-jstart) + (p-2-kstart)) + 8;
  Nsurf = Nface + Nwire;
  CHKERRQ(MatCreateSeqDense(MPI_COMM_SELF,Nint,6,NULL,&Xint));
  CHKERRQ(MatCreateSeqDense(MPI_COMM_SELF,Nsurf,6,NULL,&Xsurf));
  CHKERRQ(MatDenseGetArray(Xsurf,&xsurf));

  /*
     Require that all 12 edges and 6 faces have at least one grid point. Otherwise some of the columns of
     Xsurf will be all zero (thus making the coarse matrix singular).
  */
  PetscCheckFalse(m-istart < 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in X direction must be at least 3");
  PetscCheckFalse(n-jstart < 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in Y direction must be at least 3");
  PetscCheckFalse(p-kstart < 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in Z direction must be at least 3");

  cnt = 0;
  for (j=1; j<n-1-jstart; j++) {
    for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 0*Nsurf] = 1;
  }

  for (k=1; k<p-1-kstart; k++) {
    for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 1*Nsurf] = 1;
    for (j=1; j<n-1-jstart; j++) {
      xsurf[cnt++ + 2*Nsurf] = 1;
      /* these are the interior nodes */
      xsurf[cnt++ + 3*Nsurf] = 1;
    }
    for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 4*Nsurf] = 1;
  }
  for (j=1;j<n-1-jstart;j++) {
    for (i=1; i<m-istart-1; i++) xsurf[cnt++ + 5*Nsurf] = 1;
  }

#if defined(PETSC_USE_DEBUG_foo)
  for (i=0; i<Nsurf; i++) {
    tmp = 0.0;
    for (j=0; j<6; j++) tmp += xsurf[i+j*Nsurf];

    PetscCheckFalse(PetscAbsScalar(tmp-1.0) > 1.e-10,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong Xsurf interpolation at i %D value %g",i,(double)PetscAbsScalar(tmp));
  }
#endif
  CHKERRQ(MatDenseRestoreArray(Xsurf,&xsurf));
  /* CHKERRQ(MatView(Xsurf,PETSC_VIEWER_STDOUT_WORLD));*/

  /*
       I are the indices for all the needed vertices (in global numbering)
       Iint are the indices for the interior values, I surf for the surface values
            (This is just for the part of the global matrix obtained with MatCreateSubMatrix(), it
             is NOT the local DMDA ordering.)
       IIint and IIsurf are the same as the Iint, Isurf except they are in the global numbering
  */
#define Endpoint(a,start,b) (a == 0 || a == (b-1-start))
  CHKERRQ(PetscMalloc3(N,&II,Nint,&Iint,Nsurf,&Isurf));
  CHKERRQ(PetscMalloc2(Nint,&IIint,Nsurf,&IIsurf));
  for (k=0; k<p-kstart; k++) {
    for (j=0; j<n-jstart; j++) {
      for (i=0; i<m-istart; i++) {
        II[c++] = i + j*mwidth + k*mwidth*nwidth;

        if (!Endpoint(i,istart,m) && !Endpoint(j,jstart,n) && !Endpoint(k,kstart,p)) {
          IIint[cint]  = i + j*mwidth + k*mwidth*nwidth;
          Iint[cint++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        } else {
          IIsurf[csurf]  = i + j*mwidth + k*mwidth*nwidth;
          Isurf[csurf++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        }
      }
    }
  }
  PetscCheckFalse(c != N,PETSC_COMM_SELF,PETSC_ERR_PLIB,"c != N");
  PetscCheckFalse(cint != Nint,PETSC_COMM_SELF,PETSC_ERR_PLIB,"cint != Nint");
  PetscCheckFalse(csurf != Nsurf,PETSC_COMM_SELF,PETSC_ERR_PLIB,"csurf != Nsurf");
  CHKERRQ(DMGetLocalToGlobalMapping(da,&ltg));
  CHKERRQ(ISLocalToGlobalMappingApply(ltg,N,II,II));
  CHKERRQ(ISLocalToGlobalMappingApply(ltg,Nint,IIint,IIint));
  CHKERRQ(ISLocalToGlobalMappingApply(ltg,Nsurf,IIsurf,IIsurf));
  CHKERRQ(PetscObjectGetComm((PetscObject)da,&comm));
  CHKERRQ(ISCreateGeneral(comm,N,II,PETSC_COPY_VALUES,&is));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,Nint,Iint,PETSC_COPY_VALUES,&isint));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,Nsurf,Isurf,PETSC_COPY_VALUES,&issurf));
  CHKERRQ(PetscFree3(II,Iint,Isurf));

  CHKERRQ(ISSort(is));
  CHKERRQ(MatCreateSubMatrices(Aglobal,1,&is,&is,MAT_INITIAL_MATRIX,&Aholder));
  A    = *Aholder;
  CHKERRQ(PetscFree(Aholder));

  CHKERRQ(MatCreateSubMatrix(A,isint,isint,MAT_INITIAL_MATRIX,&Aii));
  CHKERRQ(MatCreateSubMatrix(A,isint,issurf,MAT_INITIAL_MATRIX,&Ais));
  CHKERRQ(MatCreateSubMatrix(A,issurf,isint,MAT_INITIAL_MATRIX,&Asi));

  /*
     Solve for the interpolation onto the interior Xint
  */
  CHKERRQ(MatMatMult(Ais,Xsurf,MAT_INITIAL_MATRIX,PETSC_DETERMINE,&Xint_tmp));
  CHKERRQ(MatScale(Xint_tmp,-1.0));

  if (exotic->directSolve) {
    CHKERRQ(MatGetFactor(Aii,MATSOLVERPETSC,MAT_FACTOR_LU,&iAii));
    CHKERRQ(MatFactorInfoInitialize(&info));
    CHKERRQ(MatGetOrdering(Aii,MATORDERINGND,&row,&col));
    CHKERRQ(MatLUFactorSymbolic(iAii,Aii,row,col,&info));
    CHKERRQ(ISDestroy(&row));
    CHKERRQ(ISDestroy(&col));
    CHKERRQ(MatLUFactorNumeric(iAii,Aii,&info));
    CHKERRQ(MatMatSolve(iAii,Xint_tmp,Xint));
    CHKERRQ(MatDestroy(&iAii));
  } else {
    Vec         b,x;
    PetscScalar *xint_tmp;

    CHKERRQ(MatDenseGetArray(Xint,&xint));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,Nint,NULL,&x));
    CHKERRQ(MatDenseGetArray(Xint_tmp,&xint_tmp));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,Nint,NULL,&b));
    CHKERRQ(KSPSetOperators(exotic->ksp,Aii,Aii));
    for (i=0; i<6; i++) {
      CHKERRQ(VecPlaceArray(x,xint+i*Nint));
      CHKERRQ(VecPlaceArray(b,xint_tmp+i*Nint));
      CHKERRQ(KSPSolve(exotic->ksp,b,x));
      CHKERRQ(KSPCheckSolve(exotic->ksp,pc,x));
      CHKERRQ(VecResetArray(x));
      CHKERRQ(VecResetArray(b));
    }
    CHKERRQ(MatDenseRestoreArray(Xint,&xint));
    CHKERRQ(MatDenseRestoreArray(Xint_tmp,&xint_tmp));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&b));
  }
  CHKERRQ(MatDestroy(&Xint_tmp));

#if defined(PETSC_USE_DEBUG_foo)
  CHKERRQ(MatDenseGetArrayRead(Xint,&rxint));
  for (i=0; i<Nint; i++) {
    tmp = 0.0;
    for (j=0; j<6; j++) tmp += rxint[i+j*Nint];

    PetscCheckFalse(PetscAbsScalar(tmp-1.0) > 1.e-10,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong Xint interpolation at i %D value %g",i,(double)PetscAbsScalar(tmp));
  }
  CHKERRQ(MatDenseRestoreArrayRead(Xint,&rxint));
  /* CHKERRQ(MatView(Xint,PETSC_VIEWER_STDOUT_WORLD)); */
#endif

  /*         total faces    */
  Ntotal =  mp*np*(pp+1) + mp*pp*(np+1) + np*pp*(mp+1);

  /*
      For each vertex, edge, face on process (in the same orderings as used above) determine its local number including ghost points
  */
  cnt = 0;
  { gl[cnt++] = mwidth+1;}
  {
    { gl[cnt++] = mwidth*nwidth+1;}
    { gl[cnt++] = mwidth*nwidth + mwidth; /* these are the interior nodes */ gl[cnt++] = mwidth*nwidth + mwidth+m-istart-1;}
    { gl[cnt++] = mwidth*nwidth+mwidth*(n-jstart-1)+1;}
  }
  { gl[cnt++] = mwidth*nwidth*(p-kstart-1) + mwidth+1;}

  /* PetscIntView(6,gl,PETSC_VIEWER_STDOUT_WORLD); */
  /* convert that to global numbering and get them on all processes */
  CHKERRQ(ISLocalToGlobalMappingApply(ltg,6,gl,gl));
  /* PetscIntView(6,gl,PETSC_VIEWER_STDOUT_WORLD); */
  CHKERRQ(PetscMalloc1(6*mp*np*pp,&globals));
  CHKERRMPI(MPI_Allgather(gl,6,MPIU_INT,globals,6,MPIU_INT,PetscObjectComm((PetscObject)da)));

  /* Number the coarse grid points from 0 to Ntotal */
  CHKERRQ(MatGetSize(Aglobal,&Nt,NULL));
  CHKERRQ(PetscTableCreate(Ntotal/3,Nt+1,&ht));
  for (i=0; i<6*mp*np*pp; i++) {
    CHKERRQ(PetscTableAddCount(ht,globals[i]+1));
  }
  CHKERRQ(PetscTableGetCount(ht,&cnt));
  PetscCheckFalse(cnt != Ntotal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hash table size %D not equal to total number coarse grid points %D",cnt,Ntotal);
  CHKERRQ(PetscFree(globals));
  for (i=0; i<6; i++) {
    CHKERRQ(PetscTableFind(ht,gl[i]+1,&gl[i]));
    gl[i]--;
  }
  CHKERRQ(PetscTableDestroy(&ht));
  /* PetscIntView(6,gl,PETSC_VIEWER_STDOUT_WORLD); */

  /* construct global interpolation matrix */
  CHKERRQ(MatGetLocalSize(Aglobal,&Ng,NULL));
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatCreateAIJ(PetscObjectComm((PetscObject)da),Ng,PETSC_DECIDE,PETSC_DECIDE,Ntotal,Nint+Nsurf,NULL,Nint,NULL,P));
  } else {
    CHKERRQ(MatZeroEntries(*P));
  }
  CHKERRQ(MatSetOption(*P,MAT_ROW_ORIENTED,PETSC_FALSE));
  CHKERRQ(MatDenseGetArrayRead(Xint,&rxint));
  CHKERRQ(MatSetValues(*P,Nint,IIint,6,gl,rxint,INSERT_VALUES));
  CHKERRQ(MatDenseRestoreArrayRead(Xint,&rxint));
  CHKERRQ(MatDenseGetArrayRead(Xsurf,&rxint));
  CHKERRQ(MatSetValues(*P,Nsurf,IIsurf,6,gl,rxint,INSERT_VALUES));
  CHKERRQ(MatDenseRestoreArrayRead(Xsurf,&rxint));
  CHKERRQ(MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscFree2(IIint,IIsurf));

#if defined(PETSC_USE_DEBUG_foo)
  {
    Vec         x,y;
    PetscScalar *yy;
    CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)da),Ng,PETSC_DETERMINE,&y));
    CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)da),PETSC_DETERMINE,Ntotal,&x));
    CHKERRQ(VecSet(x,1.0));
    CHKERRQ(MatMult(*P,x,y));
    CHKERRQ(VecGetArray(y,&yy));
    for (i=0; i<Ng; i++) {
      PetscCheckFalse(PetscAbsScalar(yy[i]-1.0) > 1.e-10,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong p interpolation at i %D value %g",i,(double)PetscAbsScalar(yy[i]));
    }
    CHKERRQ(VecRestoreArray(y,&yy));
    CHKERRQ(VecDestroy(x));
    CHKERRQ(VecDestroy(y));
  }
#endif

  CHKERRQ(MatDestroy(&Aii));
  CHKERRQ(MatDestroy(&Ais));
  CHKERRQ(MatDestroy(&Asi));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISDestroy(&isint));
  CHKERRQ(ISDestroy(&issurf));
  CHKERRQ(MatDestroy(&Xint));
  CHKERRQ(MatDestroy(&Xsurf));
  PetscFunctionReturn(0);
}

/*@
   PCExoticSetType - Sets the type of coarse grid interpolation to use

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - either PC_EXOTIC_FACE or PC_EXOTIC_WIREBASKET (defaults to face)

   Notes:
    The face based interpolation has 1 degree of freedom per face and ignores the
     edge and vertex values completely in the coarse problem. For any seven point
     stencil the interpolation of a constant on all faces into the interior is that constant.

     The wirebasket interpolation has 1 degree of freedom per vertex, per edge and
     per face. A constant on the subdomain boundary is interpolated as that constant
     in the interior of the domain.

     The coarse grid matrix is obtained via the Galerkin computation A_c = R A R^T, hence
     if A is nonsingular A_c is also nonsingular.

     Both interpolations are suitable for only scalar problems.

   Level: intermediate

.seealso: PCEXOTIC, PCExoticType()
@*/
PetscErrorCode  PCExoticSetType(PC pc,PCExoticType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  CHKERRQ(PetscTryMethod(pc,"PCExoticSetType_C",(PC,PCExoticType),(pc,type)));
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCExoticSetType_Exotic(PC pc,PCExoticType type)
{
  PC_MG     *mg  = (PC_MG*)pc->data;
  PC_Exotic *ctx = (PC_Exotic*) mg->innerctx;

  PetscFunctionBegin;
  ctx->type = type;
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetUp_Exotic(PC pc)
{
  Mat            A;
  PC_MG          *mg   = (PC_MG*)pc->data;
  PC_Exotic      *ex   = (PC_Exotic*) mg->innerctx;
  MatReuse       reuse = (ex->P) ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;

  PetscFunctionBegin;
  PetscCheck(pc->dm,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Need to call PCSetDM() before using this PC");
  CHKERRQ(PCGetOperators(pc,NULL,&A));
  if (ex->type == PC_EXOTIC_FACE) {
    CHKERRQ(DMDAGetFaceInterpolation(pc,pc->dm,ex,A,reuse,&ex->P));
  } else if (ex->type == PC_EXOTIC_WIREBASKET) {
    CHKERRQ(DMDAGetWireBasketInterpolation(pc,pc->dm,ex,A,reuse,&ex->P));
  } else SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Unknown exotic coarse space %d",ex->type);
  CHKERRQ(PCMGSetInterpolation(pc,1,ex->P));
  /* if PC has attached DM we must remove it or the PCMG will use it to compute incorrect sized vectors and interpolations */
  CHKERRQ(PCSetDM(pc,NULL));
  CHKERRQ(PCSetUp_MG(pc));
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_Exotic(PC pc)
{
  PC_MG          *mg  = (PC_MG*)pc->data;
  PC_Exotic      *ctx = (PC_Exotic*) mg->innerctx;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&ctx->P));
  CHKERRQ(KSPDestroy(&ctx->ksp));
  CHKERRQ(PetscFree(ctx));
  CHKERRQ(PCDestroy_MG(pc));
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_Exotic(PC pc,PetscViewer viewer)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PetscBool      iascii;
  PC_Exotic      *ctx = (PC_Exotic*) mg->innerctx;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"    Exotic type = %s\n",PCExoticTypes[ctx->type]));
    if (ctx->directSolve) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Using direct solver to construct interpolation\n"));
    } else {
      PetscViewer sviewer;
      PetscMPIInt rank;

      CHKERRQ(PetscViewerASCIIPrintf(viewer,"      Using iterative solver to construct interpolation\n"));
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ(PetscViewerASCIIPushTab(viewer));  /* should not need to push this twice? */
      CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
      if (rank == 0) {
        CHKERRQ(KSPView(ctx->ksp,sviewer));
      }
      CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
  }
  CHKERRQ(PCView_MG(pc,viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_Exotic(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscBool      flg;
  PC_MG          *mg = (PC_MG*)pc->data;
  PCExoticType   mgctype;
  PC_Exotic      *ctx = (PC_Exotic*) mg->innerctx;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Exotic coarse space options"));
  CHKERRQ(PetscOptionsEnum("-pc_exotic_type","face or wirebasket","PCExoticSetType",PCExoticTypes,(PetscEnum)ctx->type,(PetscEnum*)&mgctype,&flg));
  if (flg) {
    CHKERRQ(PCExoticSetType(pc,mgctype));
  }
  CHKERRQ(PetscOptionsBool("-pc_exotic_direct_solver","use direct solver to construct interpolation","None",ctx->directSolve,&ctx->directSolve,NULL));
  if (!ctx->directSolve) {
    if (!ctx->ksp) {
      const char *prefix;
      CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ctx->ksp));
      CHKERRQ(KSPSetErrorIfNotConverged(ctx->ksp,pc->erroriffailure));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)pc,1));
      CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)ctx->ksp));
      CHKERRQ(PCGetOptionsPrefix(pc,&prefix));
      CHKERRQ(KSPSetOptionsPrefix(ctx->ksp,prefix));
      CHKERRQ(KSPAppendOptionsPrefix(ctx->ksp,"exotic_"));
    }
    CHKERRQ(KSPSetFromOptions(ctx->ksp));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*MC
     PCEXOTIC - Two level overlapping Schwarz preconditioner with exotic (non-standard) coarse grid spaces

     This uses the PCMG infrastructure restricted to two levels and the face and wirebasket based coarse
   grid spaces.

   Notes:
    By default this uses GMRES on the fine grid smoother so this should be used with KSPFGMRES or the smoother changed to not use GMRES

   References:
+  * - These coarse grid spaces originate in the work of Bramble, Pasciak  and Schatz, "The Construction
   of Preconditioners for Elliptic Problems by Substructing IV", Mathematics of Computation, volume 53, 1989.
.  * - They were generalized slightly in "Domain Decomposition Method for Linear Elasticity", Ph. D. thesis, Barry Smith,
   New York University, 1990.
.  * - They were then explored in great detail in Dryja, Smith, Widlund, "Schwarz Analysis
   of Iterative Substructuring Methods for Elliptic Problems in Three Dimensions, SIAM Journal on Numerical
   Analysis, volume 31. 1994. These were developed in the context of iterative substructuring preconditioners.
.  * - They were then ingeniously applied as coarse grid spaces for overlapping Schwarz methods by Dohrmann and Widlund.
   They refer to them as GDSW (generalized Dryja, Smith, Widlund preconditioners). See, for example,
   Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. Extending theory for domain decomposition algorithms to irregular subdomains. In Ulrich Langer, Marco
   Discacciati, David Keyes, Olof Widlund, and Walter Zulehner, editors, Proceedings
   of the 17th International Conference on Domain Decomposition Methods in
   Science and Engineering, held in Strobl, Austria, 2006, number 60 in
   Springer Verlag, Lecture Notes in Computational Science and Engineering, 2007.
.  * -  Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. A family of energy minimizing coarse spaces for overlapping Schwarz preconditioners. In Ulrich Langer,
   Marco Discacciati, David Keyes, Olof Widlund, and Walter Zulehner, editors, Proceedings
   of the 17th International Conference on Domain Decomposition Methods
   in Science and Engineering, held in Strobl, Austria, 2006, number 60 in
   Springer Verlag, Lecture Notes in Computational Science and Engineering, 2007
.  * - Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. Domain decomposition
   for less regular subdomains: Overlapping Schwarz in two dimensions. SIAM J.
   Numer. Anal., 46(4), 2008.
-  * - Clark R. Dohrmann and Olof B. Widlund. An overlapping Schwarz
   algorithm for almost incompressible elasticity. Technical Report
   TR2008 912, Department of Computer Science, Courant Institute
   of Mathematical Sciences, New York University, May 2008. URL:

   Options Database: The usual PCMG options are supported, such as -mg_levels_pc_type <type> -mg_coarse_pc_type <type>
      -pc_mg_type <type>

   Level: advanced

.seealso:  PCMG, PCSetDM(), PCExoticType, PCExoticSetType()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Exotic(PC pc)
{
  PC_Exotic      *ex;
  PC_MG          *mg;

  PetscFunctionBegin;
  /* if type was previously mg; must manually destroy it because call to PCSetType(pc,PCMG) will not destroy it */
  if (pc->ops->destroy) {
    CHKERRQ((*pc->ops->destroy)(pc));
    pc->data = NULL;
  }
  CHKERRQ(PetscFree(((PetscObject)pc)->type_name));
  ((PetscObject)pc)->type_name = NULL;

  CHKERRQ(PCSetType(pc,PCMG));
  CHKERRQ(PCMGSetLevels(pc,2,NULL));
  CHKERRQ(PCMGSetGalerkin(pc,PC_MG_GALERKIN_PMAT));
  CHKERRQ(PetscNew(&ex)); \
  ex->type     = PC_EXOTIC_FACE;
  mg           = (PC_MG*) pc->data;
  mg->innerctx = ex;

  pc->ops->setfromoptions = PCSetFromOptions_Exotic;
  pc->ops->view           = PCView_Exotic;
  pc->ops->destroy        = PCDestroy_Exotic;
  pc->ops->setup          = PCSetUp_Exotic;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCExoticSetType_C",PCExoticSetType_Exotic));
  PetscFunctionReturn(0);
}
