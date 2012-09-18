
#include <petscpcmg.h>   /*I "petscpcmg.h" I*/
#include <petscdmda.h>   /*I "petscdmda.h" I*/
#include <../src/ksp/pc/impls/mg/mgimpl.h>

typedef struct {
  PCExoticType type;
  Mat          P;            /* the constructed interpolation matrix */
  PetscBool    directSolve;  /* use direct LU factorization to construct interpolation */
  KSP          ksp;
} PC_Exotic;

const char *const PCExoticTypes[] = {"face","wirebasket","PCExoticType","PC_Exotic",0};


#undef __FUNCT__
#define __FUNCT__ "DMDAGetWireBasketInterpolation"
/*
      DMDAGetWireBasketInterpolation - Gets the interpolation for a wirebasket based coarse space

*/
PetscErrorCode DMDAGetWireBasketInterpolation(DM da,PC_Exotic *exotic,Mat Aglobal,MatReuse reuse,Mat *P)
{
  PetscErrorCode         ierr;
  PetscInt               dim,i,j,k,m,n,p,dof,Nint,Nface,Nwire,Nsurf,*Iint,*Isurf,cint = 0,csurf = 0,istart,jstart,kstart,*II,N,c = 0;
  PetscInt               mwidth,nwidth,pwidth,cnt,mp,np,pp,Ntotal,gl[26],*globals,Ng,*IIint,*IIsurf,Nt;
  Mat                    Xint, Xsurf,Xint_tmp;
  IS                     isint,issurf,is,row,col;
  ISLocalToGlobalMapping ltg;
  MPI_Comm               comm;
  Mat                    A,Aii,Ais,Asi,*Aholder,iAii;
  MatFactorInfo          info;
  PetscScalar            *xsurf,*xint;
#if defined(PETSC_USE_DEBUG_foo)
  PetscScalar            tmp;
#endif
  PetscTable             ht;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,&dim,0,0,0,&mp,&np,&pp,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (dof != 1) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"Only for single field problems");
  if (dim != 3) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"Only coded for 3d problems");
  ierr = DMDAGetCorners(da,0,0,0,&m,&n,&p);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&istart,&jstart,&kstart,&mwidth,&nwidth,&pwidth);CHKERRQ(ierr);
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
    Symbolically one could write P = (  Xface  ) after interchanging the rows to match the natural ordering on the domain
                                        Xsurf
  */
  N     = (m - istart)*(n - jstart)*(p - kstart);
  Nint  = (m-2-istart)*(n-2-jstart)*(p-2-kstart);
  Nface = 2*( (m-2-istart)*(n-2-jstart) + (m-2-istart)*(p-2-kstart) + (n-2-jstart)*(p-2-kstart) );
  Nwire = 4*( (m-2-istart) + (n-2-jstart) + (p-2-kstart) ) + 8;
  Nsurf = Nface + Nwire;
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nint,26,PETSC_NULL,&Xint);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nsurf,26,PETSC_NULL,&Xsurf);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Xsurf,&xsurf);CHKERRQ(ierr);

  /*
     Require that all 12 edges and 6 faces have at least one grid point. Otherwise some of the columns of
     Xsurf will be all zero (thus making the coarse matrix singular).
  */
  if (m-istart < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in X direction must be at least 3");
  if (n-jstart < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in Y direction must be at least 3");
  if (p-kstart < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in Z direction must be at least 3");

  cnt = 0;
  xsurf[cnt++] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + Nsurf] = 1;} xsurf[cnt++ + 2*Nsurf] = 1;
  for (j=1;j<n-1-jstart;j++) { xsurf[cnt++ + 3*Nsurf] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 4*Nsurf] = 1;} xsurf[cnt++ + 5*Nsurf] = 1;}
  xsurf[cnt++ + 6*Nsurf] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 7*Nsurf] = 1;} xsurf[cnt++ + 8*Nsurf] = 1;
  for (k=1;k<p-1-kstart;k++) {
    xsurf[cnt++ + 9*Nsurf] = 1;  for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 10*Nsurf] = 1;}  xsurf[cnt++ + 11*Nsurf] = 1;
    for (j=1;j<n-1-jstart;j++) { xsurf[cnt++ + 12*Nsurf] = 1; /* these are the interior nodes */ xsurf[cnt++ + 13*Nsurf] = 1;}
    xsurf[cnt++ + 14*Nsurf] = 1;  for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 15*Nsurf] = 1;} xsurf[cnt++ + 16*Nsurf] = 1;
  }
  xsurf[cnt++ + 17*Nsurf] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 18*Nsurf] = 1;} xsurf[cnt++ + 19*Nsurf] = 1;
  for (j=1;j<n-1-jstart;j++) { xsurf[cnt++ + 20*Nsurf] = 1;  for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 21*Nsurf] = 1;} xsurf[cnt++ + 22*Nsurf] = 1;}
  xsurf[cnt++ + 23*Nsurf] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 24*Nsurf] = 1;} xsurf[cnt++ + 25*Nsurf] = 1;

  /* interpolations only sum to 1 when using direct solver */
#if defined(PETSC_USE_DEBUG_foo)
  for (i=0; i<Nsurf; i++) {
    tmp = 0.0;
    for (j=0; j<26; j++) {
      tmp += xsurf[i+j*Nsurf];
    }
    if (PetscAbsScalar(tmp-1.0) > 1.e-10) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong Xsurf interpolation at i %D value %G",i,PetscAbsScalar(tmp));
  }
#endif
  ierr = MatDenseRestoreArray(Xsurf,&xsurf);CHKERRQ(ierr);
  /* ierr = MatView(Xsurf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);*/


  /*
       I are the indices for all the needed vertices (in global numbering)
       Iint are the indices for the interior values, I surf for the surface values
            (This is just for the part of the global matrix obtained with MatGetSubMatrix(), it
             is NOT the local DMDA ordering.)
       IIint and IIsurf are the same as the Iint, Isurf except they are in the global numbering
  */
#define Endpoint(a,start,b) (a == 0 || a == (b-1-start))
  ierr = PetscMalloc3(N,PetscInt,&II,Nint,PetscInt,&Iint,Nsurf,PetscInt,&Isurf);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nint,PetscInt,&IIint,Nsurf,PetscInt,&IIsurf);CHKERRQ(ierr);
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
  if (c != N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"c != N");
  if (cint != Nint) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"cint != Nint");
  if (csurf != Nsurf) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"csurf != Nsurf");
  ierr = DMGetLocalToGlobalMapping(da,&ltg);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltg,N,II,II);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltg,Nint,IIint,IIint);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltg,Nsurf,IIsurf,IIsurf);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,N,II,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nint,Iint,PETSC_COPY_VALUES,&isint);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nsurf,Isurf,PETSC_COPY_VALUES,&issurf);CHKERRQ(ierr);
  ierr = PetscFree3(II,Iint,Isurf);CHKERRQ(ierr);

  ierr = MatGetSubMatrices(Aglobal,1,&is,&is,MAT_INITIAL_MATRIX,&Aholder);CHKERRQ(ierr);
  A    = *Aholder;
  ierr = PetscFree(Aholder);CHKERRQ(ierr);

  ierr = MatGetSubMatrix(A,isint,isint,MAT_INITIAL_MATRIX,&Aii);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isint,issurf,MAT_INITIAL_MATRIX,&Ais);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,issurf,isint,MAT_INITIAL_MATRIX,&Asi);CHKERRQ(ierr);

  /*
     Solve for the interpolation onto the interior Xint
  */
  ierr = MatMatMult(Ais,Xsurf,MAT_INITIAL_MATRIX,PETSC_DETERMINE,&Xint_tmp);CHKERRQ(ierr);
  ierr = MatScale(Xint_tmp,-1.0);CHKERRQ(ierr);
  if (exotic->directSolve) {
    ierr = MatGetFactor(Aii,MATSOLVERPETSC,MAT_FACTOR_LU,&iAii);CHKERRQ(ierr);
    ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
    ierr = MatGetOrdering(Aii,MATORDERINGND,&row,&col);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(iAii,Aii,row,col,&info);CHKERRQ(ierr);
    ierr = ISDestroy(&row);CHKERRQ(ierr);
    ierr = ISDestroy(&col);CHKERRQ(ierr);
    ierr = MatLUFactorNumeric(iAii,Aii,&info);CHKERRQ(ierr);
    ierr = MatMatSolve(iAii,Xint_tmp,Xint);CHKERRQ(ierr);
    ierr = MatDestroy(&iAii);CHKERRQ(ierr);
  } else {
    Vec         b,x;
    PetscScalar *xint_tmp;

    ierr = MatDenseGetArray(Xint,&xint);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,Nint,0,&x);CHKERRQ(ierr);
    ierr = MatDenseGetArray(Xint_tmp,&xint_tmp);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,Nint,0,&b);CHKERRQ(ierr);
    ierr = KSPSetOperators(exotic->ksp,Aii,Aii,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    for (i=0; i<26; i++) {
      ierr = VecPlaceArray(x,xint+i*Nint);CHKERRQ(ierr);
      ierr = VecPlaceArray(b,xint_tmp+i*Nint);CHKERRQ(ierr);
      ierr = KSPSolve(exotic->ksp,b,x);CHKERRQ(ierr);
      ierr = VecResetArray(x);CHKERRQ(ierr);
      ierr = VecResetArray(b);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(Xint,&xint);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(Xint_tmp,&xint_tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&Xint_tmp);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG_foo)
  ierr = MatDenseGetArray(Xint,&xint);CHKERRQ(ierr);
  for (i=0; i<Nint; i++) {
    tmp = 0.0;
    for (j=0; j<26; j++) {
      tmp += xint[i+j*Nint];
    }
    if (PetscAbsScalar(tmp-1.0) > 1.e-10) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong Xint interpolation at i %D value %G",i,PetscAbsScalar(tmp));
  }
  ierr = MatDenseRestoreArray(Xint,&xint);CHKERRQ(ierr);
  /* ierr =MatView(Xint,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
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
  ierr = ISLocalToGlobalMappingApply(ltg,26,gl,gl);CHKERRQ(ierr);
  /* PetscIntView(26,gl,PETSC_VIEWER_STDOUT_WORLD); */
  ierr = PetscMalloc(26*mp*np*pp*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  ierr = MPI_Allgather(gl,26,MPIU_INT,globals,26,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);

  /* Number the coarse grid points from 0 to Ntotal */
  ierr = MatGetSize(Aglobal,&Nt,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscTableCreate(Ntotal/3,Nt+1,&ht);CHKERRQ(ierr);
  for (i=0; i<26*mp*np*pp; i++){
    ierr = PetscTableAddCount(ht,globals[i]+1);CHKERRQ(ierr);
  }
  ierr = PetscTableGetCount(ht,&cnt);CHKERRQ(ierr);
  if (cnt != Ntotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hash table size %D not equal to total number coarse grid points %D",cnt,Ntotal);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  for (i=0; i<26; i++) {
    ierr = PetscTableFind(ht,gl[i]+1,&gl[i]);CHKERRQ(ierr);
    gl[i]--;
  }
  ierr = PetscTableDestroy(&ht);CHKERRQ(ierr);
  /* PetscIntView(26,gl,PETSC_VIEWER_STDOUT_WORLD); */

  /* construct global interpolation matrix */
  ierr = MatGetLocalSize(Aglobal,&Ng,PETSC_NULL);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(((PetscObject)da)->comm,Ng,PETSC_DECIDE,PETSC_DECIDE,Ntotal,Nint+Nsurf,PETSC_NULL,Nint+Nsurf,PETSC_NULL,P);CHKERRQ(ierr);
  } else {
    ierr = MatZeroEntries(*P);CHKERRQ(ierr);
  }
  ierr = MatSetOption(*P,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Xint,&xint);CHKERRQ(ierr);
  ierr = MatSetValues(*P,Nint,IIint,26,gl,xint,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Xint,&xint);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Xsurf,&xsurf);CHKERRQ(ierr);
  ierr = MatSetValues(*P,Nsurf,IIsurf,26,gl,xsurf,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Xsurf,&xsurf);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree2(IIint,IIsurf);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG_foo)
  {
    Vec         x,y;
    PetscScalar *yy;
    ierr = VecCreateMPI(((PetscObject)da)->comm,Ng,PETSC_DETERMINE,&y);CHKERRQ(ierr);
    ierr = VecCreateMPI(((PetscObject)da)->comm,PETSC_DETERMINE,Ntotal,&x);CHKERRQ(ierr);
    ierr = VecSet(x,1.0);CHKERRQ(ierr);
    ierr = MatMult(*P,x,y);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    for (i=0; i<Ng; i++) {
      if (PetscAbsScalar(yy[i]-1.0) > 1.e-10) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong p interpolation at i %D value %G",i,PetscAbsScalar(yy[i]));
    }
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
    ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = VecDestroy(y);CHKERRQ(ierr);
  }
#endif

  ierr = MatDestroy(&Aii);CHKERRQ(ierr);
  ierr = MatDestroy(&Ais);CHKERRQ(ierr);
  ierr = MatDestroy(&Asi);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&isint);CHKERRQ(ierr);
  ierr = ISDestroy(&issurf);CHKERRQ(ierr);
  ierr = MatDestroy(&Xint);CHKERRQ(ierr);
  ierr = MatDestroy(&Xsurf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetFaceInterpolation"
/*
      DMDAGetFaceInterpolation - Gets the interpolation for a face based coarse space

*/
PetscErrorCode DMDAGetFaceInterpolation(DM da,PC_Exotic *exotic,Mat Aglobal,MatReuse reuse,Mat *P)
{
  PetscErrorCode         ierr;
  PetscInt               dim,i,j,k,m,n,p,dof,Nint,Nface,Nwire,Nsurf,*Iint,*Isurf,cint = 0,csurf = 0,istart,jstart,kstart,*II,N,c = 0;
  PetscInt               mwidth,nwidth,pwidth,cnt,mp,np,pp,Ntotal,gl[6],*globals,Ng,*IIint,*IIsurf,Nt;
  Mat                    Xint, Xsurf,Xint_tmp;
  IS                     isint,issurf,is,row,col;
  ISLocalToGlobalMapping ltg;
  MPI_Comm               comm;
  Mat                    A,Aii,Ais,Asi,*Aholder,iAii;
  MatFactorInfo          info;
  PetscScalar            *xsurf,*xint;
#if defined(PETSC_USE_DEBUG_foo)
  PetscScalar            tmp;
#endif
  PetscTable             ht;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,&dim,0,0,0,&mp,&np,&pp,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (dof != 1) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"Only for single field problems");
  if (dim != 3) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"Only coded for 3d problems");
  ierr = DMDAGetCorners(da,0,0,0,&m,&n,&p);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&istart,&jstart,&kstart,&mwidth,&nwidth,&pwidth);CHKERRQ(ierr);
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
    Symbolically one could write P = (  Xface  ) after interchanging the rows to match the natural ordering on the domain
                                        Xsurf
  */
  N     = (m - istart)*(n - jstart)*(p - kstart);
  Nint  = (m-2-istart)*(n-2-jstart)*(p-2-kstart);
  Nface = 2*( (m-2-istart)*(n-2-jstart) + (m-2-istart)*(p-2-kstart) + (n-2-jstart)*(p-2-kstart) );
  Nwire = 4*( (m-2-istart) + (n-2-jstart) + (p-2-kstart) ) + 8;
  Nsurf = Nface + Nwire;
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nint,6,PETSC_NULL,&Xint);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nsurf,6,PETSC_NULL,&Xsurf);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Xsurf,&xsurf);CHKERRQ(ierr);

  /*
     Require that all 12 edges and 6 faces have at least one grid point. Otherwise some of the columns of
     Xsurf will be all zero (thus making the coarse matrix singular).
  */
  if (m-istart < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in X direction must be at least 3");
  if (n-jstart < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in Y direction must be at least 3");
  if (p-kstart < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of grid points per process in Z direction must be at least 3");

  cnt = 0;
  for (j=1;j<n-1-jstart;j++) {  for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 0*Nsurf] = 1;} }
   for (k=1;k<p-1-kstart;k++) {
    for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 1*Nsurf] = 1;}
    for (j=1;j<n-1-jstart;j++) { xsurf[cnt++ + 2*Nsurf] = 1; /* these are the interior nodes */ xsurf[cnt++ + 3*Nsurf] = 1;}
    for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 4*Nsurf] = 1;}
  }
  for (j=1;j<n-1-jstart;j++) {for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 5*Nsurf] = 1;} }

#if defined(PETSC_USE_DEBUG_foo)
  for (i=0; i<Nsurf; i++) {
    tmp = 0.0;
    for (j=0; j<6; j++) {
      tmp += xsurf[i+j*Nsurf];
    }
    if (PetscAbsScalar(tmp-1.0) > 1.e-10) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong Xsurf interpolation at i %D value %G",i,PetscAbsScalar(tmp));
  }
#endif
  ierr = MatDenseRestoreArray(Xsurf,&xsurf);CHKERRQ(ierr);
  /* ierr = MatView(Xsurf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);*/


  /*
       I are the indices for all the needed vertices (in global numbering)
       Iint are the indices for the interior values, I surf for the surface values
            (This is just for the part of the global matrix obtained with MatGetSubMatrix(), it
             is NOT the local DMDA ordering.)
       IIint and IIsurf are the same as the Iint, Isurf except they are in the global numbering
  */
#define Endpoint(a,start,b) (a == 0 || a == (b-1-start))
  ierr = PetscMalloc3(N,PetscInt,&II,Nint,PetscInt,&Iint,Nsurf,PetscInt,&Isurf);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nint,PetscInt,&IIint,Nsurf,PetscInt,&IIsurf);CHKERRQ(ierr);
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
  if (c != N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"c != N");
  if (cint != Nint) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"cint != Nint");
  if (csurf != Nsurf) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"csurf != Nsurf");
  ierr = DMGetLocalToGlobalMapping(da,&ltg);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltg,N,II,II);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltg,Nint,IIint,IIint);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltg,Nsurf,IIsurf,IIsurf);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,N,II,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nint,Iint,PETSC_COPY_VALUES,&isint);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nsurf,Isurf,PETSC_COPY_VALUES,&issurf);CHKERRQ(ierr);
  ierr = PetscFree3(II,Iint,Isurf);CHKERRQ(ierr);

  ierr = ISSort(is);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(Aglobal,1,&is,&is,MAT_INITIAL_MATRIX,&Aholder);CHKERRQ(ierr);
  A    = *Aholder;
  ierr = PetscFree(Aholder);CHKERRQ(ierr);

  ierr = MatGetSubMatrix(A,isint,isint,MAT_INITIAL_MATRIX,&Aii);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isint,issurf,MAT_INITIAL_MATRIX,&Ais);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,issurf,isint,MAT_INITIAL_MATRIX,&Asi);CHKERRQ(ierr);

  /*
     Solve for the interpolation onto the interior Xint
  */
  ierr = MatMatMult(Ais,Xsurf,MAT_INITIAL_MATRIX,PETSC_DETERMINE,&Xint_tmp);CHKERRQ(ierr);
  ierr = MatScale(Xint_tmp,-1.0);CHKERRQ(ierr);

  if (exotic->directSolve) {
    ierr = MatGetFactor(Aii,MATSOLVERPETSC,MAT_FACTOR_LU,&iAii);CHKERRQ(ierr);
    ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
    ierr = MatGetOrdering(Aii,MATORDERINGND,&row,&col);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(iAii,Aii,row,col,&info);CHKERRQ(ierr);
    ierr = ISDestroy(&row);CHKERRQ(ierr);
    ierr = ISDestroy(&col);CHKERRQ(ierr);
    ierr = MatLUFactorNumeric(iAii,Aii,&info);CHKERRQ(ierr);
    ierr = MatMatSolve(iAii,Xint_tmp,Xint);CHKERRQ(ierr);
    ierr = MatDestroy(&iAii);CHKERRQ(ierr);
  } else {
    Vec         b,x;
    PetscScalar *xint_tmp;

    ierr = MatDenseGetArray(Xint,&xint);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,Nint,0,&x);CHKERRQ(ierr);
    ierr = MatDenseGetArray(Xint_tmp,&xint_tmp);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,Nint,0,&b);CHKERRQ(ierr);
    ierr = KSPSetOperators(exotic->ksp,Aii,Aii,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    for (i=0; i<6; i++) {
      ierr = VecPlaceArray(x,xint+i*Nint);CHKERRQ(ierr);
      ierr = VecPlaceArray(b,xint_tmp+i*Nint);CHKERRQ(ierr);
      ierr = KSPSolve(exotic->ksp,b,x);CHKERRQ(ierr);
      ierr = VecResetArray(x);CHKERRQ(ierr);
      ierr = VecResetArray(b);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(Xint,&xint);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(Xint_tmp,&xint_tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&Xint_tmp);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG_foo)
  ierr = MatDenseGetArray(Xint,&xint);CHKERRQ(ierr);
  for (i=0; i<Nint; i++) {
    tmp = 0.0;
    for (j=0; j<6; j++) {
      tmp += xint[i+j*Nint];
    }
    if (PetscAbsScalar(tmp-1.0) > 1.e-10) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong Xint interpolation at i %D value %G",i,PetscAbsScalar(tmp));
  }
  ierr = MatDenseRestoreArray(Xint,&xint);CHKERRQ(ierr);
  /* ierr =MatView(Xint,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
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
  ierr = ISLocalToGlobalMappingApply(ltg,6,gl,gl);CHKERRQ(ierr);
  /* PetscIntView(6,gl,PETSC_VIEWER_STDOUT_WORLD); */
  ierr = PetscMalloc(6*mp*np*pp*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  ierr = MPI_Allgather(gl,6,MPIU_INT,globals,6,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);

  /* Number the coarse grid points from 0 to Ntotal */
  ierr = MatGetSize(Aglobal,&Nt,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscTableCreate(Ntotal/3,Nt+1,&ht);CHKERRQ(ierr);
  for (i=0; i<6*mp*np*pp; i++){
    ierr = PetscTableAddCount(ht,globals[i]+1);CHKERRQ(ierr);
  }
  ierr = PetscTableGetCount(ht,&cnt);CHKERRQ(ierr);
  if (cnt != Ntotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hash table size %D not equal to total number coarse grid points %D",cnt,Ntotal);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  for (i=0; i<6; i++) {
    ierr = PetscTableFind(ht,gl[i]+1,&gl[i]);CHKERRQ(ierr);
    gl[i]--;
  }
  ierr = PetscTableDestroy(&ht);CHKERRQ(ierr);
  /* PetscIntView(6,gl,PETSC_VIEWER_STDOUT_WORLD); */

  /* construct global interpolation matrix */
  ierr = MatGetLocalSize(Aglobal,&Ng,PETSC_NULL);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateAIJ(((PetscObject)da)->comm,Ng,PETSC_DECIDE,PETSC_DECIDE,Ntotal,Nint+Nsurf,PETSC_NULL,Nint,PETSC_NULL,P);CHKERRQ(ierr);
  } else {
    ierr = MatZeroEntries(*P);CHKERRQ(ierr);
  }
  ierr = MatSetOption(*P,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Xint,&xint);CHKERRQ(ierr);
  ierr = MatSetValues(*P,Nint,IIint,6,gl,xint,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Xint,&xint);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Xsurf,&xsurf);CHKERRQ(ierr);
  ierr = MatSetValues(*P,Nsurf,IIsurf,6,gl,xsurf,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Xsurf,&xsurf);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree2(IIint,IIsurf);CHKERRQ(ierr);


#if defined(PETSC_USE_DEBUG_foo)
  {
    Vec         x,y;
    PetscScalar *yy;
    ierr = VecCreateMPI(((PetscObject)da)->comm,Ng,PETSC_DETERMINE,&y);CHKERRQ(ierr);
    ierr = VecCreateMPI(((PetscObject)da)->comm,PETSC_DETERMINE,Ntotal,&x);CHKERRQ(ierr);
    ierr = VecSet(x,1.0);CHKERRQ(ierr);
    ierr = MatMult(*P,x,y);CHKERRQ(ierr);
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    for (i=0; i<Ng; i++) {
      if (PetscAbsScalar(yy[i]-1.0) > 1.e-10) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong p interpolation at i %D value %G",i,PetscAbsScalar(yy[i]));
    }
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
    ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = VecDestroy(y);CHKERRQ(ierr);
  }
#endif

  ierr = MatDestroy(&Aii);CHKERRQ(ierr);
  ierr = MatDestroy(&Ais);CHKERRQ(ierr);
  ierr = MatDestroy(&Asi);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&isint);CHKERRQ(ierr);
  ierr = ISDestroy(&issurf);CHKERRQ(ierr);
  ierr = MatDestroy(&Xint);CHKERRQ(ierr);
  ierr = MatDestroy(&Xsurf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCExoticSetType"
/*@
   PCExoticSetType - Sets the type of coarse grid interpolation to use

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - either PC_EXOTIC_FACE or PC_EXOTIC_WIREBASKET (defaults to face)

   Notes: The face based interpolation has 1 degree of freedom per face and ignores the
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  ierr = PetscTryMethod(pc,"PCExoticSetType_C",(PC,PCExoticType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCExoticSetType_Exotic"
PetscErrorCode  PCExoticSetType_Exotic(PC pc,PCExoticType type)
{
  PC_MG     *mg = (PC_MG*)pc->data;
  PC_Exotic *ctx = (PC_Exotic*) mg->innerctx;

  PetscFunctionBegin;
  ctx->type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_Exotic"
PetscErrorCode PCSetUp_Exotic(PC pc)
{
  PetscErrorCode ierr;
  Mat            A;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_Exotic      *ex = (PC_Exotic*) mg->innerctx;
  MatReuse       reuse = (ex->P) ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;

  PetscFunctionBegin;
  if (!pc->dm) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Need to call PCSetDM() before using this PC");
  ierr = PCGetOperators(pc,PETSC_NULL,&A,PETSC_NULL);CHKERRQ(ierr);
  if (ex->type == PC_EXOTIC_FACE) {
    ierr = DMDAGetFaceInterpolation(pc->dm,ex,A,reuse,&ex->P);CHKERRQ(ierr);
  } else if (ex->type == PC_EXOTIC_WIREBASKET) {
    ierr = DMDAGetWireBasketInterpolation(pc->dm,ex,A,reuse,&ex->P);CHKERRQ(ierr);
  } else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_PLIB,"Unknown exotic coarse space %d",ex->type);
  ierr = PCMGSetInterpolation(pc,1,ex->P);CHKERRQ(ierr);
  /* if PC has attached DM we must remove it or the PCMG will use it to compute incorrect sized vectors and interpolations */
  ierr = PCSetDM(pc,PETSC_NULL);CHKERRQ(ierr);
  ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_Exotic"
PetscErrorCode PCDestroy_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_Exotic      *ctx = (PC_Exotic*) mg->innerctx;

  PetscFunctionBegin;
  ierr = MatDestroy(&ctx->P);CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->ksp);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PCDestroy_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_Exotic"
PetscErrorCode PCView_Exotic(PC pc,PetscViewer viewer)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;
  PC_Exotic      *ctx = (PC_Exotic*) mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"    Exotic type = %s\n",PCExoticTypes[ctx->type]);CHKERRQ(ierr);
    if (ctx->directSolve) {
      ierr = PetscViewerASCIIPrintf(viewer,"      Using direct solver to construct interpolation\n");CHKERRQ(ierr);
    } else {
      PetscViewer sviewer;
      PetscMPIInt rank;

      ierr = PetscViewerASCIIPrintf(viewer,"      Using iterative solver to construct interpolation\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);  /* should not need to push this twice? */
      ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
      if (!rank) {
	ierr = KSPView(ctx->ksp,sviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  ierr = PCView_MG(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_Exotic"
PetscErrorCode PCSetFromOptions_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PC_MG          *mg = (PC_MG*)pc->data;
  PCExoticType   mgctype;
  PC_Exotic      *ctx = (PC_Exotic*) mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Exotic coarse space options");CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-pc_exotic_type","face or wirebasket","PCExoticSetType",PCExoticTypes,(PetscEnum)ctx->type,(PetscEnum*)&mgctype,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCExoticSetType(pc,mgctype);CHKERRQ(ierr);
    }
    ierr = PetscOptionsBool("-pc_exotic_direct_solver","use direct solver to construct interpolation","None",ctx->directSolve,&ctx->directSolve,PETSC_NULL);CHKERRQ(ierr);
    if (!ctx->directSolve) {
      if (!ctx->ksp) {
        const char *prefix;
        ierr = KSPCreate(PETSC_COMM_SELF,&ctx->ksp);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
        ierr = PetscLogObjectParent(pc,ctx->ksp);CHKERRQ(ierr);
        ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(ctx->ksp,prefix);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(ctx->ksp,"exotic_");CHKERRQ(ierr);
      }
      ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
     PCEXOTIC - Two level overlapping Schwarz preconditioner with exotic (non-standard) coarse grid spaces

     This uses the PCMG infrastructure restricted to two levels and the face and wirebasket based coarse
   grid spaces.

   Notes: By default this uses GMRES on the fine grid smoother so this should be used with KSPFGMRES or the smoother changed to not use GMRES

   References: These coarse grid spaces originate in the work of Bramble, Pasciak  and Schatz, "The Construction
   of Preconditioners for Elliptic Problems by Substructing IV", Mathematics of Computation, volume 53 pages 1--24, 1989.
   They were generalized slightly in "Domain Decomposition Method for Linear Elasticity", Ph. D. thesis, Barry Smith,
   New York University, 1990. They were then explored in great detail in Dryja, Smith, Widlund, "Schwarz Analysis
   of Iterative Substructuring Methods for Elliptic Problems in Three Dimensions, SIAM Journal on Numerical
   Analysis, volume 31. pages 1662-1694, 1994. These were developed in the context of iterative substructuring preconditioners.
   They were then ingeniously applied as coarse grid spaces for overlapping Schwarz methods by Dohrmann and Widlund.
   They refer to them as GDSW (generalized Dryja, Smith, Widlund preconditioners). See, for example,
   Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. Extending theory for domain decomposition algorithms to irregular subdomains. In Ulrich Langer, Marco
   Discacciati, David Keyes, Olof Widlund, and Walter Zulehner, editors, Proceedings
   of the 17th International Conference on Domain Decomposition Methods in
   Science and Engineering, held in Strobl, Austria, July 3-7, 2006, number 60 in
   Springer-Verlag, Lecture Notes in Computational Science and Engineering, pages 255-261, 2007.
   Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. A family of energy min-
   imizing coarse spaces for overlapping Schwarz preconditioners. In Ulrich Langer,
   Marco Discacciati, David Keyes, Olof Widlund, and Walter Zulehner, editors, Proceedings
   of the 17th International Conference on Domain Decomposition Methods
   in Science and Engineering, held in Strobl, Austria, July 3-7, 2006, number 60 in
   Springer-Verlag, Lecture Notes in Computational Science and Engineering, pages 247-254, 2007
   Clark R. Dohrmann, Axel Klawonn, and Olof B. Widlund. Domain decomposition
   for less regular subdomains: Overlapping Schwarz in two dimensions. SIAM J.
   Numer. Anal., 46(4):2153-2168, 2008.
   Clark R. Dohrmann and Olof B. Widlund. An overlapping Schwarz
   algorithm for almost incompressible elasticity. Technical Report
   TR2008-912, Department of Computer Science, Courant Institute
   of Mathematical Sciences, New York University, May 2008. URL:
   http://cs.nyu.edu/csweb/Research/TechReports/TR2008-912/TR2008-912.pdf

   Options Database: The usual PCMG options are supported, such as -mg_levels_pc_type <type> -mg_coarse_pc_type <type>
      -pc_mg_type <type>

   Level: advanced

.seealso:  PCMG, PCSetDM(), PCExoticType, PCExoticSetType()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_Exotic"
PetscErrorCode  PCCreate_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PC_Exotic      *ex;
  PC_MG          *mg;

  PetscFunctionBegin;
  /* if type was previously mg; must manually destroy it because call to PCSetType(pc,PCMG) will not destroy it */
  if (pc->ops->destroy) { ierr =  (*pc->ops->destroy)(pc);CHKERRQ(ierr); pc->data = 0;}
  ierr = PetscFree(((PetscObject)pc)->type_name);CHKERRQ(ierr);
  ((PetscObject)pc)->type_name = 0;

  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscNew(PC_Exotic,&ex);CHKERRQ(ierr);\
  ex->type = PC_EXOTIC_FACE;
  mg = (PC_MG*) pc->data;
  mg->innerctx = ex;


  pc->ops->setfromoptions = PCSetFromOptions_Exotic;
  pc->ops->view           = PCView_Exotic;
  pc->ops->destroy        = PCDestroy_Exotic;
  pc->ops->setup          = PCSetUp_Exotic;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCExoticSetType_C","PCExoticSetType_Exotic",PCExoticSetType_Exotic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
