#define PETSCDM_DLL
 
#include "../src/dm/da/daimpl.h" /*I      "petscda.h"     I*/
#include "petscmat.h"         /*I      "petscmat.h"    I*/


#undef __FUNCT__  
#define __FUNCT__ "DAGetWireBasket"
/*
      DAGetWireBasket - Gets the interpolation and coarse matrix for the classical wirebasket coarse
                  grid problem; for structured grids.

*/
PetscErrorCode DAGetWireBasket(DA da,Mat Aglobal)
{
  PetscErrorCode         ierr;
  PetscInt               dim,i,j,k,m,n,p,dof,Nint,Nface,Nwire,Nsurf,*Iint,*Isurf,cint = 0,csurf = 0,istart,jstart,kstart,*I,N,c = 0;
  PetscInt               mwidth,nwidth,pwidth,cnt;
  Mat                    P, Xint, Xface, Xsurf,Xface_tmp,Xint_tmp,Xint_b; 
  IS                     isint,issurf,is,row,col;
  ISLocalToGlobalMapping ltg;
  MPI_Comm               comm;
  Mat                    A,Aii,Aif,Aiw,Afi,Aff,Afw,Awi,Awf,Aww,*Aholder,iAff,iAii;
  MatFactorInfo          info;
  PetscScalar            *xsurf;
#if defined(PETSC_USE_DEBUG)
  PetscScalar            tmp;
#endif

  PetscFunctionBegin;
  ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (dof != 1) SETERRQ(PETSC_ERR_SUP,"Only for single field problems");
  if (dim != 3) SETERRQ(PETSC_ERR_SUP,"Only coded for 3d problems");
  ierr = DAGetCorners(da,0,0,0,&m,&n,&p);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&istart,&jstart,&kstart,&mwidth,&nwidth,&pwidth);CHKERRQ(ierr);
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
  ierr = MatGetArray(Xsurf,&xsurf);CHKERRQ(ierr);
  /* fill up the 8 vertex nodule basis */
  cnt = 0;
  xsurf[cnt++] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + Nsurf] = 1;} xsurf[cnt++ + 2*Nsurf] = 1;
  for (j=1;j<n-1-jstart;j++) { xsurf[cnt++ + 3*Nsurf] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 20*Nsurf] = 1;} xsurf[cnt++ + 4*Nsurf] = 1;}
  xsurf[cnt++ + 5*Nsurf] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 6*Nsurf] = 1;} xsurf[cnt++ + 7*Nsurf] = 1;
  for (k=1;k<p-1-kstart;k++) {
    xsurf[cnt++ + 8*Nsurf] = 1;  for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 21*Nsurf] = 1;}  xsurf[cnt++ + 9*Nsurf] = 1;
    for (j=1;j<n-1-jstart;j++) { xsurf[cnt++ + 22*Nsurf] = 1;xsurf[cnt++ + 23*Nsurf] = 1;}
    xsurf[cnt++ + 10*Nsurf] = 1;  for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 24*Nsurf] = 1;} xsurf[cnt++ + 11*Nsurf] = 1;
  }
  xsurf[cnt++ + 12*Nsurf] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 13*Nsurf] = 1;} xsurf[cnt++ + 14*Nsurf] = 1;
  for (j=1;j<n-1-jstart;j++) { xsurf[cnt++ + 15*Nsurf] = 1;  xsurf[cnt++ + 16*Nsurf] = 1;}
  xsurf[cnt++ + 17*Nsurf] = 1; for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 25*Nsurf] = 1;} for (i=1; i<m-istart-1; i++) { xsurf[cnt++ + 18*Nsurf] = 1;} xsurf[cnt++ + 19*Nsurf] = 1;

#if defined(PETSC_USE_DEBUG)
  for (i=0; i<Nsurf; i++) {
    tmp = 0.0;
    for (j=0; j<26; j++) {
      tmp += xsurf[i+j*Nsurf];
    }
    if (tmp != 1.0) SETERRQ2(PETSC_ERR_PLIB,"Wrong Xsurf interpolation at i %D value %G",i,tmp);
  }
#endif
  ierr = MatRestoreArray(Xsurf,&xsurf);CHKERRQ(ierr);


  /* 
       I are the indices for all the needed vertices (in global numbering)
       Ixxx are the indices for the interior values, the face values and the wirebasket values
            (in the local natural ordering on the local grid)
  */
#define Endpoint(a,start,b) (a == 0 || a == (b-1-start))
  ierr = PetscMalloc3(N,PetscInt,&I,Nint,PetscInt,&Iint,Nsurf,PetscInt,&Isurf);CHKERRQ(ierr);
  for (k=0; k<p-kstart; k++) {
    for (j=0; j<n-jstart; j++) {
      for (i=0; i<m-istart; i++) {
        I[c++] = i + j*mwidth + k*mwidth*nwidth; 

        if (!Endpoint(i,istart,m) && !Endpoint(j,jstart,n) && !Endpoint(k,kstart,p)) {
          Iint[cint++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        } else {
          Isurf[csurf++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        } 
      }
    }
  }
  if (c != N) SETERRQ(PETSC_ERR_PLIB,"c != N");
  if (cint != Nint) SETERRQ(PETSC_ERR_PLIB,"cint != Nint");
  if (surf != Nsurf) SETERRQ(PETSC_ERR_PLIB,"csurf != Nsurf");
  ierr = DAGetISLocalToGlobalMapping(da,&ltg);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltg,N,I,I);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,N,I,&is);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nint,Iint,&isint);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nsurf,Isurf,&issurf);CHKERRQ(ierr);
  ierr = PetscFree3(I,Iint,Isurf);CHKERRQ(ierr);

  ierr = MatGetSubMatrices(Aglobal,1,&is,&is,MAT_INITIAL_MATRIX,&Aholder);CHKERRQ(ierr);
  A    = *Aholder;
  ierr = PetscFree(Aholder);CHKERRQ(ierr);

  ierr = MatGetSubMatrix(A,isint,isint,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Aii);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isint,issurf,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Ais);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,issurf,isint,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Afi);CHKERRQ(ierr);

  printf("A\n");
  MatView(A,0);

  printf("Xwire\n");
  MatView(Xwire,0);
  printf("Afw\n");
  MatView(Afw,0);
  printf("Aff\n");
  MatView(Aff,0);

  /* 
     Solve for the interpolation onto the faces Xface
  */
  ierr = MatGetFactor(Aff,MAT_SOLVER_PETSC,MAT_FACTOR_LU,&iAff);CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  ierr = MatGetOrdering(Aff,MATORDERING_ND,&row,&col);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(iAff,Aff,row,col,&info);CHKERRQ(ierr);
  ierr = ISDestroy(row);CHKERRQ(ierr);
  ierr = ISDestroy(col);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(iAff,Aff,&info);CHKERRQ(ierr);
  ierr = MatDuplicate(Xface,MAT_DO_NOT_COPY_VALUES,&Xface_tmp);CHKERRQ(ierr);
  ierr = MatMatMult(Afw,Xwire,MAT_REUSE_MATRIX,PETSC_DETERMINE,&Xface_tmp);CHKERRQ(ierr);
  ierr = MatScale(Xface_tmp,-1.0);CHKERRQ(ierr);
  ierr = MatMatSolve(iAff,Xface_tmp,Xface);CHKERRQ(ierr);
  ierr = MatDestroy(Xface_tmp);CHKERRQ(ierr);
  ierr = MatDestroy(iAff);CHKERRQ(ierr);

  ierr = MatGetArray(Xface,&xwire);CHKERRQ(ierr);
  for (i=0; i<Nface; i++) {
    for (j=0; j<20; j++) {
      ;
    }
  }
  ierr = MatRestoreArray(Xface,&xwire);CHKERRQ(ierr);


#if defined(PETSC_USE_DEBUG)
  ierr = MatGetArray(Xint,&xwire);CHKERRQ(ierr);
  for (i=0; i<Nint; i++) {
    tmp = 0.0;
    for (j=0; j<20; j++) {
      tmp += xwire[i+j*Nint];
    }
    if (tmp != 1.0) printf("Wrong Xint row %d value %g\n",i,tmp); /*SETERRQ2(PETSC_ERR_PLIB,"Wrong Xint interpolation at i %D value %G",i,tmp); */
  }
  ierr = MatRestoreArray(Xint,&xwire);CHKERRQ(ierr);
#endif

#if defined(PETSC_DEBUG_WORK)
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (rank == 0) {
    PetscIntView(N,I,0);
    PetscIntView(Nint,Iint,0);
    PetscIntView(Nface,Iface,0);
    PetscIntView(Nwire,Iwire,0);
  }
#endif

  ierr = MatDestroy(Aii);CHKERRQ(ierr);
  ierr = MatDestroy(Ais);CHKERRQ(ierr);
  ierr = MatDestroy(Asi);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);
  ierr = ISDestroy(isint);CHKERRQ(ierr);
  ierr = ISDestroy(issurf);CHKERRQ(ierr);
  ierr = MatDestroy(Xint);CHKERRQ(ierr);
  ierr = MatDestroy(Xsurf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

