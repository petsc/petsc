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
  PetscInt               dim,i,j,k,m,n,p,dof,Nint,Nface,Nwire,*Iint,*Iface,*Iwire,cint = 0,cface = 0,cwire = 0,istart,jstart,kstart,*I,N,c = 0;
  PetscInt               mwidth,nwidth,pwidth,cnt;
  Mat                    P, Xint, Xface, Xwire,Xface_tmp,Xint_tmp,Xint_b; 
  IS                     isint,isface,iswire,is,row,col;
  ISLocalToGlobalMapping ltg;
  MPI_Comm               comm;
  Mat                    A,Aii,Aif,Aiw,Afi,Aff,Afw,Awi,Awf,Aww,*Aholder,iAff,iAii;
  MatFactorInfo          info;
  PetscScalar            *xwire;
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

    Xwire are the interpolation onto the vertices and edges (the wirebasket) 
                                        Xint
    Symbolically one could write P = (  Xface  ) after interchanging the rows to match the natural ordering on the domain
                                        Xwire
  */
  N     = (m - istart)*(n - jstart)*(p - kstart);
  Nint  = (m-2-istart)*(n-2-jstart)*(p-2-kstart);
  Nface = 2*( (m-2-istart)*(n-2-jstart) + (m-2-istart)*(p-2-kstart) + (n-2-jstart)*(p-2-kstart) ); 
  Nwire = 4*( (m-2-istart) + (n-2-jstart) + (p-2-kstart) ) + 8;
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nint,20,PETSC_NULL,&Xint);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nface,20,PETSC_NULL,&Xface);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nwire,20,PETSC_NULL,&Xwire);CHKERRQ(ierr);
  ierr = MatGetArray(Xwire,&xwire);CHKERRQ(ierr);
  /* fill up the 8 vertex nodule basis */
  cnt = 0;
  xwire[cnt++] = 1; for (i=1; i<m-istart-1; i++) { xwire[cnt++ + Nwire] = 1;} xwire[cnt++ + 2*Nwire] = 1;
  for (j=1;j<n-1-jstart;j++) { xwire[cnt++ + 3*Nwire] = 1;  xwire[cnt++ + 4*Nwire] = 1;}
  xwire[cnt++ + 5*Nwire] = 1; for (i=1; i<m-istart-1; i++) { xwire[cnt++ + 6*Nwire] = 1;} xwire[cnt++ + 7*Nwire] = 1;
  for (k=1;k<p-1-kstart;k++) {
    xwire[cnt++ + 8*Nwire] = 1; xwire[cnt++ + 9*Nwire] = 1;
    xwire[cnt++ + 10*Nwire] = 1; xwire[cnt++ + 11*Nwire] = 1;
  }
  xwire[cnt++ + 12*Nwire] = 1; for (i=1; i<m-istart-1; i++) { xwire[cnt++ + 13*Nwire] = 1;} xwire[cnt++ + 14*Nwire] = 1;
  for (j=1;j<n-1-jstart;j++) { xwire[cnt++ + 15*Nwire] = 1;  xwire[cnt++ + 16*Nwire] = 1;}
  xwire[cnt++ + 17*Nwire] = 1; for (i=1; i<m-istart-1; i++) { xwire[cnt++ + 18*Nwire] = 1;} xwire[cnt++ + 19*Nwire] = 1;

#if defined(PETSC_USE_DEBUG)
  for (i=0; i<Nwire; i++) {
    tmp = 0.0;
    for (j=0; j<20; j++) {
      tmp += xwire[i+j*Nwire];
    }
    if (tmp != 1.0) SETERRQ2(PETSC_ERR_PLIB,"Wrong Xwire interpolation at i %D value %G",i,tmp);
  }
#endif
  ierr = MatRestoreArray(Xwire,&xwire);CHKERRQ(ierr);


  /* 
       I are the indices for all the needed vertices (in global numbering)
       Ixxx are the indices for the interior values, the face values and the wirebasket values
            (in the local natural ordering on the local grid)
  */
#define Endpoint(a,start,b) (a == 0 || a == (b-1-start))
  ierr = PetscMalloc4(N,PetscInt,&I,Nint,PetscInt,&Iint,Nface,PetscInt,&Iface,Nwire,PetscInt,&Iwire);CHKERRQ(ierr);
  for (k=0; k<p-kstart; k++) {
    for (j=0; j<n-jstart; j++) {
      for (i=0; i<m-istart; i++) {
        I[c++] = i + j*mwidth + k*mwidth*nwidth; 

        if (!Endpoint(i,istart,m) && !Endpoint(j,jstart,n) && !Endpoint(k,kstart,p)) {
          Iint[cint++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        } else if ((Endpoint(i,istart,m) && Endpoint(j,jstart,n)) || (Endpoint(i,istart,m) && Endpoint(k,kstart,p)) || (Endpoint(j,jstart,n) && Endpoint(k,kstart,p))) {
          Iwire[cwire++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        } else {
          Iface[cface++] = i + j*(m-istart) + k*(m-istart)*(n-jstart);
        }
      }
    }
  }
  if (c != N) SETERRQ(PETSC_ERR_PLIB,"c != N");
  if (cint != Nint) SETERRQ(PETSC_ERR_PLIB,"cint != Nint");
  if (cface != Nface) SETERRQ(PETSC_ERR_PLIB,"cface != Nface");
  if (cwire != Nwire) SETERRQ(PETSC_ERR_PLIB,"cwire != Nwire");
  ierr = DAGetISLocalToGlobalMapping(da,&ltg);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltg,N,I,I);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,N,I,&is);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nint,Iint,&isint);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nface,Iface,&isface);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Nwire,Iwire,&iswire);CHKERRQ(ierr);
  ierr = PetscFree4(I,Iint,Iface,Iwire);CHKERRQ(ierr);

  ierr = MatGetSubMatrices(Aglobal,1,&is,&is,MAT_INITIAL_MATRIX,&Aholder);CHKERRQ(ierr);
  A    = *Aholder;
  ierr = PetscFree(Aholder);CHKERRQ(ierr);

  ierr = MatGetSubMatrix(A,isint,isint,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Aii);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isint,isface,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Aif);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isint,iswire,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Aiw);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isface,isint,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Afi);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isface,isface,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Aff);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isface,iswire,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Afw);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,iswire,isint,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Awi);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,iswire,isface,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Awf);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,iswire,iswire,PETSC_DECIDE,MAT_INITIAL_MATRIX,&Aww);CHKERRQ(ierr);

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
      if (PetscAbsScalar(xwire[i+j*Nface]) > .0001) xwire[i+j*Nface] = .25;
    }
  }
  ierr = MatRestoreArray(Xface,&xwire);CHKERRQ(ierr);

  printf("Xface\n");
  MatView(Xface,0);

  /*
     Solve for interpolation onto interior
  */
  ierr = MatGetFactor(Aii,MAT_SOLVER_PETSC,MAT_FACTOR_LU,&iAii);CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  ierr = MatGetOrdering(Aii,MATORDERING_ND,&row,&col);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(iAii,Aii,row,col,&info);CHKERRQ(ierr);
  ierr = ISDestroy(row);CHKERRQ(ierr);
  ierr = ISDestroy(col);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(iAii,Aii,&info);CHKERRQ(ierr);
  ierr = MatDuplicate(Xint,MAT_DO_NOT_COPY_VALUES,&Xint_tmp);CHKERRQ(ierr);
  ierr = MatDuplicate(Xint,MAT_DO_NOT_COPY_VALUES,&Xint_b);CHKERRQ(ierr);
  ierr = MatMatMult(Aif,Xface,MAT_REUSE_MATRIX,PETSC_DETERMINE,&Xint_tmp);CHKERRQ(ierr);
  printf("Xint_tmp\n");
  MatView(Xint_tmp,0);
  ierr = MatMatMult(Aiw,Xwire,MAT_REUSE_MATRIX,PETSC_DETERMINE,&Xint_b);CHKERRQ(ierr);
  printf("Xint_b\n");
  MatView(Xint_b,0);
  ierr = MatAXPY(Xint_b,1.0,Xint_tmp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(Xint_tmp);CHKERRQ(ierr);
  ierr = MatScale(Xint_b,-1.0);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  ierr = MatGetArray(Xint_b,&xwire);CHKERRQ(ierr);
  for (i=0; i<Nint; i++) {
    for (j=0; j<20; j++) {
      if (xwire[i+j*Nint] > 1.e4) printf("Bade Xint row %d value %g\n",i,xwire[i+j*Nwire]);
    }
  }
  ierr = MatRestoreArray(Xint_b,&xwire);CHKERRQ(ierr);
#endif

  ierr = MatMatSolve(iAii,Xint_b,Xint);CHKERRQ(ierr);
  ierr = MatDestroy(Xint_b);CHKERRQ(ierr);
  ierr = MatDestroy(iAii);CHKERRQ(ierr);
  printf("Xint\n");
  MatView(Xint,0);

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
  ierr = MatDestroy(Aif);CHKERRQ(ierr);
  ierr = MatDestroy(Aiw);CHKERRQ(ierr);
  ierr = MatDestroy(Afi);CHKERRQ(ierr);
  ierr = MatDestroy(Aff);CHKERRQ(ierr);
  ierr = MatDestroy(Afw);CHKERRQ(ierr);
  ierr = MatDestroy(Awi);CHKERRQ(ierr);
  ierr = MatDestroy(Awf);CHKERRQ(ierr);
  ierr = MatDestroy(Aww);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);
  ierr = ISDestroy(isint);CHKERRQ(ierr);
  ierr = ISDestroy(isface);CHKERRQ(ierr);
  ierr = ISDestroy(iswire);CHKERRQ(ierr);
  ierr = MatDestroy(Xint);CHKERRQ(ierr);
  ierr = MatDestroy(Xface);CHKERRQ(ierr);
  ierr = MatDestroy(Xwire);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

