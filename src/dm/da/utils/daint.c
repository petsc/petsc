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
  PetscInt               mwidth,nwidth,pwidth;
  Mat                    Xint, Xface, Xwire; 
  IS                     isint,isface,iswire,is;
  ISLocalToGlobalMapping ltg;
  MPI_Comm               comm;
  Mat                    A,Aii,Aif,Aiw,Afi,Aff,Afw,Awi,Awf,Aww,*Aholder;
  PetscMPIInt rank;

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

  /* 
     Solve for the interpolation onto the faces Xface
  */

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (rank == -1) {
    PetscIntView(N,I,0);
    PetscIntView(Nint,Iint,0);
    PetscIntView(Nface,Iface,0);
    PetscIntView(Nwire,Iwire,0);
  }

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

