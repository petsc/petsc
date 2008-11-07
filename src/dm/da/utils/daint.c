#define PETSCDM_DLL
 
#include "../src/dm/da/daimpl.h" /*I      "petscda.h"     I*/
#include "petscmat.h"         /*I      "petscmat.h"    I*/


#undef __FUNCT__  
#define __FUNCT__ "DAGetWireBasket"
/*
      DAGetWireBasket - Gets the interpolation and coarse matrix for the classical wirebasket coarse
                  grid problem; for structured grids.

*/
static PetscErrorCode DAGetWireBasket(DA da)
{
  PetscErrorCode ierr;
  PetscInt       dim,i,j,k,m,n,p,dof,Nint,Nface,Nwire,*Iint,*Iface,*Iwire,cint = 0,cface = 0,cwire = 0,istart,jstart,kstart,*I,N,c;
  Mat            P, Xint, Xface, Xwire; 
  IS             isint,isface,iswire,is;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,&dim,0,0,0,&m,&n,&p,&dof,0,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&istart,&jstart,&kstart,0,0,0);CHKERRQ(ierr);
  istart = istart ? -1 : 0;
  jstart = jstart ? -1 : 0;
  kstart = kstart ? -1 : 0;
  if (dof != 1) SETERRQ(PETSC_ERR_SUP,"Only for single field problems");
  if (dim != 3) SETERRQ(PETSC_ERR_SUP,"Only coded for 3d problems");

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
  Nint  = (m-2)*(n-2)*(p-2);
  Nface = 2*( (m-2)*(n-2) + (m-2)*(p-2) + (n-2)*(p-2) ); 
  Nwire = 4*( (m-2) + (n-2) + (p-2) );
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nint,20,PETSC_NULL,&Xint);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nface,20,PETSC_NULL,&Xface);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(MPI_COMM_SELF,Nwire,20,PETSC_NULL,&Xwire);CHKERRQ(ierr);

  /* 
       I are the indices for all the needed vertices (in global numbering)
       Ixxx are the indices for the interior values, the face values and the wirebasket values
            (in the local natural ordering on the local grid)
  */
#define Endpoint(a,start,b) (a == start || a == (b-1))
  ierr = PetscMalloc4(N,PetscInt,&I,Nint,PetscInt,&Iint,Nface,PetscInt,&Iface,Nwire,PetscInt,&Iwire);CHKERRQ(ierr);
  for (k=kstart; k<p; k++) {
    for (j=jstart; j<n; j++) {
      for (i=istart; i<m; i++) {
        I[c++] = i + j*m + k*m*n; /* wrong */

        if (!Endpoint(i,istart,m) && !Endpoint(j,jstart,n) && !Endpoint(k,kstart,p)) {
          Iint[cint++] = i + j*m + k*m*n;
        } else if ((Endpoint(i,istart,m) && Endpoint(j,jstart,n)) || (Endpoint(i,istart,m) && Endpoint(k,istart,p)) || (Endpoint(j,jstart,n) && Endpoint(k,kstart,p))) {
          Iwire[cwire++] = i + j*m + k*m*n;
        } else {
          Iface[cface++] = i + j*m + k*m*n;
        }
      }
    }
  }
  if (c != N) SETERRQ(PETSC_ERR_PLIB,"c != N");
  if (cint != Nint) SETERRQ(PETSC_ERR_PLIB,"cint != Nint");
  if (cface != Nface) SETERRQ(PETSC_ERR_PLIB,"cface != Nface");
  if (cwire != Nwire) SETERRQ(PETSC_ERR_PLIB,"cwire != Nwire");

  ierr = PetscFree3(Iint,Iface,Iwire);CHKERRQ(ierr);
  ierr = MatDestroy(Xint);CHKERRQ(ierr);
  ierr = MatDestroy(Xface);CHKERRQ(ierr);
  ierr = MatDestroy(Xwire);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

