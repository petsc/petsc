
#ifndef lint
static char vcid[] = "$Id: fdda.c,v 1.6 1997/01/06 20:30:50 balay Exp bsmith $";
#endif
 
#include "da.h"     /*I      "da.h"     I*/
#include "mat.h"    /*I      "mat.h"    I*/

#undef __FUNC__  
#define __FUNC__ "DAGetColoring2dBox" /* ADIC Ignore */
/*@C
      DAGetColoring2dBox - Gets the coloring required for computing the Jacobian via
          finite differences on a function defined using the nine point stencil
          on a two dimensional grid. 

          This is a utility routine that will change over time, not part of the 
          core PETSc package.

  Input Parameter:
.    da - the distributed array

  Output Parameters:
.    coloring - matrix coloring for compute Jacobians
.    J  - matrix with the correct nonzero structured 
            (obviously without the correct Jacobian values)

@*/
int DAGetColoring2dBox(DA da,ISColoring *coloring,Mat *J)
{
  int      ierr, xs,ys,nx,ny,*colors,i,j,ii,slot,gxs,gys,ys1,ny1,nx1,gnx,gny;           
  int      m,n,dim,w,s,N,indices[9],*gindices,k,xs1,flag;
  MPI_Comm comm;
  Scalar   values[9];
  /*
                                  m
          ------------------------------------------------------
         |                                                     |
         |                                                     |
         |               ----------------------                |
         |               |                    |                |
      n  |           yn  |                    |                |
         |               |                    |                |
         |               .---------------------                |
         |             (xs,ys)     xn                          |
         |            .                                        |
         |         (gxs,gys)                                   |
         |                                                     |
          -----------------------------------------------------
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&w,&s); CHKERRQ(ierr);
  if (dim != 2) SETERRQ(1,0,"2d only");
  if (w != 1)   SETERRQ(1,0,"Scalar problems only");
  if (s != 1)   SETERRQ(1,0,"Stencil width 1 only");
  /* also no support for periodic boundary conditions */
  N = m*n;
  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm); CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(da,PETSC_NULL,&gindices);

  /* create the coloring */
  colors = (int *) PetscMalloc( nx*ny*sizeof(int) ); CHKPTRQ(colors);
  ii = 0;
  for ( j=ys; j<ys+ny; j++ ) {
    for ( i=xs; i<xs+nx; i++ ) {
      colors[ii++] = (i % 3) + 3*(j % 3);
    }
  }
  ierr = ISColoringCreate(comm,nx*ny,colors,coloring); CHKERRQ(ierr);
  PetscFree(colors);

  /* create empty Jacobian matrix */
  ierr = OptionsHasName(PETSC_NULL,"-mat_seqaij",&flag); CHKERRQ(ierr);
  if (flag) {
    ierr = MatCreateSeqAIJ(comm,nx*ny,nx*ny,9,0,J);CHKERRQ(ierr);  
  } else {
    ierr = MatCreateMPIAIJ(comm,nx*ny,nx*ny,PETSC_DECIDE,PETSC_DECIDE,9,0,0,0,J);CHKERRQ(ierr);  
  }
  values[0] = 0; values[1] = 0; values[2] = 0; values[3] = 0; values[4] = 0; 
  values[5] = 0; values[6] = 0; values[7] = 0; values[8] = 0; 


  /*
      For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then map those indices to the global PETSc ordering
    before inserting in the matrix
  */
  
  /* fill up Jacobian for left edge */
  if (xs == 0) {
    ys1 = ys;
    ny1 = ny;
    /* lower left corner */
    if (ys == 0) {
      ys1++;
      ny1--;
      slot = xs - gxs + gnx*(ys - gys);
      indices[0] = slot;         indices[1] = slot + 1; 
      indices[2] = slot + gnx;   indices[3] = slot + gnx + 1;
      for ( k=0; k<4; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,4,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    /* upper left corner */
    if (ys + ny == n) {
      ny1--;
      slot = xs - gxs + gnx*(ys + (ny - 1) - gys);
      indices[0] = slot - gnx;   indices[1] = slot - gnx + 1;
      indices[2] = slot;         indices[3] = slot + 1;
      for ( k=0; k<4; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,4,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    for ( j=ys1; j<ys1+ny1; j++ ) {
      slot = xs - gxs + gnx*(j - gys);
      indices[0] = slot - gnx;   indices[1] = slot - gnx + 1;
      indices[2] = slot;         indices[3] = slot + 1;
      indices[4] = slot + gnx;   indices[5] = slot + gnx + 1;
      for ( k=0; k<6; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,6,indices,values,INSERT_VALUES); CHKERRQ(ierr);  
    }
  }      

  /* fill up Jacobian for right edge */
  if (xs + nx == m) {
    ys1 = ys;
    ny1 = ny;
    /* lower right corner */
    if (ys == 0) {
      ys1++;
      ny1--;
      slot = xs + (nx - 1) - gxs + gnx*(ys - gys);
      indices[0] = slot - 1;         indices[1] = slot; 
      indices[2] = slot + gnx - 1;   indices[3] = slot + gnx;
      for ( k=0; k<4; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,4,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    /* upper right corner */
    if (ys + ny == n) {
      ny1--;
      slot = xs + (nx - 1) - gxs + gnx*(ys + (ny - 1) - gys);
      indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;
      indices[2] = slot-1;           indices[3] = slot;
      for ( k=0; k<4; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,4,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
    for ( j=ys1; j<ys1+ny1; j++ ) {
      slot = xs + (nx - 1) - gxs + gnx*(j - gys);
      indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;
      indices[2] = slot - 1;         indices[3] = slot;
      indices[4] = slot + gnx - 1;   indices[5] = slot + gnx;
      for ( k=0; k<6; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,6,indices,values,INSERT_VALUES); CHKERRQ(ierr);   
    }
  }   

  /* fill up Jacobian for bottom */
  if (ys == 0) {
    if (xs == 0) {nx1 = nx - 1; xs1 = 1;} else {nx1 = nx; xs1 = xs;}
    if (xs + nx == m) {nx1--;}
    for ( i=xs1; i<xs1+nx1; i++ ) {
      slot = i - gxs + gnx*(ys - gys);
      indices[0] = slot - 1;       indices[1] = slot;        indices[2] = slot + 1;
      indices[3] = slot + gnx - 1; indices[4] = slot + gnx;  indices[5] = slot + gnx + 1;
      for ( k=0; k<6; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,6,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  /* fill up Jacobian for top */
  if (ys + ny == n) {
    if (xs == 0) {nx1 = nx - 1; xs1 = 1;} else {nx1 = nx; xs1 = xs;}
    if (xs + nx == m) {nx1--;}
    for ( i=xs1; i<xs1+nx1; i++ ) {
      slot = i - gxs + gnx*(ys + (ny - 1) - gys);
      indices[0] = slot - gnx - 1;   indices[1] = slot - gnx;  indices[2] = slot - gnx + 1;
      indices[3] = slot - 1;         indices[4] = slot;        indices[5] = slot  + 1;
      for ( k=0; k<6; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,6,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  /* fill up Jacobian for interior of grid */
  nx1 = nx;
  ny1 = ny;
  if (xs == 0)    {xs++; nx1--;}
  if (ys == 0)    {ys++; ny1--;}
  if (xs+nx1 == m) {nx1--;}
  if (ys+ny1 == n) {ny1--;}
  for ( i=xs; i<xs+nx1; i++ ) {
    for ( j=ys; j<ys+ny1; j++ ) {
      slot = i - gxs + gnx*(j - gys);
      indices[0] = slot - gnx - 1;indices[1] = slot - gnx; indices[2] = slot - gnx + 1;
      indices[3] = slot - 1  ;    indices[4] = slot;       indices[5] = slot + 1;
      indices[6] = slot + gnx - 1;indices[7] = slot + gnx; indices[8] = slot + gnx + 1;
      for ( k=0; k<9; k++ ) { indices[k] = gindices[indices[k]];}
      slot = gindices[slot];
      ierr = MatSetValues(*J,1,&slot,9,indices,values,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);  
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "DAGetBilinearInterpolation2dBox"
/*@C
      DAGetBilinearInterpolation2dBox - Gets the matrix representing bilinear
        interpolation from a DA grid to the next refinement.

          This is a utility routine that will change over time, not part of the 
          core PETSc package.

  Input Parameter:
.    dac - the distributed array for the coarse grid
.    daf - the distributed array for the fine grid

  Output Parameters:
.    A - restriction matrix

$         Interpolation-add is MatMultTransAdd(A,c,f,f)
$         Restriction is       MatMult(A,f,c)

@*/
int DAGetBilinearInterpolation2dBox(DA dac,DA daf,Mat *A)
{
  int      m,n,M,N,ierr,dim,w,s,xs,ys,nx,ny,gxs,gys,Nx,Ny,flag;
  MPI_Comm comm;

  ierr = DAGetInfo(dac,&dim,&M,&N,0,0,0,0,&w,&s); CHKERRQ(ierr);
  if (dim != 2) SETERRQ(1,0,"2d only");
  if (w != 1)   SETERRQ(1,0,"Scalar problems only");
  if (s != 1)   SETERRQ(1,0,"Stencil width 1 only");
  /* also no support for periodic */
  ierr = DAGetInfo(daf,&dim,&m,&n,0,0,0,0,&w,&s); CHKERRQ(ierr);
  if (dim != 2) SETERRQ(1,0,"2d only");
  if (w != 1)   SETERRQ(1,0,"Scalar problems only");
  if (s != 1)   SETERRQ(1,0,"Stencil width 1 only");
  /* also no support for periodic */

  if (m != 2*M - 1) SETERRQ(1,0,"");
  if (n != 2*N - 1) SETERRQ(1,0,"");

  /*
         Coarse grid is M by N. Fine grid is m by n
  */
  ierr = DAGetCorners(dac,&xs,&ys,0,&Nx,&Ny,0); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&gxs,&gys,0,0,0,0); CHKERRQ(ierr);
  ierr = DAGetCorners(daf,0,0,0,&nx,&ny,0); CHKERRQ(ierr);

  /* create empty matrix */
  ierr = PetscObjectGetComm((PetscObject)dac,&comm); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-mat_seqaij",&flag); CHKERRQ(ierr);
  if (flag) {
    ierr = MatCreateSeqAIJ(comm,nx*ny,Nx*Ny,9,0,A);CHKERRQ(ierr);  
  } else {
    ierr = MatCreateMPIAIJ(comm,nx*ny,Nx*Ny,PETSC_DECIDE,PETSC_DECIDE,9,0,0,0,A);CHKERRQ(ierr);  
  }

  return 0;
}





