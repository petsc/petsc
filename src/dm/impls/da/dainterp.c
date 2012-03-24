
/*
  Code for interpolating between grids represented by DMDAs
*/

/*
      For linear elements there are two branches of code to compute the interpolation. They should compute the same results but may not. The "new version" does 
   not work for periodic domains, the old does. Change NEWVERSION to 1 to compile in the new version. Eventually when we are sure the two produce identical results
   we will remove/merge the new version. Based on current tests, these both produce the same results. We are leaving NEWVERSION for now in the code since some 
   consider it cleaner, but old version is turned on since it handles periodic case.
*/
#define NEWVERSION 0

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolationScale"
/*@
    DMCreateInterpolationScale - Forms L = R*1/diag(R*1) - L.*v is like a coarse grid average of the 
      nearby fine grid points.

  Input Parameters:
+      dac - DM that defines a coarse mesh
.      daf - DM that defines a fine mesh
-      mat - the restriction (or interpolation operator) from fine to coarse

  Output Parameter:
.    scale - the scaled vector

  Level: developer

.seealso: DMCreateInterpolation()

@*/
PetscErrorCode  DMCreateInterpolationScale(DM dac,DM daf,Mat mat,Vec *scale)
{
  PetscErrorCode ierr;
  Vec            fine;
  PetscScalar    one = 1.0;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(daf,&fine);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac,scale);CHKERRQ(ierr);
  ierr = VecSet(fine,one);CHKERRQ(ierr);
  ierr = MatRestrict(mat,fine,*scale);CHKERRQ(ierr);
  ierr = VecDestroy(&fine);CHKERRQ(ierr);
  ierr = VecReciprocal(*scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation_DA_1D_Q1"
PetscErrorCode DMCreateInterpolation_DA_1D_Q1(DM dac,DM daf,Mat *A)
{
  PetscErrorCode   ierr;
  PetscInt         i,i_start,m_f,Mx,*idx_f;
  PetscInt         m_ghost,*idx_c,m_ghost_c;
  PetscInt         row,col,i_start_ghost,mx,m_c,nc,ratio;
  PetscInt         i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  PetscScalar      v[2],x;
  Mat              mat;
  DMDABoundaryType bx;
  
  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,0,&Mx,0,0,0,0,0,0,0,&bx,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&mx,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (bx == DMDA_BOUNDARY_PERIODIC){
    ratio = mx/Mx;
    if (ratio*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratio = (mx-1)/(Mx-1);
    if (ratio*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  
  ierr = DMDAGetCorners(daf,&i_start,0,0,&m_f,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,0,0,&m_ghost,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dac,&i_start_c,0,0,&m_c,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,0,0,&m_ghost_c,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);
  
  /* create interpolation matrix */
  ierr = MatCreate(((PetscObject)dac)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f,m_c,mx,Mx);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,2,PETSC_NULL,1,PETSC_NULL);CHKERRQ(ierr);
  
  /* loop over local fine grid nodes setting interpolation for those*/
  if (!NEWVERSION) {

    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(i-i_start_ghost)]/dof;
      
      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                          i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);
      
      /* 
       Only include those interpolation points that are truly 
       nonzero. Note this is very important for final grid lines
       in x direction; since they have no right neighbor
       */
      x  = ((double)(i - i_c*ratio))/((double)ratio);
      nc = 0;
      /* one left and below; or we are right on it */
      col      = dof*(i_c-i_start_ghost_c);
      cols[nc] = idx_c[col]/dof; 
      v[nc++]  = - x + 1.0;
      /* one right? */
      if (i_c*ratio != i) { 
        cols[nc] = idx_c[col+dof]/dof;
        v[nc++]  = x;
      }
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
    }
    
  } else {
    PetscScalar    *xi;
    PetscInt       li,nxi,n;
    PetscScalar    Ni[2];
    
    /* compute local coordinate arrays */
    nxi   = ratio + 1;
    ierr = PetscMalloc(sizeof(PetscScalar)*nxi,&xi);CHKERRQ(ierr);
    for (li=0; li<nxi; li++) {
      xi[li] = -1.0 + (PetscScalar)li*(2.0/(PetscScalar)(nxi-1));
    }

    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(i-i_start_ghost)]/dof;
      
      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                          i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      /* remainders */
      li = i - ratio * (i/ratio);
      if (i==mx-1){ li = nxi-1; }
      
      /* corners */
      col     = dof*(i_c-i_start_ghost_c);
      cols[0] = idx_c[col]/dof; 
      Ni[0]   = 1.0;
      if ( (li==0) || (li==nxi-1) ) {
        ierr = MatSetValue(mat,row,cols[0],Ni[0],INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }
      
      /* edges + interior */
      /* remainders */
      if (i==mx-1){ i_c--; }
      
      col     = dof*(i_c-i_start_ghost_c);
      cols[0] = idx_c[col]/dof; /* one left and below; or we are right on it */
      cols[1] = idx_c[col+dof]/dof;

      Ni[0] = 0.5*(1.0-xi[li]);
      Ni[1] = 0.5*(1.0+xi[li]);
      for (n=0; n<2; n++) {
        if( PetscAbsScalar(Ni[n])<1.0e-32) { cols[n]=-1; }
      }
      ierr = MatSetValues(mat,1,&row,2,cols,Ni,INSERT_VALUES);CHKERRQ(ierr); 
    }
    ierr = PetscFree(xi);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation_DA_1D_Q0"
PetscErrorCode DMCreateInterpolation_DA_1D_Q0(DM dac,DM daf,Mat *A)
{
  PetscErrorCode   ierr;
  PetscInt         i,i_start,m_f,Mx,*idx_f;
  PetscInt         m_ghost,*idx_c,m_ghost_c;
  PetscInt         row,col,i_start_ghost,mx,m_c,nc,ratio;
  PetscInt         i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  PetscScalar      v[2],x;
  Mat              mat;
  DMDABoundaryType bx;
  
  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,0,&Mx,0,0,0,0,0,0,0,&bx,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&mx,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (bx == DMDA_BOUNDARY_PERIODIC){
    ratio = mx/Mx;
    if (ratio*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratio = (mx-1)/(Mx-1);
    if (ratio*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }

  ierr = DMDAGetCorners(daf,&i_start,0,0,&m_f,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,0,0,&m_ghost,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,0,0,&m_c,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,0,0,&m_ghost_c,0,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreate(((PetscObject)dac)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f,m_c,mx,Mx);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,2,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (i=i_start; i<i_start+m_f; i++) {
    /* convert to local "natural" numbering and then to PETSc global numbering */
    row    = idx_f[dof*(i-i_start_ghost)]/dof;

    i_c = (i/ratio);    /* coarse grid node to left of fine grid node */

    /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x direction; since they have no right neighbor
    */
    x  = ((double)(i - i_c*ratio))/((double)ratio);
    nc = 0;
      /* one left and below; or we are right on it */
    col      = dof*(i_c-i_start_ghost_c);
    cols[nc] = idx_c[col]/dof; 
    v[nc++]  = - x + 1.0;
    /* one right? */
    if (i_c*ratio != i) { 
      cols[nc] = idx_c[col+dof]/dof;
      v[nc++]  = x;
    }
    ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(5.0*m_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation_DA_2D_Q1"
PetscErrorCode DMCreateInterpolation_DA_2D_Q1(DM dac,DM daf,Mat *A)
{
  PetscErrorCode   ierr;
  PetscInt         i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof;
  PetscInt         m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,*dnz,*onz;
  PetscInt         row,col,i_start_ghost,j_start_ghost,cols[4],mx,m_c,my,nc,ratioi,ratioj;
  PetscInt         i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c,col_shift,col_scale;
  PetscMPIInt      size_c,size_f,rank_f;
  PetscScalar      v[4],x,y;
  Mat              mat;
  DMDABoundaryType bx,by;
  
  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,&bx,&by,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (bx == DMDA_BOUNDARY_PERIODIC){
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (by == DMDA_BOUNDARY_PERIODIC){
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }
  
  
  ierr = DMDAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);
  
  /*
   Used for handling a coarse DMDA that lives on 1/4 the processors of the fine DMDA.
   The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the 
   processors). It's effective length is hence 4 times its normal length, this is
   why the col_scale is multiplied by the interpolation matrix column sizes.
   sol_shift allows each set of 1/4 processors do its own interpolation using ITS
   copy of the coarse vector. A bit of a hack but you do better.
   
   In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
   */
  ierr = MPI_Comm_size(((PetscObject)dac)->comm,&size_c);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)daf)->comm,&size_f);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)daf)->comm,&rank_f);CHKERRQ(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*(rank_f/size_c);
  
  ierr = MatPreallocateInitialize(((PetscObject)daf)->comm,m_f*n_f,col_scale*m_c*n_c,dnz,onz);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
      
      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */
      
      if (j_c < j_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                          j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                          i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);
      
      /* 
       Only include those interpolation points that are truly 
       nonzero. Note this is very important for final grid lines
       in x and y directions; since they have no right/top neighbors
       */
      nc = 0;
      /* one left and below; or we are right on it */
      col        = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc++] = col_shift + idx_c[col]/dof; 
      /* one right and below */
      if (i_c*ratioi != i) { 
        cols[nc++] = col_shift + idx_c[col+dof]/dof;
      }
      /* one left and above */
      if (j_c*ratioj != j) { 
        cols[nc++] = col_shift + idx_c[col+m_ghost_c*dof]/dof;
      }
      /* one right and above */
      if (i_c*ratioi != i && j_c*ratioj != j) {
        cols[nc++] = col_shift + idx_c[col+(m_ghost_c+1)*dof]/dof;
      }
      ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatCreate(((PetscObject)daf)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f*n_f,col_scale*m_c*n_c,mx*my,col_scale*Mx*My);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  
  /* loop over local fine grid nodes setting interpolation for those*/
  if (!NEWVERSION) {
    
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
        
        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */
        
        /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
         */
        x  = ((double)(i - i_c*ratioi))/((double)ratioi);
        y  = ((double)(j - j_c*ratioj))/((double)ratioj);
        
        nc = 0;
        /* one left and below; or we are right on it */
        col      = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc] = col_shift + idx_c[col]/dof; 
        v[nc++]  = x*y - x - y + 1.0;
        /* one right and below */
        if (i_c*ratioi != i) { 
          cols[nc] = col_shift + idx_c[col+dof]/dof;
          v[nc++]  = -x*y + x;
        }
        /* one left and above */
        if (j_c*ratioj != j) { 
          cols[nc] = col_shift + idx_c[col+m_ghost_c*dof]/dof;
          v[nc++]  = -x*y + y;
        }
        /* one right and above */
        if (j_c*ratioj != j && i_c*ratioi != i) { 
          cols[nc] = col_shift + idx_c[col+(m_ghost_c+1)*dof]/dof;
          v[nc++]  = x*y;
        }
        ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
      }
    }
    
  } else {
    PetscScalar    Ni[4];
    PetscScalar    *xi,*eta;
    PetscInt       li,nxi,lj,neta;
    
    /* compute local coordinate arrays */
    nxi  = ratioi + 1;
    neta = ratioj + 1;
    ierr = PetscMalloc(sizeof(PetscScalar)*nxi,&xi);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*neta,&eta);CHKERRQ(ierr);
    for (li=0; li<nxi; li++) {
      xi[li] = -1.0 + (PetscScalar)li*(2.0/(PetscScalar)(nxi-1));
    }
    for (lj=0; lj<neta; lj++) {
      eta[lj] = -1.0 + (PetscScalar)lj*(2.0/(PetscScalar)(neta-1));
    }

    /* loop over local fine grid nodes setting interpolation for those*/
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
        
        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */
        
        /* remainders */
        li = i - ratioi * (i/ratioi);
        if (i==mx-1){ li = nxi-1; }
        lj = j - ratioj * (j/ratioj);
        if (j==my-1){ lj = neta-1; }
        
        /* corners */
        col     = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[0] = col_shift + idx_c[col]/dof; /* left, below */
        Ni[0]   = 1.0;
        if ( (li==0) || (li==nxi-1) ) {
          if ( (lj==0) || (lj==neta-1) ) {
            ierr = MatSetValue(mat,row,cols[0],Ni[0],INSERT_VALUES);CHKERRQ(ierr); 
            continue;
          }
        }
        
        /* edges + interior */
        /* remainders */
        if (i==mx-1){ i_c--; }
        if (j==my-1){ j_c--; }

        col     = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[0] = col_shift + idx_c[col]/dof; /* left, below */
        cols[1] = col_shift + idx_c[col+dof]/dof; /* right, below */
        cols[2] = col_shift + idx_c[col+m_ghost_c*dof]/dof; /* left, above */
        cols[3] = col_shift + idx_c[col+(m_ghost_c+1)*dof]/dof; /* right, above */

        Ni[0] = 0.25*(1.0-xi[li])*(1.0-eta[lj]);
        Ni[1] = 0.25*(1.0+xi[li])*(1.0-eta[lj]);
        Ni[2] = 0.25*(1.0-xi[li])*(1.0+eta[lj]);
        Ni[3] = 0.25*(1.0+xi[li])*(1.0+eta[lj]);

        nc = 0;
        if( PetscAbsScalar(Ni[0])<1.0e-32) { cols[0]=-1; }
        if( PetscAbsScalar(Ni[1])<1.0e-32) { cols[1]=-1; }
        if( PetscAbsScalar(Ni[2])<1.0e-32) { cols[2]=-1; }
        if( PetscAbsScalar(Ni[3])<1.0e-32) { cols[3]=-1; }
        
        ierr = MatSetValues(mat,1,&row,4,cols,Ni,INSERT_VALUES);CHKERRQ(ierr); 
      }
    }
    ierr = PetscFree(xi);CHKERRQ(ierr);
    ierr = PetscFree(eta);CHKERRQ(ierr);
  }  
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
       Contributed by Andrei Draganescu <aidraga@sandia.gov>
*/
#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation_DA_2D_Q0"
PetscErrorCode DMCreateInterpolation_DA_2D_Q0(DM dac,DM daf,Mat *A)
{
  PetscErrorCode   ierr;
  PetscInt         i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof;
  PetscInt         m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,*dnz,*onz;
  PetscInt         row,col,i_start_ghost,j_start_ghost,cols[4],mx,m_c,my,nc,ratioi,ratioj;
  PetscInt         i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c,col_shift,col_scale;
  PetscMPIInt      size_c,size_f,rank_f;
  PetscScalar      v[4];
  Mat              mat;
  DMDABoundaryType bx,by;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,&bx,&by,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (bx == DMDA_BOUNDARY_PERIODIC) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Cannot handle periodic grid in x");
  if (by == DMDA_BOUNDARY_PERIODIC) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Cannot handle periodic grid in y");
  ratioi = mx/Mx;
  ratioj = my/My;
  if (ratioi*Mx != mx) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in x");
  if (ratioj*My != my) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in y");
  if (ratioi != 2) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Coarsening factor in x must be 2");
  if (ratioj != 2) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Coarsening factor in y must be 2");

  ierr = DMDAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /*
     Used for handling a coarse DMDA that lives on 1/4 the processors of the fine DMDA.
     The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the 
     processors). It's effective length is hence 4 times its normal length, this is
     why the col_scale is multiplied by the interpolation matrix column sizes.
     sol_shift allows each set of 1/4 processors do its own interpolation using ITS
     copy of the coarse vector. A bit of a hack but you do better.

     In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
  */
  ierr = MPI_Comm_size(((PetscObject)dac)->comm,&size_c);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)daf)->comm,&size_f);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)daf)->comm,&rank_f);CHKERRQ(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*(rank_f/size_c);

  ierr = MatPreallocateInitialize(((PetscObject)daf)->comm,m_f*n_f,col_scale*m_c*n_c,dnz,onz);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      if (j_c < j_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      nc = 0;
      /* one left and below; or we are right on it */
      col        = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc++] = col_shift + idx_c[col]/dof; 
      ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatCreate(((PetscObject)daf)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f*n_f,col_scale*m_c*n_c,mx*my,col_scale*Mx*My);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */
      nc = 0;
      /* one left and below; or we are right on it */
      col      = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc] = col_shift + idx_c[col]/dof; 
      v[nc++]  = 1.0;
     
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(13.0*m_f*n_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
       Contributed by Jianming Yang <jianming-yang@uiowa.edu>
*/
#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation_DA_3D_Q0"
PetscErrorCode DMCreateInterpolation_DA_3D_Q0(DM dac,DM daf,Mat *A)
{
  PetscErrorCode   ierr;
  PetscInt         i,j,l,i_start,j_start,l_start,m_f,n_f,p_f,Mx,My,Mz,*idx_f,dof;
  PetscInt         m_ghost,n_ghost,p_ghost,*idx_c,m_ghost_c,n_ghost_c,p_ghost_c,nc,*dnz,*onz;
  PetscInt         row,col,i_start_ghost,j_start_ghost,l_start_ghost,cols[8],mx,m_c,my,n_c,mz,p_c,ratioi,ratioj,ratiol;
  PetscInt         i_c,j_c,l_c,i_start_c,j_start_c,l_start_c,i_start_ghost_c,j_start_ghost_c,l_start_ghost_c,col_shift,col_scale;
  PetscMPIInt      size_c,size_f,rank_f;
  PetscScalar      v[8];
  Mat              mat;
  DMDABoundaryType bx,by,bz;
  
  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,0,&Mx,&My,&Mz,0,0,0,0,0,&bx,&by,&bz,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (bx == DMDA_BOUNDARY_PERIODIC) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Cannot handle periodic grid in x");
  if (by == DMDA_BOUNDARY_PERIODIC) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Cannot handle periodic grid in y");
  if (bz == DMDA_BOUNDARY_PERIODIC) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Cannot handle periodic grid in z");
  ratioi = mx/Mx;
  ratioj = my/My;
  ratiol = mz/Mz;
  if (ratioi*Mx != mx) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in x");
  if (ratioj*My != my) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in y");
  if (ratiol*Mz != mz) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in z");
  if (ratioi != 2 && ratioi != 1) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Coarsening factor in x must be 1 or 2");
  if (ratioj != 2 && ratioj != 1) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Coarsening factor in y must be 1 or 2");
  if (ratiol != 2 && ratiol != 1) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_WRONG,"Coarsening factor in z must be 1 or 2");

  ierr = DMDAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);
  /*
     Used for handling a coarse DMDA that lives on 1/4 the processors of the fine DMDA.
     The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the 
     processors). It's effective length is hence 4 times its normal length, this is
     why the col_scale is multiplied by the interpolation matrix column sizes.
     sol_shift allows each set of 1/4 processors do its own interpolation using ITS
     copy of the coarse vector. A bit of a hack but you do better.

     In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
  */
  ierr = MPI_Comm_size(((PetscObject)dac)->comm,&size_c);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)daf)->comm,&size_f);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)daf)->comm,&rank_f);CHKERRQ(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*Mz*(rank_f/size_c);

  ierr = MatPreallocateInitialize(((PetscObject)daf)->comm,m_f*n_f*p_f,col_scale*m_c*n_c*p_c,dnz,onz);CHKERRQ(ierr);
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
	/* convert to local "natural" numbering and then to PETSc global numbering */
	row    = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

	i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
	j_c = (j/ratioj);    /* coarse grid node below fine grid node */
	l_c = (l/ratiol);   

	if (l_c < l_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    l_start %D l_c %D l_start_ghost_c %D",l_start,l_c,l_start_ghost_c);
	if (j_c < j_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
	if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

	/* 
	   Only include those interpolation points that are truly 
	   nonzero. Note this is very important for final grid lines
	   in x and y directions; since they have no right/top neighbors
	*/
	nc = 0;
	/* one left and below; or we are right on it */
	col        = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
	cols[nc++] = col_shift + idx_c[col]/dof; 
	ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatCreate(((PetscObject)daf)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f*n_f*p_f,col_scale*m_c*n_c*p_c,mx*my*mz,col_scale*Mx*My*Mz);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
	/* convert to local "natural" numbering and then to PETSc global numbering */
	row    = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
	
	i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
	j_c = (j/ratioj);    /* coarse grid node below fine grid node */
	l_c = (l/ratiol);   
	nc = 0;
	/* one left and below; or we are right on it */
	col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
	cols[nc] = col_shift + idx_c[col]/dof; 
	v[nc++]  = 1.0;
     
	ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
      }
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(13.0*m_f*n_f*p_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation_DA_3D_Q1"
PetscErrorCode DMCreateInterpolation_DA_3D_Q1(DM dac,DM daf,Mat *A)
{
  PetscErrorCode   ierr;
  PetscInt         i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof,l;
  PetscInt         m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,Mz,mz;
  PetscInt         row,col,i_start_ghost,j_start_ghost,cols[8],mx,m_c,my,nc,ratioi,ratioj,ratiok;
  PetscInt         i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  PetscInt         l_start,p_f,l_start_ghost,p_ghost,l_start_c,p_c;
  PetscInt         l_start_ghost_c,p_ghost_c,l_c,*dnz,*onz;
  PetscScalar      v[8],x,y,z;
  Mat              mat;
  DMDABoundaryType bx,by,bz;
  
  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,0,&Mx,&My,&Mz,0,0,0,0,0,&bx,&by,&bz,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (mx == Mx) {
    ratioi = 1;
  } else if (bx == DMDA_BOUNDARY_PERIODIC) {
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (my == My) {
    ratioj = 1;
  } else if (by == DMDA_BOUNDARY_PERIODIC) {
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }
  if (mz == Mz) {
    ratiok = 1;
  } else if (bz == DMDA_BOUNDARY_PERIODIC) {
    ratiok = mz/Mz;
    if (ratiok*Mz != mz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mz/Mz  must be integer: mz %D Mz %D",mz,Mz);
  } else {
    ratiok = (mz-1)/(Mz-1);
    if (ratiok*(Mz-1) != mz-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mz - 1)/(Mz - 1) must be integer: mz %D Mz %D",mz,Mz);
  }
  
  ierr = DMDAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);
  
  /* create interpolation matrix, determining exact preallocation */
  ierr = MatPreallocateInitialize(((PetscObject)dac)->comm,m_f*n_f*p_f,m_c*n_c*p_c,dnz,onz);CHKERRQ(ierr);
  /* loop over local fine grid nodes counting interpolating points */
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
        i_c = (i/ratioi);
        j_c = (j/ratioj);
        l_c = (l/ratiok);
        if (l_c < l_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                            l_start %D l_c %D l_start_ghost_c %D",l_start,l_c,l_start_ghost_c);
        if (j_c < j_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                            j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
        if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                            i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);
        
        /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
         */
        nc       = 0;
        col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc++] = idx_c[col]/dof; 
        if (i_c*ratioi != i) { 
          cols[nc++] = idx_c[col+dof]/dof;
        }
        if (j_c*ratioj != j) { 
          cols[nc++] = idx_c[col+m_ghost_c*dof]/dof;
        }
        if (l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+m_ghost_c*n_ghost_c*dof]/dof;
        }
        if (j_c*ratioj != j && i_c*ratioi != i) { 
          cols[nc++] = idx_c[col+(m_ghost_c+1)*dof]/dof;
        }
        if (j_c*ratioj != j && l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof]/dof;
        }
        if (i_c*ratioi != i && l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof]/dof;
        }
        if (i_c*ratioi != i && l_c*ratiok != l && j_c*ratioj != j) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof]/dof;
        }
        ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatCreate(((PetscObject)dac)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m_f*n_f*p_f,m_c*n_c*p_c,mx*my*mz,Mx*My*Mz);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  
  /* loop over local fine grid nodes setting interpolation for those*/
  if (!NEWVERSION) {

    for (l=l_start; l<l_start+p_f; l++) {
      for (j=j_start; j<j_start+n_f; j++) {
        for (i=i_start; i<i_start+m_f; i++) {
          /* convert to local "natural" numbering and then to PETSc global numbering */
          row = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
          
          i_c = (i/ratioi);
          j_c = (j/ratioj);
          l_c = (l/ratiok);
          
        /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
         */
          x  = ((double)(i - i_c*ratioi))/((double)ratioi);
          y  = ((double)(j - j_c*ratioj))/((double)ratioj);
          z  = ((double)(l - l_c*ratiok))/((double)ratiok);

          nc = 0;
          /* one left and below; or we are right on it */
          col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c)+m_ghost_c*(j_c-j_start_ghost_c)+(i_c-i_start_ghost_c));
          
          cols[nc] = idx_c[col]/dof; 
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));
          
          if (i_c*ratioi != i) { 
            cols[nc] = idx_c[col+dof]/dof;
            v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));
          }
          
          if (j_c*ratioj != j) { 
            cols[nc] = idx_c[col+m_ghost_c*dof]/dof;
            v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
          }
          
          if (l_c*ratiok != l) { 
            cols[nc] = idx_c[col+m_ghost_c*n_ghost_c*dof]/dof;
            v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
          }
          
          if (j_c*ratioj != j && i_c*ratioi != i) { 
            cols[nc] = idx_c[col+(m_ghost_c+1)*dof]/dof;
            v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
          }
          
          if (j_c*ratioj != j && l_c*ratiok != l) { 
            cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof]/dof;
            v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
          }
          
          if (i_c*ratioi != i && l_c*ratiok != l) { 
            cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof]/dof;
            v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
          }
          
          if (i_c*ratioi != i && l_c*ratiok != l && j_c*ratioj != j) { 
            cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof]/dof;
            v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
          }
          ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
        }
      }
    }
    
  } else {
    PetscScalar    *xi,*eta,*zeta;
    PetscInt       li,nxi,lj,neta,lk,nzeta,n;
    PetscScalar    Ni[8];
    
    /* compute local coordinate arrays */
    nxi   = ratioi + 1;
    neta  = ratioj + 1;
    nzeta = ratiok + 1;
    ierr = PetscMalloc(sizeof(PetscScalar)*nxi,&xi);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*neta,&eta);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*nzeta,&zeta);CHKERRQ(ierr);
    for (li=0; li<nxi; li++) {
      xi[li] = -1.0 + (PetscScalar)li*(2.0/(PetscScalar)(nxi-1));
    }
    for (lj=0; lj<neta; lj++) {
      eta[lj] = -1.0 + (PetscScalar)lj*(2.0/(PetscScalar)(neta-1));
    }
    for (lk=0; lk<nzeta; lk++) {
      zeta[lk] = -1.0 + (PetscScalar)lk*(2.0/(PetscScalar)(nzeta-1));
    }
    
    for (l=l_start; l<l_start+p_f; l++) {
      for (j=j_start; j<j_start+n_f; j++) {
        for (i=i_start; i<i_start+m_f; i++) {
          /* convert to local "natural" numbering and then to PETSc global numbering */
          row = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
          
          i_c = (i/ratioi);
          j_c = (j/ratioj);
          l_c = (l/ratiok);

          /* remainders */
          li = i - ratioi * (i/ratioi);
          if (i==mx-1){ li = nxi-1; }
          lj = j - ratioj * (j/ratioj);
          if (j==my-1){ lj = neta-1; }
          lk = l - ratiok * (l/ratiok);
          if (l==mz-1){ lk = nzeta-1; }
          
          /* corners */
          col     = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c)+m_ghost_c*(j_c-j_start_ghost_c)+(i_c-i_start_ghost_c));
          cols[0] = idx_c[col]/dof; 
          Ni[0]   = 1.0;
          if ( (li==0) || (li==nxi-1) ) {
            if ( (lj==0) || (lj==neta-1) ) {
              if ( (lk==0) || (lk==nzeta-1) ) {
                ierr = MatSetValue(mat,row,cols[0],Ni[0],INSERT_VALUES);CHKERRQ(ierr);
                continue;
              }
            }
          }
          
          /* edges + interior */
          /* remainders */
          if (i==mx-1){ i_c--; }
          if (j==my-1){ j_c--; }
          if (l==mz-1){ l_c--; }
          
          col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
          cols[0] = idx_c[col]/dof; /* one left and below; or we are right on it */
          cols[1] = idx_c[col+dof]/dof; /* one right and below */
          cols[2] = idx_c[col+m_ghost_c*dof]/dof;  /* one left and above */
          cols[3] = idx_c[col+(m_ghost_c+1)*dof]/dof; /* one right and above */

          cols[4] = idx_c[col+m_ghost_c*n_ghost_c*dof]/dof; /* one left and below and front; or we are right on it */
          cols[5] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof]/dof; /* one right and below, and front */
          cols[6] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof]/dof;/* one left and above and front*/
          cols[7] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof]/dof; /* one right and above and front */

          Ni[0] = 0.125*(1.0-xi[li])*(1.0-eta[lj])*(1.0-zeta[lk]);
          Ni[1] = 0.125*(1.0+xi[li])*(1.0-eta[lj])*(1.0-zeta[lk]);
          Ni[2] = 0.125*(1.0-xi[li])*(1.0+eta[lj])*(1.0-zeta[lk]);
          Ni[3] = 0.125*(1.0+xi[li])*(1.0+eta[lj])*(1.0-zeta[lk]);

          Ni[4] = 0.125*(1.0-xi[li])*(1.0-eta[lj])*(1.0+zeta[lk]);
          Ni[5] = 0.125*(1.0+xi[li])*(1.0-eta[lj])*(1.0+zeta[lk]);
          Ni[6] = 0.125*(1.0-xi[li])*(1.0+eta[lj])*(1.0+zeta[lk]);
          Ni[7] = 0.125*(1.0+xi[li])*(1.0+eta[lj])*(1.0+zeta[lk]);

          for (n=0; n<8; n++) {
            if( PetscAbsScalar(Ni[n])<1.0e-32) { cols[n]=-1; }
          }
          ierr = MatSetValues(mat,1,&row,8,cols,Ni,INSERT_VALUES);CHKERRQ(ierr); 
          
        }
      }
    }
    ierr = PetscFree(xi);CHKERRQ(ierr);
    ierr = PetscFree(eta);CHKERRQ(ierr);
    ierr = PetscFree(zeta);CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInterpolation_DA"
PetscErrorCode  DMCreateInterpolation_DA(DM dac,DM daf,Mat *A,Vec *scale)
{
  PetscErrorCode   ierr;
  PetscInt         dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DMDABoundaryType bxc,byc,bzc,bxf,byf,bzf;
  DMDAStencilType  stc,stf;
  DM_DA            *ddc = (DM_DA*)dac->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DM_CLASSID,1);
  PetscValidHeaderSpecific(daf,DM_CLASSID,2);
  PetscValidPointer(A,3);
  if (scale) PetscValidPointer(scale,4);

  ierr = DMDAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&bxc,&byc,&bzc,&stc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&bxf,&byf,&bzf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Dimensions of DMDA do not match %D %D",dimc,dimf);CHKERRQ(ierr);
  if (dofc != doff) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"DOF of DMDA do not match %D %D",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Stencil width of DMDA do not match %D %D",sc,sf);CHKERRQ(ierr);
  if (bxc != bxf || byc != byf || bzc != bzf) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Boundary type different in two DMDAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Stencil type different in two DMDAs");CHKERRQ(ierr);
  if (Mc < 2 && Mf > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in x direction");
  if (dimc > 1 && Nc < 2 && Nf > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in y direction");
  if (dimc > 2 && Pc < 2 && Pf > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in z direction");

  if (ddc->interptype == DMDA_Q1){
    if (dimc == 1){
      ierr = DMCreateInterpolation_DA_1D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 2){
      ierr = DMCreateInterpolation_DA_2D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 3){
      ierr = DMCreateInterpolation_DA_3D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_SUP,"No support for this DMDA dimension %D for interpolation type %d",dimc,(int)ddc->interptype);
  } else if (ddc->interptype == DMDA_Q0){
    if (dimc == 1){
      ierr = DMCreateInterpolation_DA_1D_Q0(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 2){
       ierr = DMCreateInterpolation_DA_2D_Q0(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 3){
       ierr = DMCreateInterpolation_DA_3D_Q0(dac,daf,A);CHKERRQ(ierr);
    } else SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_SUP,"No support for this DMDA dimension %D for interpolation type %d",dimc,(int)ddc->interptype);
  }
  if (scale) {
    ierr = DMCreateInterpolationScale((DM)dac,(DM)daf,*A,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInjection_DA_1D"
PetscErrorCode DMCreateInjection_DA_1D(DM dac,DM daf,VecScatter *inject)
{
    PetscErrorCode   ierr;
    PetscInt         i,i_start,m_f,Mx,*idx_f,dof;
    PetscInt         m_ghost,*idx_c,m_ghost_c;
    PetscInt         row,i_start_ghost,mx,m_c,nc,ratioi;
    PetscInt         i_start_c,i_start_ghost_c;
    PetscInt         *cols;
    DMDABoundaryType bx;
    Vec              vecf,vecc;
    IS               isf;
    
    PetscFunctionBegin;
    ierr = DMDAGetInfo(dac,0,&Mx,0,0,0,0,0,0,0,&bx,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetInfo(daf,0,&mx,0,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
    if (bx == DMDA_BOUNDARY_PERIODIC) {
        ratioi = mx/Mx;
        if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
    } else {
        ratioi = (mx-1)/(Mx-1);
        if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
    }
   
    ierr = DMDAGetCorners(daf,&i_start,0,0,&m_f,0,0);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(daf,&i_start_ghost,0,0,&m_ghost,0,0);CHKERRQ(ierr);
    ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);
    
    ierr = DMDAGetCorners(dac,&i_start_c,0,0,&m_c,0,0);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,0,0,&m_ghost_c,0,0);CHKERRQ(ierr);
    ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);
    
    
    /* loop over local fine grid nodes setting interpolation for those*/
    nc = 0;
    ierr = PetscMalloc(m_f*sizeof(PetscInt),&cols);CHKERRQ(ierr);
   
   
    for (i=i_start_c; i<i_start_c+m_c; i++) {
        PetscInt i_f = i*ratioi;

           if (i_f < i_start_ghost || i_f >= i_start_ghost+m_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
 i_c %D i_f %D fine ghost range [%D,%D]",i,i_f,i_start_ghost,i_start_ghost+m_ghost);
            row = idx_f[dof*(i_f-i_start_ghost)];
            cols[nc++] = row/dof;
    }
   

    ierr = ISCreateBlock(((PetscObject)daf)->comm,dof,nc,cols,PETSC_OWN_POINTER,&isf);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dac,&vecc);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(daf,&vecf);CHKERRQ(ierr);
    ierr = VecScatterCreate(vecf,isf,vecc,PETSC_NULL,inject);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dac,&vecc);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(daf,&vecf);CHKERRQ(ierr);
    ierr = ISDestroy(&isf);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInjection_DA_2D"
PetscErrorCode DMCreateInjection_DA_2D(DM dac,DM daf,VecScatter *inject)
{
  PetscErrorCode   ierr;
  PetscInt         i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof;
  PetscInt         m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c;
  PetscInt         row,i_start_ghost,j_start_ghost,mx,m_c,my,nc,ratioi,ratioj;
  PetscInt         i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  PetscInt         *cols;
  DMDABoundaryType bx,by;
  Vec              vecf,vecc;
  IS               isf;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,&bx,&by,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);
  if (bx == DMDA_BOUNDARY_PERIODIC) {
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (by == DMDA_BOUNDARY_PERIODIC) {
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }

  ierr = DMDAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);


  /* loop over local fine grid nodes setting interpolation for those*/
  nc = 0;
  ierr = PetscMalloc(n_f*m_f*sizeof(PetscInt),&cols);CHKERRQ(ierr);
  for (j=j_start_c; j<j_start_c+n_c; j++) {
    for (i=i_start_c; i<i_start_c+m_c; i++) {
      PetscInt i_f = i*ratioi,j_f = j*ratioj;
      if (j_f < j_start_ghost || j_f >= j_start_ghost+n_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    j_c %D j_f %D fine ghost range [%D,%D]",j,j_f,j_start_ghost,j_start_ghost+n_ghost);
      if (i_f < i_start_ghost || i_f >= i_start_ghost+m_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    i_c %D i_f %D fine ghost range [%D,%D]",i,i_f,i_start_ghost,i_start_ghost+m_ghost);
      row = idx_f[dof*(m_ghost*(j_f-j_start_ghost) + (i_f-i_start_ghost))];
      cols[nc++] = row/dof;
    }
  }

  ierr = ISCreateBlock(((PetscObject)daf)->comm,dof,nc,cols,PETSC_OWN_POINTER,&isf);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = VecScatterCreate(vecf,isf,vecc,PETSC_NULL,inject);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = ISDestroy(&isf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInjection_DA_3D"
PetscErrorCode DMCreateInjection_DA_3D(DM dac,DM daf,VecScatter *inject)
{
  PetscErrorCode   ierr;
  PetscInt         i,j,k,i_start,j_start,k_start,m_f,n_f,p_f,Mx,My,Mz;
  PetscInt         m_ghost,n_ghost,p_ghost,m_ghost_c,n_ghost_c,p_ghost_c;
  PetscInt         i_start_ghost,j_start_ghost,k_start_ghost;
  PetscInt         mx,my,mz,ratioi,ratioj,ratiok;
  PetscInt         i_start_c,j_start_c,k_start_c;
  PetscInt         m_c,n_c,p_c;
  PetscInt         i_start_ghost_c,j_start_ghost_c,k_start_ghost_c;
  PetscInt         row,nc,dof;
  PetscInt         *idx_c,*idx_f;
  PetscInt         *cols;
  DMDABoundaryType bx,by,bz;
  Vec              vecf,vecc;
  IS               isf;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,0,&Mx,&My,&Mz,0,0,0,0,0,&bx,&by,&bz,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,0,&mx,&my,&mz,0,0,0,&dof,0,0,0,0,0);CHKERRQ(ierr);

  if (bx == DMDA_BOUNDARY_PERIODIC){
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (by == DMDA_BOUNDARY_PERIODIC){
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }
  if (bz == DMDA_BOUNDARY_PERIODIC){
    ratiok = mz/Mz;
    if (ratiok*Mz != mz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mz/Mz  must be integer: mz %D My %D",mz,Mz);
  } else {
    ratiok = (mz-1)/(Mz-1);
    if (ratiok*(Mz-1) != mz-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mz - 1)/(Mz - 1) must be integer: mz %D Mz %D",mz,Mz);
  }

  ierr = DMDAGetCorners(daf,&i_start,&j_start,&k_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&k_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,&k_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&k_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);


  /* loop over local fine grid nodes setting interpolation for those*/
  nc = 0;
  ierr = PetscMalloc(n_f*m_f*p_f*sizeof(PetscInt),&cols);CHKERRQ(ierr);
  for (k=k_start_c; k<k_start_c+p_c; k++) {
    for (j=j_start_c; j<j_start_c+n_c; j++) {
      for (i=i_start_c; i<i_start_c+m_c; i++) {
        PetscInt i_f = i*ratioi,j_f = j*ratioj,k_f = k*ratiok;
        if (k_f < k_start_ghost || k_f >= k_start_ghost+p_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA  "
                                                                          "k_c %D k_f %D fine ghost range [%D,%D]",k,k_f,k_start_ghost,k_start_ghost+p_ghost);
        if (j_f < j_start_ghost || j_f >= j_start_ghost+n_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA  "
                                                                          "j_c %D j_f %D fine ghost range [%D,%D]",j,j_f,j_start_ghost,j_start_ghost+n_ghost);
        if (i_f < i_start_ghost || i_f >= i_start_ghost+m_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA  "
                                                                          "i_c %D i_f %D fine ghost range [%D,%D]",i,i_f,i_start_ghost,i_start_ghost+m_ghost);
        row = idx_f[dof*(m_ghost*n_ghost*(k_f-k_start_ghost) + m_ghost*(j_f-j_start_ghost) + (i_f-i_start_ghost))];
        cols[nc++] = row/dof;
      }
    }
  }

  ierr = ISCreateBlock(((PetscObject)daf)->comm,dof,nc,cols,PETSC_OWN_POINTER,&isf);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = VecScatterCreate(vecf,isf,vecc,PETSC_NULL,inject);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = ISDestroy(&isf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateInjection_DA"
PetscErrorCode  DMCreateInjection_DA(DM dac,DM daf,VecScatter *inject)
{
  PetscErrorCode   ierr;
  PetscInt         dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DMDABoundaryType bxc,byc,bzc,bxf,byf,bzf;
  DMDAStencilType  stc,stf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DM_CLASSID,1);
  PetscValidHeaderSpecific(daf,DM_CLASSID,2);
  PetscValidPointer(inject,3);

  ierr = DMDAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&bxc,&byc,&bzc,&stc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&bxf,&byf,&bzf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Dimensions of DMDA do not match %D %D",dimc,dimf);CHKERRQ(ierr);
  if (dofc != doff) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"DOF of DMDA do not match %D %D",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Stencil width of DMDA do not match %D %D",sc,sf);CHKERRQ(ierr);
  if (bxc != bxf || byc != byf || bzc != bzf) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Boundary type different in two DMDAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Stencil type different in two DMDAs");CHKERRQ(ierr);
  if (Mc < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in x direction");
  if (dimc > 1 && Nc < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in y direction");
  if (dimc > 2 && Pc < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in z direction");

  if (dimc == 1){
    ierr = DMCreateInjection_DA_1D(dac,daf,inject);CHKERRQ(ierr);
  } else if (dimc == 2) {
    ierr = DMCreateInjection_DA_2D(dac,daf,inject);CHKERRQ(ierr);
  } else if (dimc == 3) {
    ierr = DMCreateInjection_DA_3D(dac,daf,inject);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMCreateAggregates_DA"
PetscErrorCode  DMCreateAggregates_DA(DM dac,DM daf,Mat *rest)
{
  PetscErrorCode   ierr;
  PetscInt         dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc;
  PetscInt         dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DMDABoundaryType bxc,byc,bzc,bxf,byf,bzf;
  DMDAStencilType  stc,stf;
  PetscInt         i,j,l;
  PetscInt         i_start,j_start,l_start, m_f,n_f,p_f;
  PetscInt         i_start_ghost,j_start_ghost,l_start_ghost,m_ghost,n_ghost,p_ghost;
  PetscInt         *idx_f;
  PetscInt         i_c,j_c,l_c;
  PetscInt         i_start_c,j_start_c,l_start_c, m_c,n_c,p_c;
  PetscInt         i_start_ghost_c,j_start_ghost_c,l_start_ghost_c,m_ghost_c,n_ghost_c,p_ghost_c;
  PetscInt         *idx_c;
  PetscInt         d;
  PetscInt         a;
  PetscInt         max_agg_size;
  PetscInt         *fine_nodes;
  PetscScalar      *one_vec;
  PetscInt         fn_idx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DM_CLASSID,1);
  PetscValidHeaderSpecific(daf,DM_CLASSID,2);
  PetscValidPointer(rest,3);

  ierr = DMDAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&bxc,&byc,&bzc,&stc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&bxf,&byf,&bzf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Dimensions of DMDA do not match %D %D",dimc,dimf);CHKERRQ(ierr);
  if (dofc != doff) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"DOF of DMDA do not match %D %D",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Stencil width of DMDA do not match %D %D",sc,sf);CHKERRQ(ierr);
  if (bxc != bxf || byc != byf || bzc != bzf) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Boundary type different in two DMDAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(((PetscObject)daf)->comm,PETSC_ERR_ARG_INCOMP,"Stencil type different in two DMDAs");CHKERRQ(ierr);

  if( Mf < Mc ) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Coarse grid has more points than fine grid, Mc %D, Mf %D", Mc, Mf);
  if( Nf < Nc ) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Coarse grid has more points than fine grid, Nc %D, Nf %D", Nc, Nf);
  if( Pf < Pc ) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Coarse grid has more points than fine grid, Pc %D, Pf %D", Pc, Pf);

  if (Pc < 0) Pc = 1;
  if (Pf < 0) Pf = 1;
  if (Nc < 0) Nc = 1;
  if (Nf < 0) Nf = 1;

  ierr = DMDAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* 
     Basic idea is as follows. Here's a 2D example, suppose r_x, r_y are the ratios
     for dimension 1 and 2 respectively.
     Let (i,j) be a coarse grid node. All the fine grid nodes between r_x*i and r_x*(i+1)
     and r_y*j and r_y*(j+1) will be grouped into the same coarse grid agregate.
     Each specific dof on the fine grid is mapped to one dof on the coarse grid.
  */

  max_agg_size = (Mf/Mc+1)*(Nf/Nc+1)*(Pf/Pc+1);

  /* create the matrix that will contain the restriction operator */
  ierr = MatCreateAIJ( ((PetscObject)daf)->comm, m_c*n_c*p_c*dofc, m_f*n_f*p_f*doff, Mc*Nc*Pc*dofc, Mf*Nf*Pf*doff,
			  max_agg_size, PETSC_NULL, max_agg_size, PETSC_NULL, rest);CHKERRQ(ierr);

  /* store nodes in the fine grid here */
  ierr = PetscMalloc2(max_agg_size,PetscScalar, &one_vec,max_agg_size,PetscInt, &fine_nodes);CHKERRQ(ierr);
  for(i=0; i<max_agg_size; i++) one_vec[i] = 1.0;  
  
  /* loop over all coarse nodes */
  for (l_c=l_start_c; l_c<l_start_c+p_c; l_c++) {
    for (j_c=j_start_c; j_c<j_start_c+n_c; j_c++) {
      for (i_c=i_start_c; i_c<i_start_c+m_c; i_c++) {
	for(d=0; d<dofc; d++) {
	  /* convert to local "natural" numbering and then to PETSc global numbering */
	  a = idx_c[dofc*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c))] + d;

	  fn_idx = 0;
	  /* Corresponding fine points are all points (i_f, j_f, l_f) such that
	     i_c*Mf/Mc <= i_f < (i_c+1)*Mf/Mc
	     (same for other dimensions)
	  */
	  for (l=l_c*Pf/Pc; l<PetscMin((l_c+1)*Pf/Pc,Pf); l++) {
	    for (j=j_c*Nf/Nc; j<PetscMin((j_c+1)*Nf/Nc,Nf); j++) {
	      for (i=i_c*Mf/Mc; i<PetscMin((i_c+1)*Mf/Mc,Mf); i++) {
		fine_nodes[fn_idx] = idx_f[doff*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))] + d;
		fn_idx++;
	      }
	    }
	  }
	  /* add all these points to one aggregate */
	  ierr = MatSetValues(*rest, 1, &a, fn_idx, fine_nodes, one_vec, INSERT_VALUES);CHKERRQ(ierr);
	}
      }
    }
  }
  ierr = PetscFree2(one_vec,fine_nodes);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*rest, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*rest, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
