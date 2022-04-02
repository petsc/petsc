static char help[] = "Test coloring for finite difference Jacobians with DMStag\n\n";

#include <petscdm.h>
#include <petscdmstag.h>
#include <petscsnes.h>

/*
   Note that DMStagGetValuesStencil and DMStagSetValuesStencil are inefficient,
   compared to DMStagVecGetArray() and friends, and only used here for testing
   purposes, as they allow the code for the Jacobian and residual functions to
   be more similar. In the intended application, where users are not writing
   their own Jacobian assembly routines, one should use the faster, array-based
   approach.
*/

/* A "diagonal" objective function which only couples dof living at the same "point" */
PetscErrorCode FormFunction1DNoCoupling(SNES snes, Vec x, Vec f, void *ctx)
{
  PetscInt          start,n,n_extra,N,dof[2];
  Vec               x_local;
  DM                dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N,NULL,NULL));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL));
  for (PetscInt e=start; e<start+n+n_extra; ++e) {
    for (PetscInt c=0; c<dof[0]; ++c) {
      DMStagStencil row;
      PetscScalar   x_val,val;

      row.i = e;
      row.loc = DMSTAG_LEFT;
      row.c = c;
      PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
      val = (10.0 + c) * x_val * x_val * x_val;  // f_i = (10 +c) * x_i^3
      PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
    }
    if (e < N) {
      for (PetscInt c=0; c<dof[1]; ++c) {
        DMStagStencil row;
        PetscScalar   x_val,val;

        row.i = e;
        row.loc = DMSTAG_ELEMENT;
        row.c = c;
        PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
        val = (20.0 + c) * x_val * x_val * x_val;  // f_i = (20 + c) * x_i^3
        PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian1DNoCoupling(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx)
{
  PetscInt       start,n,n_extra,N,dof[2];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N,NULL,NULL));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL));
  for (PetscInt e=start; e<start+n+n_extra; ++e) {
    for (PetscInt c=0; c<dof[0]; ++c) {
      DMStagStencil row_vertex;
      PetscScalar   x_val, val;

      row_vertex.i = e;
      row_vertex.loc = DMSTAG_LEFT;
      row_vertex.c = c;
      PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row_vertex,&x_val));
      val = 3.0 * (10.0 + c) * x_val * x_val;
      PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row_vertex,1,&row_vertex,&val,INSERT_VALUES));
    }
    if (e < N) {
      for (PetscInt c=0; c<dof[1]; ++c) {
        DMStagStencil row_element;
        PetscScalar   x_val,val;

        row_element.i = e;
        row_element.loc = DMSTAG_ELEMENT;
        row_element.c = c;
        PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row_element,&x_val));
        val = 3.0 * (20.0 + c) * x_val * x_val;
        PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row_element,1,&row_element,&val,INSERT_VALUES));
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));

  PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  PetscCheck(Amat == Pmat,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

/* Objective functions which use the DM's stencil width. */
PetscErrorCode FormFunction1D(SNES snes,Vec x,Vec f,void *ctx)
{
  Vec               x_local;
  PetscInt          dim,stencil_width,start,n,n_extra,N,dof[2];
  DMStagStencilType stencil_type;
  DM                dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim == 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"DM dimension must be 1");
  PetscCall(DMStagGetStencilType(dm,&stencil_type));
  PetscCheckFalse(stencil_type != DMSTAG_STENCIL_STAR && stencil_type != DMSTAG_STENCIL_BOX,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only star and box stencils supported");
  PetscCall(DMStagGetStencilWidth(dm,&stencil_width));

  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N,NULL,NULL));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL));

  PetscCall(VecZeroEntries(f));

  for (PetscInt e=start; e<start+n+n_extra; ++e) {
    DMStagStencil row_vertex,row_element;

    row_vertex.i = e;
    row_vertex.loc = DMSTAG_LEFT;

    row_element.i = e;
    row_element.loc = DMSTAG_ELEMENT;

    for (PetscInt offset=-stencil_width; offset<=stencil_width; ++offset) {
      const PetscInt e_offset = e + offset;

      // vertex --> vertex
      if (e_offset >=0 && e_offset < N+1) { // Does not fully wrap in the periodic case
        DMStagStencil col;
        PetscScalar   x_val,val;

        for (PetscInt c_row=0; c_row<dof[0]; ++c_row) {
          row_vertex.c = c_row;
          for (PetscInt c_col=0; c_col<dof[0]; ++c_col) {
            col.c = c_col;
            col.i = e_offset;
            col.loc = DMSTAG_LEFT;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col,&x_val));
            val = (10.0 + offset) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row_vertex,&val,ADD_VALUES));
          }
        }
      }

      // element --> vertex
      if (e_offset >=0 && e_offset < N) { // Does not fully wrap in the periodic case
        DMStagStencil col;
        PetscScalar   x_val,val;

        for (PetscInt c_row=0; c_row<dof[0]; ++c_row) {
          row_vertex.c = c_row;
          for (PetscInt c_col=0; c_col<dof[1]; ++c_col) {
            col.c = c_col;
            col.i = e_offset;
            col.loc = DMSTAG_ELEMENT;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col,&x_val));
            val = (15.0 + offset) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row_vertex,&val,ADD_VALUES));
          }
        }
      }

      if (e < N) {
        // vertex --> element
        if (e_offset >=0 && e_offset < N+1) { // Does not fully wrap in the periodic case
          DMStagStencil col;
          PetscScalar   x_val,val;

          for (PetscInt c_row=0; c_row<dof[1]; ++c_row) {
            row_element.c = c_row;
            for (PetscInt c_col=0; c_col<dof[0]; ++c_col) {
              col.c = c_col;
              col.i = e_offset;
              col.loc = DMSTAG_LEFT;
              PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col,&x_val));
              val = (25.0 + offset) * x_val * x_val * x_val;
              PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row_element,&val,ADD_VALUES));
            }
          }
        }

        // element --> element
        if (e_offset >=0 && e_offset < N) { // Does not fully wrap in the periodic case
          DMStagStencil col;
          PetscScalar   x_val,val;

          for (PetscInt c_row=0; c_row<dof[1]; ++c_row) {
            row_element.c = c_row;
            for (PetscInt c_col=0; c_col<dof[1]; ++c_col) {
              col.c = c_col;
              col.i = e_offset;
              col.loc = DMSTAG_ELEMENT;
              PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col,&x_val));
              val = (20.0 + offset) * x_val * x_val * x_val;
              PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row_element,&val,ADD_VALUES));
            }
          }
        }

      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian1D(SNES snes, Vec x, Mat Amat, Mat Pmat, void *ctx)
{
  Vec            x_local;
  PetscInt       dim,stencil_width,start,n,n_extra,N,dof[2];
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim == 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"DM dimension must be 1");
  PetscCall(DMStagGetStencilWidth(dm,&stencil_width));

  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N,NULL,NULL));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL));

  PetscCall(MatZeroEntries(Amat));

  for (PetscInt e=start; e<start+n+n_extra; ++e) {
    DMStagStencil row_vertex,row_element;

    row_vertex.i = e;
    row_vertex.loc = DMSTAG_LEFT;

    row_element.i = e;
    row_element.loc = DMSTAG_ELEMENT;

    for (PetscInt offset=-stencil_width; offset<=stencil_width; ++offset) {
      const PetscInt e_offset = e + offset;

      // vertex --> vertex
      if (e_offset >=0 && e_offset < N+1) {
        DMStagStencil col;
        PetscScalar   x_val,val;

        for (PetscInt c_row=0; c_row<dof[0]; ++c_row) {
          row_vertex.c = c_row;
          for (PetscInt c_col=0; c_col<dof[0]; ++c_col) {
            col.c = c_col;
            col.i = e_offset;
            col.loc = DMSTAG_LEFT;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col,&x_val));
            val = 3.0 * (10.0 + offset) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row_vertex,1,&col,&val,ADD_VALUES));
          }
        }
      }

      // element --> vertex
      if (e_offset >=0 && e_offset < N) {
        DMStagStencil col;
        PetscScalar   x_val,val;

        for (PetscInt c_row=0; c_row<dof[0]; ++c_row) {
          row_vertex.c = c_row;
          for (PetscInt c_col=0; c_col<dof[1]; ++c_col) {
            col.c = c_col;
            col.i = e_offset;
            col.loc = DMSTAG_ELEMENT;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col,&x_val));
            val = 3.0 * (15.0 + offset) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row_vertex,1,&col,&val,ADD_VALUES));
          }
        }
      }

      if (e < N) {
        // vertex --> element
        if (e_offset >=0 && e_offset < N+1) {
          DMStagStencil col;
          PetscScalar   x_val,val;

          for (PetscInt c_row=0; c_row<dof[1]; ++c_row) {
            row_element.c = c_row;
            for (PetscInt c_col=0; c_col<dof[0]; ++c_col) {
              col.c = c_col;
              col.i = e_offset;
              col.loc = DMSTAG_LEFT;
              PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col,&x_val));
              val = 3.0 * (25.0 + offset) * x_val * x_val;
              PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row_element,1,&col,&val,ADD_VALUES));
            }
          }
        }

        // element --> element
        if (e_offset >=0 && e_offset < N) {
          DMStagStencil col;
          PetscScalar   x_val,val;

          for (PetscInt c_row=0; c_row<dof[1]; ++c_row) {
            row_element.c = c_row;
            for (PetscInt c_col=0; c_col<dof[1]; ++c_col) {
              col.c = c_col;
              col.i = e_offset;
              col.loc = DMSTAG_ELEMENT;
              PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col,&x_val));
              val = 3.0 * (20.0 + offset) * x_val * x_val;
              PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row_element,1,&col,&val,ADD_VALUES));
            }
          }
        }
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  PetscCheck(Amat == Pmat,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction2DNoCoupling(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscInt       start[2],n[2],n_extra[2],N[2],dof[3];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,&n_extra[0],&n_extra[1],NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],NULL));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL));
  for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
    for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
      for (PetscInt c=0; c<dof[0]; ++c) {
        DMStagStencil row;
        PetscScalar   x_val,val;

        row.i = ex;
        row.j = ey;
        row.loc = DMSTAG_DOWN_LEFT;
        row.c = c;
        PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
        val = (5.0 + c) * x_val * x_val * x_val;
        PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
      }
      if (ex < N[0]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_DOWN;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = (10.0 + c) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
        }
      }
      if (ey < N[1]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_LEFT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = (15.0 + c) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
        }
      }
      if (ex < N[0] && ey < N[1]) {
        for (PetscInt c=0; c<dof[2]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_ELEMENT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = (20.0 + c) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian2DNoCoupling(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx)
{
  PetscInt       start[2],n[2],n_extra[2],N[2],dof[3];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,&n_extra[0],&n_extra[1],NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],NULL));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL));
  for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
    for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
      for (PetscInt c=0; c<dof[0]; ++c) {
        DMStagStencil row;
        PetscScalar   x_val,val;

        row.i = ex;
        row.j = ey;
        row.loc = DMSTAG_DOWN_LEFT;
        row.c = c;
        PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
        val = 3.0 * (5.0 + c) * x_val * x_val;
        PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
      }
      if (ex < N[0]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_DOWN;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = 3.0 * (10.0 + c) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
        }
      }
      if (ey < N[1]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_LEFT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = 3.0 * (15.0 + c) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
        }
      }
      if (ex < N[0] && ey < N[1]) {
        for (PetscInt c=0; c<dof[2]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_ELEMENT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = 3.0 * (20.0 + c) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));

  PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  PetscCheck(Amat == Pmat,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction2D(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscInt       start[2],n[2],n_extra[2],N[2],dof[3];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,&n_extra[0],&n_extra[1],NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],NULL));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL));

  PetscCall(VecZeroEntries(f));

  /* First, as in the simple case above */
  for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
    for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
      for (PetscInt c=0; c<dof[0]; ++c) {
        DMStagStencil row;
        PetscScalar   x_val,val;

        row.i = ex;
        row.j = ey;
        row.loc = DMSTAG_DOWN_LEFT;
        row.c = c;
        PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
        val = (5.0 + c) * x_val * x_val * x_val;
        PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
      }
      if (ex < N[0]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_DOWN;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = (10.0 + c) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
        }
      }
      if (ey < N[1]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_LEFT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = (15.0 + c) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
        }
      }
      if (ex < N[0] && ey < N[1]) {
        for (PetscInt c=0; c<dof[2]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_ELEMENT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = (20.0 + c) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
        }
      }
    }
  }

  /* Add additional terms fully coupling one interior element to another */
  {
    PetscMPIInt   rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
    if (rank == 0) {
      PetscInt      epe;
      DMStagStencil *row,*col;

      PetscCall(DMStagGetEntriesPerElement(dm,&epe));
      PetscCall(PetscMalloc1(epe,&row));
      PetscCall(PetscMalloc1(epe,&col));
      for (PetscInt i=0; i<epe; ++i) {
        row[i].i = 0;
        row[i].j = 0;
        col[i].i = 0;
        col[i].j = 1;
      }
      {
        PetscInt nrows = 0;

        for (PetscInt c=0; c<dof[0]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_DOWN_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_DOWN;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_ELEMENT;
          ++nrows;
        }
      }

      {
        PetscInt ncols = 0;

        for (PetscInt c=0; c<dof[0]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_DOWN_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_DOWN;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_ELEMENT;
          ++ncols;
        }
      }

      for (PetscInt i=0; i<epe; ++i) {
        for (PetscInt j=0; j<epe; ++j) {
          PetscScalar x_val,val;

          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col[j],&x_val));
          val = (10*i + j) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row[i],&val,ADD_VALUES));
        }
      }
      PetscCall(PetscFree(row));
      PetscCall(PetscFree(col));
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian2D(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx)
{
  PetscInt       start[2],n[2],n_extra[2],N[2],dof[3];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,&n_extra[0],&n_extra[1],NULL));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],NULL));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL));
  PetscCall(MatZeroEntries(Amat));
  for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
    for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
      for (PetscInt c=0; c<dof[0]; ++c) {
        DMStagStencil row;
        PetscScalar   x_val,val;

        row.i = ex;
        row.j = ey;
        row.loc = DMSTAG_DOWN_LEFT;
        row.c = c;
        PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
        val = 3.0 * (5.0 + c) * x_val * x_val;
        PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
      }
      if (ex < N[0]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_DOWN;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = 3.0 * (10.0 + c) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
        }
      }
      if (ey < N[1]) {
        for (PetscInt c=0; c<dof[1]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_LEFT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = 3.0 * (15.0 + c) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
        }
      }
      if (ex < N[0] && ey < N[1]) {
        for (PetscInt c=0; c<dof[2]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.loc = DMSTAG_ELEMENT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = 3.0 * (20.0 + c) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
        }
      }
    }
  }

  /* Add additional terms fully coupling one interior element to another */
  {
    PetscMPIInt   rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
    if (rank == 0) {
      PetscInt      epe;
      DMStagStencil *row,*col;

      PetscCall(DMStagGetEntriesPerElement(dm,&epe));
      PetscCall(PetscMalloc1(epe,&row));
      PetscCall(PetscMalloc1(epe,&col));
      for (PetscInt i=0; i<epe; ++i) {
        row[i].i = 0;
        row[i].j = 0;
        col[i].i = 0;
        col[i].j = 1;
      }
      {
        PetscInt nrows = 0;

        for (PetscInt c=0; c<dof[0]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_DOWN_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_DOWN;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_ELEMENT;
          ++nrows;
        }
      }

      {
        PetscInt ncols = 0;

        for (PetscInt c=0; c<dof[0]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_DOWN_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_DOWN;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_ELEMENT;
          ++ncols;
        }
      }

      for (PetscInt i=0; i<epe; ++i) {
        for (PetscInt j=0; j<epe; ++j) {
          PetscScalar x_val,val;

          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col[j],&x_val));
          val = 3.0 * (10*i + j) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row[i],1,&col[j],&val,ADD_VALUES));
        }
      }
      PetscCall(PetscFree(row));
      PetscCall(PetscFree(col));
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  PetscCheck(Amat == Pmat,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction3DNoCoupling(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscInt       start[3],n[3],n_extra[3],N[3],dof[4];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&n_extra[0],&n_extra[1],&n_extra[2]));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],&N[2]));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
  for (PetscInt ez=start[2]; ez<start[2]+n[2]+n_extra[2]; ++ez) {
    for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
      for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
        for (PetscInt c=0; c<dof[0]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.k = ez;
          row.loc = DMSTAG_BACK_DOWN_LEFT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = (5.0 + c) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
        }
        if (ez < N[2]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (50.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ey < N[1]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (55.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ex < N[0]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_DOWN;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (60.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ex < N[0] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (10.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (15.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ex < N[0] && ey < N[1]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (15.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ex < N[0] && ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[3]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_ELEMENT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (20.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,INSERT_VALUES));
          }
        }
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian3DNoCoupling(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx)
{
  PetscInt       start[3],n[3],n_extra[3],N[3],dof[4];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&n_extra[0],&n_extra[1],&n_extra[2]));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],&N[2]));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
  for (PetscInt ez=start[2]; ez<start[2]+n[2]+n_extra[2]; ++ez) {
    for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
      for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
        for (PetscInt c=0; c<dof[0]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.k = ez;
          row.loc = DMSTAG_BACK_DOWN_LEFT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = 3.0 * (5.0 + c) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
        }
        if (ez < N[2]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (50.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ey < N[1]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (55.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ex < N[0]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_DOWN;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (60.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ex < N[0] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (10.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (15.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ex < N[0] && ey < N[1]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (15.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
          }
        }
        if (ex < N[0] && ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[3]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_ELEMENT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (20.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,INSERT_VALUES));
          }
        }
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  PetscCheck(Amat == Pmat,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction3D(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscInt       start[3],n[3],n_extra[3],N[3],dof[4];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&n_extra[0],&n_extra[1],&n_extra[2]));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],&N[2]));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
  PetscCall(VecZeroEntries(f));
  for (PetscInt ez=start[2]; ez<start[2]+n[2]+n_extra[2]; ++ez) {
    for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
      for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
        for (PetscInt c=0; c<dof[0]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.k = ez;
          row.loc = DMSTAG_BACK_DOWN_LEFT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = (5.0 + c) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
        }
        if (ez < N[2]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (50.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
          }
        }
        if (ey < N[1]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (55.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
          }
        }
        if (ex < N[0]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_DOWN;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (60.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
          }
        }
        if (ex < N[0] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (10.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
          }
        }
        if (ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (15.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
          }
        }
        if (ex < N[0] && ey < N[1]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (15.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
          }
        }
        if (ex < N[0] && ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[3]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_ELEMENT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = (20.0 + c) * x_val * x_val * x_val;
            PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row,&val,ADD_VALUES));
          }
        }
      }
    }
  }

  /* Add additional terms fully coupling one interior element to another */
  {
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
    if (rank == 0) {
      PetscInt      epe;
      DMStagStencil *row,*col;

      PetscCall(DMStagGetEntriesPerElement(dm,&epe));
      PetscCall(PetscMalloc1(epe,&row));
      PetscCall(PetscMalloc1(epe,&col));
      for (PetscInt i=0; i<epe; ++i) {
        row[i].i = 0;
        row[i].j = 0;
        row[i].k = 0;
        col[i].i = 0;
        col[i].j = 0;
        col[i].k = 1;
      }

      {
        PetscInt nrows = 0;

        for (PetscInt c=0; c<dof[0]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_BACK_DOWN_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_DOWN_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_BACK_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_BACK_DOWN;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_DOWN;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_BACK;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[3]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_ELEMENT;
          ++nrows;
        }
      }

      {
        PetscInt ncols = 0;

        for (PetscInt c=0; c<dof[0]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_BACK_DOWN_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_DOWN_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_BACK_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_BACK_DOWN;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_DOWN;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_BACK;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[3]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_ELEMENT;
          ++ncols;
        }
      }

      for (PetscInt i=0; i<epe; ++i) {
        for (PetscInt j=0; j<epe; ++j) {
          PetscScalar x_val,val;

          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col[j],&x_val));
          val = (10*i + j) * x_val * x_val * x_val;
          PetscCall(DMStagVecSetValuesStencil(dm,f,1,&row[i],&val,ADD_VALUES));
        }
      }
      PetscCall(PetscFree(row));
      PetscCall(PetscFree(col));
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian3D(SNES snes,Vec x,Mat Amat,Mat Pmat,void *ctx)
{
  PetscInt       start[3],n[3],n_extra[3],N[3],dof[4];
  Vec            x_local;
  DM             dm;

  PetscFunctionBegin;
  (void) ctx;
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&x_local));
  PetscCall(DMGlobalToLocal(dm,x,INSERT_VALUES,x_local));
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&n_extra[0],&n_extra[1],&n_extra[2]));
  PetscCall(DMStagGetGlobalSizes(dm,&N[0],&N[1],&N[2]));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
  PetscCall(MatZeroEntries(Amat));
  for (PetscInt ez=start[2]; ez<start[2]+n[2]+n_extra[2]; ++ez) {
    for (PetscInt ey=start[1]; ey<start[1]+n[1]+n_extra[1]; ++ey) {
      for (PetscInt ex=start[0]; ex<start[0]+n[0]+n_extra[0]; ++ex) {
        for (PetscInt c=0; c<dof[0]; ++c) {
          DMStagStencil row;
          PetscScalar   x_val,val;

          row.i = ex;
          row.j = ey;
          row.k = ez;
          row.loc = DMSTAG_BACK_DOWN_LEFT;
          row.c = c;
          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
          val = 3.0 * (5.0 + c) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
        }
        if (ez < N[2]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (50.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
          }
        }
        if (ey < N[1]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (55.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
          }
        }
        if (ex < N[0]) {
          for (PetscInt c=0; c<dof[1]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK_DOWN;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (60.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
          }
        }
        if (ex < N[0] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_DOWN;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (10.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
          }
        }
        if (ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_LEFT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (15.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
          }
        }
        if (ex < N[0] && ey < N[1]) {
          for (PetscInt c=0; c<dof[2]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_BACK;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (15.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
          }
        }
        if (ex < N[0] && ey < N[1] && ez < N[2]) {
          for (PetscInt c=0; c<dof[3]; ++c) {
            DMStagStencil row;
            PetscScalar   x_val,val;

            row.i = ex;
            row.j = ey;
            row.k = ez;
            row.loc = DMSTAG_ELEMENT;
            row.c = c;
            PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&row,&x_val));
            val = 3.0 * (20.0 + c) * x_val * x_val;
            PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row,1,&row,&val,ADD_VALUES));
          }
        }
      }
    }
  }

  /* Add additional terms fully coupling one interior element to another */
  {
    PetscMPIInt   rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
    if (rank == 0) {
      PetscInt      epe;
      DMStagStencil *row,*col;

      PetscCall(DMStagGetEntriesPerElement(dm,&epe));
      PetscCall(PetscMalloc1(epe,&row));
      PetscCall(PetscMalloc1(epe,&col));
      for (PetscInt i=0; i<epe; ++i) {
        row[i].i = 0;
        row[i].j = 0;
        row[i].k = 0;
        col[i].i = 0;
        col[i].j = 0;
        col[i].k = 1;
      }

      {
        PetscInt nrows = 0;

        for (PetscInt c=0; c<dof[0]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_BACK_DOWN_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_DOWN_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_BACK_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_BACK_DOWN;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_DOWN;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_BACK;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[3]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_ELEMENT;
          ++nrows;
        }
      }

      {
        PetscInt ncols = 0;

        for (PetscInt c=0; c<dof[0]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_BACK_DOWN_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_DOWN_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_BACK_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_BACK_DOWN;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_DOWN;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[2]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_BACK;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[3]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_ELEMENT;
          ++ncols;
        }
      }

      for (PetscInt i=0; i<epe; ++i) {
        for (PetscInt j=0; j<epe; ++j) {
          PetscScalar x_val,val;

          PetscCall(DMStagVecGetValuesStencil(dm,x_local,1,&col[j],&x_val));
          val = 3.0 * (10*i + j) * x_val * x_val;
          PetscCall(DMStagMatSetValuesStencil(dm,Amat,1,&row[i],1,&col[j],&val,ADD_VALUES));
        }
      }
      PetscCall(PetscFree(row));
      PetscCall(PetscFree(col));
    }
  }
  PetscCall(DMRestoreLocalVector(dm,&x_local));
  PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  PetscCheck(Amat == Pmat,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for distinct Amat and Pmat");
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscInt       dim;
  PetscBool      no_coupling;
  Vec            x,b;
  SNES           snes;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  dim = 3;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  no_coupling = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-no_coupling",&no_coupling,NULL));

  switch (dim) {
    case 1:
      PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,1, 1,DMSTAG_STENCIL_BOX,1,NULL,&dm));
      break;
    case 2:
      PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,4, 3,PETSC_DECIDE, PETSC_DECIDE,1, 1, 1,DMSTAG_STENCIL_BOX,1,NULL, NULL,&dm));
      break;
    case 3:
      PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,4, 3, 3,PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,1, 1, 1, 1,
                               DMSTAG_STENCIL_BOX,1, NULL, NULL, NULL, &dm));
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,dm));
  if (no_coupling) {
    switch (dim) {
    case 1:
      PetscCall(SNESSetFunction(snes,NULL,FormFunction1DNoCoupling,NULL));
      PetscCall(SNESSetJacobian(snes,NULL,NULL,FormJacobian1DNoCoupling,NULL));
      break;
    case 2:
      PetscCall(SNESSetFunction(snes,NULL,FormFunction2DNoCoupling,NULL));
      PetscCall(SNESSetJacobian(snes,NULL,NULL,FormJacobian2DNoCoupling,NULL));
      break;
    case 3:
      PetscCall(SNESSetFunction(snes,NULL,FormFunction3DNoCoupling,NULL));
      PetscCall(SNESSetJacobian(snes,NULL,NULL,FormJacobian3DNoCoupling,NULL));
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
    }
  } else {
    switch (dim) {
      case 1:
        PetscCall(SNESSetFunction(snes,NULL,FormFunction1D,NULL));
        PetscCall(SNESSetJacobian(snes,NULL,NULL,FormJacobian1D,NULL));
        break;
      case 2:
        PetscCall(SNESSetFunction(snes,NULL,FormFunction2D,NULL));
        PetscCall(SNESSetJacobian(snes,NULL,NULL,FormJacobian2D,NULL));
        break;
      case 3:
        PetscCall(SNESSetFunction(snes,NULL,FormFunction3D,NULL));
        PetscCall(SNESSetJacobian(snes,NULL,NULL,FormJacobian3D,NULL));
        break;
      default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
    }
  }
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateGlobalVector(dm,&x));
  PetscCall(VecDuplicate(x,&b));
  PetscCall(VecSet(x,2.0)); // Initial guess
  PetscCall(VecSet(b,0.0)); // RHS
  PetscCall(SNESSolve(snes,b,x));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1d_no_coupling
      nsize: {{1 2}separate output}
      args: -dim 1 -no_coupling -stag_stencil_type none -pc_type jacobi -snes_converged_reason -snes_test_jacobian -stag_dof_0 {{1 2}separate output} -stag_dof_1 {{1 2}separate output} -snes_max_it 2
   test:
      suffix: 1d_test_jac
      nsize: {{1 2}separate output}
      args: -dim 1 -stag_stencil_width {{0 1}separate output} -pc_type jacobi -snes_converged_reason -snes_test_jacobian -snes_max_it 2
   test:
      suffix: 1d_fd_coloring
      nsize: {{1 2}separate output}
      args: -dim 1 -stag_stencil_width {{0 1 2}separate output} -pc_type jacobi -snes_converged_reason -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type {{natural sl}} -snes_max_it 2
   test:
      suffix: 1d_periodic
      nsize: {{1 2}separate output}
      args: -dim 1 -stag_boundary_type_x periodic -stag_stencil_width {{1 2}separate output} -pc_type jacobi -snes_converged_reason -snes_test_jacobian -snes_max_it 2
   test:
      suffix: 1d_multidof
      nsize: 2
      args: -dim 1 -stag_stencil_width 2 -stag_dof_0 2 -stag_dof_1 3 -pc_type jacobi -snes_converged_reason -snes_test_jacobian -snes_max_it 2
   test:
      suffix: 2d_no_coupling
      nsize: {{1 4}separate output}
      args: -dim 2 -no_coupling -stag_stencil_type none -pc_type jacobi -snes_test_jacobian -stag_dof_0 {{1 2}separate output} -stag_dof_1 {{1 2}separate output} -stag_dof_2 {{1 2}separate output} -snes_max_it 2
   test:
      suffix: 3d_no_coupling
      nsize: 2
      args: -dim 3 -no_coupling -stag_stencil_type none -pc_type jacobi -snes_test_jacobian -stag_dof_0 2 -stag_dof_1 2 -stag_dof_2 2 -stag_dof_3 2 -snes_max_it 2
   test:
      suffix: 2d_fd_coloring
      nsize: {{1 2}separate output}
      args: -dim 2 -stag_stencil_width {{1 2}separate output} -pc_type jacobi -snes_converged_reason -snes_fd_color -snes_fd_color_use_mat -stag_stencil_type {{star box}separate output} -snes_max_it 2
   test:
      suffix: 3d_fd_coloring
      nsize: {{1 2}separate output}
      args: -dim 3 -stag_stencil_width {{1 2}separate output} -pc_type jacobi -snes_converged_reason -snes_fd_color -snes_fd_color_use_mat -stag_stencil_type {{star box}separate output} -snes_max_it 2
TEST*/
