/* Internal and DMStag-specific functions related to multigrid */
#include <petsc/private/dmstagimpl.h>

/*@C
    DMStagRestrictSimple - restricts data from a fine to a coarse DMStag, in the simplest way

    Values on coarse cells are averages of all fine cells that they cover.
    Thus, values on vertices are injected, values on edges are averages
    of the underlying two fine edges, and and values on elements in
    d dimensions are averages of 2^d underlying elements.

    Input Parameters:
+   dmf - fine DM
.   xf - data on fine DM
-   dmc - coarse DM

    Output Parameter:
.   xc - data on coarse DM

    Level: advanced

.seealso: DMRestrict(), DMCoarsen(), DMSTAG, DMCreateInjection()

@*/
PetscErrorCode DMStagRestrictSimple(DM dmf,Vec xf,DM dmc,Vec xc)
{
  PetscInt       dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dmf,&dim));
  switch (dim) {
    case 1:
      PetscCall(DMStagRestrictSimple_1d(dmf,xf,dmc,xc));
      break;
    case 2:
      PetscCall(DMStagRestrictSimple_2d(dmf,xf,dmc,xc));
      break;
    case 3:
      PetscCall(DMStagRestrictSimple_3d(dmf,xf,dmc,xc));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dmf),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %" PetscInt_FMT "",dim);
      break;
  }
  PetscFunctionReturn(0);
}

/* Code duplication note: the next two functions are nearly identical, save the inclusion of the element terms */
PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation1d_a_b_Private(DM dmc,DM dmf,Mat A)
{
  PetscInt       exf,startexf,nexf,nextraxf,startexc;
  PetscInt       dof[2];
  const PetscInt dim = 1;

  PetscFunctionBegin;
  PetscCall(DMStagGetDOF(dmc,&dof[0],&dof[1],NULL,NULL));
  PetscCheck(dof[0] == 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented except for one dof per vertex");
  PetscCheck(dof[1] <= 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented for more than one dof per element");

  /* In 1D, each fine point can receive data from at most 2 coarse points, at most one of which could be off-process */
  PetscCall(MatSeqAIJSetPreallocation(A,2,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,2,NULL,1,NULL));

  PetscCall(DMStagGetCorners(dmf,&startexf,NULL,NULL,&nexf,NULL,NULL,&nextraxf,NULL,NULL));
  PetscCall(DMStagGetCorners(dmc,&startexc,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
    PetscInt exc,exf_local;
    exf_local = exf-startexf;
    exc = startexc + exf_local/2;
    /* "even" vertices are just injected, odd vertices averaged */
    if (exf_local % 2 == 0) {
      DMStagStencil     rowf,colc;
      PetscInt          ir,ic;
      const PetscScalar one = 1.0;

      rowf.i = exf; rowf.c = 0; rowf.loc = DMSTAG_LEFT;
      colc.i = exc; colc.c = 0; colc.loc = DMSTAG_LEFT;
      PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
      PetscCall(DMStagStencilToIndexLocal(dmc,dim,1,&colc,&ic));
      PetscCall(MatSetValuesLocal(A,1,&ir,1,&ic,&one,INSERT_VALUES));
    } else {
      DMStagStencil     rowf,colc[2];
      PetscInt          ir,ic[2];
      const PetscScalar weight[2] = {0.5,0.5};

      rowf.i    = exf; rowf.c    = 0; rowf.loc    = DMSTAG_LEFT;
      colc[0].i = exc; colc[0].c = 0; colc[0].loc = DMSTAG_LEFT;
      colc[1].i = exc; colc[1].c = 0; colc[1].loc = DMSTAG_RIGHT;
      PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
      PetscCall(DMStagStencilToIndexLocal(dmc,dim,2, colc, ic));
      PetscCall(MatSetValuesLocal(A,1,&ir,2,ic,weight,INSERT_VALUES));
    }
    /* Elements (excluding "extra" dummies) */
    if (dof[1] > 0 && exf < startexf+nexf) {
      DMStagStencil     rowf,colc;
      PetscInt          ir,ic;
      const PetscScalar weight = 1.0;

      rowf.i = exf;  rowf.c = 0; rowf.loc = DMSTAG_ELEMENT; /* Note that this assumes only 1 dof */
      colc.i = exc;  colc.c = 0; colc.loc = DMSTAG_ELEMENT;
      PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
      PetscCall(DMStagStencilToIndexLocal(dmc,dim,1,&colc,&ic));
      PetscCall(MatSetValuesLocal(A,1,&ir,1,&ic,&weight,INSERT_VALUES));
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation2d_0_a_b_Private(DM dmc,DM dmf,Mat A)
{
  PetscInt       exf,eyf,startexf,starteyf,nexf,neyf,nextraxf,nextrayf,startexc,starteyc,Nexf,Neyf;
  PetscInt       dof[3];
  const PetscInt dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetDOF(dmc,&dof[0],&dof[1],&dof[2],NULL));
  PetscCheck(dof[1] == 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented except for one dof per face");
  PetscCheck(dof[2] <= 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented for more than one dof per element");

  /* In 2D, each fine point can receive data from at most 4 coarse points, at most 3 of which could be off-process */
  PetscCall(MatSeqAIJSetPreallocation(A,4,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,4,NULL,3,NULL));

  PetscCall(DMStagGetCorners(dmf,&startexf,&starteyf,NULL,&nexf,&neyf,NULL,&nextraxf,&nextrayf,NULL));
  PetscCall(DMStagGetCorners(dmc,&startexc,&starteyc,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dmf,&Nexf,&Neyf,NULL));
  for (eyf=starteyf; eyf<starteyf+neyf+nextrayf; ++eyf) {
    PetscInt eyc,eyf_local;

    eyf_local = eyf-starteyf;
    eyc = starteyc + eyf_local/2;
    for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
      PetscInt exc,exf_local;

      exf_local = exf-startexf;
      exc = startexc + exf_local/2;
      /* Left edges (excluding top "extra" dummy row) */
      if (eyf < starteyf+neyf) {
        DMStagStencil rowf,colc[4];
        PetscInt      ir,ic[4],nweight;
        PetscScalar   weight[4];

        rowf.i    = exf; rowf.j    = eyf; rowf.c    = 0; rowf.loc    = DMSTAG_LEFT;
        colc[0].i = exc; colc[0].j = eyc; colc[0].c = 0; colc[0].loc = DMSTAG_LEFT;
        if (exf_local % 2 == 0) {
          if (eyf == Neyf-1 || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 1; weight[0] = 1.0;
          } else {
            nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
            if (eyf_local % 2 == 0) {
              colc[1].i = exc; colc[1].j = eyc-1; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
            } else {
              colc[1].i = exc; colc[1].j = eyc+1; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
            }
          }
        } else {
          colc[1].i = exc; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_RIGHT;
          if (eyf == Neyf-1 || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 2; weight[0] = 0.5; weight[1] = 0.5;
          } else {
            nweight = 4; weight[0] = 0.375; weight[1] = 0.375; weight[2] = 0.125; weight[3] = 0.125;
            if (eyf_local % 2 == 0) {
              colc[2].i = exc; colc[2].j = eyc-1; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
              colc[3].i = exc; colc[3].j = eyc-1; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
            } else {
              colc[2].i = exc; colc[2].j = eyc+1; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
              colc[3].i = exc; colc[3].j = eyc+1; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
            }
          }
        }
        PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
        PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,colc,ic));
        PetscCall(MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES));
      }
      /* Down edges (excluding right "extra" dummy col) */
      if (exf < startexf+nexf) {
        DMStagStencil rowf,colc[4];
        PetscInt      ir,ic[4],nweight;
        PetscScalar   weight[4];

        rowf.i    = exf; rowf.j    = eyf; rowf.c    = 0; rowf.loc    = DMSTAG_DOWN;
        colc[0].i = exc; colc[0].j = eyc; colc[0].c = 0; colc[0].loc = DMSTAG_DOWN;
        if (eyf_local % 2 == 0) {
          if (exf == Nexf-1 || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 1; weight[0] = 1.0;
          } else {
            nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
            if (exf_local % 2 == 0) {
              colc[1].i = exc-1; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
            } else {
              colc[1].i = exc+1; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
            }
          }
        } else {
          colc[1].i = exc; colc[1].j = eyc; colc[1].c = 0; colc[1].loc = DMSTAG_UP;
          if (exf == Nexf-1 || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            nweight = 2; weight[0] = 0.5; weight[1] = 0.5;
          } else {
            nweight = 4; weight[0] = 0.375; weight[1] = 0.375; weight[2] = 0.125; weight[3] = 0.125;
            if (exf_local % 2 == 0) {
              colc[2].i = exc-1; colc[2].j = eyc; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
              colc[3].i = exc-1; colc[3].j = eyc; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
            } else {
              colc[2].i = exc+1; colc[2].j = eyc; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
              colc[3].i = exc+1; colc[3].j = eyc; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
            }
          }
        }
        PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
        PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,colc,ic));
        PetscCall(MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES));
      }
      /* Elements (excluding "extra" dummy) */
      if (dof[2] > 0 && exf < startexf+nexf && eyf < starteyf+neyf) {
        DMStagStencil     rowf,colc;
        PetscInt          ir,ic;
        const PetscScalar weight = 1.0;

        rowf.i = exf; rowf.j = eyf; rowf.c = 0; rowf.loc = DMSTAG_ELEMENT;
        colc.i = exc; colc.j = eyc; colc.c = 0; colc.loc = DMSTAG_ELEMENT;
        PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
        PetscCall(DMStagStencilToIndexLocal(dmc,dim,1,&colc,&ic));
        PetscCall(MatSetValuesLocal(A,1,&ir,1,&ic,&weight,INSERT_VALUES));
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation3d_0_0_a_b_Private(DM dmc,DM dmf,Mat A)
{
  PetscInt       exf,eyf,ezf,startexf,starteyf,startezf,nexf,neyf,nezf,nextraxf,nextrayf,nextrazf,startexc,starteyc,startezc,Nexf,Neyf,Nezf;
  PetscInt       dof[4];
  const PetscInt dim = 3;

  PetscFunctionBegin;

  PetscCall(DMStagGetDOF(dmc,&dof[0],&dof[1],&dof[2],&dof[3]));
  PetscCheck(dof[2] == 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented except for one dof per face");
  PetscCheck(dof[3] <= 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented for more than one dof per element");

  /* In 3D, each fine point can receive data from at most 8 coarse points, at most 7 of which could be off-process */
  PetscCall(MatSeqAIJSetPreallocation(A,8,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,8,NULL,7,NULL));

  PetscCall(DMStagGetCorners(dmf,&startexf,&starteyf,&startezf,&nexf,&neyf,&nezf,&nextraxf,&nextrayf,&nextrazf));
  PetscCall(DMStagGetCorners(dmc,&startexc,&starteyc,&startezc,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dmf,&Nexf,&Neyf,&Nezf));
  for (ezf=startezf; ezf<startezf+nezf+nextrayf; ++ezf) {
    const PetscInt  ezf_local  = ezf - startezf;
    const PetscInt  ezc        = startezc + ezf_local/2;
    const PetscBool boundary_z = (PetscBool) (ezf == 0 || ezf == Nezf-1);

    for (eyf=starteyf; eyf<starteyf+neyf+nextrayf; ++eyf) {
      const PetscInt  eyf_local  = eyf - starteyf;
      const PetscInt  eyc        = starteyc + eyf_local/2;
      const PetscBool boundary_y = (PetscBool) (eyf == 0 || eyf == Neyf-1);

      for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
        const PetscInt  exf_local  = exf - startexf;
        const PetscInt  exc        = startexc + exf_local/2;
        const PetscBool boundary_x = (PetscBool) (exf == 0 || exf == Nexf-1);

        /* Left faces (excluding top and front "extra" dummy layers) */
        if (eyf < starteyf + neyf && ezf < startezf + nezf) {
          DMStagStencil rowf,colc[8];
          PetscInt      ir,ic[8],nweight;
          PetscScalar   weight[8];

          rowf.i    = exf; rowf.j    = eyf; rowf.k    = ezf; rowf.c    = 0; rowf.loc    = DMSTAG_LEFT;
          colc[0].i = exc; colc[0].j = eyc; colc[0].k = ezc; colc[0].c = 0; colc[0].loc = DMSTAG_LEFT;
          if (exf_local % 2 == 0) {
            if (boundary_y) {
              if (boundary_z) {
                nweight = 1; weight[0] = 1.0;
              } else {
                const PetscInt ezc_offset = ezf_local % 2 == 0 ? -1 : 1;

                nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
                colc[1].i = exc; colc[1].j = eyc; colc[1].k = ezc + ezc_offset; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
              }
            } else {
              const PetscInt eyc_offset = eyf_local % 2 == 0 ? -1 : 1;

              if (boundary_z) {
                nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
                colc[1].i = exc; colc[1].j = eyc + eyc_offset; colc[1].k = ezc; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
              } else {
                const PetscInt ezc_offset = ezf_local % 2 == 0 ? -1 : 1;

                nweight = 4; weight[0] = 0.75 * 0.75; weight[1] = weight[2] = 0.25 * 0.75; weight[3] = 0.25 * 0.25;
                colc[1].i = exc; colc[1].j = eyc;              colc[1].k = ezc + ezc_offset; colc[1].c = 0; colc[1].loc = DMSTAG_LEFT;
                colc[2].i = exc; colc[2].j = eyc + eyc_offset; colc[2].k = ezc;              colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
                colc[3].i = exc; colc[3].j = eyc + eyc_offset; colc[3].k = ezc + ezc_offset; colc[3].c = 0; colc[3].loc = DMSTAG_LEFT;
              }
            }
          } else {
            colc[1].i = exc; colc[1].j = eyc; colc[1].k = ezc; colc[1].c = 0; colc[1].loc = DMSTAG_RIGHT;
            if (boundary_y) {
              if (boundary_z) {
                nweight = 2; weight[0] = weight[1] = 0.5;
              } else {
                const PetscInt ezc_offset = ezf_local % 2 == 0 ? -1 : 1;

                nweight = 4; weight[0] = weight[1] = 0.5 * 0.75;  weight[2] = weight[3] = 0.5 * 0.25;
                colc[2].i = exc; colc[2].j = eyc; colc[2].k = ezc + ezc_offset; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
                colc[3].i = exc; colc[3].j = eyc; colc[3].k = ezc + ezc_offset; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
              }
            } else {
              const PetscInt eyc_offset = eyf_local % 2 == 0 ? -1 : 1;

              if (boundary_z) {
                nweight = 4; weight[0] = weight[1] = 0.5 * 0.75;  weight[2] = weight[3] = 0.5 * 0.25;
                colc[2].i = exc; colc[2].j = eyc + eyc_offset; colc[2].k = ezc; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
                colc[3].i = exc; colc[3].j = eyc + eyc_offset; colc[3].k = ezc; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
              } else {
                const PetscInt ezc_offset = ezf_local % 2 == 0 ? -1 : 1;

                nweight = 8;
                weight[0] = weight[1] = 0.5 * 0.75 * 0.75;
                weight[2] = weight[3] = weight[4] = weight[5] = 0.5 * 0.25 * 0.75;
                weight[6] = weight[7] = 0.5 * 0.25 * 0.25;
                colc[2].i = exc; colc[2].j = eyc;              colc[2].k = ezc + ezc_offset; colc[2].c = 0; colc[2].loc = DMSTAG_LEFT;
                colc[3].i = exc; colc[3].j = eyc;              colc[3].k = ezc + ezc_offset; colc[3].c = 0; colc[3].loc = DMSTAG_RIGHT;
                colc[4].i = exc; colc[4].j = eyc + eyc_offset; colc[4].k = ezc;              colc[4].c = 0; colc[4].loc = DMSTAG_LEFT;
                colc[5].i = exc; colc[5].j = eyc + eyc_offset; colc[5].k = ezc;              colc[5].c = 0; colc[5].loc = DMSTAG_RIGHT;
                colc[6].i = exc; colc[6].j = eyc + eyc_offset; colc[6].k = ezc + ezc_offset; colc[6].c = 0; colc[6].loc = DMSTAG_LEFT;
                colc[7].i = exc; colc[7].j = eyc + eyc_offset; colc[7].k = ezc + ezc_offset; colc[7].c = 0; colc[7].loc = DMSTAG_RIGHT;
              }
            }
          }
          PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
          PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,colc,ic));
          PetscCall(MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES));
        }

        /* Bottom faces (excluding left and front "extra" dummy layers) */
        if (exf < startexf + nexf && ezf < startezf + nezf) {
          DMStagStencil rowf,colc[8];
          PetscInt      ir,ic[8],nweight;
          PetscScalar   weight[8];

          rowf.i    = exf; rowf.j    = eyf; rowf.k    = ezf; rowf.c    = 0; rowf.loc    = DMSTAG_DOWN;
          colc[0].i = exc; colc[0].j = eyc; colc[0].k = ezc; colc[0].c = 0; colc[0].loc = DMSTAG_DOWN;
          if (eyf_local % 2 == 0) {
            if (boundary_x) {
              if (boundary_z) {
                nweight = 1; weight[0] = 1.0;
              } else {
                const PetscInt ezc_offset = ezf_local % 2 == 0 ? -1 : 1;

                nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
                colc[1].i = exc; colc[1].j = eyc; colc[1].k = ezc + ezc_offset; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
              }
            } else {
              const PetscInt exc_offset = exf_local % 2 == 0 ? -1 : 1;

              if (boundary_z) {
                nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
                colc[1].i = exc + exc_offset; colc[1].j = eyc; colc[1].k = ezc; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
              } else {
                const PetscInt ezc_offset = ezf_local % 2 == 0 ? -1 : 1;

                nweight = 4; weight[0] = 0.75 * 0.75; weight[1] = weight[2] = 0.25 * 0.75; weight[3] = 0.25 * 0.25;
                colc[1].i = exc;              colc[1].j = eyc; colc[1].k = ezc + ezc_offset; colc[1].c = 0; colc[1].loc = DMSTAG_DOWN;
                colc[2].i = exc + exc_offset; colc[2].j = eyc; colc[2].k = ezc;              colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
                colc[3].i = exc + exc_offset; colc[3].j = eyc; colc[3].k = ezc + ezc_offset; colc[3].c = 0; colc[3].loc = DMSTAG_DOWN;
              }
            }
          } else {
            colc[1].i = exc; colc[1].j = eyc; colc[1].k = ezc; colc[1].c = 0; colc[1].loc = DMSTAG_UP;
            if (boundary_x) {
              if (boundary_z) {
                nweight = 2; weight[0] = weight[1] = 0.5;
              } else {
                const PetscInt ezc_offset = ezf_local % 2 == 0 ? -1 : 1;

                nweight = 4; weight[0] = weight[1] = 0.5 * 0.75;  weight[2] = weight[3] = 0.5 * 0.25;
                colc[2].i = exc; colc[2].j = eyc; colc[2].k = ezc + ezc_offset; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
                colc[3].i = exc; colc[3].j = eyc; colc[3].k = ezc + ezc_offset; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
              }
            } else {
              const PetscInt exc_offset = exf_local % 2 == 0 ? -1 : 1;

              if (boundary_z) {
                nweight = 4; weight[0] = weight[1] = 0.5 * 0.75;  weight[2] = weight[3] = 0.5 * 0.25;
                colc[2].i = exc + exc_offset; colc[2].j = eyc; colc[2].k = ezc; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
                colc[3].i = exc + exc_offset; colc[3].j = eyc; colc[3].k = ezc; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
              } else {
                const PetscInt ezc_offset = ezf_local % 2 == 0 ? -1 : 1;

                nweight = 8;
                weight[0] = weight[1] = 0.5 * 0.75 * 0.75;
                weight[2] = weight[3] = weight[4] = weight[5] = 0.5 * 0.25 * 0.75;
                weight[6] = weight[7] = 0.5 * 0.25 * 0.25;
                colc[2].i = exc;              colc[2].j = eyc; colc[2].k = ezc + ezc_offset; colc[2].c = 0; colc[2].loc = DMSTAG_DOWN;
                colc[3].i = exc;              colc[3].j = eyc; colc[3].k = ezc + ezc_offset; colc[3].c = 0; colc[3].loc = DMSTAG_UP;
                colc[4].i = exc + exc_offset; colc[4].j = eyc; colc[4].k = ezc;              colc[4].c = 0; colc[4].loc = DMSTAG_DOWN;
                colc[5].i = exc + exc_offset; colc[5].j = eyc; colc[5].k = ezc;              colc[5].c = 0; colc[5].loc = DMSTAG_UP;
                colc[6].i = exc + exc_offset; colc[6].j = eyc; colc[6].k = ezc + ezc_offset; colc[6].c = 0; colc[6].loc = DMSTAG_DOWN;
                colc[7].i = exc + exc_offset; colc[7].j = eyc; colc[7].k = ezc + ezc_offset; colc[7].c = 0; colc[7].loc = DMSTAG_UP;
              }
            }
          }
          PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
          PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,colc,ic));
          PetscCall(MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES));
        }

        /* Back faces (excluding left and top "extra" dummy layers) */
        if (exf < startexf + nexf && ezf < startezf + nezf) {
          DMStagStencil rowf,colc[8];
          PetscInt      ir,ic[8],nweight;
          PetscScalar   weight[8];

          rowf.i    = exf; rowf.j    = eyf; rowf.k    = ezf; rowf.c    = 0; rowf.loc    = DMSTAG_BACK;
          colc[0].i = exc; colc[0].j = eyc; colc[0].k = ezc; colc[0].c = 0; colc[0].loc = DMSTAG_BACK;
          if (ezf_local % 2 == 0) {
            if (boundary_x) {
              if (boundary_y) {
                nweight = 1; weight[0] = 1.0;
              } else {
                const PetscInt eyc_offset = eyf_local % 2 == 0 ? -1 : 1;

                nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
                colc[1].i = exc; colc[1].j = eyc + eyc_offset; colc[1].k = ezc; colc[1].c = 0; colc[1].loc = DMSTAG_BACK;
              }
            } else {
              const PetscInt exc_offset = exf_local % 2 == 0 ? -1 : 1;

              if (boundary_y) {
                nweight = 2; weight[0] = 0.75; weight[1] = 0.25;
                colc[1].i = exc + exc_offset; colc[1].j = eyc; colc[1].k = ezc; colc[1].c = 0; colc[1].loc = DMSTAG_BACK;
              } else {
                const PetscInt eyc_offset = eyf_local % 2 == 0 ? -1 : 1;

                nweight = 4; weight[0] = 0.75 * 0.75; weight[1] = weight[2] = 0.25 * 0.75; weight[3] = 0.25 * 0.25;
                colc[1].i = exc + exc_offset; colc[1].j = eyc;              colc[1].k = ezc; colc[1].c = 0; colc[1].loc = DMSTAG_BACK;
                colc[2].i = exc;              colc[2].j = eyc + eyc_offset; colc[2].k = ezc; colc[2].c = 0; colc[2].loc = DMSTAG_BACK;
                colc[3].i = exc + exc_offset; colc[3].j = eyc + eyc_offset; colc[3].k = ezc; colc[3].c = 0; colc[3].loc = DMSTAG_BACK;
              }
            }
          } else {
            colc[1].i = exc; colc[1].j = eyc; colc[1].k = ezc; colc[1].c = 0; colc[1].loc = DMSTAG_FRONT;
            if (boundary_x) {
              if (boundary_y) {
                nweight = 2; weight[0] = weight[1] = 0.5;
              } else {
                const PetscInt eyc_offset = eyf_local % 2 == 0 ? -1 : 1;

                nweight = 4; weight[0] = weight[1] = 0.5 * 0.75;  weight[2] = weight[3] = 0.5 * 0.25;
                colc[2].i = exc; colc[2].j = eyc + eyc_offset; colc[2].k = ezc; colc[2].c = 0; colc[2].loc = DMSTAG_BACK;
                colc[3].i = exc; colc[3].j = eyc + eyc_offset; colc[3].k = ezc; colc[3].c = 0; colc[3].loc = DMSTAG_FRONT;
              }
            } else {
              const PetscInt exc_offset = exf_local % 2 == 0 ? -1 : 1;

              if (boundary_y) {
                nweight = 4; weight[0] = weight[1] = 0.5 * 0.75;  weight[2] = weight[3] = 0.5 * 0.25;
                colc[2].i = exc + exc_offset; colc[2].j = eyc; colc[2].k = ezc; colc[2].c = 0; colc[2].loc = DMSTAG_BACK;
                colc[3].i = exc + exc_offset; colc[3].j = eyc; colc[3].k = ezc; colc[3].c = 0; colc[3].loc = DMSTAG_FRONT;
              } else {
                const PetscInt eyc_offset = eyf_local % 2 == 0 ? -1 : 1;

                nweight = 8;
                weight[0] = weight[1] = 0.5 * 0.75 * 0.75;
                weight[2] = weight[3] = weight[4] = weight[5] = 0.5 * 0.25 * 0.75;
                weight[6] = weight[7] = 0.5 * 0.25 * 0.25;
                colc[2].i = exc;              colc[2].j = eyc+ eyc_offset; colc[2].k = ezc; colc[2].c = 0; colc[2].loc = DMSTAG_BACK;
                colc[3].i = exc;              colc[3].j = eyc+ eyc_offset; colc[3].k = ezc; colc[3].c = 0; colc[3].loc = DMSTAG_FRONT;
                colc[4].i = exc + exc_offset; colc[4].j = eyc;             colc[4].k = ezc; colc[4].c = 0; colc[4].loc = DMSTAG_BACK;
                colc[5].i = exc + exc_offset; colc[5].j = eyc;             colc[5].k = ezc; colc[5].c = 0; colc[5].loc = DMSTAG_FRONT;
                colc[6].i = exc + exc_offset; colc[6].j = eyc+ eyc_offset; colc[6].k = ezc; colc[6].c = 0; colc[6].loc = DMSTAG_BACK;
                colc[7].i = exc + exc_offset; colc[7].j = eyc+ eyc_offset; colc[7].k = ezc; colc[7].c = 0; colc[7].loc = DMSTAG_FRONT;
              }
            }
          }
          PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
          PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,colc,ic));
          PetscCall(MatSetValuesLocal(A,1,&ir,nweight,ic,weight,INSERT_VALUES));
        }
        /* Elements */
        if (dof[3] == 1 && exf < startexf + nexf && eyf < starteyf + neyf && ezf < startezf + nezf) {
          DMStagStencil     rowf,colc;
          PetscInt          ir,ic;
          const PetscScalar weight = 1.0;

          rowf.i = exf; rowf.j = eyf; rowf.k = ezf; rowf.c = 0; rowf.loc = DMSTAG_ELEMENT;
          colc.i = exc; colc.j = eyc; colc.k = ezc; colc.c = 0; colc.loc = DMSTAG_ELEMENT;
          PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&rowf,&ir));
          PetscCall(DMStagStencilToIndexLocal(dmc,dim,1,&colc,&ic));
          PetscCall(MatSetValuesLocal(A,1,&ir,1,&ic,&weight,INSERT_VALUES));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction1d_a_b_Private(DM dmc,DM dmf,Mat A)
{
  PetscInt       exf,startexf,nexf,nextraxf,startexc,Nexf;
  PetscInt       dof[2];
  const PetscInt dim = 1;

  PetscFunctionBegin;
  PetscCall(DMStagGetDOF(dmc,&dof[0],&dof[1],NULL,NULL));
  PetscCheck(dof[0] == 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented except for one dof per vertex");
  PetscCheck(dof[1] <= 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented for more than one dof per element");

  /* In 1D, each coarse point can receive from up to 3 fine points, one of which may be off-rank */
  PetscCall(MatSeqAIJSetPreallocation(A,3,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,3,NULL,1,NULL));

  PetscCall(DMStagGetCorners(dmf,&startexf,NULL,NULL,&nexf,NULL,NULL,&nextraxf,NULL,NULL));
  PetscCall(DMStagGetCorners(dmc,&startexc,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dmf,&Nexf,NULL,NULL));
  for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
    PetscInt exc,exf_local;

    exf_local = exf-startexf;
    exc = startexc + exf_local/2;
    /* "even" vertices contribute to the overlying coarse vertex, odd vertices to the two adjacent */
    if (exf_local % 2 == 0) {
      DMStagStencil colf,rowc;
      PetscInt      ir,ic;
      PetscScalar   weight;

      colf.i = exf; colf.c = 0; colf.loc = DMSTAG_LEFT;
      rowc.i = exc; rowc.c = 0; rowc.loc = DMSTAG_LEFT;
      PetscCall(DMStagStencilToIndexLocal(dmc,dim,1,&rowc,&ir));
      PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
      weight = (exf == Nexf || exf == 0) ? 0.75 : 0.5; /* Assume a Neuman-type condition */
      PetscCall(MatSetValuesLocal(A,1,&ir,1,&ic,&weight,INSERT_VALUES));
    } else {
      DMStagStencil     colf,rowc[2];
      PetscInt          ic,ir[2];
      const PetscScalar weight[2] = {0.25,0.25};

      colf.i    = exf;  colf.c    = 0; colf.loc    = DMSTAG_LEFT;
      rowc[0].i = exc;  rowc[0].c = 0; rowc[0].loc = DMSTAG_LEFT;
      rowc[1].i = exc;  rowc[1].c = 0; rowc[1].loc = DMSTAG_RIGHT;
      PetscCall(DMStagStencilToIndexLocal(dmc,dim,2, rowc, ir));
      PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
      PetscCall(MatSetValuesLocal(A,2,ir,1,&ic,weight,INSERT_VALUES));
    }
    if (dof[1] > 0 && exf < startexf+nexf) {
      DMStagStencil     rowc,colf;
      PetscInt          ir,ic;
      const PetscScalar weight = 0.5;

      rowc.i = exc; rowc.c = 0; rowc.loc = DMSTAG_ELEMENT;
      colf.i = exf; colf.c = 0; colf.loc = DMSTAG_ELEMENT;
      PetscCall(DMStagStencilToIndexLocal(dmc,dim,1,&rowc,&ir));
      PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
      PetscCall(MatSetValuesLocal(A,1,&ir,1,&ic,&weight,INSERT_VALUES));
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction2d_0_a_b_Private(DM dmc,DM dmf,Mat A)
{
  PetscInt       exf,eyf,startexf,starteyf,nexf,neyf,nextraxf,nextrayf,startexc,starteyc,Nexf,Neyf;
  PetscInt       dof[3];
  const PetscInt dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetDOF(dmc,&dof[0],&dof[1],&dof[2],NULL));
  PetscCheck(dof[1] == 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented except for one dof per face");
  PetscCheck(dof[2] <= 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented for more than one dof per element");

  /* In 2D, each coarse point can receive from up to 6 fine points,
     up to 2 of which may be off rank */
  PetscCall(MatSeqAIJSetPreallocation(A,6,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,6,NULL,2,NULL));

  PetscCall(DMStagGetCorners(dmf,&startexf,&starteyf,NULL,&nexf,&neyf,NULL,&nextraxf,&nextrayf,NULL));
  PetscCall(DMStagGetCorners(dmc,&startexc,&starteyc,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dmf,&Nexf,&Neyf,NULL));
  for (eyf=starteyf; eyf<starteyf+neyf+nextrayf; ++eyf) {
    PetscInt eyc,eyf_local;
    eyf_local = eyf-starteyf;
    eyc = starteyc + eyf_local/2;
    for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
      PetscInt exc,exf_local;
      exf_local = exf-startexf;
      exc = startexc + exf_local/2;
      /* Left edges (excluding top "extra" dummy row) */
      if (eyf < starteyf+neyf) {
        DMStagStencil rowc[2],colf;
        PetscInt      ir[2],ic,nweight;
        PetscScalar   weight[2];

        colf.i    = exf; colf.j    = eyf; colf.c = 0; colf.loc       = DMSTAG_LEFT;
        rowc[0].i = exc; rowc[0].j = eyc; rowc[0].c = 0; rowc[0].loc = DMSTAG_LEFT;
        if (exf_local % 2 == 0) {
          nweight = 1;
          if (exf == Nexf || exf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            weight[0] = 0.375;
          } else {
            weight[0] = 0.25;
          }
        } else {
          nweight = 2;
          rowc[1].i = exc; rowc[1].j = eyc; rowc[1].c = 0; rowc[1].loc = DMSTAG_RIGHT;
          weight[0] = 0.125; weight[1] = 0.125;
        }
        PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,rowc,ir));
        PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
        PetscCall(MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES));
      }
      /* Down edges (excluding right "extra" dummy col) */
      if (exf < startexf+nexf) {
        DMStagStencil rowc[2],colf;
        PetscInt      ir[2],ic,nweight;
        PetscScalar   weight[2];

        colf.i    = exf; colf.j    = eyf; colf.c = 0; colf.loc       = DMSTAG_DOWN;
        rowc[0].i = exc; rowc[0].j = eyc; rowc[0].c = 0; rowc[0].loc = DMSTAG_DOWN;
        if (eyf_local % 2 == 0) {
          nweight = 1;
          if (eyf == Neyf || eyf == 0) {
            /* Note - this presumes something like a Neumann condition, assuming
               a ghost edge with the same value as the adjacent physical edge*/
            weight[0] = 0.375;
          } else {
            weight[0] = 0.25;
          }
        } else {
          nweight = 2;
          rowc[1].i = exc; rowc[1].j = eyc; rowc[1].c = 0; rowc[1].loc = DMSTAG_UP;
          weight[0] = 0.125; weight[1] = 0.125;
        }
        PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,rowc,ir));
        PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
        PetscCall(MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES));
      }
      /* Elements (excluding "extra" dummies) */
      if (dof[2] > 0 && exf < startexf+nexf && eyf < starteyf+neyf) {
        DMStagStencil     rowc,colf;
        PetscInt          ir,ic;
        const PetscScalar cellScale = 0.25;

        rowc.i = exc; rowc.j = eyc; rowc.c = 0; rowc.loc = DMSTAG_ELEMENT;
        colf.i = exf; colf.j = eyf; colf.c = 0; colf.loc = DMSTAG_ELEMENT;
        PetscCall(DMStagStencilToIndexLocal(dmc,dim,1,&rowc,&ir));
        PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
        PetscCall(MatSetValuesLocal(A,1,&ir,1,&ic,&cellScale,INSERT_VALUES));
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction3d_0_0_a_b_Private(DM dmc,DM dmf,Mat A)
{
  PetscInt       exf,eyf,ezf,startexf,starteyf,startezf,nexf,neyf,nezf,nextraxf,nextrayf,nextrazf,startexc,starteyc,startezc,Nexf,Neyf,Nezf;
  PetscInt       dof[4];
  const PetscInt dim = 3;

  PetscFunctionBegin;

  PetscCall(DMStagGetDOF(dmc,&dof[0],&dof[1],&dof[2],&dof[3]));
  PetscCheck(dof[2] == 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented except for one dof per face");
  PetscCheck(dof[3] <= 1,PetscObjectComm((PetscObject)dmc),PETSC_ERR_SUP,"Not Implemented for more than one dof per element");

  /* In 3D, each coarse point can receive from up to 12 fine points,
     up to 4 of which may be off rank */
  PetscCall(MatSeqAIJSetPreallocation(A,12,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,12,NULL,4,NULL));

  PetscCall(DMStagGetCorners(dmf,&startexf,&starteyf,&startezf,&nexf,&neyf,&nezf,&nextraxf,&nextrayf,&nextrazf));
  PetscCall(DMStagGetCorners(dmc,&startexc,&starteyc,&startezc,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dmf,&Nexf,&Neyf,&Nezf));

  for (ezf=startezf; ezf<startezf+nezf+nextrazf; ++ezf) {
    const PetscInt ezf_local = ezf-startezf;
    const PetscInt ezc = startezc + ezf_local/2;

    for (eyf=starteyf; eyf<starteyf+neyf+nextrayf; ++eyf) {
      const PetscInt eyf_local = eyf-starteyf;
      const PetscInt eyc = starteyc + eyf_local/2;

      for (exf=startexf; exf<startexf+nexf+nextraxf; ++exf) {
        const PetscInt exf_local = exf-startexf;
        const PetscInt exc = startexc + exf_local/2;

        /* Left faces (excluding top and front "extra" dummy layers) */
        if (eyf < starteyf + neyf && ezf < startezf + nezf) {
          DMStagStencil rowc[2],colf;
          PetscInt      ir[2],ic,nweight;
          PetscScalar   weight[2];

          colf.i    = exf; colf.j    = eyf; colf.k    = ezf; colf.c    = 0; colf.loc    = DMSTAG_LEFT;
          rowc[0].i = exc; rowc[0].j = eyc; rowc[0].k = ezc; rowc[0].c = 0; rowc[0].loc = DMSTAG_LEFT;
          if (exf_local % 2 == 0) {
            nweight = 1;
            if (exf == Nexf || exf == 0) {
              /* Note - this presumes something like a Neumann condition, assuming
                 a ghost edge with the same value as the adjacent physical edge*/
              weight[0] = 0.1875;
            } else {
              weight[0] = 0.125;
            }
          } else {
            nweight = 2;
            rowc[1].i = exc; rowc[1].j = eyc; rowc[1].k = ezc; rowc[1].c = 0; rowc[1].loc = DMSTAG_RIGHT;
            weight[0] = 0.0625; weight[1] = 0.0625;
          }
          PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,rowc,ir));
          PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
          PetscCall(MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES));
        }

        /* Down faces (excluding right and front "extra" dummy layers) */
        if (exf < startexf+nexf && ezf < startezf+nezf) {
          DMStagStencil rowc[2],colf;
          PetscInt      ir[2],ic,nweight;
          PetscScalar   weight[2];

          colf.i    = exf; colf.j    = eyf; colf.k    = ezf; colf.c    = 0; colf.loc    = DMSTAG_DOWN;
          rowc[0].i = exc; rowc[0].j = eyc; rowc[0].k = ezc; rowc[0].c = 0; rowc[0].loc = DMSTAG_DOWN;
          if (eyf_local % 2 == 0) {
            nweight = 1;
            if (eyf == Neyf || eyf == 0) {
              /* Note - this presumes something like a Neumann condition, assuming
                 a ghost edge with the same value as the adjacent physical edge*/
              weight[0] = 0.1875;
            } else {
              weight[0] = 0.125;
            }
          } else {
            nweight = 2;
            rowc[1].i = exc; rowc[1].j = eyc; rowc[1].k = ezc; rowc[1].c = 0; rowc[1].loc = DMSTAG_UP;
            weight[0] = 0.0625; weight[1] = 0.0625;
          }
          PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,rowc,ir));
          PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
          PetscCall(MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES));
        }

        /* Back faces (excluding left and top "extra" dummy laers) */
        if (exf < startexf + nexf && eyf < starteyf + neyf) {
          DMStagStencil rowc[2],colf;
          PetscInt      ir[2],ic,nweight;
          PetscScalar   weight[2];

          colf.i    = exf; colf.j    = eyf; colf.k    = ezf; colf.c    = 0; colf.loc    = DMSTAG_BACK;
          rowc[0].i = exc; rowc[0].j = eyc; rowc[0].k = ezc; rowc[0].c = 0; rowc[0].loc = DMSTAG_BACK;
          if (ezf_local % 2 == 0) {
            nweight = 1;
            if (ezf == Nezf || ezf == 0) {
              /* Note - this presumes something like a Neumann condition, assuming
                 a ghost edge with the same value as the adjacent physical edge*/
              weight[0] = 0.1875;
            } else {
              weight[0] = 0.125;
            }
          } else {
            nweight = 2;
            rowc[1].i = exc; rowc[1].j = eyc; rowc[1].k = ezc; rowc[1].c = 0; rowc[1].loc = DMSTAG_FRONT;
            weight[0] = 0.0625; weight[1] = 0.0625;
          }
          PetscCall(DMStagStencilToIndexLocal(dmc,dim,nweight,rowc,ir));
          PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
          PetscCall(MatSetValuesLocal(A,nweight,ir,1,&ic,weight,INSERT_VALUES));
        }
        /* Elements */
        if (dof[3] == 1 && exf < startexf + nexf && eyf < starteyf + neyf && ezf < startezf + nezf) {
          DMStagStencil     rowc,colf;
          PetscInt          ir,ic;
          const PetscScalar weight = 0.125;

          colf.i = exf; colf.j = eyf; colf.k = ezf; colf.c = 0; colf.loc = DMSTAG_ELEMENT;
          rowc.i = exc; rowc.j = eyc; rowc.k = ezc; rowc.c = 0; rowc.loc = DMSTAG_ELEMENT;
          PetscCall(DMStagStencilToIndexLocal(dmc,dim,1,&rowc,&ir));
          PetscCall(DMStagStencilToIndexLocal(dmf,dim,1,&colf,&ic));
          PetscCall(MatSetValuesLocal(A,1,&ir,1,&ic,&weight,INSERT_VALUES));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}
