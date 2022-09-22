/* function subroutines used by power.c */

#include "power.h"

PetscErrorCode GetListofEdges_Power(PFDATA *pfdata, PetscInt *edgelist)
{
  PetscInt   i, fbus, tbus, nbranches = pfdata->nbranch;
  EDGE_Power branch  = pfdata->branch;
  PetscBool  netview = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-powernet_view", &netview));
  for (i = 0; i < nbranches; i++) {
    fbus                = branch[i].internal_i;
    tbus                = branch[i].internal_j;
    edgelist[2 * i]     = fbus;
    edgelist[2 * i + 1] = tbus;
    if (netview) PetscCall(PetscPrintf(PETSC_COMM_SELF, "branch %" PetscInt_FMT ", bus[%" PetscInt_FMT "] -> bus[%" PetscInt_FMT "]\n", i, fbus, tbus));
  }
  if (netview) {
    for (i = 0; i < pfdata->nbus; i++) {
      if (pfdata->bus[i].ngen) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, " bus %" PetscInt_FMT ": gen\n", i));
      } else if (pfdata->bus[i].nload) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, " bus %" PetscInt_FMT ": load\n", i));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_Power_private(DM networkdm, Vec localX, Mat J, PetscInt nv, PetscInt ne, const PetscInt *vtx, const PetscInt *edges, void *appctx)
{
  const PetscScalar *xarr;
  PetscInt           i, v, row[2], col[8], e, vfrom, vto;
  PetscInt           offsetfrom, offsetto, goffsetfrom, goffsetto, numComps;
  PetscScalar        values[8];
  PetscInt           j, key, offset, goffset;
  PetscScalar        Vm;
  UserCtx_Power     *user_power = (UserCtx_Power *)appctx;
  PetscScalar        Sbase      = user_power->Sbase;
  VERTEX_Power       bus;
  PetscBool          ghostvtex;
  void              *component;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(localX, &xarr));

  for (v = 0; v < nv; v++) {
    PetscCall(DMNetworkIsGhostVertex(networkdm, vtx[v], &ghostvtex));

    PetscCall(DMNetworkGetNumComponents(networkdm, vtx[v], &numComps));
    for (j = 0; j < numComps; j++) {
      PetscCall(DMNetworkGetLocalVecOffset(networkdm, vtx[v], ALL_COMPONENTS, &offset));
      PetscCall(DMNetworkGetGlobalVecOffset(networkdm, vtx[v], ALL_COMPONENTS, &goffset));
      PetscCall(DMNetworkGetComponent(networkdm, vtx[v], j, &key, &component, NULL));

      if (key == user_power->compkey_bus) {
        PetscInt        nconnedges;
        const PetscInt *connedges;

        bus = (VERTEX_Power)(component);
        if (!ghostvtex) {
          /* Handle reference bus constrained dofs */
          if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
            row[0]    = goffset;
            row[1]    = goffset + 1;
            col[0]    = goffset;
            col[1]    = goffset + 1;
            col[2]    = goffset;
            col[3]    = goffset + 1;
            values[0] = 1.0;
            values[1] = 0.0;
            values[2] = 0.0;
            values[3] = 1.0;
            PetscCall(MatSetValues(J, 2, row, 2, col, values, ADD_VALUES));
            break;
          }

          Vm = xarr[offset + 1];

          /* Shunt injections */
          row[0]    = goffset;
          row[1]    = goffset + 1;
          col[0]    = goffset;
          col[1]    = goffset + 1;
          values[0] = values[1] = values[2] = values[3] = 0.0;
          if (bus->ide != PV_BUS) {
            values[1] = 2.0 * Vm * bus->gl / Sbase;
            values[3] = -2.0 * Vm * bus->bl / Sbase;
          }
          PetscCall(MatSetValues(J, 2, row, 2, col, values, ADD_VALUES));
        }

        PetscCall(DMNetworkGetSupportingEdges(networkdm, vtx[v], &nconnedges, &connedges));

        for (i = 0; i < nconnedges; i++) {
          EDGE_Power      branch;
          VERTEX_Power    busf, bust;
          PetscInt        keyf, keyt;
          PetscScalar     Gff, Bff, Gft, Bft, Gtf, Btf, Gtt, Btt;
          const PetscInt *cone;
          PetscScalar     Vmf, Vmt, thetaf, thetat, thetaft, thetatf;

          e = connedges[i];
          PetscCall(DMNetworkGetComponent(networkdm, e, 0, &key, (void **)&branch, NULL));
          if (!branch->status) continue;

          Gff = branch->yff[0];
          Bff = branch->yff[1];
          Gft = branch->yft[0];
          Bft = branch->yft[1];
          Gtf = branch->ytf[0];
          Btf = branch->ytf[1];
          Gtt = branch->ytt[0];
          Btt = branch->ytt[1];

          PetscCall(DMNetworkGetConnectedVertices(networkdm, edges[e], &cone));
          vfrom = cone[0];
          vto   = cone[1];

          PetscCall(DMNetworkGetLocalVecOffset(networkdm, vfrom, ALL_COMPONENTS, &offsetfrom));
          PetscCall(DMNetworkGetLocalVecOffset(networkdm, vto, ALL_COMPONENTS, &offsetto));
          PetscCall(DMNetworkGetGlobalVecOffset(networkdm, vfrom, ALL_COMPONENTS, &goffsetfrom));
          PetscCall(DMNetworkGetGlobalVecOffset(networkdm, vto, ALL_COMPONENTS, &goffsetto));

          if (goffsetto < 0) goffsetto = -goffsetto - 1;

          thetaf  = xarr[offsetfrom];
          Vmf     = xarr[offsetfrom + 1];
          thetat  = xarr[offsetto];
          Vmt     = xarr[offsetto + 1];
          thetaft = thetaf - thetat;
          thetatf = thetat - thetaf;

          PetscCall(DMNetworkGetComponent(networkdm, vfrom, 0, &keyf, (void **)&busf, NULL));
          PetscCall(DMNetworkGetComponent(networkdm, vto, 0, &keyt, (void **)&bust, NULL));

          if (vfrom == vtx[v]) {
            if (busf->ide != REF_BUS) {
              /*    farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft));  */
              row[0]    = goffsetfrom;
              col[0]    = goffsetfrom;
              col[1]    = goffsetfrom + 1;
              col[2]    = goffsetto;
              col[3]    = goffsetto + 1;
              values[0] = Vmf * Vmt * (Gft * -PetscSinScalar(thetaft) + Bft * PetscCosScalar(thetaft));            /* df_dthetaf */
              values[1] = 2.0 * Gff * Vmf + Vmt * (Gft * PetscCosScalar(thetaft) + Bft * PetscSinScalar(thetaft)); /* df_dVmf */
              values[2] = Vmf * Vmt * (Gft * PetscSinScalar(thetaft) + Bft * -PetscCosScalar(thetaft));            /* df_dthetat */
              values[3] = Vmf * (Gft * PetscCosScalar(thetaft) + Bft * PetscSinScalar(thetaft));                   /* df_dVmt */

              PetscCall(MatSetValues(J, 1, row, 4, col, values, ADD_VALUES));
            }
            if (busf->ide != PV_BUS && busf->ide != REF_BUS) {
              row[0] = goffsetfrom + 1;
              col[0] = goffsetfrom;
              col[1] = goffsetfrom + 1;
              col[2] = goffsetto;
              col[3] = goffsetto + 1;
              /*    farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft)); */
              values[0] = Vmf * Vmt * (Bft * PetscSinScalar(thetaft) + Gft * PetscCosScalar(thetaft));
              values[1] = -2.0 * Bff * Vmf + Vmt * (-Bft * PetscCosScalar(thetaft) + Gft * PetscSinScalar(thetaft));
              values[2] = Vmf * Vmt * (-Bft * PetscSinScalar(thetaft) + Gft * -PetscCosScalar(thetaft));
              values[3] = Vmf * (-Bft * PetscCosScalar(thetaft) + Gft * PetscSinScalar(thetaft));

              PetscCall(MatSetValues(J, 1, row, 4, col, values, ADD_VALUES));
            }
          } else {
            if (bust->ide != REF_BUS) {
              row[0] = goffsetto;
              col[0] = goffsetto;
              col[1] = goffsetto + 1;
              col[2] = goffsetfrom;
              col[3] = goffsetfrom + 1;
              /*    farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf)); */
              values[0] = Vmt * Vmf * (Gtf * -PetscSinScalar(thetatf) + Btf * PetscCosScalar(thetaft));            /* df_dthetat */
              values[1] = 2.0 * Gtt * Vmt + Vmf * (Gtf * PetscCosScalar(thetatf) + Btf * PetscSinScalar(thetatf)); /* df_dVmt */
              values[2] = Vmt * Vmf * (Gtf * PetscSinScalar(thetatf) + Btf * -PetscCosScalar(thetatf));            /* df_dthetaf */
              values[3] = Vmt * (Gtf * PetscCosScalar(thetatf) + Btf * PetscSinScalar(thetatf));                   /* df_dVmf */

              PetscCall(MatSetValues(J, 1, row, 4, col, values, ADD_VALUES));
            }
            if (bust->ide != PV_BUS && bust->ide != REF_BUS) {
              row[0] = goffsetto + 1;
              col[0] = goffsetto;
              col[1] = goffsetto + 1;
              col[2] = goffsetfrom;
              col[3] = goffsetfrom + 1;
              /*    farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf)); */
              values[0] = Vmt * Vmf * (Btf * PetscSinScalar(thetatf) + Gtf * PetscCosScalar(thetatf));
              values[1] = -2.0 * Btt * Vmt + Vmf * (-Btf * PetscCosScalar(thetatf) + Gtf * PetscSinScalar(thetatf));
              values[2] = Vmt * Vmf * (-Btf * PetscSinScalar(thetatf) + Gtf * -PetscCosScalar(thetatf));
              values[3] = Vmt * (-Btf * PetscCosScalar(thetatf) + Gtf * PetscSinScalar(thetatf));

              PetscCall(MatSetValues(J, 1, row, 4, col, values, ADD_VALUES));
            }
          }
        }
        if (!ghostvtex && bus->ide == PV_BUS) {
          row[0]    = goffset + 1;
          col[0]    = goffset + 1;
          values[0] = 1.0;
          if (user_power->jac_error) values[0] = 50.0;
          PetscCall(MatSetValues(J, 1, row, 1, col, values, ADD_VALUES));
        }
      }
    }
  }

  PetscCall(VecRestoreArrayRead(localX, &xarr));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_Power(SNES snes, Vec X, Mat J, Mat Jpre, void *appctx)
{
  DM              networkdm;
  Vec             localX;
  PetscInt        nv, ne;
  const PetscInt *vtx, *edges;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(J));

  PetscCall(SNESGetDM(snes, &networkdm));
  PetscCall(DMGetLocalVector(networkdm, &localX));

  PetscCall(DMGlobalToLocalBegin(networkdm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(networkdm, X, INSERT_VALUES, localX));

  PetscCall(DMNetworkGetSubnetwork(networkdm, 0, &nv, &ne, &vtx, &edges));
  PetscCall(FormJacobian_Power_private(networkdm, localX, J, nv, ne, vtx, edges, appctx));

  PetscCall(DMRestoreLocalVector(networkdm, &localX));

  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction_Power(DM networkdm, Vec localX, Vec localF, PetscInt nv, PetscInt ne, const PetscInt *vtx, const PetscInt *edges, void *appctx)
{
  UserCtx_Power     *User = (UserCtx_Power *)appctx;
  PetscInt           e, v, vfrom, vto;
  const PetscScalar *xarr;
  PetscScalar       *farr;
  PetscInt           offsetfrom, offsetto, offset, i, j, key, numComps;
  PetscScalar        Vm;
  PetscScalar        Sbase = User->Sbase;
  VERTEX_Power       bus   = NULL;
  GEN                gen;
  LOAD               load;
  PetscBool          ghostvtex;
  void              *component;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(localX, &xarr));
  PetscCall(VecGetArray(localF, &farr));

  for (v = 0; v < nv; v++) {
    PetscCall(DMNetworkIsGhostVertex(networkdm, vtx[v], &ghostvtex));
    PetscCall(DMNetworkGetNumComponents(networkdm, vtx[v], &numComps));
    PetscCall(DMNetworkGetLocalVecOffset(networkdm, vtx[v], ALL_COMPONENTS, &offset));

    for (j = 0; j < numComps; j++) {
      PetscCall(DMNetworkGetComponent(networkdm, vtx[v], j, &key, &component, NULL));
      if (key == User->compkey_bus) {
        PetscInt        nconnedges;
        const PetscInt *connedges;

        bus = (VERTEX_Power)(component);
        /* Handle reference bus constrained dofs */
        if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
          farr[offset]     = xarr[offset] - bus->va * PETSC_PI / 180.0;
          farr[offset + 1] = xarr[offset + 1] - bus->vm;
          break;
        }

        if (!ghostvtex) {
          Vm = xarr[offset + 1];

          /* Shunt injections */
          farr[offset] += Vm * Vm * bus->gl / Sbase;
          if (bus->ide != PV_BUS) farr[offset + 1] += -Vm * Vm * bus->bl / Sbase;
        }

        PetscCall(DMNetworkGetSupportingEdges(networkdm, vtx[v], &nconnedges, &connedges));
        for (i = 0; i < nconnedges; i++) {
          EDGE_Power      branch;
          PetscInt        keye;
          PetscScalar     Gff, Bff, Gft, Bft, Gtf, Btf, Gtt, Btt;
          const PetscInt *cone;
          PetscScalar     Vmf, Vmt, thetaf, thetat, thetaft, thetatf;

          e = connedges[i];
          PetscCall(DMNetworkGetComponent(networkdm, e, 0, &keye, (void **)&branch, NULL));
          if (!branch->status) continue;
          Gff = branch->yff[0];
          Bff = branch->yff[1];
          Gft = branch->yft[0];
          Bft = branch->yft[1];
          Gtf = branch->ytf[0];
          Btf = branch->ytf[1];
          Gtt = branch->ytt[0];
          Btt = branch->ytt[1];

          PetscCall(DMNetworkGetConnectedVertices(networkdm, e, &cone));
          vfrom = cone[0];
          vto   = cone[1];

          PetscCall(DMNetworkGetLocalVecOffset(networkdm, vfrom, ALL_COMPONENTS, &offsetfrom));
          PetscCall(DMNetworkGetLocalVecOffset(networkdm, vto, ALL_COMPONENTS, &offsetto));

          thetaf  = xarr[offsetfrom];
          Vmf     = xarr[offsetfrom + 1];
          thetat  = xarr[offsetto];
          Vmt     = xarr[offsetto + 1];
          thetaft = thetaf - thetat;
          thetatf = thetat - thetaf;

          if (vfrom == vtx[v]) {
            farr[offsetfrom] += Gff * Vmf * Vmf + Vmf * Vmt * (Gft * PetscCosScalar(thetaft) + Bft * PetscSinScalar(thetaft));
            farr[offsetfrom + 1] += -Bff * Vmf * Vmf + Vmf * Vmt * (-Bft * PetscCosScalar(thetaft) + Gft * PetscSinScalar(thetaft));
          } else {
            farr[offsetto] += Gtt * Vmt * Vmt + Vmt * Vmf * (Gtf * PetscCosScalar(thetatf) + Btf * PetscSinScalar(thetatf));
            farr[offsetto + 1] += -Btt * Vmt * Vmt + Vmt * Vmf * (-Btf * PetscCosScalar(thetatf) + Gtf * PetscSinScalar(thetatf));
          }
        }
      } else if (key == User->compkey_gen) {
        if (!ghostvtex) {
          gen = (GEN)(component);
          if (!gen->status) continue;
          farr[offset] += -gen->pg / Sbase;
          farr[offset + 1] += -gen->qg / Sbase;
        }
      } else if (key == User->compkey_load) {
        if (!ghostvtex) {
          load = (LOAD)(component);
          farr[offset] += load->pl / Sbase;
          farr[offset + 1] += load->ql / Sbase;
        }
      }
    }
    if (bus && bus->ide == PV_BUS) farr[offset + 1] = xarr[offset + 1] - bus->vm;
  }
  PetscCall(VecRestoreArrayRead(localX, &xarr));
  PetscCall(VecRestoreArray(localF, &farr));
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess_Power(DM networkdm, Vec localX, PetscInt nv, PetscInt ne, const PetscInt *vtx, const PetscInt *edges, void *appctx)
{
  VERTEX_Power   bus;
  PetscInt       i;
  GEN            gen;
  PetscBool      ghostvtex, sharedv;
  PetscScalar   *xarr;
  PetscInt       key, numComps, j, offset;
  void          *component;
  PetscMPIInt    rank;
  MPI_Comm       comm;
  UserCtx_Power *User = (UserCtx_Power *)appctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)networkdm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecGetArray(localX, &xarr));
  for (i = 0; i < nv; i++) {
    PetscCall(DMNetworkIsGhostVertex(networkdm, vtx[i], &ghostvtex));
    PetscCall(DMNetworkIsSharedVertex(networkdm, vtx[i], &sharedv));
    if (ghostvtex || sharedv) continue;

    PetscCall(DMNetworkGetLocalVecOffset(networkdm, vtx[i], ALL_COMPONENTS, &offset));
    PetscCall(DMNetworkGetNumComponents(networkdm, vtx[i], &numComps));
    for (j = 0; j < numComps; j++) {
      PetscCall(DMNetworkGetComponent(networkdm, vtx[i], j, &key, &component, NULL));
      if (key == User->compkey_bus) {
        bus              = (VERTEX_Power)(component);
        xarr[offset]     = bus->va * PETSC_PI / 180.0;
        xarr[offset + 1] = bus->vm;
      } else if (key == User->compkey_gen) {
        gen = (GEN)(component);
        if (!gen->status) continue;
        xarr[offset + 1] = gen->vs;
        break;
      }
    }
  }
  PetscCall(VecRestoreArray(localX, &xarr));
  PetscFunctionReturn(0);
}
