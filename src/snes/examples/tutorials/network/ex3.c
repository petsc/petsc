static char help[] = "This example demonstrates the use of DMNetwork interface with subnetworks for solving a coupled nonlinear \n\
                      electric power grid and water pipe problem.\n\
                      The available solver options are in the ex1options file \n\
                      and the data files are in the datafiles of subdirectories.\n\
                      This example shows the use of subnetwork feature in DMNetwork. \n\
                      Run this program: mpiexec -n <n> ./ex1 \n\\n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

#include "pflow/pf.h"
#include "waternet/waternet.h"

typedef struct{
  UserCtx_Power user_power;
  PetscInt      subsnes_id; /* snes solver id */
  PetscInt      it;         /* iteration number */
  Vec           localXold;  /* store previous solution, used by FormFunction_Dummy() */
} UserCtx;

PetscErrorCode UserMonitor(SNES snes,PetscInt its,PetscReal fnorm ,void *appctx)
{
  PetscErrorCode ierr;
  UserCtx        *user = (UserCtx*)appctx;
  Vec            X,localXold=user->localXold;
  PetscInt       subsnes_id=user->subsnes_id;
  DM             networkdm;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    if (subsnes_id == 2) {
      ierr = PetscPrintf(PETSC_COMM_SELF," it %d, subsnes_id %d, fnorm %g\n",user->it,user->subsnes_id,fnorm);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF,"       subsnes_id %d, fnorm %g\n",user->subsnes_id,fnorm);CHKERRQ(ierr);
    }
  }
  ierr = SNESGetSolution(snes,&X);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localXold);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localXold);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_Power_private(DM networkdm,Vec localX, Mat J,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarr;
  PetscInt          i,v,row[2],col[8],e,vfrom,vto;
  PetscInt          offsetfrom,offsetto,goffsetfrom,goffsetto,numComps;
  PetscScalar       values[8];
  PetscInt          j,key,offset,goffset;
  PetscScalar       Vm;
  UserCtx_Power     *user_power=(UserCtx_Power*)appctx;
  PetscScalar       Sbase=user_power->Sbase;
  VERTEX_Power      bus;
  PetscBool         ghostvtex;
  void*             component;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);

  for (v=0; v<nv; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex);CHKERRQ(ierr);

    ierr = DMNetworkGetNumComponents(networkdm,vtx[v],&numComps);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetVariableOffset(networkdm,vtx[v],&offset);CHKERRQ(ierr);
      ierr = DMNetworkGetVariableGlobalOffset(networkdm,vtx[v],&goffset);CHKERRQ(ierr);
      ierr = DMNetworkGetComponent(networkdm,vtx[v],j,&key,&component);CHKERRQ(ierr);

      if (key == 1) {
        PetscInt       nconnedges;
	const PetscInt *connedges;

	bus = (VERTEX_Power)(component);
        if (!ghostvtex) {
          /* Handle reference bus constrained dofs */
          if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
            row[0] = goffset; row[1] = goffset+1;
            col[0] = goffset; col[1] = goffset+1; col[2] = goffset; col[3] = goffset+1;
            values[0] = 1.0; values[1] = 0.0; values[2] = 0.0; values[3] = 1.0;
            ierr = MatSetValues(J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);
            break;
          }

          Vm = xarr[offset+1];

          /* Shunt injections */
          row[0] = goffset; row[1] = goffset+1;
          col[0] = goffset; col[1] = goffset+1;
          values[0] = values[1] = values[2] = values[3] = 0.0;
          if (bus->ide != PV_BUS) {
            values[1] = 2.0*Vm*bus->gl/Sbase;
            values[3] = -2.0*Vm*bus->bl/Sbase;
          }
          ierr = MatSetValues(J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);
        }

	ierr = DMNetworkGetSupportingEdges(networkdm,vtx[v],&nconnedges,&connedges);CHKERRQ(ierr);

	for (i=0; i < nconnedges; i++) {
	  EDGE_Power     branch;
	  VERTEX_Power   busf,bust;
	  PetscInt       keyf,keyt;
          PetscScalar    Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
          const PetscInt *cone;
          PetscScalar    Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

	  e = connedges[i];
	  ierr = DMNetworkGetComponent(networkdm,e,0,&key,(void**)&branch);CHKERRQ(ierr);
	  if (!branch->status) continue;

	  Gff = branch->yff[0];
	  Bff = branch->yff[1];
	  Gft = branch->yft[0];
	  Bft = branch->yft[1];
	  Gtf = branch->ytf[0];
	  Btf = branch->ytf[1];
	  Gtt = branch->ytt[0];
	  Btt = branch->ytt[1];

	  ierr = DMNetworkGetConnectedVertices(networkdm,edges[e],&cone);CHKERRQ(ierr);
	  vfrom = cone[0];
	  vto   = cone[1];

	  ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offsetfrom);CHKERRQ(ierr);
	  ierr = DMNetworkGetVariableOffset(networkdm,vto,&offsetto);CHKERRQ(ierr);
	  ierr = DMNetworkGetVariableGlobalOffset(networkdm,vfrom,&goffsetfrom);CHKERRQ(ierr);
	  ierr = DMNetworkGetVariableGlobalOffset(networkdm,vto,&goffsetto);CHKERRQ(ierr);

	  if (goffsetto < 0) goffsetto = -goffsetto - 1;

	  thetaf  = xarr[offsetfrom];
	  Vmf     = xarr[offsetfrom+1];
	  thetat  = xarr[offsetto];
	  Vmt     = xarr[offsetto+1];
	  thetaft = thetaf - thetat;
	  thetatf = thetat - thetaf;

	  ierr = DMNetworkGetComponent(networkdm,vfrom,0,&keyf,(void**)&busf);CHKERRQ(ierr);
	  ierr = DMNetworkGetComponent(networkdm,vto,0,&keyt,(void**)&bust);CHKERRQ(ierr);

	  if (vfrom == vtx[v]) {
	    if (busf->ide != REF_BUS) {
	      /*    farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft));  */
	      row[0]  = goffsetfrom;
	      col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
	      values[0] =  Vmf*Vmt*(Gft*-PetscSinScalar(thetaft) + Bft*PetscCosScalar(thetaft)); /* df_dthetaf */
	      values[1] =  2.0*Gff*Vmf + Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft)); /* df_dVmf */
	      values[2] =  Vmf*Vmt*(Gft*PetscSinScalar(thetaft) + Bft*-PetscCosScalar(thetaft)); /* df_dthetat */
	      values[3] =  Vmf*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft)); /* df_dVmt */

	      ierr = MatSetValues(J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	    }
	    if (busf->ide != PV_BUS && busf->ide != REF_BUS) {
	      row[0] = goffsetfrom+1;
	      col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
	      /*    farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft)); */
	      values[0] =  Vmf*Vmt*(Bft*PetscSinScalar(thetaft) + Gft*PetscCosScalar(thetaft));
	      values[1] =  -2.0*Bff*Vmf + Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft));
	      values[2] =  Vmf*Vmt*(-Bft*PetscSinScalar(thetaft) + Gft*-PetscCosScalar(thetaft));
	      values[3] =  Vmf*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft));

	      ierr = MatSetValues(J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	    }
	  } else {
	    if (bust->ide != REF_BUS) {
	      row[0] = goffsetto;
	      col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
	      /*    farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf)); */
	      values[0] =  Vmt*Vmf*(Gtf*-PetscSinScalar(thetatf) + Btf*PetscCosScalar(thetaft)); /* df_dthetat */
	      values[1] =  2.0*Gtt*Vmt + Vmf*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf)); /* df_dVmt */
	      values[2] =  Vmt*Vmf*(Gtf*PetscSinScalar(thetatf) + Btf*-PetscCosScalar(thetatf)); /* df_dthetaf */
	      values[3] =  Vmt*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf)); /* df_dVmf */

	      ierr = MatSetValues(J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	    }
	    if (bust->ide != PV_BUS && bust->ide != REF_BUS) {
	      row[0] = goffsetto+1;
	      col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
	      /*    farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf)); */
	      values[0] =  Vmt*Vmf*(Btf*PetscSinScalar(thetatf) + Gtf*PetscCosScalar(thetatf));
	      values[1] =  -2.0*Btt*Vmt + Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf));
	      values[2] =  Vmt*Vmf*(-Btf*PetscSinScalar(thetatf) + Gtf*-PetscCosScalar(thetatf));
	      values[3] =  Vmt*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf));

	      ierr = MatSetValues(J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	    }
	  }
	}
	if (!ghostvtex && bus->ide == PV_BUS) {
	  row[0] = goffset+1; col[0] = goffset+1; values[0]  = 1.0;
          if (user_power->jac_error) values[0] = 50.0;
          ierr = MatSetValues(J,1,row,1,col,values,ADD_VALUES);CHKERRQ(ierr);
	}
      }
    }
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_subPower(SNES snes,Vec X, Mat J,Mat Jpre,void *appctx)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            localX;
  PetscInt       nv,ne,i,j,offset,nvar,row;
  const PetscInt *vtx,*edges;
  PetscBool      ghostvtex;
  PetscScalar    one = 1.0;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  //printf("FormJacobian_subPower...\n");
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  /* Power subnetwork: copied from pflow/FormJacobian_Power() */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormJacobian_Power_private(networkdm,localX,J,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  /* Water subnetwork: Identity */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  for (i=0; i<nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableGlobalOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumVariables(networkdm,vtx[i],&nvar);CHKERRQ(ierr);
    for (j=0; j<nvar; j++) {
      row = offset + j;
      ierr = MatSetValues(J,1,&row,1,&row,&one,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Dummy equation localF(X) = localX - localXold */
PetscErrorCode FormFunction_Dummy(DM networkdm,Vec localX, Vec localF,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarr,*xoldarr;
  PetscScalar       *farr;
  PetscInt          i,j,offset,nvar;
  PetscBool         ghostvtex;
  UserCtx           *user = (UserCtx*)appctx;
  Vec               localXold = user->localXold;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localXold,&xoldarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  for (i=0; i<nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if(ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumVariables(networkdm,vtx[i],&nvar);CHKERRQ(ierr);
    for (j=0; j<nvar; j++) {
      farr[offset+j] = xarr[offset+j] - xoldarr[offset+j];
    }
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localXold,&xoldarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction_Power(DM networkdm,Vec localX, Vec localF,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  PetscErrorCode    ierr;
  UserCtx_Power     *User=(UserCtx_Power*)appctx;
  PetscInt          e,v,vfrom,vto;
  const PetscScalar *xarr;
  PetscScalar       *farr;
  PetscInt          offsetfrom,offsetto,offset,i,j,key,numComps;
  PetscScalar       Vm;
  PetscScalar       Sbase=User->Sbase;
  VERTEX_Power      bus=NULL;
  GEN               gen;
  LOAD              load;
  PetscBool         ghostvtex;
  void*             component;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  for (v=0; v<nv; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[v],&numComps);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,vtx[v],&offset);CHKERRQ(ierr);

    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetComponent(networkdm,vtx[v],j,&key,&component);CHKERRQ(ierr);
      if (key == 1) {
        PetscInt       nconnedges;
	const PetscInt *connedges;

	bus = (VERTEX_Power)(component);
	/* Handle reference bus constrained dofs */
	if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
	  farr[offset] = xarr[offset] - bus->va*PETSC_PI/180.0;
	  farr[offset+1] = xarr[offset+1] - bus->vm;
	  break;
	}

	if (!ghostvtex) {
	  Vm = xarr[offset+1];

	  /* Shunt injections */
	  farr[offset] += Vm*Vm*bus->gl/Sbase;
	  if(bus->ide != PV_BUS) farr[offset+1] += -Vm*Vm*bus->bl/Sbase;
	}

	ierr = DMNetworkGetSupportingEdges(networkdm,vtx[v],&nconnedges,&connedges);CHKERRQ(ierr);
	for (i=0; i < nconnedges; i++) {
	  EDGE_Power       branch;
	  PetscInt       keye;
          PetscScalar    Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
          const PetscInt *cone;
          PetscScalar    Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

	  e = connedges[i];
	  ierr = DMNetworkGetComponent(networkdm,e,0,&keye,(void**)&branch);CHKERRQ(ierr);
	  if (!branch->status) continue;
	  Gff = branch->yff[0];
	  Bff = branch->yff[1];
	  Gft = branch->yft[0];
	  Bft = branch->yft[1];
	  Gtf = branch->ytf[0];
	  Btf = branch->ytf[1];
	  Gtt = branch->ytt[0];
	  Btt = branch->ytt[1];

	  ierr = DMNetworkGetConnectedVertices(networkdm,e,&cone);CHKERRQ(ierr);
	  vfrom = cone[0];
	  vto   = cone[1];

	  ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offsetfrom);CHKERRQ(ierr);
	  ierr = DMNetworkGetVariableOffset(networkdm,vto,&offsetto);CHKERRQ(ierr);

	  thetaf = xarr[offsetfrom];
	  Vmf     = xarr[offsetfrom+1];
	  thetat = xarr[offsetto];
	  Vmt     = xarr[offsetto+1];
	  thetaft = thetaf - thetat;
	  thetatf = thetat - thetaf;

	  if (vfrom == vtx[v]) {
	    farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*PetscCosScalar(thetaft) + Bft*PetscSinScalar(thetaft));
	    farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*PetscCosScalar(thetaft) + Gft*PetscSinScalar(thetaft));
	  } else {
	    farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*PetscCosScalar(thetatf) + Btf*PetscSinScalar(thetatf));
	    farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*PetscCosScalar(thetatf) + Gtf*PetscSinScalar(thetatf));
	  }
	}
      } else if (key == 2) {
	if (!ghostvtex) {
	  gen = (GEN)(component);
	  if (!gen->status) continue;
	  farr[offset] += -gen->pg/Sbase;
	  farr[offset+1] += -gen->qg/Sbase;
	}
      } else if (key == 3) {
	if (!ghostvtex) {
	  load = (LOAD)(component);
	  farr[offset] += load->pl/Sbase;
	  farr[offset+1] += load->ql/Sbase;
	}
      }
    }
    if (bus && bus->ide == PV_BUS) {
      farr[offset+1] = xarr[offset+1] - bus->vm;
    }
  }
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *appctx)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            localX,localF;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  PetscMPIInt    rank;
  MPI_Comm       comm;
  UserCtx        *user = (UserCtx*)appctx;
  UserCtx_Power  user_power = (*user).user_power;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecSet(localF,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Form Function for power subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (user->subsnes_id == 1) { /* snes_water only */
    ierr = FormFunction_Dummy(networkdm,localX,localF,nv,ne,vtx,edges,user);CHKERRQ(ierr);
  } else {
    ierr = FormFunction_Power(networkdm,localX,localF,nv,ne,vtx,edges,&user_power);CHKERRQ(ierr);
  }

  /* Form Function for water subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (user->subsnes_id == 0) { /* snes_power only */
    ierr = FormFunction_Dummy(networkdm,localX,localF,nv,ne,vtx,edges,user);CHKERRQ(ierr);
  } else {
    ierr = FormFunction_Water(networkdm,localX,localF,nv,ne,vtx,edges,NULL);CHKERRQ(ierr);
  }

  /* Form Function for the coupling subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,2,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (ne) {
    const PetscInt *cone;
    PetscInt       key,offset,i,j,numComps;
    PetscBool      ghostvtex;
    PetscScalar    *farr;
    void*          component;

    ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);
    ierr = DMNetworkGetConnectedVertices(networkdm,edges[0],&cone);CHKERRQ(ierr);
#if 0
    PetscInt       vid[2];
    ierr = DMNetworkGetGlobalVertexIndex(networkdm,cone[0],&vid[0]);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVertexIndex(networkdm,cone[1],&vid[1]);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Formfunction, coupling subnetwork: nv %d, ne %d; connected vertices %d %d\n",rank,nv,ne,vid[0],vid[1]);CHKERRQ(ierr);
#endif

    for (i=0; i<2; i++) {
      ierr = DMNetworkIsGhostVertex(networkdm,cone[i],&ghostvtex);CHKERRQ(ierr);
      if (!ghostvtex) {
        ierr = DMNetworkGetNumComponents(networkdm,cone[i],&numComps);CHKERRQ(ierr);
        for (j=0; j<numComps; j++) {
          ierr = DMNetworkGetComponent(networkdm,cone[i],j,&key,&component);CHKERRQ(ierr);
          if (key == 3) { /* a load vertex in power subnet */
            ierr = DMNetworkGetVariableOffset(networkdm,cone[i],&offset);CHKERRQ(ierr);
            //printf("[%d] v_power load: ...Flocal[%d]= %g, %g\n",rank,offset,farr[offset],farr[offset+1]);
          } else if (key == 5) { /* a vertex in water subnet */
            ierr = DMNetworkGetVariableOffset(networkdm,cone[i],&offset);CHKERRQ(ierr);
            //printf("[%d] v_water:     ...Flocal[%d]= %g\n",rank,offset,farr[offset]);
          }
        }
      }
    }
    ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  }

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);
  /* ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess_Power(DM networkdm,Vec localX,PetscInt nv,PetscInt ne, const PetscInt *vtx, const PetscInt *edges,void* appctx)
{
  PetscErrorCode ierr;
  VERTEX_Power   bus;
  PetscInt       i;
  GEN            gen;
  PetscBool      ghostvtex;
  PetscScalar    *xarr;
  PetscInt       key,numComps,j,offset;
  void*          component;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[i],&numComps);CHKERRQ(ierr);
    for (j=0; j < numComps; j++) {
      ierr = DMNetworkGetComponent(networkdm,vtx[i],j,&key,&component);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEX_Power)(component);
	xarr[offset] = bus->va*PETSC_PI/180.0;
	xarr[offset+1] = bus->vm;
      } else if(key == 2) {
	gen = (GEN)(component);
	if (!gen->status) continue;
	xarr[offset+1] = gen->vs;
	break;
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess_Water(DM networkdm,Vec localX,PetscInt nv,PetscInt ne, const PetscInt *vtx, const PetscInt *edges,void* appctx)
{
  PetscErrorCode ierr;
  PetscInt       i,offset,key;
  PetscBool      ghostvtex;
  VERTEX_Water   vertex;
  PetscScalar    *xarr;
  //PetscMPIInt    rank;
  //MPI_Comm       comm;

  PetscFunctionBegin;
  //ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
  //ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  for (i=0; i < nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;
    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(networkdm,vtx[i],0,&key,(void**)&vertex);CHKERRQ(ierr);
    if (key != 5) SETERRQ1(PETSC_COMM_SELF,0,"not a VERTEX_Water, key = %d",key);

    if (vertex->type == VERTEX_TYPE_JUNCTION) {
      xarr[offset] = 100;
    } else if (vertex->type == VERTEX_TYPE_RESERVOIR) {
      xarr[offset] = vertex->res.head;
    } else {
      xarr[offset] = vertex->tank.initlvl + vertex->tank.elev;
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess(DM networkdm,Vec X,void* appctx)
{
  PetscErrorCode ierr;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  UserCtx        *user = (UserCtx*)appctx;
  Vec            localX = user->localXold;
  //PetscMPIInt    rank;
  //MPI_Comm       comm;

  PetscFunctionBegin;
  //ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
  //ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = VecSet(localX,0.0);CHKERRQ(ierr);

  /* Set initial guess for power subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialGuess_Power(networkdm,localX,nv,ne,vtx,edges,NULL);CHKERRQ(ierr);
  //ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Power subnetwork: nv %d, ne %d\n",rank,nv,ne);
  //ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  /* Set initial guess for water subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialGuess_Water(networkdm,localX,nv,ne,vtx,edges,NULL);CHKERRQ(ierr);
  //ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Water subnetwork: nv %d, ne %d\n",rank,nv,ne);CHKERRQ(ierr);
  //ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
#if 0
  /* Set initial guess for the coupling subnet */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,2,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (ne) {
    const PetscInt *cone;
    ierr = DMNetworkGetConnectedVertices(networkdm,edges[0],&cone);CHKERRQ(ierr);
  }
#endif

  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode   ierr;
  DM               networkdm;
  PetscLogStage    stage[4];
  PetscMPIInt      crank;
  PetscInt         nsubnet = 3,numVertices[3],NumVertices[3],numEdges[3],NumEdges[3];
  PetscInt         i,j,*edgelist[3],nv,ne,componentkey[6];
  const PetscInt   *vtx,*edges;
  Vec              X,F;
  SNES             snes,snes_power,snes_water;
  Mat              Jac;
  PetscBool        viewJ=PETSC_FALSE,viewX=PETSC_FALSE;
  UserCtx          user;
  PetscInt         it_max=10;
  SNESConvergedReason reason=-1;

  /* Power subnetwork */
  char             pfdata_file[PETSC_MAX_PATH_LEN]="pflow/datafiles/case9.m";
  PFDATA           *pfdata;
  PetscInt         genj,loadj;
  int              *edgelist_power=NULL;
  PetscScalar      Sbase;

  /* Water subnetwork */
  WATERDATA        *waterdata;
  char             waterdata_file[PETSC_MAX_PATH_LEN]="waternet/sample1.inp";
  int              *edgelist_water=NULL;

  /* Coupling subnetwork */
  int              *edgelist_couple=NULL;

  ierr = PetscInitialize(&argc,&argv,"ex1options",help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&crank);CHKERRQ(ierr);

  /* (1) Read Data - Only rank 0 reads the data */
  /*--------------------------------------------*/
  ierr = PetscLogStageRegister("Read Data",&stage[0]);CHKERRQ(ierr);
  PetscLogStagePush(stage[0]);

  for (i=0; i<nsubnet; i++) {
    numVertices[i] = 0; NumVertices[i] = PETSC_DETERMINE;
    numEdges[i]    = 0; NumEdges[i]    = PETSC_DETERMINE;
  }

  /* READ THE DATA FOR THE FIRST SUBNETWORK: Electric Power Grid */
  if (!crank) {
    ierr = PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
    ierr = PetscNew(&pfdata);CHKERRQ(ierr);
    ierr = PFReadMatPowerData(pfdata,pfdata_file);CHKERRQ(ierr);
    Sbase = pfdata->sbase;

    numEdges[0]    = pfdata->nbranch;
    numVertices[0] = pfdata->nbus;

    ierr = PetscMalloc1(2*numEdges[0],&edgelist_power);CHKERRQ(ierr);
    ierr = GetListofEdges_Power(pfdata,edgelist_power);CHKERRQ(ierr);
    printf("edgelist_power:\n");
    for (i=0; i<numEdges[0]; i++) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%D %D]",edgelist_power[2*i],edgelist_power[2*i+1]);CHKERRQ(ierr);
    }
    printf("\n");
  }
  /* Broadcast power Sbase to all processors */
  ierr = MPI_Bcast(&Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  user.user_power.Sbase = Sbase;
  user.user_power.jac_error = PETSC_FALSE;
  /* If external option activated. Introduce error in jacobian */
  ierr = PetscOptionsHasName(NULL,NULL, "-jac_error", &user.user_power.jac_error);CHKERRQ(ierr);

  /* GET DATA FOR THE SECOND SUBNETWORK: Waternet */
  ierr = PetscNew(&waterdata);CHKERRQ(ierr);
  if (!crank) {
    ierr = PetscOptionsGetString(NULL,NULL,"-waterdata",waterdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
    ierr = WaterNetReadData(waterdata,waterdata_file);CHKERRQ(ierr);

    ierr = PetscCalloc1(2*waterdata->nedge,&edgelist_water);CHKERRQ(ierr);
    ierr = GetListofEdges_Water(waterdata,edgelist_water);CHKERRQ(ierr);
    numEdges[1]    = waterdata->nedge;
    numVertices[1] = waterdata->nvertex;

    printf("edgelist_water:\n");
    for (i=0; i<numEdges[1]; i++) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%D %D]",edgelist_water[2*i],edgelist_water[2*i+1]);CHKERRQ(ierr);
    }
    printf("\n");
  }

  /* Get data for the coupling subnetwork */
  if (!crank) {
    numEdges[2] = 1; numVertices[2] = 0;
    ierr = PetscMalloc1(4*numEdges[2],&edgelist_couple);CHKERRQ(ierr);
    edgelist_couple[0] = 0; edgelist_couple[1] = 4; /* from node: net[0] vertex[4] */
    edgelist_couple[2] = 1; edgelist_couple[3] = 0; /* to node:   net[1] vertex[0] */
  }
  PetscLogStagePop();

  /* (2) Create network */
  /*--------------------*/
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Net Setup",&stage[1]);CHKERRQ(ierr);
  PetscLogStagePush(stage[1]);

  /* Create an empty network object */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);

  /* Register the components in the network */
  ierr = DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&componentkey[0]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&componentkey[1]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&componentkey[2]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&componentkey[3]);CHKERRQ(ierr);

  ierr = DMNetworkRegisterComponent(networkdm,"edge_water",sizeof(struct _p_EDGE_Water),&componentkey[4]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"vertex_water",sizeof(struct _p_VERTEX_Water),&componentkey[5]);CHKERRQ(ierr);


  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] local nvertices %d %d; nedges %d %d\n",crank,numVertices[0],numVertices[1],numEdges[0],numEdges[1]);
  ierr = DMNetworkSetSizes(networkdm,nsubnet,numVertices,numEdges,NumVertices,NumEdges);CHKERRQ(ierr);

  /* Add edge connectivity */
  edgelist[0] = edgelist_power;
  edgelist[1] = edgelist_water;
  edgelist[2] = edgelist_couple;
  ierr = DMNetworkSetEdgeList(networkdm,edgelist);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUpCoupled(networkdm);CHKERRQ(ierr);

  /* Add network components - only process[0] has any data to add */
  /* ADD VARIABLES AND COMPONENTS FOR THE POWER SUBNETWORK */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (!crank) printf("[%d] Power network: nv %d, ne %d\n",crank,nv,ne);
  genj = 0; loadj = 0;
  for (i = 0; i < ne; i++) {
    ierr = DMNetworkAddComponent(networkdm,edges[i],componentkey[0],&pfdata->branch[i]);CHKERRQ(ierr);
  }

  for (i = 0; i < nv; i++) {
    ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[1],&pfdata->bus[i]);CHKERRQ(ierr);
    if (pfdata->bus[i].ngen) {
      for (j = 0; j < pfdata->bus[i].ngen; j++) {
        ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[2],&pfdata->gen[genj++]);CHKERRQ(ierr);
      }
    }
    if (pfdata->bus[i].nload) {
      for (j=0; j < pfdata->bus[i].nload; j++) {
        ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[3],&pfdata->load[loadj++]);CHKERRQ(ierr);
      }
    }
    /* Add number of variables */
    ierr = DMNetworkAddNumVariables(networkdm,vtx[i],2);CHKERRQ(ierr);
  }

  /* ADD VARIABLES AND COMPONENTS FOR THE WATER SUBNETWORK */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (!crank) printf("[%d] Water network: nv %d, ne %d\n",crank,nv,ne);
  for (i = 0; i < ne; i++) {
    ierr = DMNetworkAddComponent(networkdm,edges[i],componentkey[4],&waterdata->edge[i]);CHKERRQ(ierr);
  }

  for (i = 0; i < nv; i++) {
    ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[5],&waterdata->vertex[i]);CHKERRQ(ierr);
    /* Add number of variables */
    ierr = DMNetworkAddNumVariables(networkdm,vtx[i],1);CHKERRQ(ierr);
  }

  /* Set up DM for use */
  ierr = DMSetUp(networkdm);CHKERRQ(ierr);

  /* Distribute networkdm to multiple processes */
  ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&user.localXold);CHKERRQ(ierr);

  PetscLogStagePop();

  /* (3) Setup Solvers */
  /*-------------------*/
  ierr = PetscOptionsGetBool(NULL,NULL,"-viewJ",&viewJ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-viewX",&viewX,NULL);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("SNES Setup",&stage[2]);CHKERRQ(ierr);
  PetscLogStagePush(stage[2]);

#if 1
  ierr = SetInitialGuess(networkdm,X,&user);CHKERRQ(ierr);
  //ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Create coupled snes */
  /*-------------------- */
  if (!crank) printf("SNES setup ......\n");
  user.subsnes_id = nsubnet-1;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,"coupled_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESMonitorSet(snes,UserMonitor,&user,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  //ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

  if (viewJ) {
    /* View Jac structure */
    ierr = SNESGetJacobian(snes,&Jac,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatView(Jac,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  if (viewX) {
    if (!crank) printf("Solution:\n");
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (viewJ) {
    /* View assembled Jac */
    ierr = MatView(Jac,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
#endif

#if 1
  /* Create snes_power */
  /*-------------------*/
  if (!crank) printf("SNES_power setup ......\n");
  ierr = SetInitialGuess(networkdm,X,&user);CHKERRQ(ierr);

  user.subsnes_id = 0;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes_power);CHKERRQ(ierr);
  ierr = SNESSetDM(snes_power,networkdm);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes_power,"power_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_power,F,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESMonitorSet(snes_power,UserMonitor,&user,NULL);CHKERRQ(ierr);

  /* Use user-provide Jacobian */
  ierr = DMCreateMatrix(networkdm,&Jac);CHKERRQ(ierr);
  //ierr = FormJacobian_subPower(snes_power,X,Jac,Jac,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_power,Jac,Jac,FormJacobian_subPower,&user);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes_power);CHKERRQ(ierr);
  //ierr = SNESSolve(snes_power,NULL,X);CHKERRQ(ierr);

  if (viewX) {
    if (!crank) printf("Power Solution:\n");
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (viewJ) {
    if (!crank) printf("Power Jac:\n");
    ierr = SNESGetJacobian(snes_power,&Jac,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatView(Jac,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    /* ierr = MatView(Jac,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  }
#endif

#if 1
  /* Create snes_water */
  /*-------------------*/
  if (!crank) printf("SNES_water setup......\n");
  ierr = SetInitialGuess(networkdm,X,&user);CHKERRQ(ierr);

  user.subsnes_id = 1;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes_water);CHKERRQ(ierr);
  ierr = SNESSetDM(snes_water,networkdm);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes_water,"water_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_water,F,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESMonitorSet(snes_water,UserMonitor,&user,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_water);CHKERRQ(ierr);
  //ierr = SNESSolve(snes_water,NULL,X);CHKERRQ(ierr);

  if (viewX) {
    if (!crank) printf("Water Solution:\n");
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
#endif
  PetscLogStagePop();

#if 1
  /* (4) Solve */
  /*-----------*/
  ierr = PetscLogStageRegister("SNES Solve",&stage[3]);CHKERRQ(ierr);
  PetscLogStagePush(stage[3]);
  user.it = 0;
  while (user.it < it_max && reason<0) {
    user.subsnes_id = 0;
    ierr = SNESSolve(snes_power,NULL,X);CHKERRQ(ierr);

    user.subsnes_id = 1;
    ierr = SNESSolve(snes_water,NULL,X);CHKERRQ(ierr);

    user.subsnes_id = nsubnet-1;
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

    ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
    user.it++;
  }
  if (!crank) printf("SNES converged in %d iterations\n",user.it);
  if (viewX) {
    if (!crank) printf("Final Solution:\n");
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   }
#endif
  PetscLogStagePop();

  /* Free objects */
  /* -------------*/
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&user.localXold);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = MatDestroy(&Jac);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes_power);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes_water);CHKERRQ(ierr);

  if (!crank) {
    ierr = PetscFree(edgelist_power);CHKERRQ(ierr);
    ierr = PetscFree(pfdata->bus);CHKERRQ(ierr);
    ierr = PetscFree(pfdata->gen);CHKERRQ(ierr);
    ierr = PetscFree(pfdata->branch);CHKERRQ(ierr);
    ierr = PetscFree(pfdata->load);CHKERRQ(ierr);
    ierr = PetscFree(pfdata);CHKERRQ(ierr);

    ierr = PetscFree(edgelist_water);CHKERRQ(ierr);
    ierr = PetscFree(waterdata->vertex);CHKERRQ(ierr);
    ierr = PetscFree(waterdata->edge);CHKERRQ(ierr);

    ierr = PetscFree(edgelist_couple);CHKERRQ(ierr);
  }
  ierr = PetscFree(waterdata);CHKERRQ(ierr);
  ierr = DMDestroy(&networkdm);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
