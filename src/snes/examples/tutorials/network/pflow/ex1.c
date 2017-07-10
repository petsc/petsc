static char help[] = "This example demonstrates the use of DMNetwork interface with subnetworks for solving a coupled nonlinear \n\
                      electric power grid and water pipe problem.\n\
                      The available solver options are in the pfoptions file and the data files are in the datafiles directory.\n\
                      The electric power grid data file format used is from the MatPower package \n\
                      (http://www.pserc.cornell.edu//matpower/).\n\
                      This example shows the use of subnetwork feature in DMNetwork. \n\
                      Run this program: mpiexec -n <n> ./ex1 \n\\n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

#include "pf.h"
#include "wash.h"

PetscErrorCode GetListofEdges(PetscInt nbranches, EDGEDATA branch,int edges[])
{
  PetscInt       i, fbus,tbus;

  PetscFunctionBegin;
  for (i=0; i < nbranches; i++) {
    fbus = branch[i].internal_i;
    tbus = branch[i].internal_j;
    edges[2*i]   = fbus;
    edges[2*i+1] = tbus;
  }
  PetscFunctionReturn(0);
}

typedef struct{
  PetscScalar  Sbase;
}UserCtx;

PetscErrorCode FormFunction_Power(DM networkdm,Vec localX, Vec localF,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  PetscErrorCode ierr;
  UserCtx       *User=(UserCtx*)appctx;
  PetscInt      e;
  PetscInt      v,vfrom,vto;
  const PetscScalar *xarr;
  PetscScalar   *farr;
  PetscInt      offsetfrom,offsetto,offset;
  DMNetworkComponentGenericDataType *arr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);

  for (v=0; v < nv; v++) {
    PetscInt    i,j,offsetd,key;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEXDATA  bus=NULL;
    GEN         gen;
    LOAD        load;
    PetscBool   ghostvtex;
    PetscInt    numComps;

    ierr = DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[v],&numComps);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,vtx[v],&offset);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetComponentKeyOffset(networkdm,vtx[v],j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 1) {
        PetscInt       nconnedges;
	const PetscInt *connedges;

	bus = (VERTEXDATA)(arr+offsetd);
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
	  EDGEDATA       branch;
	  PetscInt       keye;
          PetscScalar    Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
          const PetscInt *cone;
          PetscScalar    Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

	  e = connedges[i];
	  ierr = DMNetworkGetComponentKeyOffset(networkdm,e,0,&keye,&offsetd);CHKERRQ(ierr);
	  branch = (EDGEDATA)(arr+offsetd);
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
	  gen = (GEN)(arr+offsetd);
	  if (!gen->status) continue;
	  farr[offset] += -gen->pg/Sbase;
	  farr[offset+1] += -gen->qg/Sbase;
	}
      } else if (key == 3) {
	if (!ghostvtex) {
	  load = (LOAD)(arr+offsetd);
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

PetscErrorCode FormFunction_Wash(DM networkdm,Vec localX, Vec localF,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarr;
  PetscScalar       *farr,*juncf,*pipef,aux,dx;
  PetscInt          varoffset,offsetd,v,e,numComps,j,key;
  Junction          junction;
  Pipe              pipe;
  PipeField         *pipex,*juncx;
  PetscBool         ghostvtex;
  const PetscInt    *cone;
  PetscInt          vfrom,vto,offsetfrom,offsetto,junctoffset,nend,i;
  DMNetworkComponentGenericDataType *arr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);
  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);

  /* Vertex/junction loop */
  for (v=0; v<nv; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetNumComponents(networkdm,vtx[v],&numComps);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,vtx[v],&varoffset);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetComponentKeyOffset(networkdm,vtx[v],j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 5) {
        //printf(" v %d: junction\n",vtx[v]);
        junction = (Junction)(arr + offsetd);
        juncx    = (PipeField*)(xarr + varoffset);
        juncf    = (PetscScalar*)(farr + varoffset);

        /* junction:
           juncf[0] = -qJ + sum(q_downstream); juncf[1] = qJ - sum(q_upstream) */
        if (junction->type == NONE) {
          juncf[0] = -juncx[0].q;
          juncf[1] =  juncx[0].q;
        } else if (junction->type == RESERVOIR) { /* upstream reservoir */
          juncf[0] = juncx[0].q;
          juncf[1] = juncx[0].h - junction->reservoir.hres;
        } else if (junction->type == DEMAND) { /* discharge valve */
          juncf[0] = -juncx[0].q;
          juncf[1] = juncx[0].q - junction->demand.q0;
        } else if (junction->type == VALVE) { /* discharge valve*/
          juncf[0] = -juncx[0].q;
          juncf[1] = pow(junction->valve.cdag, 2)*(2*GRAV*juncx[0].h) - pow(juncx[0].q,2);
        }
      }
    }
  }

  /* Edge/pipe loop */
  for (e=0; e<ne; e++) {
    ierr = DMNetworkGetNumComponents(networkdm,edges[e],&numComps);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,edges[e],&varoffset);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetComponentKeyOffset(networkdm,edges[e],j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 4) {
        pipe     = (Pipe)(arr + offsetd);
        pipex    = (PipeField*)(xarr + varoffset);
        pipef    = (PetscScalar*)(farr + varoffset);

        aux = (pipe->length/(pipe->nnodes - 1))*((pipe->R)/(GRAV*pipe->A));

        /* Get boundary values from connected vertices */
        ierr = DMNetworkGetConnectedVertices(networkdm,edges[e],&cone);CHKERRQ(ierr); 
        vfrom = cone[0]; /* local ordering */
        vto   = cone[1];
        /* printf(" e %d: pipe, aux %g; vfrom/to: %d %d\n",edges[e],aux,vfrom,vto); */
        ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offsetfrom);CHKERRQ(ierr);
        ierr = DMNetworkGetVariableOffset(networkdm,vto,&offsetto);CHKERRQ(ierr);

        /* Upstream boundary */
        ierr = DMNetworkGetComponentKeyOffset(networkdm,vfrom,5,NULL,&junctoffset);CHKERRQ(ierr);
        junction = (Junction)(arr + junctoffset);
        juncx    = (PipeField*)(xarr + offsetfrom);
        juncf    = (PetscScalar*)(farr + offsetfrom);

        pipef[1] = pipex[0].h - juncx[0].h;
        if (junction->type != RESERVOIR) juncf[0] -= pipex[0].q;

        /* Downstream boundary */
        ierr = DMNetworkGetComponentKeyOffset(networkdm,vto,5,NULL,&junctoffset);CHKERRQ(ierr);
        junction = (Junction)(arr + junctoffset);
        juncx    = (PipeField*)(xarr + offsetto);
        juncf    = (PetscScalar*)(farr + offsetto);
        nend     = pipe->nnodes - 1;

        pipef[2*nend + 1] = pipex[nend].h - juncx[0].h;
        if (junction->type != RESERVOIR) juncf[0] += pipex[nend].q;

        /* Now, write internal node equations 
         q                 = constant
         dh + dx*aux*q*|q| = 0;
         */
        dx  = 1./(pipe->nnodes - 1);
        aux = pipe->length * pipe->R / (GRAV*pipe->A);

        /* the ending nodes */
        pipef[0]      = pipex[0].q - pipex[nend].q;
        pipef[2*nend] = pipex[nend].h - pipex[0].h + aux*pipex[nend].q*PetscAbsScalar(pipex[nend].q);

        /* the internal nodes */
        for (i = 1; i < pipe->nnodes - 1; i++) {
          pipef[2*i]     = pipex[i].q - pipex[i+1].q;
          pipef[2*i + 1] = pipex[i].h - pipex[i - 1].h + dx*aux*pipex[i].q*PetscAbsScalar(pipex[i].q);
        }
      }
    }
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FormFunction(SNES snes,Vec X, Vec F,void *appctx)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            localX,localF;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecSet(localF,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Form Function for first subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormFunction_Power(networkdm,localX,localF,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  /* Form Function for second subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormFunction_Wash(networkdm,localX,localF,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);
  /* ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_Subnet(DM networkdm,Vec localX, Mat J, Mat Jpre, PetscInt nv, PetscInt ne, const PetscInt *vtx, const PetscInt *edges, void *appctx)
{
  PetscErrorCode ierr;
  UserCtx       *User=(UserCtx*)appctx;
  PetscInt      e;
  PetscInt      v,vfrom,vto;
  const PetscScalar *xarr;
  PetscInt      offsetfrom,offsetto,goffsetfrom,goffsetto;
  DMNetworkComponentGenericDataType *arr;
  PetscInt      row[2],col[8];
  PetscScalar   values[8];

  PetscFunctionBegin;

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);

  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);

  for (v=0; v < nv; v++) {
    PetscInt    i,j,offsetd,key;
    PetscInt    offset,goffset;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEXDATA  bus;
    PetscBool   ghostvtex;
    PetscInt    numComps;

    ierr = DMNetworkIsGhostVertex(networkdm,vtx[v],&ghostvtex);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[v],&numComps);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetVariableOffset(networkdm,vtx[v],&offset);CHKERRQ(ierr);
      ierr = DMNetworkGetVariableGlobalOffset(networkdm,vtx[v],&goffset);CHKERRQ(ierr);
      ierr = DMNetworkGetComponentKeyOffset(networkdm,vtx[v],j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 1) {
        PetscInt       nconnedges;
	const PetscInt *connedges;

	bus = (VERTEXDATA)(arr+offsetd);
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
	  EDGEDATA       branch;
	  VERTEXDATA     busf,bust;
	  PetscInt       offsetfd,offsettd,keyf,keyt;
          PetscScalar    Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
          const PetscInt *cone;
          PetscScalar    Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

	  e = connedges[i];
	  ierr = DMNetworkGetComponentKeyOffset(networkdm,e,0,&key,&offsetd);CHKERRQ(ierr);
	  branch = (EDGEDATA)(arr+offsetd);
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
	  ierr = DMNetworkGetVariableGlobalOffset(networkdm,vfrom,&goffsetfrom);CHKERRQ(ierr);
	  ierr = DMNetworkGetVariableGlobalOffset(networkdm,vto,&goffsetto);CHKERRQ(ierr);

	  if (goffsetto < 0) goffsetto = -goffsetto - 1;

	  thetaf = xarr[offsetfrom];
	  Vmf     = xarr[offsetfrom+1];
	  thetat = xarr[offsetto];
	  Vmt     = xarr[offsetto+1];
	  thetaft = thetaf - thetat;
	  thetatf = thetat - thetaf;

	  ierr = DMNetworkGetComponentKeyOffset(networkdm,vfrom,0,&keyf,&offsetfd);CHKERRQ(ierr);
	  ierr = DMNetworkGetComponentKeyOffset(networkdm,vto,0,&keyt,&offsettd);CHKERRQ(ierr);
	  busf = (VERTEXDATA)(arr+offsetfd);
	  bust = (VERTEXDATA)(arr+offsettd);

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
	  row[0] = goffset+1; col[0] = goffset+1;
	  values[0]  = 1.0;
	  ierr = MatSetValues(J,1,row,1,col,values,ADD_VALUES);CHKERRQ(ierr);
	}
      }
    }
  }
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian(SNES snes,Vec X, Mat J,Mat Jpre,void *appctx)
{
  PetscErrorCode ierr;
  DM            networkdm;
  Vec           localX;
  PetscInt      ne,nv;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Form Jacobian for first subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormJacobian_Subnet(networkdm,localX,J,Jpre,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  /* Form Jacobian for second subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormJacobian_Subnet(networkdm,localX,J,Jpre,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues_Power(DM networkdm,Vec localX,PetscInt nv,PetscInt ne, const PetscInt *vtx, const PetscInt *edges,void* appctx)
{
  PetscErrorCode ierr;
  VERTEXDATA     bus;
  PetscInt       i;
  GEN            gen;
  PetscBool      ghostvtex;
  PetscScalar    *xarr;
  PetscInt       key,numComps,j,offset,offsetd;
  DMNetworkComponentGenericDataType *arr;

  PetscFunctionBegin;
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[i],&numComps);CHKERRQ(ierr);
    for (j=0; j < numComps; j++) {
      ierr = DMNetworkGetComponentKeyOffset(networkdm,vtx[i],j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEXDATA)(arr+offsetd);
	xarr[offset] = bus->va*PETSC_PI/180.0;
	xarr[offset+1] = bus->vm;
      } else if(key == 2) {
	gen = (GEN)(arr+offsetd);
	if (!gen->status) continue;
	xarr[offset+1] = gen->vs;
	break;
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues_Wash(DM networkdm,Vec localX,PetscInt nv,PetscInt ne, const PetscInt *vtx, const PetscInt *edges,void* appctx)
{
  PetscErrorCode ierr;
  PetscBool      ghostvtex;
  PetscScalar    *xarr;
  PetscInt       i,key,numComps,j,offset,offsetd,k,nvar;
  Pipe           pipe;
  DMNetworkComponentGenericDataType *arr;

  PetscFunctionBegin;

  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);
  for (i = 0; i < ne; i++) {
    ierr = DMNetworkGetVariableOffset(networkdm,edges[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,edges[i],&numComps);CHKERRQ(ierr);
    for (j=0; j < numComps; j++) {
      ierr = DMNetworkGetComponentKeyOffset(networkdm,edges[i],j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 4) { /* pipe */
        pipe = (Pipe)(arr + offsetd);
        ierr = PipeSetUp(pipe);CHKERRQ(ierr);  /* creates pipe->da, must be called after DMNetworkDistribute() */

        ierr = DMNetworkGetNumVariables(networkdm,edges[i],&nvar);CHKERRQ(ierr);
        /* printf("SetInitialValues_Wash edge %d, nvar %d\n",edges[i],nvar); */
        for (k=0; k<nvar; k++) xarr[offset + k] = 1.0;
      }
    }
  }

  for (i = 0; i < nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,vtx[i],&numComps);CHKERRQ(ierr);
    for (j=0; j < numComps; j++) {
      ierr = DMNetworkGetComponentKeyOffset(networkdm,vtx[i],j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 5) { /* junction */
        ierr = DMNetworkGetNumVariables(networkdm,vtx[i],&nvar);CHKERRQ(ierr);
        /* printf("SetInitialValues_Wash vertex %d, nvar %d\n",vtx[i],nvar); */
        for (k=0; k<nvar; k++) xarr[offset + k] = 1.0;
      }
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues(DM networkdm, Vec X,void* appctx) 
{
  PetscErrorCode ierr;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Set initial guess for first subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialValues_Power(networkdm,localX,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  /* Set initial guess for second subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialValues_Wash(networkdm,localX,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscErrorCode   ierr;
  PetscInt         numEdges1=0,numVertices1=0,numEdges2=0,numVertices2=0;
  int              *edgelist1 = NULL,*edgelist2 = NULL;
  DM               networkdm;
  PetscInt         componentkey[6];
  UserCtx          User;
  PetscLogStage    stage1,stage2;
  PetscMPIInt      rank;
  PetscInt         nsubnet = 2;
  PetscInt         numVertices[2],NumVertices[2];
  PetscInt         numEdges[2],NumEdges[2];
  PetscInt         *edgelist[2]; 
  PetscInt         nv,ne;
  const PetscInt   *vtx;
  const PetscInt   *edges;
  PetscInt         i,j;
  Vec              X,F;
  SNES             snes;

  char             pfdata_file[PETSC_MAX_PATH_LEN]="datafiles/case9.m";
  PFDATA           *pfdata1;
  PetscInt         genj,loadj;

  Wash             wash;
  PetscInt         KeyPipe,KeyJunction,washCase=0;
  Pipe             pipe;
  Junction         junction;
  PetscBool        parseflg=PETSC_FALSE;
  char             filename[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc,&argv,"pfoptions",help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  {
    /* introduce the const crank so the clang static analyzer realizes that if it enters any of the if (crank) then it must have entered the first */
    /* this is an experiment to see how the analyzer reacts */
    const PetscMPIInt crank = rank;

    /* Create an empty network object */
    ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);

    /* Register the components in the network */
    ierr = DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGEDATA),&componentkey[0]);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEXDATA),&componentkey[1]);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&componentkey[2]);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&componentkey[3]);CHKERRQ(ierr);

    ierr = DMNetworkRegisterComponent(networkdm,"pipestruct",sizeof(struct _p_Pipe),&componentkey[4]);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"junctionstruct",sizeof(struct _p_Junction),&componentkey[5]);CHKERRQ(ierr);
    KeyPipe     = componentkey[4];
    KeyJunction = componentkey[5];

    ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
    PetscLogStagePush(stage1);

    /* READ THE DATA - Only rank 0 reads the data */
    if (!crank) {
      /* READ DATA FOR THE FIRST SUBNETWORK - Electric Power Grid */
      ierr = PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
      ierr = PetscNew(&pfdata1);CHKERRQ(ierr);
      ierr = PFReadMatPowerData(pfdata1,pfdata_file);CHKERRQ(ierr);
      User.Sbase = pfdata1->sbase;

      numEdges1 = pfdata1->nbranch;
      numVertices1 = pfdata1->nbus;

      ierr = PetscMalloc1(2*numEdges1,&edgelist1);CHKERRQ(ierr);
      ierr = GetListofEdges(pfdata1->nbranch,pfdata1->branch,edgelist1);CHKERRQ(ierr);

      /* GET DATA FOR THE SECOND SUBNETWORK - Water Pipes */
      ierr = PetscOptionsGetInt(NULL,PETSC_NULL, "-washcase", &washCase, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetString(NULL,PETSC_NULL, "-washdata",filename,PETSC_MAX_PATH_LEN-1,&parseflg);CHKERRQ(ierr);
      if (parseflg) {
        ierr = WashNetworkCreate(PETSC_COMM_SELF,-1,filename,&wash);CHKERRQ(ierr);
      } else {
        washCase = 0;
        ierr = PetscOptionsGetInt(NULL,PETSC_NULL, "-washcase", &washCase, PETSC_NULL);CHKERRQ(ierr);
        ierr = WashNetworkCreate(PETSC_COMM_SELF,washCase,NULL,&wash);CHKERRQ(ierr);
      }
      numEdges2    = wash->npipes;
      numVertices2 = wash->njunctions;
      edgelist2    = wash->edgelist;
      printf("Wash subnetwork: npipes %d, njunctions %d\n",numEdges2,numVertices2);
    }

    PetscLogStagePop();
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
    PetscLogStagePush(stage2);

    /* Set number of nodes/edges */
    numVertices[0] = numVertices1; numVertices[1] = numVertices2;
    NumVertices[0] = PETSC_DETERMINE; NumVertices[1] = PETSC_DETERMINE;
    numEdges[0] = numEdges1; numEdges[1] = numEdges2;
    NumEdges[0] = PETSC_DETERMINE; NumEdges[1] = PETSC_DETERMINE;
    ierr = DMNetworkSetSizes(networkdm,nsubnet,numVertices,numEdges,NumVertices,NumEdges);CHKERRQ(ierr);

    /* Add edge connectivity */
    edgelist[0] = edgelist1; edgelist[1] = edgelist2;
    ierr = DMNetworkSetEdgeList(networkdm,edgelist);CHKERRQ(ierr);

    /* Set up the network layout */
    ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

    /* Add network components only process 0 has any data to add*/
    if (!crank) {
      genj=0; loadj=0;

      /* ADD VARIABLES AND COMPONENTS FOR THE POWER SUBNETWORK */
      ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);

      for (i = 0; i < ne; i++) {
        ierr = DMNetworkAddComponent(networkdm,edges[i],componentkey[0],&pfdata1->branch[i]);CHKERRQ(ierr);
      }

      for (i = 0; i < nv; i++) {
        ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[1],&pfdata1->bus[i]);CHKERRQ(ierr);
        if (pfdata1->bus[i].ngen) {
          for (j = 0; j < pfdata1->bus[i].ngen; j++) {
            ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[2],&pfdata1->gen[genj++]);CHKERRQ(ierr);
          }
        }
        if (pfdata1->bus[i].nload) {
          for (j=0; j < pfdata1->bus[i].nload; j++) {
            ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[3],&pfdata1->load[loadj++]);CHKERRQ(ierr);
          }
        }
        /* Add number of variables */
        ierr = DMNetworkAddNumVariables(networkdm,vtx[i],2);CHKERRQ(ierr);
      }

      /* ADD VARIABLES AND COMPONENTS FOR THE WASH SUBNETWORK */
      ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);

      pipe = wash->pipe;
      for (i = 0; i < ne; i++) {
        ierr = DMNetworkAddComponent(networkdm,edges[i],KeyPipe,&pipe[i]);CHKERRQ(ierr);

        /* Add number of variables to each pipe */
        /* printf(" edge %d, num var %d\n",edges[i],2*pipe[i].nnodes); */
        ierr = DMNetworkAddNumVariables(networkdm,edges[i],2*pipe[i].nnodes);CHKERRQ(ierr);
      }

      junction = wash->junction;
      for (i = 0; i < nv; i++) {
        ierr = DMNetworkAddComponent(networkdm,vtx[i],KeyJunction,&junction[i]);CHKERRQ(ierr);

        /* Add number of variables */
        ierr = DMNetworkAddNumVariables(networkdm,vtx[i],2);CHKERRQ(ierr);
      }
    }

    /* Set up DM for use */
    ierr = DMSetUp(networkdm);CHKERRQ(ierr);

    if (!crank) {
      ierr = PetscFree(edgelist1);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->load);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1);CHKERRQ(ierr);

      ierr = WashNetworkCleanUp(wash);CHKERRQ(ierr);
      ierr = WashNetworkDestroy(&wash);CHKERRQ(ierr);
    }

    /* Distribute networkdm to multiple processes */
    ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

    PetscLogStagePop();

    /* Broadcast Sbase to all processors */
    ierr = MPI_Bcast(&User.Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
    ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

    ierr = SetInitialValues(networkdm,X,&User);CHKERRQ(ierr);
    /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

    /* HOOK UP SOLVER */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,F,FormFunction,&User);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
    /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = VecDestroy(&F);CHKERRQ(ierr);

    /* Destroy objects from each pipe that are created in PipeSetUp()-- See FormFunction_Wash() */
    {
    PetscInt e,numComps,offsetd,keypipe;
    DMNetworkComponentGenericDataType *arr;

    ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);
    ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    for (e=0; e<ne; e++) {
      ierr = DMNetworkGetNumComponents(networkdm,edges[e],&numComps);CHKERRQ(ierr);
      for (j = 0; j < numComps; j++) {
        ierr = DMNetworkGetComponentKeyOffset(networkdm,edges[e],j,&keypipe,&offsetd);CHKERRQ(ierr);
        if (keypipe == 4) {
          Pipe pipe = (Pipe)(arr + offsetd);
          ierr = PipeDestroy(&pipe);CHKERRQ(ierr);
        }
      }
    }
    }

    ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}
