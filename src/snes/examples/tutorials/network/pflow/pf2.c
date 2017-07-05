static char help[] = "This example demonstrates the use of DMNetwork interface with subnetworks for solving a nonlinear electric power grid problem.\n\
                      The available solver options are in the pfoptions file and the data files are in the datafiles directory.\n\
                      The data file format used is from the MatPower package (http://www.pserc.cornell.edu//matpower/).\n\
                      This example shows the use of subnetwork feature in DMNetwork. It creates duplicates of the same network which are treated as subnetworks.\n\
                      Run this program: mpiexec -n <n> ./pf\n\
                      mpiexec -n <n> ./pf \n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

#include "pf.h"
#include <petscdmnetwork.h>

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
  PetscBool    jac_error; /* introduce error in the jacobian */
}UserCtx;

PetscErrorCode FormFunction(SNES snes,Vec X, Vec F,void *appctx)
{
  PetscErrorCode ierr;
  DM             networkdm;
  UserCtx       *User=(UserCtx*)appctx;
  Vec           localX,localF;
  PetscInt      e;
  PetscInt      v,vStart,vEnd,vfrom,vto;
  const PetscScalar *xarr;
  PetscScalar   *farr;
  PetscInt      offsetfrom,offsetto,offset;
  DMNetworkComponentGenericDataType *arr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);

  for (v=vStart; v < vEnd; v++) {
    PetscInt    i,j,offsetd,key;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEXDATA  bus=NULL;
    GEN         gen;
    LOAD        load;
    PetscBool   ghostvtex;
    PetscInt    numComps;

    ierr = DMNetworkIsGhostVertex(networkdm,v,&ghostvtex);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,v,&numComps);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,v,&offset);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetComponentKeyOffset(networkdm,v,j,&key,&offsetd);CHKERRQ(ierr);
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

	ierr = DMNetworkGetSupportingEdges(networkdm,v,&nconnedges,&connedges);CHKERRQ(ierr);
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

	  if (vfrom == v) {
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
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian(SNES snes,Vec X, Mat J,Mat Jpre,void *appctx)
{
  PetscErrorCode ierr;
  DM            networkdm;
  UserCtx       *User=(UserCtx*)appctx;
  Vec           localX;
  PetscInt      e;
  PetscInt      v,vStart,vEnd,vfrom,vto;
  const PetscScalar *xarr;
  PetscInt      offsetfrom,offsetto,goffsetfrom,goffsetto;
  DMNetworkComponentGenericDataType *arr;
  PetscInt      row[2],col[8];
  PetscScalar   values[8];

  PetscFunctionBegin;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);

  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);

  for (v=vStart; v < vEnd; v++) {
    PetscInt    i,j,offsetd,key;
    PetscInt    offset,goffset;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEXDATA  bus;
    PetscBool   ghostvtex;
    PetscInt    numComps;

    ierr = DMNetworkIsGhostVertex(networkdm,v,&ghostvtex);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,v,&numComps);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMNetworkGetVariableOffset(networkdm,v,&offset);CHKERRQ(ierr);
      ierr = DMNetworkGetVariableGlobalOffset(networkdm,v,&goffset);CHKERRQ(ierr);
      ierr = DMNetworkGetComponentKeyOffset(networkdm,v,j,&key,&offsetd);CHKERRQ(ierr);
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

	ierr = DMNetworkGetSupportingEdges(networkdm,v,&nconnedges,&connedges);CHKERRQ(ierr);
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

	  if (vfrom == v) {
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
    if (User->jac_error) values[0] = 50.0;
	  ierr = MatSetValues(J,1,row,1,col,values,ADD_VALUES);CHKERRQ(ierr);
	}
      }
    }
  }
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues(DM networkdm,Vec X,void* appctx)
{
  PetscErrorCode ierr;
  VERTEXDATA     bus;
  GEN            gen;
  PetscInt       v, vStart, vEnd, offset;
  PetscBool      ghostvtex;
  Vec            localX;
  PetscScalar    *xarr;
  PetscInt       key,numComps,j,offsetd;
  DMNetworkComponentGenericDataType *arr;
  
  PetscFunctionBegin;
  ierr = DMNetworkGetVertexRange(networkdm,&vStart, &vEnd);CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm,v,&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,v,&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumComponents(networkdm,v,&numComps);CHKERRQ(ierr);
    for (j=0; j < numComps; j++) {
      ierr = DMNetworkGetComponentKeyOffset(networkdm,v,j,&key,&offsetd);CHKERRQ(ierr);
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
  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int main(int argc,char ** argv)
{
  PetscErrorCode   ierr;
  char             pfdata_file[PETSC_MAX_PATH_LEN]="datafiles/case9.m";
  PFDATA           *pfdata1,*pfdata2;
  PetscInt         numEdges1=0,numVertices1=0,numEdges2=0,numVertices2=0;
  int              *edges1 = NULL,*edges2 = NULL;
  DM               networkdm;
  PetscInt         componentkey[4];
  UserCtx          User;
  PetscLogStage    stage1,stage2;
  PetscMPIInt      rank;
  PetscInt         nsubnet = 2;

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

    ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
    PetscLogStagePush(stage1);
    /* READ THE DATA */
    if (!crank) {
      /* Only rank 0 reads the data */
      ierr = PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
      /* HERE WE CREATE COPIES OF THE SAME NETWORK THAT WILL BE TREATED AS SUBNETWORKS */

      /*    READ DATA FOR THE FIRST SUBNETWORK */
      ierr = PetscNew(&pfdata1);CHKERRQ(ierr);
      ierr = PFReadMatPowerData(pfdata1,pfdata_file);CHKERRQ(ierr);
      User.Sbase = pfdata1->sbase;

      numEdges1 = pfdata1->nbranch;
      numVertices1 = pfdata1->nbus;

      ierr = PetscMalloc1(2*numEdges1,&edges1);CHKERRQ(ierr);
      ierr = GetListofEdges(pfdata1->nbranch,pfdata1->branch,edges1);CHKERRQ(ierr);

      /*    READ DATA FOR THE SECOND SUBNETWORK */
      ierr = PetscNew(&pfdata2);CHKERRQ(ierr);
      ierr = PFReadMatPowerData(pfdata2,pfdata_file);CHKERRQ(ierr);
      User.Sbase = pfdata2->sbase;

      numEdges2 = pfdata2->nbranch;
      numVertices2 = pfdata2->nbus;

      ierr = PetscMalloc1(2*numEdges2,&edges2);CHKERRQ(ierr);
      ierr = GetListofEdges(pfdata2->nbranch,pfdata2->branch,edges2);CHKERRQ(ierr);
    }

    PetscLogStagePop();
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
    PetscLogStagePush(stage2);

    PetscInt numVertices[2],NumVertices[2];
    PetscInt numEdges[2],NumEdges[2];
    PetscInt *edges[2]; 

    numVertices[0] = numVertices1; numVertices[1] = numVertices2;
    NumVertices[0] = PETSC_DETERMINE; NumVertices[1] = PETSC_DETERMINE;
    numEdges[0] = numEdges1; numEdges[1] = numEdges2;
    NumEdges[0] = PETSC_DETERMINE; NumEdges[1] = PETSC_DETERMINE;
    /* Set number of nodes/edges */
    ierr = DMNetworkSetSizes(networkdm,nsubnet,numVertices,numEdges,NumVertices,NumEdges);CHKERRQ(ierr);

    edges[0] = edges1; edges[1] = edges2;

    /* Add edge connectivity */
    ierr = DMNetworkSetEdgeList(networkdm,edges);CHKERRQ(ierr);

    /* Set up the network layout */
    ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

    ierr = DMView(networkdm,0);CHKERRQ(ierr);
    if (!crank) {
      ierr = PetscFree(edges1);CHKERRQ(ierr);
      ierr = PetscFree(edges2);CHKERRQ(ierr);
    }

    if (!crank) {
      ierr = PetscFree(pfdata1->bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1->load);CHKERRQ(ierr);
      ierr = PetscFree(pfdata1);CHKERRQ(ierr);

      ierr = PetscFree(pfdata2->bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata2->gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata2->branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata2->load);CHKERRQ(ierr);
      ierr = PetscFree(pfdata2);CHKERRQ(ierr);
    }

    ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}
