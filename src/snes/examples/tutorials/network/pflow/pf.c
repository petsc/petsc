static char help[] = "This example demonstrates the use of DMNetwork interface for solving a nonlinear electric power grid problem.\n\
                      The available solver options are in the pfoptions file and the data files are in the datafiles directory.\n\
                      The data file format used is from the MatPower package (http://www.pserc.cornell.edu//matpower/).\n\
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
  PFDATA           *pfdata;
  PetscInt         numEdges=0,numVertices=0;
  int              *edges = NULL;
  PetscInt         i;
  DM               networkdm;
  PetscInt         componentkey[4];
  UserCtx          User;
  PetscLogStage    stage1,stage2;
  PetscMPIInt      rank;
  PetscInt         eStart, eEnd, vStart, vEnd,j;
  PetscInt         genj,loadj;
  Vec              X,F;
  Mat              J;
  SNES             snes;

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
      /*    READ DATA */
      /* Only rank 0 reads the data */
      ierr = PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
      ierr = PetscNew(&pfdata);CHKERRQ(ierr);
      ierr = PFReadMatPowerData(pfdata,pfdata_file);CHKERRQ(ierr);
      User.Sbase = pfdata->sbase;

      numEdges = pfdata->nbranch;
      numVertices = pfdata->nbus;

      ierr = PetscMalloc1(2*numEdges,&edges);CHKERRQ(ierr);
      ierr = GetListofEdges(pfdata->nbranch,pfdata->branch,edges);CHKERRQ(ierr);
    }

    /* If external option activated. Introduce error in jacobian */
    ierr = PetscOptionsHasName(NULL,NULL, "-jac_error", &User.jac_error);CHKERRQ(ierr);

    PetscLogStagePop();
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
    PetscLogStagePush(stage2);
    /* Set number of nodes/edges */
    ierr = DMNetworkSetSizes(networkdm,numVertices,numEdges,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    /* Add edge connectivity */
    ierr = DMNetworkSetEdgeList(networkdm,edges);CHKERRQ(ierr);
    /* Set up the network layout */
    ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

    if (!crank) {
      ierr = PetscFree(edges);CHKERRQ(ierr);
    }

    /* Add network components only process 0 has any data to add*/
    if (!crank) {
      genj=0; loadj=0;
      ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
      for (i = eStart; i < eEnd; i++) {
        ierr = DMNetworkAddComponent(networkdm,i,componentkey[0],&pfdata->branch[i-eStart]);CHKERRQ(ierr);
      }
      ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
      for (i = vStart; i < vEnd; i++) {
        ierr = DMNetworkAddComponent(networkdm,i,componentkey[1],&pfdata->bus[i-vStart]);CHKERRQ(ierr);
        if (pfdata->bus[i-vStart].ngen) {
          for (j = 0; j < pfdata->bus[i-vStart].ngen; j++) {
            ierr = DMNetworkAddComponent(networkdm,i,componentkey[2],&pfdata->gen[genj++]);CHKERRQ(ierr);
          }
        }
        if (pfdata->bus[i-vStart].nload) {
          for (j=0; j < pfdata->bus[i-vStart].nload; j++) {
            ierr = DMNetworkAddComponent(networkdm,i,componentkey[3],&pfdata->load[loadj++]);CHKERRQ(ierr);
          }
        }
        /* Add number of variables */
        ierr = DMNetworkAddNumVariables(networkdm,i,2);CHKERRQ(ierr);
      }
    }

    /* Set up DM for use */
    ierr = DMSetUp(networkdm);CHKERRQ(ierr);

    if (!crank) {
      ierr = PetscFree(pfdata->bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->load);CHKERRQ(ierr);
      ierr = PetscFree(pfdata);CHKERRQ(ierr);
    }

    /* Distribute networkdm to multiple processes */
    ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

    PetscLogStagePop();
    ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);

#if 0
    PetscInt numComponents;
    EDGEDATA edge;
    PetscInt offset,key,kk;
    DMNetworkComponentGenericDataType *arr;
    VERTEXDATA     bus;
    GEN            gen;
    LOAD           load;

    for (i = eStart; i < eEnd; i++) {
      ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);
      ierr = DMNetworkGetComponentKeyOffset(networkdm,i,0,&key,&offset);CHKERRQ(ierr);
      edge = (EDGEDATA)(arr+offset);
      ierr = DMNetworkGetNumComponents(networkdm,i,&numComponents);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Line %d ---- %d\n",crank,numComponents,edge->internal_i,edge->internal_j);CHKERRQ(ierr);
    }

    for (i = vStart; i < vEnd; i++) {
      ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);
      ierr = DMNetworkGetNumComponents(networkdm,i,&numComponents);CHKERRQ(ierr);
      for (kk=0; kk < numComponents; kk++) {
        ierr = DMNetworkGetComponentKeyOffset(networkdm,i,kk,&key,&offset);CHKERRQ(ierr);
        if (key == 1) {
          bus = (VERTEXDATA)(arr+offset);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Bus %d\n",crank,numComponents,bus->internal_i);CHKERRQ(ierr);
        } else if (key == 2) {
          gen = (GEN)(arr+offset);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Gen pg = %f qg = %f\n",crank,gen->pg,gen->qg);CHKERRQ(ierr);
        } else if (key == 3) {
          load = (LOAD)(arr+offset);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Load pl = %f ql = %f\n",crank,load->pl,load->ql);CHKERRQ(ierr);
        }
      }
    }
#endif
    /* Broadcast Sbase to all processors */
    ierr = MPI_Bcast(&User.Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
    ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

    ierr = DMCreateMatrix(networkdm,&J);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    
    ierr = SetInitialValues(networkdm,X,&User);CHKERRQ(ierr);
    
    /* HOOK UP SOLVER */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,F,FormFunction,&User);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,&User);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
    /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    
    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = VecDestroy(&F);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}
