static char help[] = "This example demonstrates the DMCircuit interface for a per phase steady state power flow problem.\n\
                      The available solver options are in the pfoptions file and the data files are in the datafiles directory.\n\
                      The data file format for the reader is the MatPower data format.\n\
                      Run this program: mpiexec -n <n> ./PF\n					\
                      mpiexec -n <n> ./PF -pfdata <filename>\n";

/* T
   Concepts: DMCircuit
   Concepts: PETSc SNES solver
*/

#include "pf.h"
#include <petscdmcircuit.h>

PetscMPIInt rank;

#undef __FUNCT__
#define __FUNCT__ "GetListofEdges"
PetscErrorCode GetListofEdges(PetscInt nbranches, EDGEDATA branch,PetscInt *edges)
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

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X, Vec F,void *appctx)
{
  PetscErrorCode ierr;
  DM             circuitdm;
  UserCtx       *User=(UserCtx*)appctx;
  Vec           localX,localF;
  PetscInt      e;
  PetscInt      v,vStart,vEnd,vfrom,vto;
  const PetscScalar *xarr;
  PetscScalar   *farr;
  PetscInt      offsetfrom,offsetto,offset;
  DMCircuitComponentGenericDataType *arr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&circuitdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(circuitdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(circuitdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(circuitdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(circuitdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(circuitdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(circuitdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  ierr = DMCircuitGetVertexRange(circuitdm,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMCircuitGetComponentDataArray(circuitdm,&arr);CHKERRQ(ierr);

  for (v=vStart; v < vEnd; v++) {
    PetscInt    i,j,offsetd,key;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEXDATA  bus;
    GEN         gen;
    LOAD        load;
    PetscBool   ghostvtex;
    PetscInt    numComps;

    ierr = DMCircuitIsGhostVertex(circuitdm,v,&ghostvtex);CHKERRQ(ierr);
    ierr = DMCircuitGetNumComponents(circuitdm,v,&numComps);CHKERRQ(ierr);
    ierr = DMCircuitGetVariableOffset(circuitdm,v,&offset);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMCircuitGetComponentTypeOffset(circuitdm,v,j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEXDATA)(arr+offsetd);
	/* Handle reference bus constrained dofs */
	if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
	  farr[offset] = 0.0;
	  farr[offset+1] = 0.0;
	  break;
	}

	if (!ghostvtex) {
	  Vm = xarr[offset+1];

	  /* Shunt injections */
	  farr[offset] += Vm*Vm*bus->gl/Sbase;
	  farr[offset+1] += -Vm*Vm*bus->bl/Sbase;
	}
	PetscInt nconnedges;
	const PetscInt *connedges;

	ierr = DMCircuitGetSupportingEdges(circuitdm,v,&nconnedges,&connedges);CHKERRQ(ierr);
	for (i=0; i < nconnedges; i++) {
	  EDGEDATA branch;
	  PetscInt keye;
	  e = connedges[i];
	  ierr = DMCircuitGetComponentTypeOffset(circuitdm,e,0,&keye,&offsetd);CHKERRQ(ierr);
	  branch = (EDGEDATA)(arr+offsetd);
	  if (!branch->status) continue;
	  PetscScalar Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
	  Gff = branch->yff[0];
	  Bff = branch->yff[1];
	  Gft = branch->yft[0];
	  Bft = branch->yft[1];
	  Gtf = branch->ytf[0];
	  Btf = branch->ytf[1];
	  Gtt = branch->ytt[0];
	  Btt = branch->ytt[1];

	  const PetscInt *cone;
	  ierr = DMCircuitGetConnectedNodes(circuitdm,e,&cone);CHKERRQ(ierr);
	  vfrom = cone[0];
	  vto   = cone[1];

	  ierr = DMCircuitGetVariableOffset(circuitdm,vfrom,&offsetfrom);CHKERRQ(ierr);
	  ierr = DMCircuitGetVariableOffset(circuitdm,vto,&offsetto);CHKERRQ(ierr);

	  PetscScalar Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

	  thetaf = xarr[offsetfrom];
	  Vmf     = xarr[offsetfrom+1];
	  thetat = xarr[offsetto];
	  Vmt     = xarr[offsetto+1];
	  thetaft = thetaf - thetat;
	  thetatf = thetat - thetaf;

	  if (vfrom == v) {
	    farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*cos(thetaft) + Bft*sin(thetaft));
	    farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*cos(thetaft) + Gft*sin(thetaft));
	  } else {
	    farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*cos(thetatf) + Btf*sin(thetatf));
	    farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*cos(thetatf) + Gtf*sin(thetatf));
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
    if (bus->ide == PV_BUS) {
      farr[offset+1] = 0.0;
    }
  }
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(circuitdm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(circuitdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(circuitdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(circuitdm,&localF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes,Vec X, Mat *J,Mat *Jpre,MatStructure *flg,void *appctx)
{
  PetscErrorCode ierr;
  DM            circuitdm;
  UserCtx       *User=(UserCtx*)appctx;
  Vec           localX;
  PetscInt      e;
  PetscInt      v,vStart,vEnd,vfrom,vto;
  const PetscScalar *xarr;
  PetscInt      offsetfrom,offsetto,goffsetfrom,goffsetto;
  DMCircuitComponentGenericDataType *arr;
  PetscInt      row[2],col[8];
  PetscScalar   values[8];

  PetscFunctionBegin;
  *flg = SAME_NONZERO_PATTERN;
  ierr = MatZeroEntries(*J);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&circuitdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(circuitdm,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(circuitdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(circuitdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);

  ierr = DMCircuitGetVertexRange(circuitdm,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMCircuitGetComponentDataArray(circuitdm,&arr);CHKERRQ(ierr);

  for (v=vStart; v < vEnd; v++) {
    PetscInt    i,j,offsetd,key;
    PetscInt    offset,goffset;
    PetscScalar Vm;
    PetscScalar Sbase=User->Sbase;
    VERTEXDATA  bus;
    PetscBool   ghostvtex;
    PetscInt    numComps;

    ierr = DMCircuitIsGhostVertex(circuitdm,v,&ghostvtex);CHKERRQ(ierr);
    ierr = DMCircuitGetNumComponents(circuitdm,v,&numComps);CHKERRQ(ierr);
    for (j = 0; j < numComps; j++) {
      ierr = DMCircuitGetVariableOffset(circuitdm,v,&offset);CHKERRQ(ierr);
      ierr = DMCircuitGetVariableGlobalOffset(circuitdm,v,&goffset);CHKERRQ(ierr);
      ierr = DMCircuitGetComponentTypeOffset(circuitdm,v,j,&key,&offsetd);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEXDATA)(arr+offsetd);
	if (!ghostvtex) {
	  /* Handle reference bus constrained dofs */
	  if (bus->ide == REF_BUS || bus->ide == ISOLATED_BUS) {
	    row[0] = goffset; row[1] = goffset+1;
	    col[0] = goffset; col[1] = goffset+1; col[2] = goffset; col[3] = goffset+1;
	    values[0] = 1.0; values[1] = 0.0; values[2] = 0.0; values[3] = 1.0;
	    ierr = MatSetValues(*J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);
	    break;
	  }
	  
	  Vm = xarr[offset+1];
	  
	  /* Shunt injections */
	  row[0] = goffset; row[1] = goffset+1;
	  col[0] = goffset; col[1] = goffset+1; col[2] = goffset; col[3] = goffset+1;
	  values[0] = 2*Vm*bus->gl/Sbase;
	  values[1] = values[2] = 0.0;
	  values[3] = -2*Vm*bus->bl/Sbase;
	  ierr = MatSetValues(*J,2,row,2,col,values,ADD_VALUES);CHKERRQ(ierr);
	}

	PetscInt nconnedges;
	const PetscInt *connedges;

	ierr = DMCircuitGetSupportingEdges(circuitdm,v,&nconnedges,&connedges);CHKERRQ(ierr);
	for (i=0; i < nconnedges; i++) {
	  EDGEDATA branch;
	  VERTEXDATA busf,bust;
	  PetscInt   offsetfd,offsettd,keyf,keyt;
	  e = connedges[i];
	  ierr = DMCircuitGetComponentTypeOffset(circuitdm,e,0,&key,&offsetd);CHKERRQ(ierr);
	  branch = (EDGEDATA)(arr+offsetd);
	  if (!branch->status) continue;
	  PetscScalar Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt;
	  Gff = branch->yff[0];
	  Bff = branch->yff[1];
	  Gft = branch->yft[0];
	  Bft = branch->yft[1];
	  Gtf = branch->ytf[0];
	  Btf = branch->ytf[1];
	  Gtt = branch->ytt[0];
	  Btt = branch->ytt[1];

	  const PetscInt *cone;
	  ierr = DMCircuitGetConnectedNodes(circuitdm,e,&cone);CHKERRQ(ierr);
	  vfrom = cone[0];
	  vto   = cone[1];

	  ierr = DMCircuitGetVariableOffset(circuitdm,vfrom,&offsetfrom);CHKERRQ(ierr);
	  ierr = DMCircuitGetVariableOffset(circuitdm,vto,&offsetto);CHKERRQ(ierr);
	  ierr = DMCircuitGetVariableGlobalOffset(circuitdm,vfrom,&goffsetfrom);CHKERRQ(ierr);
	  ierr = DMCircuitGetVariableGlobalOffset(circuitdm,vto,&goffsetto);CHKERRQ(ierr);

	  if (goffsetfrom < 0) goffsetfrom = -goffsetfrom - 1; /* Convert to actual global offset for ghost nodes, global offset is -(gstart+1) */
	  if (goffsetto < 0) goffsetto = -goffsetto - 1;
	  PetscScalar Vmf,Vmt,thetaf,thetat,thetaft,thetatf;

	  thetaf = xarr[offsetfrom];
	  Vmf     = xarr[offsetfrom+1];
	  thetat = xarr[offsetto];
	  Vmt     = xarr[offsetto+1];
	  thetaft = thetaf - thetat;
	  thetatf = thetat - thetaf;

	  ierr = DMCircuitGetComponentTypeOffset(circuitdm,vfrom,0,&keyf,&offsetfd);CHKERRQ(ierr);
	  ierr = DMCircuitGetComponentTypeOffset(circuitdm,vto,0,&keyt,&offsettd);CHKERRQ(ierr);
	  busf = (VERTEXDATA)(arr+offsetfd);
	  bust = (VERTEXDATA)(arr+offsettd);

	  if (vfrom == v) {
	    /*    farr[offsetfrom]   += Gff*Vmf*Vmf + Vmf*Vmt*(Gft*cos(thetaft) + Bft*sin(thetaft));  */
	    row[0]  = goffsetfrom;
	    col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
	    values[0] =  Vmf*Vmt*(Gft*-sin(thetaft) + Bft*cos(thetaft)); /* df_dthetaf */    
	    values[1] =  2*Gff*Vmf + Vmt*(Gft*cos(thetaft) + Bft*sin(thetaft)); /* df_dVmf */
	    values[2] =  Vmf*Vmt*(Gft*sin(thetaft) + Bft*-cos(thetaft)); /* df_dthetat */
	    values[3] =  Vmf*(Gft*cos(thetaft) + Bft*sin(thetaft)); /* df_dVmt */
	    
	    ierr = MatSetValues(*J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	    
	    if (busf->ide != PV_BUS) {
	      row[0] = goffsetfrom+1;
	      col[0]  = goffsetfrom; col[1] = goffsetfrom+1; col[2] = goffsetto; col[3] = goffsetto+1;
	      /*    farr[offsetfrom+1] += -Bff*Vmf*Vmf + Vmf*Vmt*(-Bft*cos(thetaft) + Gft*sin(thetaft)); */
	      values[0] =  Vmf*Vmt*(Bft*sin(thetaft) + Gft*cos(thetaft));
	      values[1] =  -2*Bff*Vmf + Vmt*(-Bft*cos(thetaft) + Gft*sin(thetaft));
	      values[2] =  Vmf*Vmt*(-Bft*sin(thetaft) + Gft*-cos(thetaft));
	      values[3] =  Vmf*(-Bft*cos(thetaft) + Gft*sin(thetaft));
	      
	      ierr = MatSetValues(*J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	    }
	  } else {
	    row[0] = goffsetto;
	    col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
	    /*    farr[offsetto]   += Gtt*Vmt*Vmt + Vmt*Vmf*(Gtf*cos(thetatf) + Btf*sin(thetatf)); */
	    values[0] =  Vmt*Vmf*(Gtf*-sin(thetatf) + Btf*cos(thetaft)); /* df_dthetat */
	    values[1] =  2*Gtt*Vmt + Vmf*(Gtf*cos(thetatf) + Btf*sin(thetatf)); /* df_dVmt */
	    values[2] =  Vmt*Vmf*(Gtf*sin(thetatf) + Btf*-cos(thetatf)); /* df_dthetaf */
	    values[3] =  Vmt*(Gtf*cos(thetatf) + Btf*sin(thetatf)); /* df_dVmf */
	    
	    ierr = MatSetValues(*J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	    
	    if (bust->ide != PV_BUS) {
	      row[0] = goffsetto+1;
	      col[0] = goffsetto; col[1] = goffsetto+1; col[2] = goffsetfrom; col[3] = goffsetfrom+1;
	      /*    farr[offsetto+1] += -Btt*Vmt*Vmt + Vmt*Vmf*(-Btf*cos(thetatf) + Gtf*sin(thetatf)); */
	      values[0] =  Vmt*Vmf*(Btf*sin(thetatf) + Gtf*cos(thetatf));
	      values[1] =  -2*Btt*Vmt + Vmf*(-Btf*cos(thetatf) + Gtf*sin(thetatf));
	      values[2] =  Vmt*Vmf*(-Btf*sin(thetatf) + Gtf*-cos(thetatf));
	      values[3] =  Vmt*(-Btf*cos(thetatf) + Gtf*sin(thetatf));
	      
	      ierr = MatSetValues(*J,1,row,4,col,values,ADD_VALUES);CHKERRQ(ierr);
	    }
	  }
	}
	if (!ghostvtex && bus->ide == PV_BUS) {
	  row[0] = goffset+1; col[0] = goffset+1;
	  values[0]  = 1.0;
	  ierr = MatSetValues(*J,1,row,1,col,values,ADD_VALUES);CHKERRQ(ierr);
	}
      }
    }
  }
  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(circuitdm,&localX);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialValues"
PetscErrorCode SetInitialValues(DM circuitdm,Vec X,void* appctx)
{
  PetscErrorCode ierr;
  VERTEXDATA     bus;
  GEN            gen;
  PetscInt       v, vStart, vEnd, offset;
  PetscBool      ghostvtex;
  Vec            localX;
  PetscScalar    *xarr;
  PetscInt       key;
  DMCircuitComponentGenericDataType *arr;
  
  PetscFunctionBegin;
  ierr = DMCircuitGetVertexRange(circuitdm,&vStart, &vEnd);CHKERRQ(ierr);

  ierr = DMGetLocalVector(circuitdm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(circuitdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(circuitdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMCircuitGetComponentDataArray(circuitdm,&arr);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    ierr = DMCircuitIsGhostVertex(circuitdm,v,&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;
    PetscInt numComps;
    PetscInt j;
    PetscInt offsetd;
    ierr = DMCircuitGetVariableOffset(circuitdm,v,&offset);CHKERRQ(ierr);
    ierr = DMCircuitGetNumComponents(circuitdm,v,&numComps);CHKERRQ(ierr);
    for (j=0; j < numComps; j++) {
      ierr = DMCircuitGetComponentTypeOffset(circuitdm,v,j,&key,&offsetd);CHKERRQ(ierr);
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
  ierr = DMLocalToGlobalBegin(circuitdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(circuitdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(circuitdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char ** argv)
{
  PetscErrorCode       ierr;
  char                 pfdata_file[PETSC_MAX_PATH_LEN]="datafiles/case9.m";
  PFDATA               pfdata;
  PetscInt             numEdges=0,numVertices=0;
  PetscInt             *edges = NULL;
  PetscInt             i;  
  DM                   circuitdm;
  PetscInt             componentkey[4];
  UserCtx              User;
  PetscLogStage        stage1,stage2;
  PetscInt             size;

  PetscInitialize(&argc,&argv,"pfoptions",help);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Create an empty circuit object */
  ierr = DMCircuitCreate(PETSC_COMM_WORLD,&circuitdm);CHKERRQ(ierr);
  /* Register the components in the circuit */
  ierr = DMCircuitRegisterComponent(circuitdm,"branchstruct",sizeof(struct _p_EDGEDATA),&componentkey[0]);CHKERRQ(ierr);
  ierr = DMCircuitRegisterComponent(circuitdm,"busstruct",sizeof(struct _p_VERTEXDATA),&componentkey[1]);CHKERRQ(ierr);
  ierr = DMCircuitRegisterComponent(circuitdm,"genstruct",sizeof(struct _p_GEN),&componentkey[2]);CHKERRQ(ierr);
  ierr = DMCircuitRegisterComponent(circuitdm,"loadstruct",sizeof(struct _p_LOAD),&componentkey[3]);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
  PetscLogStagePush(stage1);
  /* READ THE DATA */
  if (!rank) {
    /*    READ DATA */
    /* Only rank 0 reads the data */
    ierr = PetscOptionsGetString(PETSC_NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
    ierr = PFReadMatPowerData(&pfdata,pfdata_file);CHKERRQ(ierr);
    User.Sbase = pfdata.sbase;

    numEdges = pfdata.nbranch;
    numVertices = pfdata.nbus;

    ierr = PetscMalloc(2*numEdges*sizeof(PetscInt),&edges);CHKERRQ(ierr);
    ierr = GetListofEdges(pfdata.nbranch,pfdata.branch,edges);CHKERRQ(ierr);
  }
  PetscLogStagePop();
  MPI_Barrier(PETSC_COMM_WORLD);
  ierr = PetscLogStageRegister("Create circuit",&stage2);CHKERRQ(ierr);
  PetscLogStagePush(stage2);
  /* Set number of nodes/edges */
  ierr = DMCircuitSetSizes(circuitdm,numVertices,numEdges,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  /* Add edge connectivity */
  ierr = DMCircuitSetEdgeList(circuitdm,edges);CHKERRQ(ierr);
  /* Set up the circuit layout */
  ierr = DMCircuitLayoutSetUp(circuitdm);CHKERRQ(ierr);

  if (!rank) {
    ierr = PetscFree(edges);CHKERRQ(ierr);
  }
  /* Add circuit components */
  PetscInt eStart, eEnd, vStart, vEnd,j;
  PetscInt genj=0,loadj=0;
  ierr = DMCircuitGetEdgeRange(circuitdm,&eStart,&eEnd);CHKERRQ(ierr);
  for (i = eStart; i < eEnd; i++) {
    ierr = DMCircuitAddComponent(circuitdm,i,componentkey[0],&pfdata.branch[i-eStart]);CHKERRQ(ierr);
  }
  ierr = DMCircuitGetVertexRange(circuitdm,&vStart,&vEnd);CHKERRQ(ierr);
  for (i = vStart; i < vEnd; i++) {
    ierr = DMCircuitAddComponent(circuitdm,i,componentkey[1],&pfdata.bus[i-vStart]);CHKERRQ(ierr);
    if (pfdata.bus[i-vStart].ngen) {
      for (j = 0; j < pfdata.bus[i-vStart].ngen; j++) {
	ierr = DMCircuitAddComponent(circuitdm,i,componentkey[2],&pfdata.gen[genj++]);CHKERRQ(ierr);
      }
    }
    if (pfdata.bus[i-vStart].nload) {
      for (j=0; j < pfdata.bus[i-vStart].nload; j++) {
	ierr = DMCircuitAddComponent(circuitdm,i,componentkey[3],&pfdata.load[loadj++]);CHKERRQ(ierr);
      }
    }
    /* Add number of variables */
    ierr = DMCircuitAddNumVariables(circuitdm,i,2);CHKERRQ(ierr);
  }
  /* Set up DM for use */
  ierr = DMSetUp(circuitdm);CHKERRQ(ierr);

  if (!rank) {
    ierr = PetscFree(pfdata.bus);CHKERRQ(ierr);
    ierr = PetscFree(pfdata.gen);CHKERRQ(ierr);
    ierr = PetscFree(pfdata.branch);CHKERRQ(ierr);
    ierr = PetscFree(pfdata.load);CHKERRQ(ierr);
  }


  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) {
    DM distcircuitdm;
    /* Circuit partitioning and distribution of data */
    ierr = DMCircuitDistribute(circuitdm,&distcircuitdm);CHKERRQ(ierr);
    ierr = DMDestroy(&circuitdm);CHKERRQ(ierr);
    circuitdm = distcircuitdm;
  }

  PetscLogStagePop();
  ierr = DMCircuitGetEdgeRange(circuitdm,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMCircuitGetVertexRange(circuitdm,&vStart,&vEnd);CHKERRQ(ierr);
  
#if 0
  PetscInt numComponents;
  EDGEDATA edge;
  PetscInt offset,key;
  DMCircuitComponentGenericDataType *arr;
  for (i = eStart; i < eEnd; i++) {
    ierr = DMCircuitGetComponentDataArray(circuitdm,&arr);CHKERRQ(ierr);
    ierr = DMCircuitGetComponentTypeOffset(circuitdm,i,0,&key,&offset);CHKERRQ(ierr);
    edge = (EDGEDATA)(arr+offset);
    ierr = DMCircuitGetNumComponents(circuitdm,i,&numComponents);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Line %d ---- %d\n",rank,numComponents,edge->internal_i,edge->internal_j);CHKERRQ(ierr);
  }    

  VERTEXDATA bus;
  GEN        gen;
  LOAD       load;
  PetscInt   kk;
  for (i = vStart; i < vEnd; i++) {
    ierr = DMCircuitGetComponentDataArray(circuitdm,&arr);CHKERRQ(ierr);
    ierr = DMCircuitGetNumComponents(circuitdm,i,&numComponents);CHKERRQ(ierr);
    for (kk=0; kk < numComponents; kk++) {
      ierr = DMCircuitGetComponentTypeOffset(circuitdm,i,kk,&key,&offset);CHKERRQ(ierr);
      if (key == 1) {
	bus = (VERTEXDATA)(arr+offset);
	ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Bus %d\n",rank,numComponents,bus->internal_i);CHKERRQ(ierr);
      } else if (key == 2) {
	gen = (GEN)(arr+offset);
	ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Gen pg = %f qg = %f\n",rank,gen->pg,gen->qg);CHKERRQ(ierr);
      } else if (key == 3) {
	load = (LOAD)(arr+offset);
	ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Load pd = %f qd = %f\n",rank,load->pl,load->ql);CHKERRQ(ierr);
      }
    }
  }  
#endif  
  /* Broadcast Sbase to all processors */
  ierr = MPI_Bcast(&User.Sbase,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  Vec X,F;
  ierr = DMCreateGlobalVector(circuitdm,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

  Mat J;
  ierr = DMCreateMatrix(circuitdm,&J);CHKERRQ(ierr);
  ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

  ierr = SetInitialValues(circuitdm,X,&User);CHKERRQ(ierr);

  SNES snes;
  /* HOOK UP SOLVER */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,circuitdm);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,FormFunction,&User);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,&User);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&circuitdm);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
