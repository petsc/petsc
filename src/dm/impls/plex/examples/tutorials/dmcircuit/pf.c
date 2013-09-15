static char help[] = "Main routine for the per phase steady state power in power balance form.\n\
Run this program: mpiexec -n <n> ./PF\n\
                  mpiexec -n <n> ./PF -pfdata datafiles/<filename>\n";

/* T
   Concepts: Per-phase steady state power flow
   Concepts: PETSc SNES solver
*/

#include "pf.h"

PetscMPIInt rank;

#define MAX_DATA_AT_POINT 10

typedef PetscInt DataArrayType;

typedef struct _p_PetscCircuitComponentHeader *PetscCircuitComponentHeader;
struct _p_PetscCircuitComponentHeader {
  PetscInt ndata; 
  PetscInt size[MAX_DATA_AT_POINT];
  PetscInt key[MAX_DATA_AT_POINT];
  PetscInt offset[MAX_DATA_AT_POINT];
};

typedef struct _p_PetscCircuitComponentValue *PetscCircuitComponentValue;
struct _p_PetscCircuitComponentValue {
  void* data[MAX_DATA_AT_POINT];
};

typedef struct {
  char name[20];
  PetscInt size;
}PetscCircuitComponent;

typedef struct{
  PFDATA *pfdata;
}UserCtx;

struct _p_PetscCircuit{
  PetscInt              NEdges; /* Number of global edges */
  PetscInt              NNodes; /* Number of global nodes */
  PetscInt              nEdges; /* Number of local edges */
  PetscInt              nNodes; /* Number of local nodes */
  PetscInt              *edges; /* Edge list */
  PetscInt              pStart,pEnd,vStart,vEnd,eStart,eEnd;
  DM                    dm;     /* DM */
  PetscSection          DataSection; /* Section for managing parameter distribution */
  PetscSection          DofSection;  /* Section for managing data distribution */
  PetscInt              *arr; /* Array for holding and distributing the data */
  PetscInt              ncomponent;
  PetscCircuitComponent  component[10];
  PetscCircuitComponentHeader header;  
  PetscCircuitComponentValue  cvalue;
  PetscInt               dataheadersize;
  DataArrayType         *dataarray; /* Array to hold the data */
};

typedef struct _p_PetscCircuit *PetscCircuit;

/* The interface needs a major rehaul. This is a quick implementation of the PetscCircuit interface */
/* Creates an empty PetscCircuit object */
#undef __FUNCT__ 
#define __FUNCT__ "PetscCircuitCreate"
PetscErrorCode PetscCircuitCreate(PetscCircuit *circuit)
{
  PetscErrorCode ierr;
  PetscCircuit   circ;
  PetscFunctionBegin;
  ierr = PetscNew(struct _p_PetscCircuit,&circ);CHKERRQ(ierr);
  circ->NNodes = 0;
  circ->nNodes = 0;
  circ->NEdges = 0;
  circ->nEdges = 0;
  *circuit = circ;
  PetscFunctionReturn(0);
}

/* Set the number of local/global edges and nodes */
#undef __FUNCT__
#define __FUNCT__ "PetscCircuitSetSizes"
PetscErrorCode PetscCircuitSetSizes(PetscCircuit circuit,PetscInt nNodes,PetscInt nEdges,PetscInt NNodes,PetscInt NEdges)
{

  PetscFunctionBegin;
  circuit->nNodes = nNodes;
  circuit->nEdges = nEdges;
  circuit->NNodes = NNodes;
  circuit->NEdges = NEdges;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitSetEdges"
PetscErrorCode PetscCircuitSetEdges(PetscCircuit circuit,PetscInt *edges)
{
  PetscFunctionBegin;
  circuit->edges = edges;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitLayoutSetUp"
PetscErrorCode PetscCircuitLayoutSetUp(PetscCircuit circuit)
{
  PetscErrorCode ierr;
  PetscInt       dim = 1; /* One dimensional circuit */
  PetscInt       numCorners=2;
  PetscInt       spacedim=2;
  double         *vertexcoords=NULL;
  PetscInt       i;

  PetscFunctionBegin;
  if (circuit->nNodes) {
    ierr = PetscMalloc(numCorners*circuit->nNodes*sizeof(PetscInt),&vertexcoords);CHKERRQ(ierr);
  }
  ierr = DMPlexCreateFromCellList(PETSC_COMM_WORLD,dim,circuit->nEdges,circuit->nNodes,numCorners,PETSC_FALSE,circuit->edges,spacedim,vertexcoords,&circuit->dm);CHKERRQ(ierr);
  ierr = DMPlexGetChart(circuit->dm,&circuit->pStart,&circuit->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(circuit->dm,0,&circuit->eStart,&circuit->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(circuit->dm,1,&circuit->vStart,&circuit->vEnd);CHKERRQ(ierr);
  
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&circuit->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&circuit->DofSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(circuit->DataSection,circuit->pStart,circuit->pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(circuit->DofSection,circuit->pStart,circuit->pEnd);CHKERRQ(ierr);

  circuit->dataheadersize = sizeof(struct _p_PetscCircuitComponentHeader)/sizeof(DataArrayType);
  ierr = PetscMalloc((circuit->pEnd-circuit->pStart)*sizeof(struct _p_PetscCircuitComponentHeader),&circuit->header);CHKERRQ(ierr);
  for (i = circuit->pStart; i < circuit->pEnd; i++) {
    PetscInt ndata;
    circuit->header[i].ndata = 0;
    ndata = circuit->header[i].ndata;
    ierr = PetscSectionAddDof(circuit->DataSection,i,circuit->dataheadersize);CHKERRQ(ierr);
    circuit->header[i].offset[ndata] = 0;
  }
  ierr = PetscMalloc((circuit->pEnd-circuit->pStart)*sizeof(struct _p_PetscCircuitComponentValue),&circuit->cvalue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitRegisterComponent"
PetscErrorCode PetscCircuitRegisterComponent(PetscCircuit circuit,const char *name,PetscInt size,PetscInt *componentkey)
{
  PetscErrorCode ierr;
  PetscCircuitComponent *component=&circuit->component[circuit->ncomponent];
  PetscFunctionBegin;

  /* Skipping string comparison for now to check if the parameter already exists */
  ierr = PetscStrcpy(component->name,name);CHKERRQ(ierr);
  component->size = size/sizeof(DataArrayType);
  *componentkey = circuit->ncomponent;
  circuit->ncomponent++;
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetVertexRange"
PetscErrorCode PetscCircuitGetVertexRange(PetscCircuit circuit,PetscInt *vStart,PetscInt *vEnd)
{

  PetscFunctionBegin;
  if (vStart) *vStart = circuit->vStart;
  if (vEnd) *vEnd = circuit->vEnd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetEdgeRange"
PetscErrorCode PetscCircuitGetEdgeRange(PetscCircuit circuit,PetscInt *eStart,PetscInt *eEnd)
{

  PetscFunctionBegin;
  if (eStart) *eStart = circuit->eStart;
  if (eEnd) *eEnd = circuit->eEnd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitAddComponent"
PetscErrorCode PetscCircuitAddComponent(PetscCircuit circuit, PetscInt p,PetscInt componentkey,void* compvalue)
{
  PetscCircuitComponent component=circuit->component[componentkey];
  PetscCircuitComponentHeader header=&circuit->header[p];
  PetscCircuitComponentValue  cvalue=&circuit->cvalue[p];
  PetscErrorCode         ierr;
  
  PetscFunctionBegin;
  header->size[header->ndata] = component.size;
  ierr = PetscSectionAddDof(circuit->DataSection,p,component.size);CHKERRQ(ierr);
  header->key[header->ndata] = componentkey;
  if (header->ndata != 0) header->offset[header->ndata] = header->offset[header->ndata-1] + header->size[header->ndata-1]; 

  cvalue->data[header->ndata] = (void*)compvalue;
  header->ndata++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetNumComponents"
PetscErrorCode PetscCircuitGetNumComponents(PetscCircuit circuit,PetscInt p,PetscInt *numcomponents)
{
  PetscErrorCode ierr;
  PetscInt       offset;
  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DataSection,p,&offset);CHKERRQ(ierr);
  *numcomponents = ((PetscCircuitComponentHeader)(circuit->dataarray+offset))->ndata;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetComponentTypeOffset"
PetscErrorCode PetscCircuitGetComponentTypeOffset(PetscCircuit circuit,PetscInt p, PetscInt compnum, PetscInt *compkey, PetscInt *offset)
{
  PetscErrorCode ierr;
  PetscInt       offsetp;
  PetscCircuitComponentHeader header;
  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DataSection,p,&offsetp);CHKERRQ(ierr);
  header = (PetscCircuitComponentHeader)(circuit->dataarray+offsetp);
  *compkey = header->key[compnum];
  *offset  = header->offset[compnum];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitAddNumVariables"
PetscErrorCode PetscCircuitAddNumVariables(PetscCircuit circuit,PetscInt p,PetscInt nvar)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscSectionAddDof(circuit->DofSection,p,nvar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This can be removed later */
#undef __FUNCT__
#define __FUNCT__ "PetscCircuitComponentSetUp"
PetscErrorCode PetscCircuitComponentSetUp(PetscCircuit circuit)
{
  PetscErrorCode              ierr;
  PetscInt                    arr_size;
  PetscInt                    p,offset;
  PetscCircuitComponentHeader header;
  PetscCircuitComponentValue  cvalue;
  DataArrayType               *dataarray;
  PetscFunctionBegin;
  ierr = PetscSectionSetUp(circuit->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(circuit->DataSection,&arr_size);CHKERRQ(ierr);
  ierr = PetscMalloc(arr_size*sizeof(DataArrayType),&circuit->dataarray);CHKERRQ(ierr);
  dataarray = circuit->dataarray;
  for (p = circuit->pStart; p < circuit->pEnd; p++) {
    ierr = PetscSectionGetOffset(circuit->DataSection,p,&offset);CHKERRQ(ierr);
    /* Copy header */
    header = &circuit->header[p];
    ierr = PetscMemcpy(dataarray+offset,header,circuit->dataheadersize*sizeof(DataArrayType));
    /* Copy data */
    offset += circuit->dataheadersize;
    cvalue = &circuit->cvalue[p];
    PetscInt ncomp=header->ndata,i;
    for (i = 0; i < ncomp; i++) {
      offset += header->offset[i];
      ierr = PetscMemcpy(dataarray+offset,cvalue->data[i],header->size[i]*sizeof(DataArrayType));
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitVariablesSetUp"
PetscErrorCode PetscCircuitVariablesSetUp(PetscCircuit circuit)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSectionSetUp(circuit->DofSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCircuitGetDataArray"
PetscErrorCode PetscCircuitGetDataArray(PetscCircuit circuit,PetscInt p, DataArrayType **dataarray)
{
  PetscErrorCode ierr;
  PetscInt       offset;
  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DataSection,p,&offset);CHKERRQ(ierr);
  *dataarray = circuit->dataarray+offset+circuit->dataheadersize;
  PetscFunctionReturn(0);
}

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

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char ** argv)
{
  PetscErrorCode       ierr;
  char                 pfdata_file[PETSC_MAX_PATH_LEN]="datafiles/case9.m";
  UserCtx              User;
  PFDATA               pfdata;
  PetscInt             numEdges=0,numVertices=0;
  PetscInt             *edges = NULL;
  PetscInt             i;  
  PetscCircuit         circuit;
  PetscInt             componentkey[4];

  PetscInitialize(&argc,&argv,"pfoptions",help);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscCircuitCreate(&circuit);CHKERRQ(ierr);
  /* READ THE DATA */
  if (!rank) {
    /*    READ DATA */
    /* Only rank 0 reads the data */
    ierr = PetscOptionsGetString(PETSC_NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
    ierr = PFReadMatPowerData(&pfdata,pfdata_file);CHKERRQ(ierr);
    User.pfdata = &pfdata;

    /* Reorder the generator data structure according to bus numbers */
    GEN  newgen;
    LOAD newload;
    PetscInt genj=0,loadj=0,j;
    ierr = PetscMalloc(pfdata.ngen*sizeof(struct _p_GEN),&newgen);CHKERRQ(ierr);
    ierr = PetscMalloc(pfdata.nload*sizeof(struct _p_LOAD),&newload);CHKERRQ(ierr);
    for (i = 0; i < pfdata.nbus; i++) {
      for (j = 0; j < pfdata.bus[i].ngen; j++) {
	ierr = PetscMemcpy(&newgen[genj++],&pfdata.gen[pfdata.bus[i].gidx[j]],sizeof(struct _p_GEN));
      }
      for (j = 0; j < pfdata.bus[i].nload; j++) {
	ierr = PetscMemcpy(&newload[loadj++],&pfdata.load[pfdata.bus[i].lidx[j]],sizeof(struct _p_LOAD));
      }
      
    }
    ierr = PetscFree(pfdata.gen);CHKERRQ(ierr);
    ierr = PetscFree(pfdata.load);CHKERRQ(ierr);
    pfdata.gen = newgen;
    pfdata.load = newload;
    numEdges = pfdata.nbranch;
    
    numVertices = pfdata.nbus;
    ierr = PetscMalloc(2*numEdges*sizeof(PetscInt),&edges);CHKERRQ(ierr);
    ierr = GetListofEdges(pfdata.nbranch,pfdata.branch,edges);CHKERRQ(ierr);
  }
  ierr = PetscCircuitSetSizes(circuit,numVertices,numEdges,numVertices,numEdges);CHKERRQ(ierr);
  ierr = PetscCircuitSetEdges(circuit,edges);CHKERRQ(ierr);
  ierr = PetscCircuitLayoutSetUp(circuit);CHKERRQ(ierr);

  ierr = PetscCircuitRegisterComponent(circuit,"branchstruct",sizeof(struct _p_EDGEDATA),&componentkey[0]);CHKERRQ(ierr);
  ierr = PetscCircuitRegisterComponent(circuit,"busstruct",sizeof(struct _p_VERTEXDATA),&componentkey[1]);CHKERRQ(ierr);
  ierr = PetscCircuitRegisterComponent(circuit,"genstruct",sizeof(struct _p_GEN),&componentkey[2]);CHKERRQ(ierr);
  ierr = PetscCircuitRegisterComponent(circuit,"loadstruct",sizeof(struct _p_LOAD),&componentkey[3]);CHKERRQ(ierr);

  PetscInt eStart, eEnd, vStart, vEnd,j;
  PetscInt genj=0,loadj=0;
  ierr = PetscCircuitGetEdgeRange(circuit,&eStart,&eEnd);CHKERRQ(ierr);
  for (i = eStart; i < eEnd; i++) {
    ierr = PetscCircuitAddComponent(circuit,i,componentkey[0],&pfdata.branch[i-eStart]);CHKERRQ(ierr);
  }
  ierr = PetscCircuitGetVertexRange(circuit,&vStart,&vEnd);CHKERRQ(ierr);
  for (i = vStart; i < vEnd; i++) {
    ierr = PetscCircuitAddComponent(circuit,i,componentkey[1],&pfdata.bus[i-vStart]);CHKERRQ(ierr);
    if (pfdata.bus[i-vStart].ngen) {
      for (j = 0; j < pfdata.bus[i-vStart].ngen; j++) {
	ierr = PetscCircuitAddComponent(circuit,i,componentkey[2],&pfdata.gen[genj++]);CHKERRQ(ierr);
      }
    }
    if (pfdata.bus[i-vStart].nload) {
      for (j=0; j < pfdata.bus[i-vStart].nload; j++) {
	ierr = PetscCircuitAddComponent(circuit,i,componentkey[3],&pfdata.load[loadj++]);CHKERRQ(ierr);
      }
    }
    /* Add number of variables */
    ierr = PetscCircuitAddNumVariables(circuit,i,2);CHKERRQ(ierr);
  }
  ierr = PetscCircuitComponentSetUp(circuit);CHKERRQ(ierr);
  ierr = PetscCircuitVariablesSetUp(circuit);CHKERRQ(ierr);

  PetscInt numComponents;
  EDGEDATA edge;
  for (i = eStart; i < eEnd; i++) {
    ierr = PetscCircuitGetDataArray(circuit,i,(DataArrayType**)&edge);CHKERRQ(ierr);
    ierr = PetscCircuitGetNumComponents(circuit,i,&numComponents);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Line %d ---- %d\n",rank,numComponents,edge->internal_i,edge->internal_j);CHKERRQ(ierr);
  }    

  VERTEXDATA bus;
  for (i = vStart; i < vEnd; i++) {
    ierr = PetscCircuitGetDataArray(circuit,i,(DataArrayType**)&bus);CHKERRQ(ierr);
    ierr = PetscCircuitGetNumComponents(circuit,i,&numComponents);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Bus %d\n",rank,numComponents,bus->internal_i);CHKERRQ(ierr);
  }    

  
  /* Broadcast Sbase to all processors */
  ierr = MPI_Bcast(&pfdata.sbase,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
