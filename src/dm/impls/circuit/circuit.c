#include <petsc-private/dmcircuitimpl.h>  /*I  "petscdmcircuit.h"  I*/
#include <petscdmplex.h>
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "DMCircuitSetSizes"
/*@
  DMCircuitSetSizes - Sets the local and global vertices and edges.

  Collective on DM
  
  Input Parameters:
+ dm - the dm object
. nV - number of local vertices
. nE - number of local edges
. NV - number of global vertices (or PETSC_DETERMINE)
- NE - number of global edges (or PETSC_DETERMINE)

   Notes
   If one processor calls this with NV (NE) of PETSC_DECIDE then all processors must, otherwise the prgram will hang.

   You cannot change the sizes once they have been set

   Level: intermediate

.seealso: DMCircuitCreate
@*/
PetscErrorCode DMCircuitSetSizes(DM dm, PetscInt nV, PetscInt nE, PetscInt NV, PetscInt NE)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;
  PetscInt       a[2],b[2];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (NV > 0) PetscValidLogicalCollectiveInt(dm,NV,4);
  if (NE > 0) PetscValidLogicalCollectiveInt(dm,NE,5);
  if (NV > 0 && nV > NV) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local vertex size %D cannot be larger than global vertex size %D",nV,NV);
  if (NE > 0 && nE > NE) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local edge size %D cannot be larger than global edge size %D",nE,NE);
  if ((circuit->nNodes >= 0 || circuit->NNodes >= 0) && (circuit->nNodes != nV || circuit->NNodes != NV)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset vertex sizes to %D local %D global after previously setting them to %D local %D global",nV,NV,circuit->nNodes,circuit->NNodes);
  if ((circuit->nEdges >= 0 || circuit->NEdges >= 0) && (circuit->nEdges != nE || circuit->NEdges != NE)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot change/reset edge sizes to %D local %D global after previously setting them to %D local %D global",nE,NE,circuit->nEdges,circuit->NEdges);
  if (NE < 0 || NV < 0) {
    a[0] = nV; a[1] = nE;
    ierr = MPI_Allreduce(a,b,2,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    NV = b[0]; NE = b[1];
  }
  circuit->nNodes = nV;
  circuit->NNodes = NV;
  circuit->nEdges = nE;
  circuit->NEdges = NE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitSetEdgeList"
/*@
  DMCircuitSetEdgeList - Sets the list of local edges (vertex connectivity) for the circuit

  Logically collective on DM

  Input Parameters:
. edges - list of edges

  Notes:
  There is no copy involved in this operation, only the pointer is referenced. The edgelist should
  not be destroyed before the call to DMCircuitLayoutSetUp

  Level: intermediate

.seealso: DMCircuitCreate, DMCircuitSetSizes
@*/
PetscErrorCode DMCircuitSetEdgeList(DM dm, int edgelist[])
{
  DM_Circuit *circuit = (DM_Circuit*) dm->data;
  
  PetscFunctionBegin;
  circuit->edges = edgelist;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitLayoutSetUp"
/*@
  DMCircuitLayoutSetUp - Sets up the bare layout (graph) for the circuit

  Collective on DM

  Input Parameters
. DM - the dmcircuit object

  Notes:
  This routine should be called after the circuit sizes and edgelists have been provided. It creates
  the bare layout of the circuit and sets up the circuit to begin insertion of components.

  All the components should be registered before calling this routine.

  Level: intermediate

.seealso: DMCircuitSetSizes, DMCircuitSetEdgeList
@*/
PetscErrorCode DMCircuitLayoutSetUp(DM dm)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;
  PetscInt       dim = 1; /* One dimensional circuit */
  PetscInt       numCorners=2;
  PetscInt       spacedim=2;
  double         *vertexcoords=NULL;
  PetscInt       i;
  PetscInt       ndata;

  PetscFunctionBegin;
  if (circuit->nNodes) {
    ierr = PetscMalloc(numCorners*circuit->nNodes*sizeof(PetscInt),&vertexcoords);CHKERRQ(ierr);
  }
  ierr = DMPlexCreateFromCellList(PetscObjectComm((PetscObject)dm),dim,circuit->nEdges,circuit->nNodes,numCorners,PETSC_FALSE,circuit->edges,spacedim,vertexcoords,&circuit->plex);CHKERRQ(ierr);
  if (circuit->nNodes) {
    ierr = PetscFree(vertexcoords);CHKERRQ(ierr);
  }
  ierr = DMPlexGetChart(circuit->plex,&circuit->pStart,&circuit->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(circuit->plex,0,&circuit->eStart,&circuit->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(circuit->plex,1,&circuit->vStart,&circuit->vEnd);CHKERRQ(ierr);
  
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm),&circuit->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm),&circuit->DofSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(circuit->DataSection,circuit->pStart,circuit->pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(circuit->DofSection,circuit->pStart,circuit->pEnd);CHKERRQ(ierr);

  circuit->dataheadersize = sizeof(struct _p_DMCircuitComponentHeader)/sizeof(DMCircuitComponentGenericDataType);
  ierr = PetscMalloc((circuit->pEnd-circuit->pStart)*sizeof(struct _p_DMCircuitComponentHeader),&circuit->header);CHKERRQ(ierr);
  for (i = circuit->pStart; i < circuit->pEnd; i++) {
    circuit->header[i].ndata = 0;
    ndata = circuit->header[i].ndata;
    ierr = PetscSectionAddDof(circuit->DataSection,i,circuit->dataheadersize);CHKERRQ(ierr);
    circuit->header[i].offset[ndata] = 0;
  }
  ierr = PetscMalloc((circuit->pEnd-circuit->pStart)*sizeof(struct _p_DMCircuitComponentValue),&circuit->cvalue);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitRegisterComponent"
/*@
  DMCircuitRegisterComponent - Registers the circuit component

  Logically collective on DM

  Input Parameters
+ dm   - the circuit object
. name - the component name
- size - the storage size in bytes for this component data

   Output Parameters
.   key - an integer key that defines the component

   Notes
   This routine should be called by all processors before calling DMCircuitLayoutSetup().

   Level: intermediate

.seealso: DMCircuitLayoutSetUp, DMCircuitCreate
@*/
PetscErrorCode DMCircuitRegisterComponent(DM dm,const char *name,PetscInt size,PetscInt *key)
{
  PetscErrorCode        ierr;
  DM_Circuit            *circuit = (DM_Circuit*) dm->data;
  DMCircuitComponent    *component=&circuit->component[circuit->ncomponent];
  PetscBool             flg=PETSC_FALSE;
  PetscInt              i;

  PetscFunctionBegin;

  for (i=0; i < circuit->ncomponent; i++) {
    ierr = PetscStrcmp(component->name,name,&flg);CHKERRQ(ierr);
    if (flg) {
      *key = i;
      PetscFunctionReturn(0);
    }
  }
  
  ierr = PetscStrcpy(component->name,name);CHKERRQ(ierr);
  component->size = size/sizeof(DMCircuitComponentGenericDataType);
  *key = circuit->ncomponent;
  circuit->ncomponent++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitGetVertexRange"
/*@
  DMCircuitGetVertexRange - Get the bounds [start, end) for the vertices.

  Not Collective

  Input Parameters:
+ dm - The DMCircuit object

  Output Paramters:
+ vStart - The first vertex point
- vEnd   - One beyond the last vertex point

  Level: intermediate

.seealso: DMCircuitGetEdgeRange
@*/
PetscErrorCode DMCircuitGetVertexRange(DM dm,PetscInt *vStart,PetscInt *vEnd)
{
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  if (vStart) *vStart = circuit->vStart;
  if (vEnd) *vEnd = circuit->vEnd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitGetEdgeRange"
/*@
  DMCircuitGetEdgeRange - Get the bounds [start, end) for the edges.

  Not Collective

  Input Parameters:
+ dm - The DMCircuit object

  Output Paramters:
+ eStart - The first edge point
- eEnd   - One beyond the last edge point

  Level: intermediate

.seealso: DMCircuitGetVertexRange
@*/
PetscErrorCode DMCircuitGetEdgeRange(DM dm,PetscInt *eStart,PetscInt *eEnd)
{
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  if (eStart) *eStart = circuit->eStart;
  if (eEnd) *eEnd = circuit->eEnd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitAddComponent"
/*@
  DMCircuitAddComponent - Adds a circuit component at the given point (vertex/edge)

  Not Collective

  Input Parameters:
+ dm           - The DMCircuit object
. p            - vertex/edge point
. componentkey - component key returned while registering the component
- compvalue    - pointer to the data structure for the component

  Level: intermediate

.seealso: DMCircuitGetVertexRange, DMCircuitGetEdgeRange, DMCircuitRegisterComponent
@*/
PetscErrorCode DMCircuitAddComponent(DM dm, PetscInt p,PetscInt componentkey,void* compvalue)
{
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;
  DMCircuitComponent component=circuit->component[componentkey];
  DMCircuitComponentHeader header=&circuit->header[p];
  DMCircuitComponentValue  cvalue=&circuit->cvalue[p];
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
#define __FUNCT__ "DMCircuitGetNumComponents"
/*@
  DMCircuitGetNumComponents - Get the number of components at a vertex/edge

  Not Collective 

  Input Parameters:
+ dm - The DMCircuit object
. p  - vertex/edge point

  Output Parameters:
. numcomponents - Number of components at the vertex/edge

  Level: intermediate

.seealso: DMCircuitRegisterComponent, DMCircuitAddComponent
@*/
PetscErrorCode DMCircuitGetNumComponents(DM dm,PetscInt p,PetscInt *numcomponents)
{
  PetscErrorCode ierr;
  PetscInt       offset;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DataSection,p,&offset);CHKERRQ(ierr);
  *numcomponents = ((DMCircuitComponentHeader)(circuit->componentdataarray+offset))->ndata;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitGetComponentTypeOffset"
/*@
  DMCircuitGetComponentTypeOffset - Gets the type along with the offset for indexing the 
                                    component value from the component data array

  Not Collective

  Input Parameters:
+ dm      - The DMCircuit object
. p       - vertex/edge point
- compnum - component number
	
  Output Parameters:
+ compkey - the key obtained when registering the component
- offset  - offset into the component data array associated with the vertex/edge point

  Notes:
  Typical usage:

  DMCircuitGetComponentDataArray(dm, &arr);
  DMCircuitGetVertex/EdgeRange(dm,&Start,&End);
  Loop over vertices or edges
    DMCircuitGetNumComponents(dm,v,&numcomps);
    Loop over numcomps
      DMCircuitGetComponentTypeOffset(dm,v,compnum,&key,&offset);
      compdata = (UserCompDataType)(arr+offset);
  
  Level: intermediate

.seealso: DMCircuitGetNumComponents, DMCircuitGetComponentDataArray, 
@*/
PetscErrorCode DMCircuitGetComponentTypeOffset(DM dm,PetscInt p, PetscInt compnum, PetscInt *compkey, PetscInt *offset)
{
  PetscErrorCode ierr;
  PetscInt       offsetp;
  DMCircuitComponentHeader header;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DataSection,p,&offsetp);CHKERRQ(ierr);
  header = (DMCircuitComponentHeader)(circuit->componentdataarray+offsetp);
  *compkey = header->key[compnum];
  *offset  = offsetp+circuit->dataheadersize+header->offset[compnum];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitGetVariableOffset"
/*@
  DMCircuitGetVariableOffset - Get the offset for accessing the variable associated with the given vertex/edge from the local vector.

  Not Collective

  Input Parameters:
+ dm     - The DMCircuit object
- p      - the edge/vertex point

  Output Parameters:
. offset - the offset

  Level: intermediate

.seealso: DMCircuitGetVariableGlobalOffset, DMGetLocalVector
@*/
PetscErrorCode DMCircuitGetVariableOffset(DM dm,PetscInt p,PetscInt *offset)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->DofSection,p,offset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitGetVariableGlobalOffset"
/*@
  DMCircuitGetVariableGlobalOffset - Get the global offset for the variable associated with the given vertex/edge from the global vector.

  Not Collective

  Input Parameters:
+ dm      - The DMCircuit object
- p       - the edge/vertex point

  Output Parameters:
. offsetg - the offset

  Level: intermediate

.seealso: DMCircuitGetVariableOffset, DMGetLocalVector
@*/
PetscErrorCode DMCircuitGetVariableGlobalOffset(DM dm,PetscInt p,PetscInt *offsetg)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionGetOffset(circuit->GlobalDofSection,p,offsetg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitAddNumVariables"
/*@ 
  DMCircuitAddNumVariables - Add number of variables associated with a given point.

  Not Collective

  Input Parameters:
+ dm   - The DMCircuitObject
. p    - the vertex/edge point
- nvar - number of additional variables

  Level: intermediate

.seealso: DMCircuitSetNumVariables
@*/
PetscErrorCode DMCircuitAddNumVariables(DM dm,PetscInt p,PetscInt nvar)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionAddDof(circuit->DofSection,p,nvar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitAddNumVariables"
/*@ 
  DMCircuitSetNumVariables - Sets number of variables for a vertex/edge point.

  Not Collective

  Input Parameters:
+ dm   - The DMCircuitObject
. p    - the vertex/edge point
- nvar - number of variables

  Level: intermediate

.seealso: DMCircuitAddNumVariables
@*/
PetscErrorCode DMCircuitSetNumVariables(DM dm,PetscInt p,PetscInt nvar)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionSetDof(circuit->DofSection,p,nvar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Sets up the array that holds the data for all components and its associated section. This
   function is called during DMSetUp() */
#undef __FUNCT__
#define __FUNCT__ "DMCircuitComponentSetUp"
PetscErrorCode DMCircuitComponentSetUp(DM dm)
{
  PetscErrorCode              ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;
  PetscInt                    arr_size;
  PetscInt                    p,offset,offsetp;
  DMCircuitComponentHeader header;
  DMCircuitComponentValue  cvalue;
  DMCircuitComponentGenericDataType      *componentdataarray;
  PetscInt ncomp, i;

  PetscFunctionBegin;
  ierr = PetscSectionSetUp(circuit->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(circuit->DataSection,&arr_size);CHKERRQ(ierr);
  ierr = PetscMalloc(arr_size*sizeof(DMCircuitComponentGenericDataType),&circuit->componentdataarray);CHKERRQ(ierr);
  componentdataarray = circuit->componentdataarray;
  for (p = circuit->pStart; p < circuit->pEnd; p++) {
    ierr = PetscSectionGetOffset(circuit->DataSection,p,&offsetp);CHKERRQ(ierr);
    /* Copy header */
    header = &circuit->header[p];
    ierr = PetscMemcpy(componentdataarray+offsetp,header,circuit->dataheadersize*sizeof(DMCircuitComponentGenericDataType));
    /* Copy data */
    cvalue = &circuit->cvalue[p];
    ncomp = header->ndata;
    for (i = 0; i < ncomp; i++) {
      offset = offsetp + circuit->dataheadersize + header->offset[i];
      ierr = PetscMemcpy(componentdataarray+offset,cvalue->data[i],header->size[i]*sizeof(DMCircuitComponentGenericDataType));
    }
  }
  PetscFunctionReturn(0);
}

/* Sets up the section for dofs. This routine is called during DMSetUp() */
#undef __FUNCT__
#define __FUNCT__ "DMCircuitVariablesSetUp"
PetscErrorCode DMCircuitVariablesSetUp(DM dm)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = PetscSectionSetUp(circuit->DofSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitGetComponentDataArray"
/*@C
  DMCircuitGetComponentDataArray - Returns the component data array

  Not Collective

  Input Parameters:
. dm - The DMCircuit Object

  Output Parameters:
. componentdataarray - array that holds data for all components

  Level: intermediate

.seealso: DMCircuitGetComponentTypeOffset, DMCircuitGetNumComponents
@*/
PetscErrorCode DMCircuitGetComponentDataArray(DM dm,DMCircuitComponentGenericDataType **componentdataarray)
{
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  *componentdataarray = circuit->componentdataarray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitDistribute"
/*@
  DMCircuitDistribute - Distributes the circuit and moves associated component data.

  Collective

  Input Parameter:
. oldDM - the original DMCircuit object

  Output Parameter:
. distDM - the distributed DMCircuit object

  Notes:
  This routine should be called only when using multiple processors.

  Distributes the circuit with a non-overlapping partitioning of the edges.

  Level: intermediate

.seealso: DMCircuitCreate
@*/
PetscErrorCode DMCircuitDistribute(DM oldDM,DM *distDM)
{
  PetscErrorCode ierr;
  DM_Circuit     *oldDMcircuit = (DM_Circuit*)oldDM->data;
  const char*    partitioner="chaco";
  PetscSF        pointsf;
  DM             newDM;
  DM_Circuit     *newDMcircuit;

  PetscFunctionBegin;
  ierr = DMCircuitCreate(PetscObjectComm((PetscObject)oldDM),&newDM);CHKERRQ(ierr);
  newDMcircuit = (DM_Circuit*)newDM->data;
  newDMcircuit->dataheadersize = sizeof(struct _p_DMCircuitComponentHeader)/sizeof(DMCircuitComponentGenericDataType);
  /* Distribute plex dm and dof section */
  ierr = DMPlexDistribute(oldDMcircuit->plex,partitioner,0,&pointsf,&newDMcircuit->plex);CHKERRQ(ierr);
  /* Distribute dof section */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)oldDM),&newDMcircuit->DofSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointsf,oldDMcircuit->DofSection,NULL,newDMcircuit->DofSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)oldDM),&newDMcircuit->DataSection);CHKERRQ(ierr);
  /* Distribute data and associated section */
  ierr = DMPlexDistributeData(newDMcircuit->plex,pointsf,oldDMcircuit->DataSection,MPI_INT,(void*)oldDMcircuit->componentdataarray,newDMcircuit->DataSection,(void**)&newDMcircuit->componentdataarray);CHKERRQ(ierr);
  /* Destroy point SF */
  ierr = PetscSFDestroy(&pointsf);CHKERRQ(ierr);
  
  ierr = PetscSectionGetChart(newDMcircuit->DataSection,&newDMcircuit->pStart,&newDMcircuit->pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(newDMcircuit->plex,0, &newDMcircuit->eStart,&newDMcircuit->eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(newDMcircuit->plex,1,&newDMcircuit->vStart,&newDMcircuit->vEnd);CHKERRQ(ierr);
  newDMcircuit->nEdges = newDMcircuit->eEnd - newDMcircuit->eStart;
  newDMcircuit->nNodes = newDMcircuit->vEnd - newDMcircuit->vStart;
  newDMcircuit->NNodes = oldDMcircuit->NNodes;
  newDMcircuit->NEdges = oldDMcircuit->NEdges;
  /* Set Dof section as the default section for dm */
  ierr = DMSetDefaultSection(newDMcircuit->plex,newDMcircuit->DofSection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(newDMcircuit->plex,&newDMcircuit->GlobalDofSection);CHKERRQ(ierr);

  *distDM = newDM;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitGetSupportingEdges"
/*@C
  DMCircuitGetSupportingEdges - Return the supporting edges for this vertex point

  Not Collective

  Input Parameters:
+ dm - The DMCircuit object
- p  - the vertex point

  Output Paramters:
+ nedges - number of edges connected to this vertex point
- edges  - List of edge points

  Level: intermediate

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMCircuitCreate, DMCircuitGetConnectedNodes
@*/
PetscErrorCode DMCircuitGetSupportingEdges(DM dm,PetscInt vertex,PetscInt *nedges,const PetscInt *edges[])
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = DMPlexGetSupportSize(circuit->plex,vertex,nedges);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(circuit->plex,vertex,edges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitGetConnectedNodes"
/*@C
  DMCircuitGetConnectedNodes - Return the connected edges for this edge point

  Not Collective

  Input Parameters:
+ dm - The DMCircuit object
- p  - the edge point

  Output Paramters:
. vertices  - vertices connected to this edge

  Level: intermediate

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMCircuitCreate, DMCircuitGetSupportingEdges
@*/
PetscErrorCode DMCircuitGetConnectedNodes(DM dm,PetscInt edge,const PetscInt *vertices[])
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = DMPlexGetCone(circuit->plex,edge,vertices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitIsGhostVertex"
/*@
  DMCircuitIsGhostVertex - Returns TRUE if the vertex is a ghost vertex

  Not Collective

  Input Parameters:
+ dm - The DMCircuit object
. p  - the vertex point

  Output Parameter:
. isghost - TRUE if the vertex is a ghost point 

  Level: intermediate

.seealso: DMCircuitCreate, DMCircuitGetConnectedNodes, DMCircuitGetVertexRange
@*/
PetscErrorCode DMCircuitIsGhostVertex(DM dm,PetscInt p,PetscBool *isghost)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*)dm->data;
  PetscInt       offsetg;
  PetscSection   sectiong;

  PetscFunctionBegin;
  *isghost = PETSC_FALSE;
  ierr = DMGetDefaultGlobalSection(circuit->plex,&sectiong);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(sectiong,p,&offsetg);CHKERRQ(ierr);
  if (offsetg < 0) *isghost = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_Circuit"
PetscErrorCode DMSetUp_Circuit(DM dm)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit=(DM_Circuit*)dm->data;

  PetscFunctionBegin;
  ierr = DMCircuitComponentSetUp(dm);CHKERRQ(ierr);
  ierr = DMCircuitVariablesSetUp(dm);CHKERRQ(ierr);

  ierr = DMSetDefaultSection(circuit->plex,circuit->DofSection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(circuit->plex,&circuit->GlobalDofSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Circuit"
PetscErrorCode DMCreateMatrix_Circuit(DM dm,Mat *J)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMCreateMatrix(circuit->plex,J);CHKERRQ(ierr);
  ierr = MatSetDM(*J,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Circuit"
PetscErrorCode DMDestroy_Circuit(DM dm)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMDestroy(&circuit->plex);CHKERRQ(ierr);
  circuit->edges = NULL;
  ierr = PetscSectionDestroy(&circuit->DataSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&circuit->DofSection);CHKERRQ(ierr);
  /*  ierr = PetscSectionDestroy(&circuit->GlobalDofSection);CHKERRQ(ierr); */
  ierr = PetscFree(circuit->componentdataarray);CHKERRQ(ierr);
  ierr = PetscFree(circuit->cvalue);CHKERRQ(ierr);
  ierr = PetscFree(circuit->header);CHKERRQ(ierr);
  ierr = PetscFree(circuit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_Circuit"
PetscErrorCode DMView_Circuit(DM dm, PetscViewer viewer)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMView(circuit->plex,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "DMGlobalToLocalBegin_Circuit"
PetscErrorCode DMGlobalToLocalBegin_Circuit(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMGlobalToLocalBegin(circuit->plex,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "DMGlobalToLocalEnd_Circuit"
PetscErrorCode DMGlobalToLocalEnd_Circuit(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMGlobalToLocalEnd(circuit->plex,g,mode,l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "DMLocalToGlobalBegin_Circuit"
PetscErrorCode DMLocalToGlobalBegin_Circuit(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMLocalToGlobalBegin(circuit->plex,l,mode,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "DMLocalToGlobalEnd_Circuit"
PetscErrorCode DMLocalToGlobalEnd_Circuit(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMLocalToGlobalEnd(circuit->plex,l,mode,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
