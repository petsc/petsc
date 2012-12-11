#include <petsc-private/compleximpl.h>
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>

typedef struct {
  PetscInt nvertices;
  PetscInt ncells;
  PetscInt nconn;               /* number of entries in cell->vertex connectivity array */
} PieceInfo;

#if defined(PETSC_USE_REAL_SINGLE)
  static const char precision[]  = "Float32";
#elif defined(PETSC_USE_REAL_DOUBLE)
  static const char precision[]  = "Float64";
#else
  static const char precision[]  = "UnknownPrecision";
#endif

#undef __FUNCT__
#define __FUNCT__ "TransferWrite"
static PetscErrorCode TransferWrite(PetscViewer viewer,FILE *fp,PetscMPIInt srank,PetscMPIInt root,const void *send,void *recv,PetscMPIInt count,PetscDataType datatype,PetscMPIInt tag)
{
  PetscMPIInt rank;
  PetscErrorCode ierr;
  MPI_Comm comm = ((PetscObject)viewer)->comm;
  MPI_Datatype mpidatatype;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscDataTypeToMPIDataType(datatype,&mpidatatype);CHKERRQ(ierr);

  if (rank == srank && rank != root) {
    ierr = MPI_Send((void*)send,count,mpidatatype,root,tag,comm);CHKERRQ(ierr);
  } else if (rank == root) {
    const void *buffer;
    if (root == srank) {        /* self */
      buffer = send;
    } else {
      MPI_Status status;
      PetscMPIInt nrecv;
      ierr = MPI_Recv(recv,count,mpidatatype,srank,tag,comm,&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,mpidatatype,&nrecv);CHKERRQ(ierr);
      if (count != nrecv) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Array size mismatch");
      buffer = recv;
    }
    ierr = PetscViewerVTKFWrite(viewer,fp,buffer,count,datatype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetVTKConnectivity"
static PetscErrorCode DMComplexGetVTKConnectivity(DM dm,PieceInfo *piece,PetscVTKInt **oconn,PetscVTKInt **ooffsets,PetscVTKType **otypes)
{
  PetscErrorCode ierr;
  PetscVTKInt *conn,*offsets;
  PetscVTKType *types;
  PetscInt dim,vStart,vEnd,cStart,cEnd,pStart,pEnd,cellHeight,cMax,numLabelCells,hasLabel,c,v,countcell,countconn;

  PetscFunctionBegin;
  ierr = PetscMalloc3(piece->nconn,PetscVTKInt,&conn,piece->ncells,PetscVTKInt,&offsets,piece->ncells,PetscVTKType,&types);CHKERRQ(ierr);

  ierr = DMComplexGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMComplexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMComplexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMComplexGetVTKBounds(dm, &cMax, PETSC_NULL);CHKERRQ(ierr);
  if (cMax >= 0) {cEnd = PetscMin(cEnd, cMax);}
  ierr = DMComplexGetStratumSize(dm, "vtk", 1, &numLabelCells);CHKERRQ(ierr);
  hasLabel = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;

  countcell = 0;
  countconn = 0;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = PETSC_NULL;
    PetscInt  closureSize,nverts,celltype,startoffset;

    if (hasLabel) {
      PetscInt value;

      ierr = DMComplexGetLabelValue(dm, "vtk", c, &value);CHKERRQ(ierr);
      if (value != 1) continue;
    }
    startoffset = countconn;
    ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (v = 0; v < closureSize*2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) {
        conn[countconn++] = closure[v] - vStart;
      }
    }
    ierr = DMComplexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    offsets[countcell] = countconn;
    nverts = countconn - startoffset;
    ierr = DMComplexVTKGetCellType(dm,dim,nverts,&celltype);CHKERRQ(ierr);
    types[countcell] = celltype;
    countcell++;
  }
  if (countcell != piece->ncells) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent cell count");
  if (countconn != piece->nconn) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistent connectivity count");
  *oconn = conn;
  *ooffsets = offsets;
  *otypes = types;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexVTKWriteAll_VTU"
/*
  Write all fields that have been provided to the viewer
  Multi-block XML format with binary appended data.
*/
PetscErrorCode DMComplexVTKWriteAll_VTU(DM dm,PetscViewer viewer)
{
  MPI_Comm comm = ((PetscObject)dm)->comm;
  PetscViewer_VTK          *vtk = (PetscViewer_VTK*)viewer->data;
  PetscViewerVTKObjectLink link;
  FILE                     *fp;
  PetscMPIInt              rank,size,tag;
  PetscErrorCode ierr;
  PetscInt                 dim,cellHeight,cStart,cEnd,vStart,vEnd,cMax,numLabelCells,hasLabel,c,v,r,i;
  PieceInfo                piece,*gpiece = PETSC_NULL;
  void                     *buffer = PETSC_NULL;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,"Complex values not supported");
#endif
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  tag = ((PetscObject)viewer)->tag;

  ierr = PetscFOpen(comm,vtk->filename,"wb",&fp);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"<?xml version=\"1.0\"?>\n");CHKERRQ(ierr);
#ifdef PETSC_WORDS_BIGENDIAN
  ierr = PetscFPrintf(comm,fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");CHKERRQ(ierr);
#else
  ierr = PetscFPrintf(comm,fp,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");CHKERRQ(ierr);
#endif
  ierr = PetscFPrintf(comm,fp,"  <UnstructuredGrid>\n");CHKERRQ(ierr);

  ierr = DMComplexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMComplexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMComplexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMComplexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMComplexGetVTKBounds(dm, &cMax, PETSC_NULL);CHKERRQ(ierr);
  if (cMax >= 0) {cEnd = PetscMin(cEnd, cMax);}
  ierr = DMComplexGetStratumSize(dm, "vtk", 1, &numLabelCells);CHKERRQ(ierr);
  hasLabel = numLabelCells > 0 ? PETSC_TRUE : PETSC_FALSE;
  piece.nvertices = vEnd - vStart;
  piece.ncells = 0;
  piece.nconn = 0;
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = PETSC_NULL;
    PetscInt  closureSize;

    if (hasLabel) {
      PetscInt value;

      ierr = DMComplexGetLabelValue(dm, "vtk", c, &value);CHKERRQ(ierr);
      if (value != 1) continue;
    }
    ierr = DMComplexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (v = 0; v < closureSize*2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) piece.nconn++;
    }
    ierr = DMComplexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    piece.ncells++;
  }
  if (!rank) {ierr = PetscMalloc(size*sizeof(piece),&gpiece);CHKERRQ(ierr);}
  ierr = MPI_Gather(&piece,sizeof(piece)/sizeof(PetscInt),MPIU_INT,gpiece,sizeof(piece)/sizeof(PetscInt),MPIU_INT,0,comm);CHKERRQ(ierr);

  /*
   * Write file header
   */
  if (!rank) {
    PetscInt boffset = 0;

    for (r=0; r<size; r++) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"    <Piece NumberOfPoints=\"%D\" NumberOfCells=\"%D\">\n",gpiece[r].nvertices,gpiece[r].ncells);CHKERRQ(ierr);
      /* Coordinate positions */
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      <Points>\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"%s\" Name=\"Position\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%D\" />\n",precision,boffset);CHKERRQ(ierr);
      boffset += gpiece[r].nvertices*3*sizeof(PetscScalar) + sizeof(int);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      </Points>\n");CHKERRQ(ierr);
      /* Cell connectivity */
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      <Cells>\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset);CHKERRQ(ierr);
      boffset += gpiece[r].nconn*sizeof(PetscInt) + sizeof(int);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"offsets\"      NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset);CHKERRQ(ierr);
      boffset += gpiece[r].ncells*sizeof(PetscInt) + sizeof(int);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"UInt8\" Name=\"types\"        NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset);CHKERRQ(ierr);
      boffset += gpiece[r].ncells*sizeof(unsigned char) + sizeof(int);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      </Cells>\n");CHKERRQ(ierr);

      /*
       * Cell Data headers
       */
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      <CellData>\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"        <DataArray type=\"Int32\" Name=\"Rank\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",boffset);CHKERRQ(ierr);
      boffset += gpiece[r].ncells*sizeof(int) + sizeof(int);
      /* all the vectors */
      for (link=vtk->link; link; link=link->next) {
        Vec X = (Vec)link->vec;
        PetscInt bs;
        const char *vecname = "";
        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        if (((PetscObject)X)->name || link != vtk->link) { /* If the object is already named, use it. If it is past the first link, name it to disambiguate. */
          ierr = PetscObjectGetName((PetscObject)X,&vecname);CHKERRQ(ierr);
        }
        ierr = VecGetBlockSize(X,&bs);CHKERRQ(ierr);
        for (i=0; i<bs; i++) {
          char buf[256];
          const char *fieldname = PETSC_NULL;
          /* ierr = DMDAGetFieldName(da,i,&fieldname);CHKERRQ(ierr); */
          if (!fieldname) {
            ierr = PetscSNPrintf(buf,sizeof(buf),"Unnamed%D",i);CHKERRQ(ierr);
            fieldname = buf;
          }
          ierr = PetscFPrintf(comm,fp,"        <DataArray type=\"%s\" Name=\"%s%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%D\" />\n",precision,vecname,fieldname,boffset);CHKERRQ(ierr);
          boffset += gpiece[r].ncells*sizeof(PetscScalar) + sizeof(int);
        }
      }
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      </CellData>\n");CHKERRQ(ierr);

      /*
       * TODO: Point Data headers
       */
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      <PointData>\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"      </PointData>\n");CHKERRQ(ierr);

      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"    </Piece>\n");CHKERRQ(ierr);
    }
  }

  ierr = PetscFPrintf(comm,fp,"  </UnstructuredGrid>\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"  <AppendedData encoding=\"raw\">\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"_");CHKERRQ(ierr);

  if (!rank) {
    PetscInt maxsize = 0;
    for (r=0; r<size; r++) {
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].nvertices*3*sizeof(PetscScalar)));
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].ncells*sizeof(PetscScalar)));
      maxsize = PetscMax(maxsize, (PetscInt) (gpiece[r].nconn*sizeof(PetscVTKInt)));
    }
    ierr = PetscMalloc(maxsize,&buffer);CHKERRQ(ierr);
  }
  for (r=0; r<size; r++) {
    if (r == rank) {
      PetscInt nsend;
      {                         /* Position */
        const PetscScalar *x;
        PetscScalar *y = PETSC_NULL;
        Vec coords;
        nsend = piece.nvertices*3;
        ierr = DMGetCoordinatesLocal(dm,&coords);CHKERRQ(ierr);
        ierr = VecGetArrayRead(coords,&x);CHKERRQ(ierr);
        if (dim != 3) {
          ierr = PetscMalloc(piece.nvertices*3*sizeof(PetscScalar),&y);CHKERRQ(ierr);
          for (i=0; i<piece.nvertices; i++) {
            y[i*3+0] = x[i*dim+0];
            y[i*3+1] = (dim > 1) ? x[i*dim+1] : 0;
            y[i*3+2] = 0;
          }
        }
        ierr = TransferWrite(viewer,fp,r,0,y?y:x,buffer,nsend,PETSC_SCALAR,tag);CHKERRQ(ierr);
        ierr = PetscFree(y);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(coords,&x);CHKERRQ(ierr);
      }
      {                           /* Connectivity, offsets, types */
        PetscVTKInt *connectivity,*offsets;
        PetscVTKType *types;
        ierr = DMComplexGetVTKConnectivity(dm,&piece,&connectivity,&offsets,&types);CHKERRQ(ierr);
        ierr = TransferWrite(viewer,fp,r,0,connectivity,buffer,piece.nconn,PETSC_INT32,tag);CHKERRQ(ierr);
        ierr = TransferWrite(viewer,fp,r,0,offsets,buffer,piece.ncells,PETSC_INT32,tag);CHKERRQ(ierr);
        ierr = TransferWrite(viewer,fp,r,0,types,buffer,piece.ncells,PETSC_UINT8,tag);CHKERRQ(ierr);
        ierr = PetscFree3(connectivity,offsets,types);CHKERRQ(ierr);
      }
      {                         /* Owners (cell data) */
        PetscVTKInt *owners;
        ierr = PetscMalloc(piece.ncells*sizeof(PetscVTKInt),&owners);CHKERRQ(ierr);
        for (i=0; i<piece.ncells; i++) owners[i] = rank;
        ierr = TransferWrite(viewer,fp,r,0,owners,buffer,piece.ncells,PETSC_INT32,tag);CHKERRQ(ierr);
        ierr = PetscFree(owners);CHKERRQ(ierr);
      }
                                /* Cell data */
      for (link=vtk->link; link; link=link->next) {
        Vec X = (Vec)link->vec;
        const PetscScalar *x;
        PetscScalar *y;
        PetscInt bs;
        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        ierr = VecGetBlockSize(X,&bs);CHKERRQ(ierr);
        ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
        ierr = PetscMalloc(piece.ncells*sizeof(PetscScalar),&y);CHKERRQ(ierr);
        for (i=0; i<bs; i++) {
          PetscInt cnt;
          for (c=cStart,cnt=0; c<cEnd; c++) {
            const PetscScalar *xpoint;
            if (hasLabel) {     /* Ignore some cells */
              PetscInt value;
              ierr = DMComplexGetLabelValue(dm, "vtk", c, &value);CHKERRQ(ierr);
              if (value != 1) continue;
            }
            ierr = DMComplexPointLocalRead(dm,c,x,&xpoint);CHKERRQ(ierr);
            y[cnt++] = xpoint[i];
          }
          if (cnt != piece.ncells) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Count does not match");
          ierr = TransferWrite(viewer,fp,r,0,y,buffer,piece.ncells,PETSC_SCALAR,tag);CHKERRQ(ierr);
        }
        ierr = PetscFree(y);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
      }

      for (link=vtk->link; link; link=link->next) {
        if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for point data yet");
      }
    } else if (!rank) {
      ierr = TransferWrite(viewer,fp,r,0,PETSC_NULL,buffer,gpiece[r].nvertices*3,PETSC_SCALAR,tag);CHKERRQ(ierr); /* positions */
      ierr = TransferWrite(viewer,fp,r,0,PETSC_NULL,buffer,gpiece[r].nconn,PETSC_INT32,tag);CHKERRQ(ierr); /* connectivity */
      ierr = TransferWrite(viewer,fp,r,0,PETSC_NULL,buffer,gpiece[r].ncells,PETSC_INT32,tag);CHKERRQ(ierr); /* offsets */
      ierr = TransferWrite(viewer,fp,r,0,PETSC_NULL,buffer,gpiece[r].ncells,PETSC_UINT8,tag);CHKERRQ(ierr); /* types */
      ierr = TransferWrite(viewer,fp,r,0,PETSC_NULL,buffer,gpiece[r].ncells,PETSC_INT32,tag);CHKERRQ(ierr); /* owner rank (cells) */
      /* all cell data */
      for (link=vtk->link; link; link=link->next) {
        Vec X = (Vec)link->vec;
        PetscInt bs;
        if ((link->ft != PETSC_VTK_CELL_FIELD) && (link->ft != PETSC_VTK_CELL_VECTOR_FIELD)) continue;
        ierr = VecGetBlockSize(X,&bs);CHKERRQ(ierr);
        for (i=0; i<bs; i++) {
          ierr = TransferWrite(viewer,fp,r,0,PETSC_NULL,buffer,gpiece[r].ncells,PETSC_SCALAR,tag);CHKERRQ(ierr);
        }
      }
      /* all point data */
      for (link=vtk->link; link; link=link->next) {
        if ((link->ft != PETSC_VTK_POINT_FIELD) && (link->ft != PETSC_VTK_POINT_VECTOR_FIELD)) continue;
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for point data yet");
      }
    }
  }
  ierr = PetscFree(gpiece);CHKERRQ(ierr);
  ierr = PetscFree(buffer);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"\n  </AppendedData>\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,fp,"</VTKFile>\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
