PetscErrorCode outputSieve(const char basename[])
{
  PetscFunctionBegin;

  sieve = m->getSieve();
    SaveMesh2ParallelFiles(basename);//dump coordinates
    if (debug)
    {
        sieve->view("mesh sieve", comm);
    }

    typedef PETSC_MESH_TYPE::sieve_type ISieve;
    std::ofstream fs;
    ostringstream filename;
    filename<<basename<<"_IsieveNP"<<size<<"Rank"<<rank<<".txt";
    fs.open(filename.str().c_str());
    const Obj<ISieve::chart_type> & chart=sieve->getChart();
    ISieve::point_type min = chart->min();
    ISieve::point_type max = chart->max();
    fs<<"chart: min= "<<min<<", max= "<<max<<"\n";
    fs<<"point, coneSize, supportSize\n";
    for (int i=min; i<max; i++)
    {
        fs<<i<<": "<<sieve->getConeSize(i)<<", "<<sieve->getSupportSize(i)
                <<"\n";
    }
    //output cone and support
    typedef ALE::ISieveVisitor::PointRetriever<ISieve> Visitor;
    Visitor pV((int) pow(sieve->getMaxConeSize(), m->depth())+1, true);
    for (int i=min; i<max; i++)
    {

        sieve->cone(i, pV);
        const PETSC_MESH_TYPE::point_type *cPoints = pV.getPoints();
        const int cSize = pV.getSize();
        if (cSize>0)
        {
            fs<<"cone for point "<<i<<": ";
            for (int cl=0; cl<cSize-1; cl++)
            {
                fs<<cPoints[cl]<<" ";
            }
            fs<<cPoints[cSize-1]<<"\n";
        }
        pV.clear();

        sieve->orientedCone(i, pV);
        const Visitor::oriented_point_type *oPoints = pV.getOrientedPoints();
        const int oSize = pV.getOrientedSize();
        if (oSize>0)
        {
            fs<<"coneOrientation for point "<<i<<": ";
            for (int cl=0; cl<oSize-1; cl++)
            {
                fs<<oPoints[cl].second<<" ";
            }
            fs<<oPoints[oSize-1].second<<"\n";
        }
        pV.clear();

        sieve->support(i, pV);
        const PETSC_MESH_TYPE::point_type *sPoints = pV.getPoints();
        const int sSize = pV.getSize();
        if (sSize>0)
        {
            fs<<"support for point "<<i<<": ";
            for (int cl=0; cl<sSize-1; cl++)
            {
                fs<<sPoints[cl]<<" ";
            }
            fs<<sPoints[sSize-1]<<"\n";
        }
        pV.clear();

    }
    fs.close();

    //output renumbering
    if (size>1)
    {
        ostringstream nfilename;
        nfilename<<basename<<"_renumberingNP"<<size<<"Rank"<<rank<<".txt";
        fs.open(nfilename.str().c_str());

        std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::sieve_type::point_type>
                &renumbering= m->getRenumbering();
        fs
                <<"renumbring: first=serial number (second=sequential starting from 0)\n";
        for (std::map<int,int>::iterator it=renumbering.begin(); it
                !=renumbering.end(); it++)
        {
            fs<<it->first<<std::endl;
            //fs<<it->first<<" "<<it->second<<std::endl;
        }
        fs.close();
    }
    MPI_Barrier(comm);

    PetscFunctionReturn(0);
}

template<typename Mesh>
PetscErrorCode WriteCoordinates(const Mesh& mesh, const char basename[])
{
  PetscViewer    viewer;
  ostringstream  filename;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  filename << basename << ".p" << mesh.commSize() << ".r" << mesh.commRank() << ".coord";
  ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename.str().c_str(), &viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Global mesh: %d elements, %d faces, %d edges and %d vertices\n",
                                NumberofElements, NumberofFaces, NumberofEdges, NumberofVertexes);CHKERRQ(ierr);

  const ALE::Obj<typename Mesh::label_sequence>&    vertices    = mesh.depthStratum(0);
  const ALE::Obj<typename Mesh::real_section_type>& coordinates = mesh.getRealSection("coordinates");

  ierr = PetscViewerASCIIPrintf(viewer, "%d local vertices\n", vertices->size());CHKERRQ(ierr);
  for(typename Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const typename Mesh::real_section_type::value_type *coords = coordinates->restrictPoint(*v_iter);

    ierr = PetscViewerASCIIPrintf(viewer, "%9d %10.5f %10.5f %10.5f\n", *v_iter, coords[0], coords[1], coords[2]);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode USG::loadSieve(const char *basename)
{
  PetscFunctionBegin;

  const Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(comm,debug);
  m = PETSC_MESH_TYPE(comm, 3, debug);
  FILE *fp;

  //isieve
  typedef PETSC_MESH_TYPE::sieve_type ISieve;

  ostringstream ifilename;
  ifilename << basename << ".p" << mesh.commSize() << ".r" << mesh.commRank() << ".isieve";
    fp=fopen(ifilename.str().c_str(), "r");
    printf("loading isieve from %s\n", ifilename.str().c_str());
    int min, max;
    fscanf(fp, "chart: min= %d, max= %d\n", &min, &max);
    sieve->setChart(ISieve::chart_type(min, max));
    fscanf(fp, "%*[^\n]");//skip a line
    int p, coneSize, supportSize;
    for (int i=min; i<max; i++)
    {
        fscanf(fp, "%d: %d, %d\n", &p, &coneSize, &supportSize);
        sieve->setConeSize(PETSC_MESH_TYPE::sieve_type::point_type(p), coneSize);
        sieve->setSupportSize(PETSC_MESH_TYPE::sieve_type::point_type(p),
                supportSize);
    }
    sieve->allocate();
    int maxSize=std::max(sieve->getMaxConeSize(), sieve->getMaxSupportSize());

    ISieve::point_type *points = new ISieve::point_type[maxSize];

    for (int i=min; i<max; i++)
    {
        int psize=sieve->getConeSize(i);
        int pt;
        if (psize>0)
        {
            fscanf(fp, "\ncone for point %d:", &pt);
            assert(i==pt);
            for (int p=0; p<psize; p++)
            {
                fscanf(fp, "%d", &points[p]);
            }
            sieve->setCone(points, pt);

        }

        if (psize>0)
        {
            fscanf(fp, "\nconeOrientation for point %d:", &pt);
            assert(i==pt);
            for (int p=0; p<psize; p++)
            {
                fscanf(fp, "%d", &points[p]);
            }
            sieve->setConeOrientation(points, pt);

        }

        psize=sieve->getSupportSize(i);
        if (psize>0)
        {
            fscanf(fp, "\nsupport for point %d:", &pt);
            assert(i==pt);
            for (int p=0; p<psize; p++)
            {
                fscanf(fp, "%d", &points[p]);
            }
            sieve->setSupport(pt, points);

        }
    }

    delete [] points;
    fclose(fp);

    m->setSieve(sieve);
    //outputSieve("loaded");
    m->stratify();

    //sieve->view("test",comm);
    MPI_Barrier(comm);
    printf("%d: done with ISieve loading\n", rank);

    //read the parallel mesh files for coordinates
    std::map<int,double*> coords;
    int numCells = 0;
    int numVertices=0;
    ostringstream meshname;
    meshname<<basename<<"_coordinatesNP"<<size<<"Rank"<<rank<<".txt";

    fp=fopen(meshname.str().c_str(), "r");
    printf("loading parallel mesh from %s\n", meshname.str().c_str());
    fscanf(fp, "Global mesh: %d elements, %d faces, %d edges and %d vertexes",
            &NumberofElements, &NumberofFaces, &NumberofEdges,
            &NumberofVertexes);
    fscanf(fp, "%d local vertexes", &numVertices);
    int index;
    double *xyz;
    for (int i=0; i<numVertices; i++)
    {
        xyz=(double*)malloc(sizeof(double)*3);
        fscanf(fp, "%d %lf %lf %lf", &index, &xyz[0], &xyz[1], &xyz[2]);
        coords.insert(std::pair<int,double*>(index,xyz));
    }
    fscanf(fp,"%d local cells, %d global cells",&numCells,&NumberofElements);
    fclose(fp);
    //printf("[%d]: starting with coordinates construction\n",rank);
    //construct coordinates   
    const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
    coordinates->setChart(ISieve::chart_type(min, max));

    const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& vertices = m->depthStratum(0);
    m->setupCoordinates(coordinates);
    coordinates->setFiberDimension(vertices, 3);
    m->allocate(coordinates);
    for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter)
    {
        //printf("v_iter=%d\n",*v_iter);       
        coordinates->updatePoint(*v_iter, coords[*v_iter]);
    }

    MPI_Barrier(comm);
    printf("[%d]: finished with coordinates construction\n",rank);
    ierr = MeshSetMesh(mesh, m);CHKERRQ(ierr);

    for(std::map<int, double*>::const_iterator it = coords.begin(); it != coords.end(); ++it)
    {
        free(it->second);
    }
    //outputSieve("loaded");

    MPI_Barrier(comm);
    m->setCalculatedOverlap(true);
    //m->view("loaded mesh",comm);

   
    if(size>1) //compute the overlap for parallel simulations
    {       
        //load renumbering
        std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::sieve_type::point_type>
        &renumbering= m->getRenumbering();
        ostringstream numbername;
        numbername<<basename<<"_renumberingNP"<<size<<"Rank"<<rank<<".txt";
        printf("loading parallel renumbering  from %s\n", numbername.str().c_str());
        fp=fopen(numbername.str().c_str(), "r");
        fscanf(fp, "%*[^\n]");//skip a line
        int serialNumber;
        for (int i=min; i<max; i++)
        {
            fscanf(fp,"%d\n",&serialNumber);
            renumbering.insert(std::pair<int,int>(serialNumber,i));
        }
        fclose(fp);
        // Create the parallel overlap

        Obj<PETSC_MESH_TYPE::send_overlap_type> sendParallelMeshOverlap = m->getSendOverlap();
        Obj<PETSC_MESH_TYPE::recv_overlap_type> recvParallelMeshOverlap = m->getRecvOverlap();

        ALE::SetFromMap<std::map<PETSC_MESH_TYPE::point_type,PETSC_MESH_TYPE::point_type> > globalPoints(renumbering);

        ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
        m->setCalculatedOverlap(true);
    }
  PetscFunctionReturn(0);
}
