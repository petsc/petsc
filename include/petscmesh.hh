#if !defined(__PETSCMESH_HH)
#define __PETSCMESH_HH

#include <petscmesh.h>

using ALE::Obj;

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateMatrix" 
template<typename Section>
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateMatrix(const Obj<ALE::Mesh>& mesh, const Obj<Section>& section, MatType mtype, Mat *J)
{
  const ALE::Obj<typename ALE::Mesh::order_type>& order = mesh->getFactory()->getGlobalOrder(mesh, "default", section);
  int            localSize  = order->getLocalSize();
  int            globalSize = order->getGlobalSize();
  PetscTruth     isShell, isBlock, isSeqBlock, isMPIBlock;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(mesh->comm(), J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J, localSize, localSize, globalSize, globalSize);CHKERRQ(ierr);
  ierr = MatSetType(*J, mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*J);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSHELL, &isShell);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATBAIJ, &isBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSEQBAIJ, &isSeqBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATMPIBAIJ, &isMPIBlock);CHKERRQ(ierr);
  if (!isShell) {
    int bs = 1;

    if (isBlock || isSeqBlock || isMPIBlock) {
      bs = section->getFiberDimension(*section->getChart().begin());
    }
    ierr = preallocateOperator(mesh, bs, section->getAtlas(), order, *J);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MeshCreateGlobalScatter"
template<typename Section>
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalScatter(const ALE::Obj<ALE::Mesh>& m, const ALE::Obj<Section>& s, VecScatter *scatter)
{
  typedef ALE::Mesh::real_section_type::index_type index_type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_GetGlobalScatter,0,0,0,0);CHKERRQ(ierr);
  const ALE::Mesh::real_section_type::chart_type& chart       = s->getChart();
  const ALE::Obj<ALE::Mesh::order_type>&          globalOrder = m->getFactory()->getGlobalOrder(m, s->getName(), s);
  int *localIndices, *globalIndices;
  int  localSize = s->size();
  int  localIndx = 0, globalIndx = 0;
  Vec  globalVec, localVec;
  IS   localIS, globalIS;

  ierr = VecCreate(m->comm(), &globalVec);CHKERRQ(ierr);
  ierr = VecSetSizes(globalVec, globalOrder->getLocalSize(), PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(globalVec);CHKERRQ(ierr);
  // Loop over all local points
  ierr = PetscMalloc(localSize*sizeof(int), &localIndices); CHKERRQ(ierr);
  ierr = PetscMalloc(localSize*sizeof(int), &globalIndices); CHKERRQ(ierr);
  for(ALE::Mesh::real_section_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
    // Map local indices to global indices
    s->getIndices(*p_iter, localIndices, &localIndx, 0, true, true);
    s->getIndices(*p_iter, globalOrder, globalIndices, &globalIndx, 0, true, false);
    //numConstraints += s->getConstraintDimension(*p_iter);
  }
  // Local arrays also have constraints, which are not mapped
  if (localIndx  != localSize) SETERRQ2(PETSC_ERR_ARG_SIZ, "Invalid number of local indices %d, should be %d", localIndx, localSize);
  if (globalIndx != localSize) SETERRQ2(PETSC_ERR_ARG_SIZ, "Invalid number of global indices %d, should be %d", globalIndx, localSize);
  if (m->debug()) {
    globalOrder->view("Global Order");
    for(int i = 0; i < localSize; ++i) {
      printf("[%d] localIndex[%d]: %d globalIndex[%d]: %d\n", m->commRank(), i, localIndices[i], i, globalIndices[i]);
    }
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, localSize, localIndices,  &localIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, localSize, globalIndices, &globalIS);CHKERRQ(ierr);
  ierr = PetscFree(localIndices);CHKERRQ(ierr);
  ierr = PetscFree(globalIndices);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, s->sizeWithBC(), s->restrict(), &localVec);CHKERRQ(ierr);
  ierr = VecScatterCreate(localVec, localIS, globalVec, globalIS, scatter);CHKERRQ(ierr);
  ierr = ISDestroy(globalIS);CHKERRQ(ierr);
  ierr = ISDestroy(localIS);CHKERRQ(ierr);
  ierr = VecDestroy(localVec);CHKERRQ(ierr);
  ierr = VecDestroy(globalVec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mesh_GetGlobalScatter,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocateOperator"
template<typename Atlas>
PetscErrorCode preallocateOperator(const ALE::Obj<ALE::Mesh>& mesh, const int bs, const ALE::Obj<Atlas>& atlas, const ALE::Obj<ALE::Mesh::order_type>& globalOrder, Mat A)
{
  typedef ALE::SieveAlg<ALE::Mesh> sieve_alg_type;
  MPI_Comm                              comm      = mesh->comm();
  const ALE::Obj<ALE::Mesh>             adjBundle = new ALE::Mesh(comm, mesh->debug());
  const ALE::Obj<ALE::Mesh::sieve_type> adjGraph  = new ALE::Mesh::sieve_type(comm, mesh->debug());
  PetscInt       numLocalRows, firstRow;
  PetscInt      *dnz, *onz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  adjBundle->setSieve(adjGraph);
  numLocalRows = globalOrder->getLocalSize();
  firstRow     = globalOrder->getGlobalOffsets()[mesh->commRank()];
  ierr = PetscMalloc2(numLocalRows, PetscInt, &dnz, numLocalRows, PetscInt, &onz);CHKERRQ(ierr);
  /* Create local adjacency graph */
  /*   In general, we need to get FIAT info that attaches dual basis vectors to sieve points */
  const typename Atlas::chart_type& chart = atlas->getChart();

  for(typename Atlas::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const Obj<typename sieve_alg_type::supportArray>& star = sieve_alg_type::star(mesh, *c_iter);

    for(typename sieve_alg_type::supportArray::const_iterator s_iter = star->begin(); s_iter != star->end(); ++s_iter) {
      const Obj<typename sieve_alg_type::coneArray>& closure = sieve_alg_type::closure(mesh, *s_iter);

      for(typename sieve_alg_type::coneArray::const_iterator cl_iter = closure->begin(); cl_iter != closure->end(); ++cl_iter) {
        adjGraph->addCone(*cl_iter, *c_iter);
      }
    }
  }
  /* Distribute adjacency graph */
  adjBundle->constructOverlap();
  typedef typename ALE::Mesh::sieve_type::point_type point_type;
  typedef typename ALE::Mesh::send_overlap_type send_overlap_type;
  typedef typename ALE::Mesh::recv_overlap_type recv_overlap_type;
  typedef typename ALE::Field<send_overlap_type, int, ALE::Section<point_type, point_type> > send_section_type;
  typedef typename ALE::Field<recv_overlap_type, int, ALE::Section<point_type, point_type> > recv_section_type;
  const Obj<send_overlap_type>& vertexSendOverlap = mesh->getSendOverlap();
  const Obj<recv_overlap_type>& vertexRecvOverlap = mesh->getRecvOverlap();
  const Obj<send_overlap_type>  nbrSendOverlap    = new send_overlap_type(comm, mesh->debug());
  const Obj<recv_overlap_type>  nbrRecvOverlap    = new recv_overlap_type(comm, mesh->debug());
  const Obj<send_section_type>  sendSection       = new send_section_type(comm, mesh->debug());
  const Obj<recv_section_type>  recvSection       = new recv_section_type(comm, sendSection->getTag(), mesh->debug());

  ALE::Distribution<ALE::Mesh>::coneCompletion(vertexSendOverlap, vertexRecvOverlap, adjBundle, sendSection, recvSection);
  /* Distribute indices for new points */
  ALE::Distribution<ALE::Mesh>::updateOverlap(sendSection, recvSection, nbrSendOverlap, nbrRecvOverlap);
  mesh->getFactory()->completeOrder(globalOrder, nbrSendOverlap, nbrRecvOverlap, true);
  /* Read out adjacency graph */
  const ALE::Obj<ALE::Mesh::sieve_type> graph = adjBundle->getSieve();

  ierr = PetscMemzero(dnz, numLocalRows/bs * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(onz, numLocalRows/bs * sizeof(PetscInt));CHKERRQ(ierr);
  for(typename Atlas::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const typename Atlas::point_type& point = *c_iter;

    if (globalOrder->isLocal(point)) {
      const ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence>& adj   = graph->cone(point);
      const ALE::Mesh::order_type::value_type&                     rIdx  = globalOrder->restrictPoint(point)[0];
      const int                                                    row   = rIdx.prefix;
      const int                                                    rSize = rIdx.index/bs;

      //if (rIdx.index%bs) std::cout << "["<<graph->commRank()<<"]: row "<<row<<": size " << rIdx.index << " bs "<<bs<<std::endl;
      if (rSize == 0) continue;
      for(ALE::Mesh::sieve_type::traits::coneSequence::iterator v_iter = adj->begin(); v_iter != adj->end(); ++v_iter) {
        const ALE::Mesh::point_type&             neighbor = *v_iter;
        const ALE::Mesh::order_type::value_type& cIdx     = globalOrder->restrictPoint(neighbor)[0];
        const int&                               cSize    = cIdx.index/bs;

        //if (cIdx.index%bs) std::cout << "["<<graph->commRank()<<"]:   col "<<cIdx.prefix<<": size " << cIdx.index << " bs "<<bs<<std::endl;
        if (cSize > 0) {
          if (globalOrder->isLocal(neighbor)) {
            for(int r = 0; r < rSize; ++r) {dnz[(row - firstRow)/bs + r] += cSize;}
          } else {
            for(int r = 0; r < rSize; ++r) {onz[(row - firstRow)/bs + r] += cSize;}
          }
        }
      }
    }
  }
  if (mesh->debug()) {
    int rank = mesh->commRank();
    for(int r = 0; r < numLocalRows/bs; r++) {
      std::cout << "["<<rank<<"]: dnz["<<r<<"]: " << dnz[r] << " onz["<<r<<"]: " << onz[r] << std::endl;
    }
  }
  ierr = MatSeqAIJSetPreallocation(A, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A, bs, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A, bs, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = PetscFree2(dnz, onz);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif // __PETSCMESH_HH
