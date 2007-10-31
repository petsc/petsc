// -*- C++ -*-
//
// ----------------------------------------------------------------------
//
//                           Brad T. Aagaard
//                        U.S. Geological Survey
//
// {LicenseText}
//
// ----------------------------------------------------------------------
//

#include "CohesiveTopology.hh" // implementation of object methods

#include <assert.h> // USES assert()

// ----------------------------------------------------------------------
void
pylith::faults::CohesiveTopology::create(ALE::Obj<Mesh>* fault,
					 const ALE::Obj<Mesh>& mesh,
					 const ALE::Obj<Mesh::int_section_type>& groupField,
					 const int materialId,
					 const bool constraintCell)
{ // create
  assert(0 != fault);
  static PetscEvent CreateVertexSet_Event = 0, CreateFaultMesh_Event = 0, OrientFaultMesh_Event = 0, AddCohesivePoints_Event = 0, SplitMesh_Event = 0;

  if (!CreateVertexSet_Event) {
    PetscLogEventRegister(&CreateVertexSet_Event, "CreateVertexSet", 0);
  }
  if (!CreateFaultMesh_Event) {
    PetscLogEventRegister(&CreateFaultMesh_Event, "CreateFaultMesh", 0);
  }
  if (!OrientFaultMesh_Event) {
    PetscLogEventRegister(&OrientFaultMesh_Event, "OrientFaultMesh", 0);
  }
  if (!AddCohesivePoints_Event) {
    PetscLogEventRegister(&AddCohesivePoints_Event, "AddCohesivePoints", 0);
  }
  if (!SplitMesh_Event) {
    PetscLogEventRegister(&SplitMesh_Event, "SplitMesh", 0);
  }

  typedef ALE::SieveAlg<Mesh> sieveAlg;
  typedef ALE::Selection<Mesh> selection;
  typedef std::set<Mesh::point_type> PointSet;

  const int_section_type::chart_type& chart = groupField->getChart();
  PointSet faultVertices; // Vertices on fault
  const ALE::Obj<sieve_type>& sieve = mesh->getSieve();
  *fault = new Mesh(mesh->comm(), mesh->getDimension()-1, mesh->debug());
  const ALE::Obj<sieve_type> faultSieve = new sieve_type(sieve->comm(), 
                                                         sieve->debug());
  const int  numCells   = mesh->heightStratum(0)->size();
  int        numCorners = 0;    // The number of vertices in a mesh cell
  int        faceSize   = 0;    // The number of vertices in a mesh face
  int       *indices    = NULL; // The indices of a face vertex set in a cell
  int        oppositeVertex;    // For simplices, the vertex opposite a given face
  PointArray origVertices;
  PointArray faceVertices;
  PointArray neighborVertices;

  if (!(*fault)->commRank()) {
    numCorners = sieve->nCone(*mesh->heightStratum(0)->begin(), mesh->depth())->size();
    faceSize   = selection::numFaceVertices(*mesh->heightStratum(0)->begin(), mesh);
    indices    = new int[faceSize];
  }

  // Create set with vertices on fault
  PetscLogEventBegin(CreateVertexSet_Event,0,0,0,0);
  for(int_section_type::chart_type::iterator c_iter = chart.begin();
      c_iter != chart.end();
      ++c_iter) {
    if (mesh->depth(*c_iter)) {
      std::cout << "Vertex " << *c_iter << " depth " << mesh->depth(*c_iter) << std::endl;
      const ALE::Obj<sieve_type::traits::coneSequence>& cone = sieve->cone(*c_iter);

      for(sieve_type::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
        std::cout << "  cone point " << *p_iter << std::endl;
      }
      const ALE::Obj<sieve_type::traits::supportSequence>& support = sieve->support(*c_iter);

      for(sieve_type::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
        std::cout << "  support point " << *p_iter << std::endl;
      }
      throw ALE::Exception("Invalid pset vertex");
    }
    faultVertices.insert(*c_iter);
  } // for
  PetscLogEventEnd(CreateVertexSet_Event,0,0,0,0);

  const PointSet::const_iterator fvBegin = faultVertices.begin();
  const PointSet::const_iterator fvEnd   = faultVertices.end();

  int f = sieve->base()->size() + sieve->cap()->size();
  int debug = mesh->debug();
  ALE::Obj<PointSet> face = new PointSet();
  PointSet faultCells;
  
  // Create a sieve which captures the fault
  PetscLogEventBegin(CreateFaultMesh_Event,0,0,0,0);
  for(PointSet::const_iterator fv_iter = fvBegin; fv_iter != fvEnd; ++fv_iter) {
    const ALE::Obj<sieveAlg::supportArray>& cells =
      sieveAlg::nSupport(mesh, *fv_iter, mesh->depth());
    const sieveAlg::supportArray::iterator cBegin = cells->begin();
    const sieveAlg::supportArray::iterator cEnd   = cells->end();
    
    if (debug)
      std::cout << "Checking fault vertex " << *fv_iter << std::endl;
    for(sieveAlg::supportArray::iterator c_iter = cBegin;
        c_iter != cEnd;
        ++c_iter) {
      if (debug) std::cout << "  Checking cell " << *c_iter << std::endl;
      if (faultCells.find(*c_iter) != faultCells.end())	continue;
      const ALE::Obj<sieveAlg::coneArray>& cone =
        sieveAlg::nCone(mesh, *c_iter, mesh->height());
      const sieveAlg::coneArray::iterator vBegin = cone->begin();
      const sieveAlg::coneArray::iterator vEnd   = cone->end();

      face->clear();
      for(sieveAlg::coneArray::iterator v_iter = vBegin;
          v_iter != vEnd;
          ++v_iter) {
        if (faultVertices.find(*v_iter) != fvEnd) {
          if (debug) std::cout << "    contains fault vertex " << *v_iter << std::endl;
          face->insert(face->end(), *v_iter);
        } // if
      } // for
      if ((int) face->size() > faceSize)
        throw ALE::Exception("Invalid fault mesh: Too many vertices of an "
                             "element on the fault");
      if ((int) face->size() == faceSize) {
        if (debug)
          std::cout << "  Contains a face on the fault" << std::endl;
        const ALE::Obj<sieve_type::supportSet> preFace = faultSieve->nJoin1(face);

        if (preFace->size() > 1) {
          throw ALE::Exception("Invalid fault sieve: Multiple faces from "
                               "vertex set");
        } else if (preFace->size() == 1) {
          // Add the other cell neighbor for this face
          faultSieve->addArrow(*preFace->begin(), *c_iter);
        } else if (preFace->size() == 0) {
          if (debug) std::cout << "  Orienting face " << f << std::endl;
          selection::getOrientedFace(mesh, *c_iter, face, numCorners, indices, &origVertices, &faceVertices);
          if (debug) std::cout << "  Adding face " << f << std::endl;
          int color = 0;
          for(PointArray::const_iterator f_iter = faceVertices.begin();
              f_iter != faceVertices.end(); ++f_iter) {
            if (debug) std::cout << "    vertex " << *f_iter << std::endl;
            faultSieve->addArrow(*f_iter, f, color++);
          } // for
          faultSieve->addArrow(f, *c_iter);
          f++;
        } // if/else
        faultCells.insert(*c_iter);
      } // if
    } // for
  } // for
  (*fault)->setSieve(faultSieve);
  (*fault)->stratify();
  faultCells.clear();
  if (debug) (*fault)->view("Fault mesh");
  PetscLogEventEnd(CreateFaultMesh_Event,0,0,0,0);
  // Orient the fault sieve
  PetscLogEventBegin(OrientFaultMesh_Event,0,0,0,0);
  const ALE::Obj<Mesh::label_sequence>& fFaces = (*fault)->heightStratum(1);
  int faultDepth      = (*fault)->depth()-1; // Depth of fault cells
  int numFaultCorners = 0; // The number of vertices in a fault cell
  int faultFaceSize   = 0; // The number of vertices in a face between fault cells
  PointSet flippedFaces;   // Incorrectly oriented fault cells

  if (!(*fault)->commRank()) {
    numFaultCorners = faultSieve->nCone(*fFaces->begin(), faultDepth)->size();
    if (debug) std::cout << "  Fault corners " << numFaultCorners << std::endl;
    assert(numFaultCorners == faceSize);
    faultFaceSize = selection::numFaceVertices(*fFaces->begin(), (*fault), faultDepth);
  }
  if (debug) std::cout << "  Fault face size " << faultFaceSize << std::endl;
  // Do BFS on the fault mesh
  //   
  PointSet facesSeen;
  ALE::Obj<PointSet> level     = new PointSet();
  ALE::Obj<PointSet> nextLevel = new PointSet();
  ALE::Obj<PointSet> tmpLevel;
  int levelNum = 0;

  if (fFaces->size()) nextLevel->insert(*fFaces->begin());
  while(nextLevel->size()) {
    if (debug) std::cout << "Level " << levelNum << std::endl;
    tmpLevel = level; level = nextLevel; nextLevel = tmpLevel;
    for(PointSet::iterator e_iter = level->begin(); e_iter != level->end(); ++e_iter) {
      if (debug) std::cout << "  Checking fault face " << *e_iter << std::endl;
      const ALE::Obj<sieve_type::traits::coneSequence>& vertices = faultSieve->cone(*e_iter);
      sieve_type::traits::coneSequence::iterator   vEnd     = vertices->end();

      for(sieve_type::traits::coneSequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
        const ALE::Obj<sieve_type::traits::supportSequence>& neighbors  = faultSieve->support(*v_iter);
        const sieve_type::traits::supportSequence::iterator nBegin = neighbors->begin();
        const sieve_type::traits::supportSequence::iterator nEnd   = neighbors->end();

        for(sieve_type::traits::supportSequence::iterator n_iter = nBegin; n_iter != nEnd; ++n_iter) {
          if (facesSeen.find(*n_iter) != facesSeen.end()) continue;
          if (*e_iter == *n_iter) continue;
          if (debug) std::cout << "  Checking fault neighbor " << *n_iter << std::endl;
          const ALE::Obj<sieve_type::coneSet>& meet = faultSieve->nMeet(*e_iter, *n_iter, faultDepth);

          if (debug) {
            for(sieve_type::coneSet::iterator c_iter = meet->begin(); c_iter != meet->end(); ++c_iter)
              std::cout << "    meet " << *c_iter << std::endl;
          }
          if ((int) meet->size() == faultFaceSize) {
            if (debug) std::cout << "    Found neighboring fault face " << *n_iter << std::endl;
            bool compatible = _compatibleOrientation(*fault, *e_iter, *n_iter, numFaultCorners, faultFaceSize, faultDepth,
                                                     meet, indices, &origVertices, &faceVertices, &neighborVertices);
            if (!compatible ^ (flippedFaces.find(*e_iter) != flippedFaces.end())) {
              if (debug) std::cout << "  Scheduling fault face " << *n_iter << " to be flipped" << std::endl;
              flippedFaces.insert(*n_iter);
            }
            nextLevel->insert(*n_iter);
          }
        }
      }
      facesSeen.insert(*e_iter);
    }
    level->clear();
    levelNum++;
  }
  assert(facesSeen.size() == fFaces->size());
  for(PointSet::const_iterator f_iter = flippedFaces.begin(); f_iter != flippedFaces.end(); ++f_iter) {
    if (debug) std::cout << "  Reversing fault face " << *f_iter << std::endl;
    faceVertices.clear();
    const ALE::Obj<sieve_type::traits::coneSequence>& cone = faultSieve->cone(*f_iter);
    for(sieve_type::traits::coneSequence::iterator v_iter = cone->begin();
        v_iter != cone->end(); ++v_iter) {
      faceVertices.insert(faceVertices.begin(), *v_iter);
    }
    faultSieve->clearCone(*f_iter);
    int color = 0;
    for(PointArray::const_iterator v_iter = faceVertices.begin();
        v_iter != faceVertices.end(); ++v_iter) {
      faultSieve->addArrow(*v_iter, *f_iter, color++);
    } // for
  }
  flippedFaces.clear();
  if (debug) (*fault)->view("Oriented Fault mesh");
  for(Mesh::label_sequence::iterator e_iter = fFaces->begin(); e_iter != fFaces->end(); ++e_iter) {
    if (debug) std::cout << "  Checking fault face " << *e_iter << std::endl;
    const ALE::Obj<sieve_type::traits::coneSequence>& vertices = faultSieve->cone(*e_iter);
    sieve_type::traits::coneSequence::iterator   vEnd     = vertices->end();
    PointSet facesSeen;

    for(sieve_type::traits::coneSequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
      const ALE::Obj<sieve_type::traits::supportSequence>& neighbors  = faultSieve->support(*v_iter);
      const sieve_type::traits::supportSequence::iterator nBegin = neighbors->begin();
      const sieve_type::traits::supportSequence::iterator nEnd   = neighbors->end();

      for(sieve_type::traits::supportSequence::iterator n_iter = nBegin; n_iter != nEnd; ++n_iter) {
        if (facesSeen.find(*n_iter) != facesSeen.end()) continue;
        facesSeen.insert(*n_iter);
        if (debug) std::cout << "  Checking fault neighbor " << *n_iter << std::endl;
        if (*e_iter >= *n_iter) continue;
        const ALE::Obj<sieve_type::coneSet>& meet = faultSieve->nMeet(*e_iter, *n_iter, faultDepth);

        if (debug) {
          for(sieve_type::coneSet::iterator c_iter = meet->begin(); c_iter != meet->end(); ++c_iter)
            std::cout << "    meet " << *c_iter << std::endl;
        }
        if ((int) meet->size() == faultFaceSize) {
          if (debug) std::cout << "    Found neighboring fault face " << *n_iter << std::endl;
          bool eOrient = selection::getOrientedFace(*fault, *e_iter, meet, numFaultCorners, indices, &origVertices, &faceVertices);
          bool nOrient = selection::getOrientedFace(*fault, *n_iter, meet, numFaultCorners, indices, &origVertices, &neighborVertices);

          if (faultFaceSize > 1) {
            if (debug) {
              for(PointArray::iterator v_iter = faceVertices.begin(); v_iter != faceVertices.end(); ++v_iter) {
                std::cout << "  face vertex " << *v_iter << std::endl;
              }
              for(PointArray::iterator v_iter = neighborVertices.begin(); v_iter != neighborVertices.end(); ++v_iter) {
                std::cout << "  neighbor vertex " << *v_iter << std::endl;
              }
            }
            // Here we use the fact that fault faces are only 1D
            if (*faceVertices.begin() == *neighborVertices.begin()) {
              if (debug) std::cout << "  Scheduling fault face " << *n_iter << " to be flipped" << std::endl;
              assert(false);
            }
          } else {
            // For 0D, we use the orientation returned (not sure if we have to do this)
            if (nOrient == eOrient) {
              if (debug) std::cout << "  Scheduling fault face " << *n_iter << " to be flipped" << std::endl;
              assert(false);
            }
          }
        }
      }
    }
  }
  PetscLogEventEnd(OrientFaultMesh_Event,0,0,0,0);

  // Add new shadow vertices and possibly Lagrange multipler vertices
  PetscLogEventBegin(AddCohesivePoints_Event,0,0,0,0);
  const ALE::Obj<Mesh::label_sequence>& fVertices = (*fault)->depthStratum(0);
  const ALE::Obj<Mesh::label_sequence>& vertices = mesh->depthStratum(0);
  const ALE::Obj<std::set<std::string> >& groupNames = mesh->getIntSections();
  Mesh::point_type newPoint = sieve->base()->size() + sieve->cap()->size();
  std::map<int,int> vertexRenumber;
  
  for(Mesh::label_sequence::iterator v_iter = fVertices->begin();
      v_iter != fVertices->end();
      ++v_iter, ++newPoint) {
    vertexRenumber[*v_iter] = newPoint;
    if (debug) 
      std::cout << "Duplicating " << *v_iter << " to "
		<< vertexRenumber[*v_iter] << std::endl;

    // Add shadow and constraint vertices (if they exist) to group
    // associated with fault
    groupField->addPoint(newPoint, 1);
    if (constraintCell)
      groupField->addPoint(newPoint+1, 1);

    // Add shadow vertices to other groups, don't add constraint
    // vertices (if they exist) because we don't want BC, etc to act
    // on constraint vertices
    for(std::set<std::string>::const_iterator name = groupNames->begin();
       name != groupNames->end(); ++name) {
      const ALE::Obj<int_section_type>& group = mesh->getIntSection(*name);
      if (group->hasPoint(*v_iter))
        group->addPoint(newPoint, 1);
    } // for
    if (constraintCell) newPoint++;
  } // for
  for(std::set<std::string>::const_iterator name = groupNames->begin();
      name != groupNames->end(); ++name) {
    mesh->reallocate(mesh->getIntSection(*name));
  } // for

  // Split the mesh along the fault sieve and create cohesive elements
  const ALE::Obj<Mesh::label_sequence>& faces = (*fault)->depthStratum(1);
  const ALE::Obj<Mesh::label_type>& material = mesh->hasLabel("material-id") ? mesh->getLabel("material-id") : mesh->createLabel("material-id");
  int firstCohesiveCell = newPoint;
  PointSet replaceCells;
  PointSet noReplaceCells;
  PointSet replaceVertices;

  for(Mesh::label_sequence::iterator f_iter = faces->begin();
      f_iter != faces->end(); ++f_iter, ++newPoint) {
    if (debug) std::cout << "Considering fault face " << *f_iter << std::endl;
    const ALE::Obj<sieve_type::traits::supportSequence>& cells =
      faultSieve->support(*f_iter);
    Mesh::point_type cell = *cells->begin();
    Mesh::point_type otherCell;

    if (debug) std::cout << "  Checking orientation against cell " << cell << std::endl;
    selection::getOrientedFace(mesh, cell, &vertexRenumber, numCorners, indices, &origVertices, &faceVertices);

    const ALE::Obj<sieve_type::traits::coneSequence>& faceCone = faultSieve->cone(*f_iter);
    bool found = true;

    if (numFaultCorners == 2) {
      if (faceVertices[0] != *faceCone->begin()) found = false;
    } else {
      int v = 0;
      // Locate first vertex
      while((v < numFaultCorners) && (faceVertices[v] != *faceCone->begin())) ++v;
      for(sieve_type::traits::coneSequence::iterator v_iter = faceCone->begin(); v_iter != faceCone->end(); ++v_iter, ++v) {
        if (debug) std::cout << "    Checking " << *v_iter << " against " << faceVertices[v%numFaultCorners] << std::endl;
        if (faceVertices[v%numFaultCorners] != *v_iter) {
          found = false;
          break;
        }
      }
    }
    if (found) {
      if (debug) std::cout << "  Choosing other cell" << std::endl;
      otherCell = cell;
      cell = *(++cells->begin());
    } else {
      otherCell = *(++cells->begin());
      if (debug) std::cout << "  Verifing reverse orientation" << std::endl;
      found = true;
      int v = 0;
      // Locate first vertex
      while((v < numFaultCorners) && (faceVertices[v] != *faceCone->rbegin())) ++v;
      for(sieve_type::traits::coneSequence::reverse_iterator v_iter = faceCone->rbegin(); v_iter != faceCone->rend(); ++v_iter, ++v) {
        if (debug) std::cout << "    Checking " << *v_iter << " against " << faceVertices[v%numFaultCorners] << std::endl;
        if (faceVertices[v%numFaultCorners] != *v_iter) {
          found = false;
          break;
        }
      }
      assert(found);
    }
    noReplaceCells.insert(otherCell);
    replaceCells.insert(cell);
    replaceVertices.insert(faceCone->begin(), faceCone->end());
    // Adding cohesive cell (not interpolated)
    const ALE::Obj<sieve_type::traits::coneSequence>& fCone  = faultSieve->cone(*f_iter);
    const sieve_type::traits::coneSequence::iterator  fBegin = fCone->begin();
    const sieve_type::traits::coneSequence::iterator  fEnd   = fCone->end();
    int color = 0;

	if (debug)
	  std::cout << "  Creating cohesive cell " << newPoint << std::endl;
    //for(PointArray::iterator v_iter = fBegin; v_iter != fEnd; ++v_iter) {
    for(sieve_type::traits::coneSequence::iterator v_iter = fBegin; v_iter != fEnd; ++v_iter) {
      if (debug)
        std::cout << "    vertex " << *v_iter << std::endl;
      sieve->addArrow(*v_iter, newPoint, color++);
    }
    //for(PointArray::iterator v_iter = fBegin; v_iter != fEnd; ++v_iter) {
    for(sieve_type::traits::coneSequence::iterator v_iter = fBegin; v_iter != fEnd; ++v_iter) {
      if (debug)
        std::cout << "    shadow vertex " << vertexRenumber[*v_iter] << std::endl;
      sieve->addArrow(vertexRenumber[*v_iter], newPoint, color++);
    }
    if (constraintCell) {
      //for(PointArray::iterator v_iter = fBegin; v_iter != fEnd; ++v_iter) {
      for(sieve_type::traits::coneSequence::iterator v_iter = fBegin; v_iter != fEnd; ++v_iter) {
        if (debug)
          std::cout << "    Lagrange vertex " << vertexRenumber[*v_iter]+1 << std::endl;
        sieve->addArrow(vertexRenumber[*v_iter]+1, newPoint, color++);
      }
    }
    mesh->setValue(material, newPoint, materialId);
  } // for
  PetscLogEventEnd(AddCohesivePoints_Event,0,0,0,0);
  // Replace all cells with a vertex on the fault that share a face with this one, or with one that does
  PetscLogEventBegin(SplitMesh_Event,0,0,0,0);
  for(PointSet::const_iterator v_iter = replaceVertices.begin(); v_iter != replaceVertices.end(); ++v_iter) {
    bool modified = true;

    while(modified) {
      modified = false;
      const ALE::Obj<sieve_type::traits::supportSequence>& neighbors = sieve->support(*v_iter);
      const sieve_type::traits::supportSequence::iterator  end       = neighbors->end();

      for(sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != end; ++n_iter) {
        if (replaceCells.find(*n_iter)   != replaceCells.end())   continue;
        if (noReplaceCells.find(*n_iter) != noReplaceCells.end()) continue;
        if (*n_iter >= firstCohesiveCell) continue;
        if (debug) std::cout << "  Checking fault neighbor " << *n_iter << std::endl;
        // If neighbors shares a faces with anyone in replaceCells, then add
        for(PointSet::const_iterator c_iter = replaceCells.begin(); c_iter != replaceCells.end(); ++c_iter) {
          const ALE::Obj<sieve_type::coneSet>& preFace = sieve->nMeet(*c_iter, *n_iter, mesh->depth());

          if ((int) preFace->size() == faceSize) {
            if (debug) std::cout << "    Scheduling " << *n_iter << " for replacement" << std::endl;
            replaceCells.insert(*n_iter);
            modified = true;
            break;
          }
        }
      }
    }
  }
  for(PointSet::const_iterator c_iter = replaceCells.begin(); c_iter != replaceCells.end(); ++c_iter) {
    _replaceCell(sieve, *c_iter, &vertexRenumber, debug);
  }
  if (!(*fault)->commRank()) delete [] indices;
  mesh->stratify();
  const ALE::Obj<Mesh::label_type>& label          = mesh->createLabel(std::string("censored depth"));
  const ALE::Obj<PointSet>          modifiedPoints = new PointSet();
  _computeCensoredDepth(mesh, label, mesh->getSieve(), mesh->getSieve()->roots(), firstCohesiveCell, modifiedPoints);
  if (debug) mesh->view("Mesh with Cohesive Elements");

  // Fix coordinates
  const ALE::Obj<real_section_type>& coordinates = 
    mesh->getRealSection("coordinates");
  const ALE::Obj<Mesh::label_sequence>& fVertices2 = (*fault)->depthStratum(0);

  for(Mesh::label_sequence::iterator v_iter = fVertices2->begin();
      v_iter != fVertices2->end();
      ++v_iter) {
    coordinates->addPoint(vertexRenumber[*v_iter],
			  coordinates->getFiberDimension(*v_iter));
    if (constraintCell) {
      coordinates->addPoint(vertexRenumber[*v_iter]+1,
			  coordinates->getFiberDimension(*v_iter));
    }
  } // for
  mesh->reallocate(coordinates);
  for(Mesh::label_sequence::iterator v_iter = fVertices2->begin();
      v_iter != fVertices2->end();
      ++v_iter) {
    coordinates->updatePoint(vertexRenumber[*v_iter], 
			     coordinates->restrictPoint(*v_iter));
    if (constraintCell) {
      coordinates->updatePoint(vertexRenumber[*v_iter]+1,
			     coordinates->restrictPoint(*v_iter));
    }
  }
  if (debug) coordinates->view("Coordinates with shadow vertices");
  PetscLogEventEnd(SplitMesh_Event,0,0,0,0);
} // createCohesiveCells

template<class InputPoints>
bool
pylith::faults::CohesiveTopology::_compatibleOrientation(const ALE::Obj<Mesh>& mesh,
                                                         const Mesh::point_type& p,
                                                         const Mesh::point_type& q,
                                                         const int numFaultCorners,
                                                         const int faultFaceSize,
                                                         const int faultDepth,
                                                         const ALE::Obj<InputPoints>& points,
                                                         int indices[],
                                                         PointArray *origVertices,
                                                         PointArray *faceVertices,
                                                         PointArray *neighborVertices)
{
  typedef ALE::Selection<Mesh> selection;
  const int debug = mesh->debug();
  bool compatible;

  bool eOrient = selection::getOrientedFace(mesh, p, points, numFaultCorners, indices, origVertices, faceVertices);
  bool nOrient = selection::getOrientedFace(mesh, q, points, numFaultCorners, indices, origVertices, neighborVertices);

  if (faultFaceSize > 1) {
    if (debug) {
      for(PointArray::iterator v_iter = faceVertices->begin(); v_iter != faceVertices->end(); ++v_iter) {
        std::cout << "  face vertex " << *v_iter << std::endl;
      }
      for(PointArray::iterator v_iter = neighborVertices->begin(); v_iter != neighborVertices->end(); ++v_iter) {
        std::cout << "  neighbor vertex " << *v_iter << std::endl;
      }
    }
    compatible = !(*faceVertices->begin() == *neighborVertices->begin());
  } else {
    compatible = !(nOrient == eOrient);
  }
  return compatible;
}

void
pylith::faults::CohesiveTopology::_replaceCell(const ALE::Obj<sieve_type>& sieve,
                                               const Mesh::point_type cell,
                                               std::map<int,int> *vertexRenumber,
                                               const int debug)
{
  bool       replace = false;
  PointArray newVertices;

  const ALE::Obj<sieve_type::traits::coneSequence>& cCone = sieve->cone(cell);

  for(sieve_type::traits::coneSequence::iterator v_iter = cCone->begin();
      v_iter != cCone->end(); ++v_iter) {
    if (vertexRenumber->find(*v_iter) != vertexRenumber->end()) {
      if (debug) std::cout << "    vertex " << (*vertexRenumber)[*v_iter] << std::endl;
      newVertices.insert(newVertices.end(), (*vertexRenumber)[*v_iter]);
      replace = true;
    } else {
      if (debug) std::cout << "    vertex " << *v_iter << std::endl;
      newVertices.insert(newVertices.end(), *v_iter);
    } // if/else
  } // for
  if (replace) {
    if (debug) std::cout << "  Replacing cell " << cell << std::endl;
    sieve->clearCone(cell);
    int color = 0;
    for(PointArray::const_iterator v_iter = newVertices.begin(); v_iter != newVertices.end(); ++v_iter) {
      sieve->addArrow(*v_iter, cell, color++);
    } // for
  }
}

template<class InputPoints>
void
pylith::faults::CohesiveTopology::_computeCensoredDepth(const ALE::Obj<Mesh>& mesh,
                                                        const ALE::Obj<Mesh::label_type>& depth,
                                                        const ALE::Obj<Mesh::sieve_type>& sieve,
                                                        const ALE::Obj<InputPoints>& points,
                                                        const Mesh::point_type& firstCohesiveCell,
                                                        const ALE::Obj<std::set<Mesh::point_type> >& modifiedPoints)
{
  modifiedPoints->clear();

  for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
    if (*p_iter >= firstCohesiveCell) continue;
    // Compute the max depth of the points in the cone of p, and add 1
    int d0 = mesh->getValue(depth, *p_iter, -1);
    int d1 = mesh->getMaxValue(depth, sieve->cone(*p_iter), -1) + 1;

    if(d1 != d0) {
      mesh->setValue(depth, *p_iter, d1);
      modifiedPoints->insert(*p_iter);
    }
  }
  // FIX: We would like to avoid the copy here with support()
  if(modifiedPoints->size() > 0) {
    _computeCensoredDepth(mesh, depth, sieve, sieve->support(modifiedPoints), firstCohesiveCell, modifiedPoints);
  }
};


// End of file
