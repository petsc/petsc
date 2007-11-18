//routines for doing various pieces of mesh surgery

//Inputs: The mesh, a supportSet containing all the relevant cell names for the ear,
//        and the former maxindex;
//Returns: the new maxindex.


namespace ALE {

/*

b
|
|- 
| \ angle
|  |
a------c

*/

  double corner_angle (int dim, double * a, double * b, double * c) {

  // o is our origin point, p is one endpoint, q is the other; find their angle given the coordinates
    double tmp_b = 0, tmp_c = 0, mag_b = 0, mag_c = 0, dot_product = 0, angle;
    for (int i = 0; i < dim; i++) {
      tmp_b = (a[i] - b[i]);
      tmp_c = (a[i] - c[i]);
      mag_b += tmp_b*tmp_b;
      mag_c += tmp_c*tmp_c;
      dot_product += tmp_b*tmp_c;
    }
    if (mag_b == 0 || mag_c == 0) return 0.;
    angle = acos(dot_product/sqrt(mag_b*mag_c));
    //    PetscPrintf(PETSC_COMM_WORLD, "angle: %f\n", angle);
    return angle;
  } 

/*
     ^
    /|\
   / | \
  /  |  \  volume
 /   |   \
/____|____\


*/

  double tetrahedron_volume (int dim, double * a, double * b, double * c, double * d) {
    double x[dim];
    double y[dim];
    double z[dim];
    for (int i = 0; i < dim; i++) {
      x[i] = b[i] - a[i];
      y[i] = c[i] - a[i];
      z[i] = d[i] - a[i];
    }
    double volume = x[0]*y[1]*z[2] + x[1]*y[2]*z[0] + x[2]*y[1]*z[2] 
                  - x[0]*y[2]*z[1] - x[1]*y[0]*z[2] - x[2]*y[0]*z[1];
    volume = volume / 2;
    return fabs(volume);
  }



//noninterpolated for now

/*
     a              a
    /|\            / \
   / | \          /   \
  /  |  \        /  1  \
 /   |   \      /       \
c  1 | 2  d -> c---------d
 \   |   /      \       /
  \  |  /        \  2  /
   \ | /          \   /
    \|/            \ /
     b              b

a = vertices[0];
b = vertices[1];
c = vertices[2];
d = vertices[3];

1 = cells[0];
2 = cells[1];

*/


  void Surgery_2D_22Flip_Setup(Obj<Mesh> m, Mesh::point_type a, Mesh::point_type b, Mesh::point_type * cells, Mesh::point_type * vertices) {
    //given a and b, set up the rest of the data structure needed for the 2-2 ear flip
    vertices[0] = a;
    vertices[1] = b;
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    line->clear();
    line->insert(a);
    line->insert(b);
    Obj<Mesh::sieve_type::supportSet> doublet = m->getSieve()->nJoin1(line); //check to see if this works
    //PetscPrintf(m->comm(), "%d in line, %d in doublet %d, %d\n", line->size(), doublet->size(), a, b);
    if (doublet->size() != 2) throw Exception("bad flip setup 2-2");
    Mesh::sieve_type::supportSet::iterator d_iter = doublet->begin();
    cells[0] = *d_iter;
    d_iter++;
    cells[1] = *d_iter;
    Obj<Mesh::sieve_type::coneSequence> corners = m->getSieve()->cone(cells[0]);
    Mesh::sieve_type::coneSequence::iterator c_iter     = corners->begin();
    Mesh::sieve_type::coneSequence::iterator c_iter_end = corners->end();
    while (c_iter != c_iter_end) {
      if (*c_iter != a && *c_iter != b) vertices[2] = *c_iter;
      c_iter++;
    }
    corners = m->getSieve()->cone(cells[1]);
    c_iter      = corners->begin();
    c_iter_end  = corners->end();
    while (c_iter != c_iter_end) {
      if (*c_iter != a && *c_iter != b) vertices[3] = *c_iter;
      c_iter++;
    }
    //PetscPrintf(m->comm(), "2-2 Ear: %d %d %d %d, %d %d\n", vertices[0], vertices[1], vertices[2], vertices[3], cells[0], cells[1]);
    return;
  }

  PetscTruth Surgery_2D_22Flip_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices) {
    double pi = 3.14159;
    //VALIDITY CONDITION FOR THIS FLIP: cad and cbd must be (much) less than 180. 
    //we could probably have some angle heuristic for local delaunay approximation, but whatever.
    //must compute it in terms of acb + bad etc.
    int dim = 2;
    double a_coords[2], b_coords[2], c_coords[2], d_coords[2];
    const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    double current_angle = corner_angle(dim, a_coords, c_coords, b_coords) + corner_angle(dim, a_coords, d_coords, b_coords);
    //PetscPrintf(m->comm(), "%f angle\n", current_angle);
    if (current_angle > pi) return PETSC_FALSE;
    current_angle = corner_angle(dim, b_coords, c_coords, a_coords) + corner_angle(dim, b_coords, d_coords, a_coords);
    //PetscPrintf(m->comm(), "%f angle\n", current_angle);
    if (current_angle > pi) return PETSC_FALSE;
    return PETSC_TRUE;
  }

  int Surgery_2D_22Flip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    if (m->getDimension() != 2) {
      throw Exception("Wrong Flip Performed - Wrong ear or dimension");
    }
    Obj<Mesh::sieve_type> s = m->getSieve();
    //get the triangles
    s->removeBasePoint(cells[0]);
    s->removeCapPoint(cells[0]);
    s->removeBasePoint(cells[1]);
    s->removeCapPoint(cells[1]);  

    s->addArrow(vertices[0], cells[0]);
    s->addArrow(vertices[2], cells[0]);
    s->addArrow(vertices[3], cells[0]);
    
    s->addArrow(vertices[1], cells[1]);
    s->addArrow(vertices[2], cells[1]);
    s->addArrow(vertices[3], cells[1]);
    if (m->debug()) {
      //TEST the validity of the flip
      if (s->cone(cells[0])->size() != 3 || s->cone(cells[1])->size() != 3) throw Exception("Flip Failed -> 2->2");
  
      Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
      line->insert(vertices[0]);
      line->insert(vertices[2]);
      Obj<Mesh::sieve_type::supportSet> join = s->nJoin1(line);
      if (join->size() > 2) {
        throw Exception("ERROR: Bad edge: a -- c in 2-2 flip");
      }
      line->clear();
      line->insert(vertices[0]);
      line->insert(vertices[3]);
      join = s->nJoin1(line);
      if (join->size() > 2) {
        throw Exception("Bad edge: a -- d in 2-2 flip");
      }
      line->clear();
      line->insert(vertices[1]);
      line->insert(vertices[2]);
      join = s->nJoin1(line);
      if (join->size() > 2) {
        throw Exception("Bad edge: b -- c in 2-2 flip");
      }
      line->clear();
      line->insert(vertices[1]);
      line->insert(vertices[3]);
      join = s->nJoin1(line);
      if (join->size() > 2) {
        throw Exception("Bad edge: b -- d in 2-2 flip");
      }
      line->clear();
      line->insert(vertices[2]);
      line->insert(vertices[3]);
      join = s->nJoin1(line);
      if (join->size() != 2) {
        PetscPrintf(m->comm(), "join size(%d, %d): %d\n", vertices[2], vertices[3], join->size());
        throw Exception("Bad edge: c -- d in 2-2 flip");
      }
    }
    return maxIndex;
  }

/*
a---------------b     a---------------b
 \\           //       \             /
  \ \   1   / /         \           /
   \  \   /  /           \    1    /
    \   d   /      ->     \       /
     \2 | 3/               \     /
      \ | /                 \   /
       \|/                   \ /
        c                     c
*/


  void Surgery_2D_31Flip_Setup(Obj<Mesh> m, Mesh::point_type d, Mesh::point_type * cells, Mesh::point_type * vertices) {
    vertices[3] = d;
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::sieve_type::supportSequence> cell_points = s->support(d);
    Mesh::sieve_type::supportSequence::iterator cp_iter = cell_points->begin();
    Mesh::sieve_type::supportSequence::iterator cp_iter_end = cell_points->end();
    Obj<Mesh::sieve_type::coneSet> vertex_points = s->cone(cell_points);
    if (vertex_points->size() != 4) throw Exception("Bad vertex set in 3-1 flip");
    cells[0] = *cp_iter;
    cp_iter++;
    cells[1] = *cp_iter;
    cp_iter++;
    cells[2] = *cp_iter;
    Mesh::sieve_type::coneSet::iterator v_iter = vertex_points->begin();
    Mesh::sieve_type::coneSet::iterator v_iter_end = vertex_points->end();
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    while (v_iter != v_iter_end) {
      if (*v_iter != vertices[3]) {
        line->clear(); 
        line->insert(*v_iter);
        line->insert(d);
        Obj<Mesh::sieve_type::supportSet> join = s->nJoin1(line);
        bool j_1 = (join->find(cells[0]) != join->end());
	bool j_2 = (join->find(cells[1]) != join->end());
        bool j_3 = (join->find(cells[2]) != join->end());
        if        (j_1 && j_2) { vertices[0] = *v_iter;
	} else if (j_1 && j_3) { vertices[1] = *v_iter;
        } else if (j_2 && j_3) { vertices[2] = *v_iter;
        }
      }
      v_iter++;
    }

    //check the validity of what we just did
    //PetscPrintf(m->comm(), "3-1 Flip: %d %d %d %d, %d %d %d\n", vertices[0], vertices[1], vertices[2], vertices[3], cells[0], cells[1], cells[2]);
  }

  PetscTruth Surgery_2D_31Flip_Possible(Obj<Mesh> m, Mesh::point_type * vertices, Mesh::point_type * cells) {
    //VALIDITY CONDITION FOR THIS FLIP: acd, dcb, and acb MUST be (much) less than 180.
    // BUT this should be essentially guaranteed and we don't need to test it
    return PETSC_TRUE;
  }

  int Surgery_2D_31Flip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    s->removeBasePoint(cells[0]);
    s->removeBasePoint(cells[1]);
    s->removeBasePoint(cells[2]);
    //s->removeCapPoint(vertices[3]);
    s->addArrow(vertices[0], cells[0]);
    s->addArrow(vertices[1], cells[0]);
    s->addArrow(vertices[2], cells[0]);
    return maxIndex;
  }



/*
 = c          = a
a-----------b      a-----------b
|\         /|      |\          |
| \   1   / |      | \         |
|  \   =2/  |      |  \   1    |
|   \   /   |      |   \       |
|    \ /    |      |    \      |
| 2   e  3  |  ->  |     \     |
|  =4/ \  =1|      |      \    |
|   /   \   |      |  2    \   |
|  /     \  |      |        \  |
| /   4   \ |      |         \ |
|/     =3  \|      |          \|
c-----------d      c-----------d
 = d         = b
*/


  void Surgery_2D_42Flip_Setup(Obj<Mesh> m, Mesh::point_type e, Mesh::point_type * cells, Mesh::point_type * vertices) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    Obj<Mesh::sieve_type::supportSequence> cell_points = s->support(e);
    if (cell_points->size() != 4) {
      PetscPrintf(m->comm(), "%d\n", cell_points->size());
      throw Exception("Bad Ear 4-2");
    }
    Obj<Mesh::sieve_type::coneSet> vertex_points = s->cone(cell_points);
    Mesh::sieve_type::supportSequence::iterator cp_iter = cell_points->begin();
    vertices[4] = e;
    cells[0] = *cp_iter;
    Obj<Mesh::sieve_type::coneSequence> c_corners = s->cone(cells[0]);
    if (c_corners->size() != 3) throw Exception("Bad Mesh in 4-2 flip");
    Mesh::sieve_type::coneSequence::iterator cc_iter = c_corners->begin();
    Mesh::sieve_type::coneSequence::iterator cc_iter_end = c_corners->end();
    bool first = true;
    //find a and b
    while (cc_iter != cc_iter_end) {
      if ((*cc_iter != e) && first) {
        vertices[0] = *cc_iter;
        first = false;
      } else if (*cc_iter != e) {
        vertices[1] = *cc_iter;
      }
      cc_iter++;
    }
    line->clear();
    line->insert(vertices[0]);
    line->insert(vertices[4]);
    //PetscPrintf(m->comm(), "current line: %d, %d\n", vertices[0], vertices[4]);
    Obj<Mesh::sieve_type::supportSet> join = s->nJoin1(line);
    if (join->size() != 2) throw Exception("bad join in 4-2 flip");
    Mesh::sieve_type::supportSet::iterator j_iter = join->begin();
    //find 2
    while (j_iter != join->end()) {
      if (*j_iter != cells[0]) cells[1] = *j_iter;
      j_iter++;
    }
    c_corners = s->cone(cells[1]);
    cc_iter = c_corners->begin();
    cc_iter_end = c_corners->end();
    //find c
    while (cc_iter != cc_iter_end) {
      if (*cc_iter != vertices[0] && *cc_iter != e)vertices[2] = *cc_iter;
      cc_iter++;
    }
    line->clear();
    line->insert(vertices[1]);
    line->insert(vertices[4]);
    join = s->nJoin1(line);
    if (join->size() != 2) throw Exception("bad join in 4-2 flip");
    j_iter = join->begin();
    //find 3;
    while (j_iter != join->end()) {
      if (*j_iter != cells[0]) cells[2] = *j_iter;
      j_iter++;
    }
    c_corners = s->cone(cells[2]);
    cc_iter = c_corners->begin();
    cc_iter_end = c_corners->end();
    //find d
    while (cc_iter != cc_iter_end) {
      if (*cc_iter != vertices[1] && *cc_iter != vertices[4])vertices[3] = *cc_iter;
      cc_iter++;
    }
    //find 4
    line->clear();
    line->insert(vertices[2]);
    line->insert(vertices[4]);
    join = s->nJoin1(line);
    j_iter = join->begin();
    while (j_iter != join->end()) {
      if (*j_iter != cells[1]) cells[3] = *j_iter;
      j_iter++;
    }
    //for some reason this screws up... cp_iter might be incoherent at this point.
    //while (cp_iter != cell_points->end()) {
    //  if (*cp_iter != cells[0] && *cp_iter != cells[1] && *cp_iter != cells[2]) cells[3] = *cp_iter;
    //  cp_iter++;
    //}
    //PetscPrintf(m->comm(), "Completed Ear: %d %d %d %d %d, %d %d %d %d\n", vertices[0], vertices[1], vertices[2], vertices[3], vertices[4],cells[0], cells[1], cells[2], cells[3]);
    return;
  }


  PetscTruth Surgery_2D_42Flip_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices) {
    //criterion for this flip:  
    // 1. bac and bdc must be < 180 degrees
    //    if this isn't true, we can reorient the flip here and make it true. (a four-sided figure may have one concave vertex)
    //    a->c b->a d->b c->d, 1->3, 3->4, 4->2, 2->1
    double pi = 3.14158;
    int dim = 2;
    double a_coords[2], b_coords[2], c_coords[2], d_coords[2], e_coords[2];
    const Obj<Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    PetscMemcpy(e_coords, coordinates->restrictPoint(vertices[4]), dim*sizeof(double));
    if (corner_angle(dim, a_coords, b_coords, e_coords) + corner_angle(dim, a_coords, c_coords, e_coords) > pi ||
        corner_angle(dim, d_coords, b_coords, e_coords) + corner_angle(dim, d_coords, c_coords, e_coords) > pi) {
      //rotate the ear
      //PetscPrintf(m->comm(), "rotating the ear.\n");
      Mesh::point_type temp_point = cells[0];
      cells[0] = cells[1];
      cells[1] = cells[3];
      cells[3] = cells[2];
      cells[2] = temp_point;
      temp_point = vertices[0];
      vertices[0] = vertices[2];
      vertices[2] = vertices[3];
      vertices[3] = vertices[1];
      vertices[1] = temp_point;
      //check the rotation
      Obj<Mesh::sieve_type> s = m->getSieve();
      Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
      Obj<Mesh::sieve_type::supportSet> join;
      line->clear();
      line->insert(vertices[4]);
      line->insert(vertices[0]);
      join = s->nJoin1(line);
      if (join->find(cells[0]) == join->end() || join->find(cells[1]) == join->end() || join->size() != 2) {
        PetscPrintf(m->comm(), "join(%d, %d) size: %d\n", vertices[0], vertices[4], join->size());
        throw Exception("rotation. BAD");
      }
      line->clear();
      line->insert(vertices[4]);
      line->insert(vertices[1]);
      join = s->nJoin1(line);
      if (join->find(cells[0]) == join->end() || join->find(cells[2]) == join->end() || join->size() != 2) {
        PetscPrintf(m->comm(), "join(%d, %d) size: %d\n", vertices[1], vertices[4], join->size());
        throw Exception("rotation. BAD");
      }
      line->clear();
      line->insert(vertices[4]);
      line->insert(vertices[2]);
      join = s->nJoin1(line);
      if (join->find(cells[3]) == join->end() && join->find(cells[1]) == join->end() || join->size() != 2) {
        PetscPrintf(m->comm(), "join(%d, %d) size: %d\n", vertices[2], vertices[4], join->size());
        throw Exception("rotation. BAD");
      }
      line->clear();
      line->insert(vertices[4]);
      line->insert(vertices[3]);
      join = s->nJoin1(line);
      if (join->find(cells[2]) == join->end() && join->find(cells[3]) == join->end() || join->size() != 2) {
        PetscPrintf(m->comm(), "join(%d, %d) size: %d\n", vertices[0], vertices[4], join->size());
        throw Exception("rotation. BAD");
      }
    }
    return PETSC_TRUE;
  }



  Mesh::point_type Surgery_2D_42Flip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, ALE::Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    //s->removeBasePoint(vertices[4]);
    //s->removeCapPoint(vertices[4]);
    s->removeBasePoint(cells[2]);
    s->removeCapPoint(cells[2]);
    s->removeBasePoint(cells[3]);
    s->removeCapPoint(cells[3]);
    s->removeBasePoint(cells[0]);
    s->removeCapPoint(cells[0]);
    s->removeBasePoint(cells[1]);
    s->removeCapPoint(cells[1]);
    if (s->hasPoint(cells[2]) || s->hasPoint(cells[3])) throw Exception("problem in 4-2 flip: not deleting points!");

    s->addArrow(vertices[0], cells[0]);
    s->addArrow(vertices[3], cells[0]);
    s->addArrow(vertices[1], cells[0]);

    s->addArrow(vertices[0], cells[1]);
    s->addArrow(vertices[3], cells[1]);
    s->addArrow(vertices[2], cells[1]);

    //check the coherence of the modified mesh
    if (m->debug()) {
      //PetscPrintf(m->comm(), "cone sizes: %d, %d", s->cone(cells[0])->size(), s->cone(cells[1])->size());
      if (s->cone(cells[0])->size() != 3 || s->cone(cells[1])->size() != 3) throw Exception("problem in 4-2 flip: incoherent resulting mesh");

      //check edges of the resultant 2-ear for coherency
      Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
      line->insert(vertices[0]);
      line->insert(vertices[1]);
      Obj<Mesh::sieve_type::supportSet> join = s->nJoin1(line);
      if (join->size() > 2) {
        PetscPrintf(m->comm(), "Join Size: %d, a: %d, b: %d\n", join->size(), vertices[0], vertices[1]);
        Mesh::sieve_type::supportSet::iterator jerr_iter = join->begin();
        while (jerr_iter != join->end()) {
          PetscPrintf(m->comm(), "Join Component: %d\n", *jerr_iter);
          jerr_iter++;
        }
        throw Exception("ERROR: Bad edge: a -- b in 4-2 flip");
      }
      line->clear();
      line->insert(vertices[0]);
      line->insert(vertices[2]);
      join = s->nJoin1(line);
      if (join->size() > 2) {
        PetscPrintf(m->comm(), "Join Size: %d, a: %d, c: %d\n", join->size(), vertices[0], vertices[2]);
        Mesh::sieve_type::supportSet::iterator jerr_iter = join->begin();
        while (jerr_iter != join->end()) {
          PetscPrintf(m->comm(), "Join Component: %d\n", *jerr_iter);
          jerr_iter++;
        }
        throw Exception("Bad edge: a -- c in 4-2 flip");
      }
      line->clear();
      line->insert(vertices[1]);
      line->insert(vertices[3]);
      join = s->nJoin1(line);
      if (join->size() > 2) {
        PetscPrintf(m->comm(), "Join Size: %d, b: %d, c: %d\n", join->size(), vertices[1], vertices[3]);
        Mesh::sieve_type::supportSet::iterator jerr_iter = join->begin();
        while (jerr_iter != join->end()) {
          PetscPrintf(m->comm(), "Join Component: %d\n", *jerr_iter);
          jerr_iter++;
        }
        throw Exception("Bad edge: b -- d in 4-2 flip");
      }
      line->clear();
      line->insert(vertices[2]);
      line->insert(vertices[3]);
      join = s->nJoin1(line);
      if (join->size() > 2) {
        PetscPrintf(m->comm(), "Join Size: %d, c: %d, d: %d\n", join->size(), vertices[2], vertices[3]);
        Mesh::sieve_type::supportSet::iterator jerr_iter = join->begin();
        while (jerr_iter != join->end()) {
          PetscPrintf(m->comm(), "Join Component: %d\n", *jerr_iter);
          jerr_iter++;
        }
        throw Exception("Bad edge: c -- d in 4-2 flip");
      }
      line->clear();
      line->insert(vertices[0]);
      line->insert(vertices[3]);
      join = s->nJoin1(line);
      if (join->size() != 2) {
        PetscPrintf(m->comm(), "Join Size: %d, a: %d, d: %d\n", join->size(), vertices[0], vertices[3]);
        Mesh::sieve_type::supportSet::iterator jerr_iter = join->begin();
        while (jerr_iter != join->end()) {
          PetscPrintf(m->comm(), "Join Component: %d\n", *jerr_iter);
          jerr_iter++;
        }
        throw Exception("Bad edge: a -- d in 4-2 flip");
      }
    }
    return maxIndex;

  } 


/*

exterior           exterior

a------d------c   a-------------c
 \     |     /     \           /
  \ 1  |  2 /       \    1    /
   \   |   /    ->   \       /
    \  |  /           \     /
     \ | /             \   /
      \|/               \ /
       b                 b
*/

  void Surgery_2D_21BoundFlip_Setup(Obj<Mesh> m, Mesh::point_type d, Mesh::point_type * cells, Mesh::point_type * vertices) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    vertices[3] = d;
    Obj<Mesh::sieve_type::supportSequence> cell_points = s->support(d);
    if (cell_points->size() != 2) throw Exception("Wrong flip (2->1 boundary)");
    Mesh::sieve_type::supportSequence::iterator cp_iter=  cell_points->begin();
    cells[0] = *cp_iter;
    cp_iter++;
    cells[1] = *cp_iter;
    Obj<Mesh::sieve_type::coneSet> cell_vertices = s->cone(cell_points);
    Mesh::sieve_type::coneSet::iterator cv_iter = cell_vertices->begin();
    Mesh::sieve_type::coneSet::iterator cv_iter_end = cell_vertices->end();
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    Obj<Mesh::sieve_type::supportSet> join;
    while (cv_iter != cv_iter_end) {
      if (*cv_iter != vertices[3]) {
        line->clear();
        line->insert(vertices[3]);
        line->insert(*cv_iter);
        join = s->nJoin1(line);
        if (join->size() == 2) {
          vertices[1] = *cv_iter;
        } else if (join->find(cells[0]) != join->end()) {
          vertices[0] = *cv_iter;
        } else {
          vertices[2] = *cv_iter;
        }
      }
      cv_iter++;
    }
    //PetscPrintf(m->comm(), "Completed 2-1 ear: %d %d %d %d, %d %d\n", vertices[0], vertices[1], vertices[2], vertices[3], cells[0], cells[1]);
    return;
  }

  PetscTruth Surgery_2D_21BoundFlip_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices) {
    //criterion for this flip: abc must be convex, otherwise the interior point would be exposed
    double pi = 3.14158;
    const Obj<Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    int dim = m->getDimension();
    double coords_a[dim], coords_b[dim], coords_c[dim], coords_d[dim];
    PetscMemcpy(coords_a, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(coords_b, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(coords_c, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(coords_d, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    if (corner_angle(dim, coords_b, coords_a, coords_d) + 
        corner_angle(dim, coords_b, coords_c, coords_d) > pi) return PETSC_FALSE;
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_2D_21BoundFlip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    s->removeBasePoint(cells[1]);
    s->removeBasePoint(cells[0]);
    //s->removeCapPoint(vertices[3]);
    s->addArrow(vertices[2], cells[0]);
    s->addArrow(vertices[1], cells[0]);
    s->addArrow(vertices[0], cells[0]);
    return maxIndex;
  }


/*

interior           interior

b-------------c   b-------------c
 \           /    
  \         /      
   \       /    -> 
    \     /         
     \   /         
      \ /         
       a         
*/


  void Surgery_2D_10BoundFlip_Setup(Obj<Mesh> m, Mesh::point_type a, Mesh::point_type * cells, Mesh::point_type * vertices) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    vertices[0] = a;
    Obj<Mesh::sieve_type::supportSequence> cell_points = s->support(a);
    if (cell_points->size() != 1) throw Exception("Wrong flip performed (1->0 Boundary)");
    cells[0] = *cell_points->begin();
  }

  PetscTruth Surgery_2D_10BoundFlip_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices) {
    //no degenerate cases;
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_2D_10BoundFlip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    //s->removeCapPoint(vertices[0]);
    s->removeBasePoint(cells[0]);
    return maxIndex;
  }

//Two interior tets to three

/*
     d              d
    / \            /|\
   / 1 \          / | \
  /  |  \        /  |  \
 /   |   \      /   2   \
b----a----c -> b1---a---3c
 \   |   /      \   |   /
  \  |  /        \  |  /
   \ 2 /          \ | /
    \|/            \|/
     e              e
*/

  void Surgery_3D_23Flip_Setup(Obj<Mesh> m, Mesh::point_type a, Mesh::point_type b, Mesh::point_type c, Mesh::point_type * vertices, Mesh::point_type * cells) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    Obj<Mesh::sieve_type::supportSet> join;
    line->clear();
    vertices[0] = a;
    vertices[1] = b;
    vertices[2] = c;
    line->insert(a);
    line->insert(b);
    line->insert(c);
    join = s->nJoin1(line);
    if (join->size() != 2) throw Exception("Incoherent mesh in 23 flip");
    Mesh::sieve_type::supportSet::iterator j_iter = join->begin();
    cells[0] = *j_iter;
    j_iter++;
    cells[1] = *j_iter;
    Obj<Mesh::sieve_type::coneSequence> corners = s->cone(cells[0]);
    Mesh::sieve_type::coneSequence::iterator c_iter = corners->begin();
    Mesh::sieve_type::coneSequence::iterator c_iter_end = corners->end();
    while (c_iter != c_iter_end) {
      if (*c_iter != a && *c_iter != b && *c_iter != c) {
        vertices[3] = *c_iter;
      }
      c_iter++;
    }
    corners = s->cone(cells[1]);
    c_iter = corners->begin();
    c_iter_end = corners->end();
    while (c_iter != c_iter_end) {
      if (*c_iter != a && *c_iter != b && *c_iter != c) {
        vertices[4] = *c_iter;
      }
      c_iter++;
    }
    return;

  }

  PetscTruth Surgery_3D_23Flip_Possible (Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices){
    //The 2-3 flip is only possible if the line between a and d lies inside the volume; this occurs when, for example,
    //volume(abde + bcde + cade) <== volume(abcd + abce)
    int dim = m->getDimension();
    const Obj<Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    double a_coords[dim], b_coords[dim], c_coords[dim], d_coords[dim], e_coords[dim];
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    PetscMemcpy(e_coords, coordinates->restrictPoint(vertices[4]), dim*sizeof(double));
    double volume_1 = tetrahedron_volume(dim, a_coords, b_coords, d_coords, e_coords);
    double volume_2 =  tetrahedron_volume(dim, b_coords, c_coords, d_coords, e_coords);
    double volume_3 =  tetrahedron_volume(dim, c_coords, a_coords, d_coords, e_coords);
    double now_volume = tetrahedron_volume(dim, a_coords, b_coords, c_coords, d_coords)
      + tetrahedron_volume(dim, a_coords, b_coords, c_coords, d_coords);
    if   ((volume_1 + volume_2 > now_volume) 
       || (volume_2 + volume_3 > now_volume) 
       || (volume_3 + volume_1 > now_volume)) return PETSC_FALSE;
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_3D_23Flip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Mesh::point_type newCell = maxIndex+1;
    Obj<Mesh::sieve_type> s = m->getSieve();
    s->removeBasePoint(cells[0]);
    s->removeBasePoint(cells[1]);

    s->addArrow(vertices[0], cells[0]);
    s->addArrow(vertices[1], cells[0]);
    s->addArrow(vertices[3], cells[0]);
    s->addArrow(vertices[4], cells[0]);

    s->addArrow(vertices[1], cells[1]);
    s->addArrow(vertices[2], cells[1]);
    s->addArrow(vertices[3], cells[1]);
    s->addArrow(vertices[4], cells[1]);

    s->addArrow(vertices[2], newCell);
    s->addArrow(vertices[0], newCell);
    s->addArrow(vertices[3], newCell);
    s->addArrow(vertices[4], newCell);

    return newCell;
  }


//Three interior tets to two

/*
     a              a
    /|\            /|\
   / | \          / | \
  /  |  \        /  1  \
 /   d   \      /   |   \
c1---2---3e -> c----d----e
 \   |   /      \   |   /
  \  |  /        \  2  /
   \ | /          \ | /
    \|/            \|/
     b              b
*/


  void Surgery_3D_32Flip_Setup (Obj<Mesh> m, Mesh::point_type a, Mesh::point_type b, Mesh::point_type * cells, Mesh::point_type * vertices) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    vertices[0] = a;
    vertices[1] = b;
    line->clear();
    line->insert(a);
    line->insert(b);
    Obj<Mesh::sieve_type::supportSet> join = s->nJoin1(line);
    if (join->size() != 3) throw Exception("Wrong flip: 3-2");
    Mesh::sieve_type::supportSet::iterator j_iter = join->begin();
    cells[0] = *j_iter;
    j_iter++;
    cells[1] = *j_iter;
    j_iter++;
    cells[2] = *j_iter;
    Obj<Mesh::sieve_type::coneSequence> corners = s->cone(cells[0]);
    Mesh::sieve_type::coneSequence::iterator c_iter = corners->begin();
    Mesh::sieve_type::coneSequence::iterator c_iter_end = corners->end();
    bool first = true;
    Mesh::point_type p;
    while (c_iter != c_iter_end) {
      p = *c_iter;
      if (p != a && p != b) {
        if (first) {
          vertices[2] = p;
        } else {
          vertices[3] = p;
        }
      }
      c_iter++;
    }
    corners = s->cone(cells[1]);
    c_iter = corners->begin();
    c_iter_end = corners->end();
    while (c_iter != c_iter_end) {
      p = *c_iter;
      if (p == vertices[3]) { //we've got 2 instead of 1.
	Mesh::point_type swap = cells[1];
        cells[1] = cells[2];
        cells[2] = swap;
      }
      if (p != a && p != b && p != vertices[2] && p != vertices[3]) {
        vertices[4] = p;
      }
      c_iter++;
    }
  }
  PetscTruth Surgery_3D_32Flip_Possible (Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices){
    //The 3-2 flip is only possible if each new subvolume will have volume less than the previous three combined (convex)
    int dim = m->getDimension();
    const Obj<Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    double a_coords[dim], b_coords[dim], c_coords[dim], d_coords[dim], e_coords[dim];
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    PetscMemcpy(e_coords, coordinates->restrictPoint(vertices[4]), dim*sizeof(double));
    double cur_volume = tetrahedron_volume(dim, a_coords, b_coords, c_coords, d_coords) + 
      tetrahedron_volume(dim, a_coords, b_coords, d_coords, e_coords) +
      tetrahedron_volume(dim, a_coords, b_coords, e_coords, c_coords);
    double volume_1 = tetrahedron_volume(dim, a_coords, c_coords, d_coords, e_coords);
    double volume_2 = tetrahedron_volume(dim, b_coords, c_coords, d_coords, e_coords);
    if (volume_1 > cur_volume) return PETSC_FALSE;
    if (volume_2 > cur_volume) return PETSC_FALSE;
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_3D_32Flip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, ALE::Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    s->removeBasePoint(cells[0]);
    s->removeBasePoint(cells[1]);
    s->removeBasePoint(cells[2]);
    
    s->addArrow(vertices[0], cells[0]);
    s->addArrow(vertices[2], cells[0]);
    s->addArrow(vertices[3], cells[0]);
    s->addArrow(vertices[4], cells[0]);

    s->addArrow(vertices[1], cells[1]);
    s->addArrow(vertices[2], cells[1]);
    s->addArrow(vertices[3], cells[1]);
    s->addArrow(vertices[4], cells[1]);
     
    return maxIndex;
  }

//Four interior tets to four, on the line


/*
     a              a
    /|\            /|\
   / | \          / | \
  /1 | 2\        /  1  \
 /   |f  \      /   |f  \
b----e----d -> b----e----d
 \   |   /      \   |   /
  \3 | 4/        \  |  /
   \ | /          \ 2 /
    \|/            \|/
     c              c
*/

int Surgery_3D_44Flip(Obj<Mesh>, Obj<Mesh::sieve_type> ear, ALE::Mesh::point_type maxIndex) {

  return 0;

}

//Four interior tets get collapsed into a single tet.

/*      d                     d
a---------------b     b---------------a
 \\     2     //       \             /
  \ \       / /         \           /
   \  \ 1 /  /           \    1    /
    \ 4 e 3 /      ->     \       /
     \  |  /               \     /
      \ | /                 \   /
       \|/                   \ /
        c                     c
*/


  void Surgery_3D_41Flip_Setup(Obj<Mesh> m, Mesh::point_type e, Mesh::point_type * cells, Mesh::point_type * vertices) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::sieve_type::supportSequence> cell_points = s->support(e);
    vertices[4] = e;
    if (cell_points->size() != 4) throw Exception("Wrong flip: 4-1");
    cells[0] = *cell_points->begin();
    Obj<Mesh::sieve_type::coneSequence> corners = s->cone(cells[0]);
    Mesh::sieve_type::coneSequence::iterator c_iter = corners->begin();
    Mesh::sieve_type::coneSequence::iterator c_iter_end = corners->end();
    //name the first three vertices
    int index = 0;
    while (c_iter != c_iter_end) {
      if (*c_iter != e) {
        vertices[index] = *c_iter;
        index++;
      }
      c_iter++;
    }
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    line->clear();
    //build the face between 
    line->insert(vertices[4]);
    line->insert(vertices[0]);
    line->insert(vertices[1]);
    Obj<Mesh::sieve_type::supportSet> join = s->nJoin1(line);
    Mesh::sieve_type::supportSet::iterator j_iter = join->begin();
    Mesh::sieve_type::supportSet::iterator j_iter_end = join->end();
    if (join->size() != 2) throw Exception("4-1 flip: bad join");
    while (j_iter != j_iter_end) {
      if (*j_iter != cells[0]) cells[1] = *j_iter;
      j_iter++;
    }
    line->clear();
    line->insert(vertices[4]);
    line->insert(vertices[1]);
    line->insert(vertices[2]);
    join = s->nJoin1(line);
    if (join->size() != 2) throw Exception("4-1 flip: bad join");
    j_iter = join->begin();
    j_iter_end = join->end();
    while (j_iter != j_iter_end) {
      if (*j_iter != cells[0]) cells[2] = *j_iter;
      j_iter++;
    }
    line->clear();
    line->insert(vertices[4]);
    line->insert(vertices[2]);
    line->insert(vertices[0]);
    join = s->nJoin1(line);
    if (join->size() != 2) throw Exception("4-1 flip: bad join");
    j_iter = join->begin();
    j_iter_end = join->end();
    while (j_iter != j_iter_end) {
      if (*j_iter != cells[0]) cells[3] = *j_iter;
      j_iter++;
    }
    //find d
    corners = s->cone(cells[1]);
    c_iter = corners->begin();
    c_iter_end = corners->end(); 
    while (c_iter != c_iter_end) {
      Mesh::point_type p = *c_iter;
      if (p != vertices[0] && p != vertices[1] && vertices[2] && p != vertices[4]) vertices[3] = p;
      c_iter++;
    }
    return;
  }

  PetscTruth Surgery_3D_41Flip_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices) {
    //criteria have to be true if we've gotten this far;
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_3D_41Flip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {

    Obj<Mesh::sieve_type> s = m->getSieve();
    s->removeBasePoint(cells[0]);
    s->removeBasePoint(cells[1]);
    s->removeBasePoint(cells[2]);
    s->removeBasePoint(cells[3]);
 
    s->addArrow(vertices[0], cells[0]);
    s->addArrow(vertices[1], cells[0]);
    s->addArrow(vertices[2], cells[0]);
    s->addArrow(vertices[3], cells[0]);
    return maxIndex;
  }


/*

a-----------b      a-----------b
|\\  \ /  //|      |\   \ /   /|
| \\  f  // |      | \   f   / |
|  \\/|\//  |      |  \ /|\ /  |
|   \\g//   |      |   \ | /   |
|   /\|/\   |      |   /\|/\   |
|   / e \   |  ->  |   / e \   |
|  / / \ \  |      |  / / \ \  |
|  //   \\  |      |  //   \\  |
| //     \\ |      | //     \\ |
|//       \\|      |//       \\|
|/         \|      |/         \|
c-----------d      c-----------d

*/

int Surgery_3D_84Flip(Obj<Mesh>, Obj<Mesh::sieve_type> ear, ALE::Mesh::point_type maxIndex) {

  return 0;

}

  //just like the 4-1 except one of the 4 is missing.

  void Surgery_3D_31BoundFlip_Setup(Obj<Mesh> m, Mesh::point_type e, Mesh::point_type * cells, Mesh::point_type * vertices) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    
  }

  Mesh::point_type Surgery_3D_31BoundFlip(Obj<Mesh>, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
  
  return maxIndex;
}

  //add this one later

int Surgery_3D_42BoundFlip(Obj<Mesh>, Obj<Mesh::sieve_type> ear, ALE::Mesh::point_type maxIndex) {

  return 0;

}

//remove a 2D point

int Surgery_2D_Remove_Vertex(Obj<Mesh> m, Mesh::point_type vertex, ALE::Mesh::point_type maxIndex) {
//get potential ears
  Obj<Mesh::sieve_type> s = m->getSieve();
  Obj<Mesh::label_type> boundary = m->getLabel("marker"); //this could be erroneously marked, unfortunately; it's used for other things.
  bool on_boundary = (m->getValue(boundary, vertex) == 1);
  //if (on_boundary) PetscPrintf(m->comm(), "Boundary node.");
  Mesh::point_type cells[4];
  Mesh::point_type vertices[5];
  Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
  Obj<Mesh::sieve_type::coneSet> neighbors = s->cone(s->support(vertex));
//go through neighbors seeing if they are legitimate two-ears; if so then flip and repeat.
  bool remove_finished = false;
  while (!remove_finished) {
    int neighbor_size = neighbors->size() - 1;
    //PetscPrintf(m->comm(), "Neighbor size = %d\n", neighbor_size);
    bool changed_neighbors = false;
    Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    if (changed_neighbors) neighbors = s->cone(s->support(vertex));
/*
    if (neighbor_size == 4 && !on_boundary) {
      //CHANGE: 4-2 is a last resort; 
        //PetscPrintf(m->comm(), "flip: 4-2\n");
        Surgery_2D_42Flip_Setup(m, vertex, cells, vertices);
        if (Surgery_2D_42Flip_Possible(m, cells, vertices) == PETSC_TRUE) {
          Surgery_2D_42Flip(m, cells, vertices, maxIndex);
      }
    } else 
*/
      if (neighbor_size == 3 && !on_boundary) {
      
	//PetscPrintf(m->comm(), "flip: 3-1\n");      
      Surgery_2D_31Flip_Setup(m, vertex, cells, vertices);
      if (Surgery_2D_31Flip_Possible(m, cells, vertices) == PETSC_TRUE) {
      Surgery_2D_31Flip(m, cells, vertices, maxIndex);
      }
    } else if (neighbor_size == 3 && on_boundary) {

	//PetscPrintf(m->comm(), "2-1\n");
      Surgery_2D_21BoundFlip_Setup(m, vertex, cells, vertices);
      Surgery_2D_21BoundFlip(m, cells, vertices, maxIndex);

    } else if (neighbor_size == 2 && on_boundary) {

	//PetscPrintf(m->comm(), "1-0\n");
      Surgery_2D_10BoundFlip_Setup(m, vertex, cells, vertices);
      Surgery_2D_10BoundFlip(m, cells, vertices, maxIndex);

    } else while (n_iter != n_iter_end && !changed_neighbors) {
      if (*n_iter != vertex) {
	//find the line, find the on-link link, and decimate it
        line->clear();
        line->insert(*n_iter);
        line->insert(vertex);
        if (s->nJoin1(line)->size() != 2) {
          //do nothing
        } else {
          //2->2 flip
          //PetscPrintf(m->comm(), "2-2 attempt\n");
          Surgery_2D_22Flip_Setup(m, vertex, *n_iter, cells, vertices); 
          if (Surgery_2D_22Flip_Possible(m, cells, vertices) == PETSC_TRUE) {
            Surgery_2D_22Flip(m, cells, vertices, maxIndex);  //in 2D there are no cases where we have to take the maxIndex into account
            changed_neighbors = true;
          }
        }
      }
      n_iter++;
    }
    if (n_iter == n_iter_end && neighbors->size() == 4) {  //last ditch 4-2 flip
      Surgery_2D_42Flip_Setup(m, vertex, cells, vertices);
      if (Surgery_2D_42Flip_Possible(m, cells, vertices) == PETSC_TRUE) {
        Surgery_2D_42Flip(m, cells, vertices, maxIndex);    
      }
    }
    if (!changed_neighbors) {
      //we're done if the local topology hasn't changed
      remove_finished = true;
    } else {
      neighbors = s->cone(s->support(vertex));
    }
  }
  return maxIndex;
}

//remove a 3D point

  Mesh::point_type Surgery_3D_Remove_Vertex(Obj<Mesh> m, Mesh::point_type vertex, ALE::Mesh::point_type maxIndex) {
    //get potential ears
    Mesh::point_type cur_maxIndex = maxIndex;
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::label_type> boundary = m->getLabel("marker"); 
    //this could be erroneously marked, unfortunately; it's used for other things.
    bool on_boundary = (m->getValue(boundary, vertex) == 1);
    //if (on_boundary) PetscPrintf(m->comm(), "Boundary node.");
    Mesh::point_type cells[4];
    Mesh::point_type vertices[7];
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    Obj<Mesh::sieve_type::supportSet> link;
    Obj<Mesh::sieve_type::coneSet> neighbors = s->cone(s->support(vertex));
  //go through neighbors seeing if they are legitimate two-ears; if so then flip and repeat.
    bool remove_finished = false;
    //Algorithm: Do 2-3 flips until the cardinality of the neighbors of n on the link is 3, then do a 3-2 flip to remove it
    //do this until the cardinality of the set of neighbors is 0
    while (!remove_finished) {
      int neighbor_size = neighbors->size() - 1;
      //PetscPrintf(m->comm(), "Neighbor size = %d\n", neighbor_size);
      bool changed_neighbors = false;
      Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
      Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
      if (changed_neighbors) neighbors = s->cone(s->support(vertex));
      if (neighbor_size == 4 && !on_boundary) {
        Surgery_3D_41Flip_Setup(m, vertex, cells, vertices);
        if (Surgery_3D_41Flip_Possible(m, vertices, cells) == PETSC_TRUE) {
          cur_maxIndex = Surgery_3D_41Flip(m, vertices, cells, cur_maxIndex);
        }
      } else {
        while (n_iter != n_iter_end && !changed_neighbors) {
          if (*n_iter != vertex) {
          line->clear();
          line->insert(*n_iter);
          line->insert(vertex);
          link = s->nJoin1(line);
          bool line_done = false;
          while (!line_done) {
            bool line_changed = false;
            if (line->size() == 3) {
            //we're done with this line; successful or not.
            Surgery_3D_32Flip_Setup(m, *n_iter, vertex, cells, vertices);
              if (Surgery_3D_32Flip_Possible(m, cells, vertices) == PETSC_TRUE) {
                cur_maxIndex = Surgery_3D_32Flip(m, cells, vertices, cur_maxIndex);
              }
            } else { //2-3 flip to remove an edge from this line
              Obj<Mesh::sieve_type::coneSet> link_neighbors = s->cone(link);
	      Mesh::sieve_type::coneSet::iterator ln_iter = link_neighbors->begin();
	      Mesh::sieve_type::coneSet::iterator ln_iter_end = link_neighbors->end();
              while (ln_iter != ln_iter_end && !line_changed) {
                if (*ln_iter != vertex && *ln_iter != *n_iter) {
                  Surgery_3D_23Flip_Setup(m, vertex, *ln_iter, *n_iter, cells, vertices);
                  if (Surgery_3D_23Flip_Possible(m, cells, vertices) == PETSC_TRUE) {
                    cur_maxIndex = Surgery_3D_23Flip(m, cells, vertices, cur_maxIndex);
                  }
		}
                ln_iter++;
              }
            }
            if (line_changed) {
              link = s->nJoin1(line);
	    } else {
              //we're done here
              line_done = true;
            }
	  } //end of line_done loop   
          n_iter++;  
	} //end of loop over neighbors 
        if (!changed_neighbors) {
        //we're done if the local topology hasn't changed
        remove_finished = true;
        } else {
          neighbors = s->cone(s->support(vertex));
        }
      }  
    } // end of overall loop
  } //end overall while
    return cur_maxIndex;
  }
}
