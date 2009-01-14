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
       c
      /|
n1   //
/\  /e|..
 \ /  /  ..
  a---b   :  <-  angle
 / \  \  ..
\/  \f|..
n2   \\
      \|
       d
  */

  double doublet_angle (int dim, double * a, double * b, double * c, double * d) {
    if (dim != 3) return 0.;
    const int nDim = 3;
    double b_r[nDim], c_r[nDim], d_r[nDim], n1[nDim], n2[nDim], n1mag = 0, n2mag = 0, n1dotn2 = 0, angle;
    for (int i = 0; i < dim; i++) {
      b_r[i] = b[i] - a[i];
      c_r[i] = c[i] - a[i];
      d_r[i] = d[i] - a[i];
    }
    n1[0] = b_r[1]*c_r[2] - b_r[2]*c_r[1];
    n1[1] = b_r[2]*c_r[0] - b_r[0]*c_r[2];
    n1[2] = b_r[0]*c_r[1] - b_r[1]*c_r[0];
    n2[0] = b_r[1]*d_r[2] - b_r[2]*d_r[1];
    n2[1] = b_r[2]*d_r[0] - b_r[0]*d_r[2];
    n2[2] = b_r[0]*d_r[1] - b_r[1]*d_r[0];
    for (int i = 0; i < dim; i++) {
      n1mag += n1[i]*n1[i];
      n2mag += n2[i]*n2[i];
      n1dotn2 += n1[i]*n2[i];
    }
    if (n1mag == 0. || n2mag == 0.) return 0;
    angle = acos(fabs(n1dotn2)/sqrt(n1mag*n2mag));
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
    if (dim != 3) return 0.;
    const int nDim = 3;
    double x[nDim];
    double y[nDim];
    double z[nDim];
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

a = order[0];
b = order[1];
c = order[2];

a = order[3];
b = order[4];
d = order[5];

//NOTE: REWRITE TO BE ORDERED

*/


  void Surgery_2D_22Flip_Setup(Obj<Mesh> m, Mesh::point_type a, Mesh::point_type b, Mesh::point_type * cells, Mesh::point_type * vertices, int * order) {
    //given a and b, set up the rest of the data structure needed for the 2-2 ear flip
    vertices[0] = a;
    vertices[1] = b;
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    line->clear();
    line->insert(a);
    line->insert(b);
    Obj<Mesh::sieve_type::supportSet> doublet = m->getSieve()->nJoin1(line); //check to see if this works
    //PetscPrintf(m->comm(), "%d in line, %d in doublet %d, %d\n", line->size(), doublet->size(), a, b);
    if (doublet->size() != 2) throw Exception("bad flip setup 2-2"); //CHECK EXTERNALLY TO MAKE SURE WE'RE NOT FLIPPING ACROSS INTERIOR BOUND
    Mesh::sieve_type::supportSet::iterator d_iter = doublet->begin();
    cells[0] = *d_iter;
    d_iter++;
    cells[1] = *d_iter;
    Obj<Mesh::sieve_type::coneSequence> corners = m->getSieve()->cone(cells[0]);
    Mesh::sieve_type::coneSequence::iterator c_iter     = corners->begin();
    Mesh::sieve_type::coneSequence::iterator c_iter_end = corners->end();
    int index = 0;
    while (c_iter != c_iter_end) {
      //PetscPrintf(m->comm(), "color: %d\n", c_iter.color());
      if (*c_iter != a && *c_iter != b) {
	vertices[2] = *c_iter;
	order[2] = index;
      } else if (*c_iter == a) {
	order[0] = index;
      } else {
	order[1] = index;
      }
      c_iter++;
      index++;
    }
    index = 0;
    corners = m->getSieve()->cone(cells[1]);
    c_iter      = corners->begin();
    c_iter_end  = corners->end();
    while (c_iter != c_iter_end) {
      if (*c_iter != a && *c_iter != b) {
	vertices[3] = *c_iter;
	order[5] = index;
      } else if (*c_iter == a) {
	order[3] = index;
      } else order[4] = index;
      c_iter++;
      index++;
    }
    //PetscPrintf(m->comm(), "2-2 Ear: %d %d %d %d, %d %d\n", vertices[0], vertices[1], vertices[2], vertices[3], cells[0], cells[1]);
    return;
  }

  PetscTruth Surgery_2D_22Flip_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, int * order) {
    const double pi = M_PI;
    //VALIDITY CONDITION FOR THIS FLIP: cad and cbd must be (much) less than 180. 
    //we could probably have some angle heuristic for local delaunay approximation, but whatever.
    //must compute it in terms of acb + bad etc.
    int dim = m->getDimension();
    //if (dim != 2) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    double a_coords[nDim], b_coords[nDim], c_coords[nDim], d_coords[nDim];
    const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    double current_angle = corner_angle(dim, a_coords, c_coords, b_coords) + corner_angle(dim, a_coords, d_coords, b_coords);
    //PetscPrintf(m->comm(), "%f angle\n", current_angle);
    if (current_angle >= pi) return PETSC_FALSE; //give this vertex some "wiggle" as it's likely to just disappear anyways
    current_angle = corner_angle(dim, b_coords, c_coords, a_coords) + corner_angle(dim, b_coords, d_coords, a_coords);
    //PetscPrintf(m->comm(), "%f angle\n", current_angle);
    if (current_angle >= pi) return PETSC_FALSE;
    return PETSC_TRUE;
  }

  //if this flip improves the ratio between the maximum angle, do it

  PetscTruth Surgery_2D_22Flip_Preferable(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, int * order) {
    //does a BASIC test to see if this edge would do better flipped
    //change of plan: ALWAYS chuck the maximum angle
    //if abc + abd + bac + bad < cda + cdb + dca + dcb then flip (divide the larger angle)
    const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    int dim = m->getDimension();
    //if (dim != 2) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    double a_coords[nDim], b_coords[nDim], c_coords[nDim], d_coords[nDim];
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    double angle_a = corner_angle(dim, a_coords, b_coords, c_coords) + corner_angle(dim, a_coords, b_coords, d_coords); 
    double angle_b = corner_angle(dim, b_coords, a_coords, c_coords) + corner_angle(dim, b_coords, a_coords, d_coords);
    double angle_c = corner_angle(dim, c_coords, d_coords, a_coords) + corner_angle(dim, c_coords, d_coords, b_coords);
    double angle_d = corner_angle(dim, d_coords, c_coords, a_coords) + corner_angle(dim, d_coords, c_coords, b_coords);
    //double ang_threshhold = M_PI - 0.01;
    //if (angle_c > ang_threshhold && angle_a < ang_threshhold && angle_b < ang_threshhold) return PETSC_TRUE;
    //if (angle_d > ang_threshhold && angle_a < ang_threshhold && angle_b < ang_threshhold) return PETSC_TRUE;
    if (angle_a + angle_b < angle_c + angle_d){
      return PETSC_TRUE;
    }
    return PETSC_FALSE;
  }

  int Surgery_2D_22Flip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, int * order, Mesh::point_type maxIndex) {
    //if (m->getDimension() != 2) {
    //  throw Exception("Wrong Flip Performed - Wrong ear or dimension");
    //}
    Obj<Mesh::sieve_type> s = m->getSieve();
    //get the triangles
    s->removeBasePoint(cells[0]);
    s->removeCapPoint(cells[0]);
    s->removeBasePoint(cells[1]);
    s->removeCapPoint(cells[1]);  
    s->addArrow(vertices[3], cells[0]);
    if (order[0] < order[2]) {
      s->addArrow(vertices[0], cells[0]);
      s->addArrow(vertices[2], cells[0]);
    } else {
      s->addArrow(vertices[2], cells[0]);
      s->addArrow(vertices[0], cells[0]);
    }
    s->addArrow(vertices[2], cells[1]);
    if (order[4] < order[5]) {
      s->addArrow(vertices[1], cells[1]);
      s->addArrow(vertices[3], cells[1]);
    } else {
      s->addArrow(vertices[3], cells[1]);
      s->addArrow(vertices[1], cells[1]);
    }
    if (1) {
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

  void Surgery_2D_22Flip_FixBoundary(Obj<Mesh> b_m, Mesh::point_type * cells, Mesh::point_type * vertices) {
    //fixes up a boundary mesh if it's flipped across -- just remove anything in the njoin1
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    line->clear();
    line->insert(vertices[0]);
    line->insert(vertices[1]);
    Obj<Mesh::sieve_type::supportSet> ab_join = b_m->getSieve()->nJoin1(line);
    Mesh::sieve_type::supportSet::iterator ab_iter = ab_join->begin();
    Mesh::sieve_type::supportSet::iterator ab_iter_end = ab_join->end();
    while (ab_iter != ab_iter_end) {
      b_m->getSieve()->removeBasePoint(*ab_iter);
      ab_iter++;
    }
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

  //EDIT: SEE IF THIS NOW PRESERVES ORDERINGS
  int Surgery_2D_31Flip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    //s->removeBasePoint(cells[0]);
    s->removeBasePoint(cells[1]);
    s->removeBasePoint(cells[2]);
    s->removeCapPoint(vertices[3]);
    //s->addArrow(vertices[0], cells[0]);
    //s->addArrow(vertices[1], cells[0]);
    s->addArrow(vertices[2], cells[0]);
    return maxIndex;
  }

/*

EDIT: not using really
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
    double pi = M_PI;
    //    double pi = 3.141592653589793238;
    int dim = m->getDimension();
    //if (dim != 2) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    double a_coords[nDim], b_coords[nDim], c_coords[nDim], d_coords[nDim], e_coords[nDim];
    const Obj<Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    PetscMemcpy(e_coords, coordinates->restrictPoint(vertices[4]), dim*sizeof(double));
    if (corner_angle(dim, a_coords, b_coords, e_coords) + corner_angle(dim, a_coords, c_coords, e_coords) >= pi ||
        corner_angle(dim, d_coords, b_coords, e_coords) + corner_angle(dim, d_coords, c_coords, e_coords) >= pi) {
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
    s->removeBasePoint(vertices[4]);
    s->removeCapPoint(vertices[4]);
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
    f-----g             f-----g
   /\     /\           /    _-'\
  /  \   /  \         /  _-'    \
 /    \ /    \       /_-'        \
b------a------c ->  b-------------c
 \    / \    /       \'-_        /
  \  /   \  /         \  '-_    /
   \/     \/           \    '-_/
    d-----e             d-----e


  */

  void Surgery_2D_LineContract_Setup(Obj<Mesh> m, Mesh::point_type a, Mesh::point_type b, Mesh::point_type * cells, Mesh::point_type * vertices) {
    //Line contract is there to make it easier to have internal edges expressedly represented in 2D or 3D boundary meshes.  One can have an arbitrary number of 
    //facets "circling" the line, and if they all have convex joins with respect to b, then the line may be shrunk from bac to bc with interior edges shunted over
    //order does NOT matter, as the legitimacy test is done on the joins of each of the vertices involved; the cells are not going to be filled in.
    
    //JUST record a and b.  The cone(support) of a will be used for the rest
    vertices[0] = a;
    vertices[1] = b;
  }

  //in the case where we have an interior vertex, we may have it find b such that it can contract from a (nearest-neighbor).  The criterion should be the LARGEST angle
  void Surgery_2D_LineContract_Direction(Obj<Mesh> m, Mesh::point_type a, Mesh::point_type * cells, Mesh::point_type * vertices) {
    Obj<Mesh::real_section_type> coordinates = m->getRealSection("coordinates");
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    vertices[0] = a;
    int dim = m->getDimension();
    //if (dim != 2) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    double v_coords[nDim], n_coords[nDim], e1_coords[nDim], e2_coords[nDim];
    PetscMemcpy(v_coords, coordinates->restrictPoint(a), dim*sizeof(double));
    //go through the neighbors of the vertex and find the nearest one
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::sieve_type::coneSet> neighbors = s->cone(s->support(a));
    Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    Mesh::point_type p;
    //    bool first = true;
    double p_angle = 0.0;
    while (n_iter != n_iter_end) {
      /*
      if (*n_iter != a) {
        const double * tmpcoords = coordinates->restrictPoint(*n_iter);
        double dist = 0;
        for (int i = 0; i < dim; i++) {
          dist += (tmpcoords[i] - coords[i])*(tmpcoords[i] - coords[i]);
        }
        dist = sqrt(dist);
        if (first) {
          first = false;
          p_dist = dist;
          p = *n_iter;
        } else {
          if (dist < p_dist) { p_dist = dist;
            p = *n_iter;
          }
        }
      }
      */
      //treat each like a doublet, and find the maximum angle
      if (*n_iter != a) {
        PetscMemcpy(n_coords, coordinates->restrictPoint(*n_iter), dim*sizeof(double));
        line->clear();
        line->insert(*n_iter);
        line->insert(vertices[0]);
        Obj<Mesh::sieve_type::supportSet> join = s->nJoin1(line);
        Obj<Mesh::sieve_type::coneSet> corners = s->cone(join);
        Mesh::sieve_type::coneSet::iterator c_iter = corners->begin();
        Mesh::sieve_type::coneSet::iterator c_iter_end = corners->end();
        Mesh::point_type earpoints[2];
        int index = 0;
        while (c_iter != c_iter_end) {
          if (*c_iter != vertices[0] && *c_iter != *n_iter) {
            earpoints[index] = *c_iter;
            index++;
          }
          c_iter++;
        }
        PetscMemcpy(e1_coords, coordinates->restrictPoint(earpoints[0]), dim*sizeof(double));
        PetscMemcpy(e2_coords, coordinates->restrictPoint(earpoints[1]), dim*sizeof(double));
        double cur_angle = corner_angle(dim, n_coords, v_coords, e1_coords) + 
                           corner_angle(dim, n_coords, v_coords, e2_coords);
        if (cur_angle > p_angle) {
          p = *n_iter;
          p_angle = cur_angle;
        }
      }
      n_iter++;
    }
    vertices[1] = p;
    return;
  }

  PetscTruth Surgery_2D_LineContract_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices) {
    //for now this is last-ditch for 3D... do this to form the exterior and let tetgen deal with collisions
    //TODO: Tetgen flakes out; fix this
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_2D_LineContract(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::sieve_type::supportSequence> a_support = s->support(vertices[0]);
    Mesh::sieve_type::supportSequence::iterator as_iter = a_support->begin();
    Mesh::sieve_type::supportSequence::iterator as_iter_end = a_support->end();
    Mesh::sieve_type::supportSet remove_elements;
    while (as_iter != as_iter_end) {
      //shrink every element to b
      s->addArrow(vertices[1], *as_iter, as_iter.color());
      if (s->cone(*as_iter)->size() != 4) remove_elements.insert(*as_iter);
      as_iter++;
    }
    Mesh::sieve_type::supportSet::iterator re_iter = remove_elements.begin();
    Mesh::sieve_type::supportSet::iterator re_iter_end = remove_elements.end();
    while (re_iter != re_iter_end) {
      s->removeBasePoint(*re_iter);
      s->removeCapPoint(*re_iter);
      re_iter++;
    }
    s->removeBasePoint(vertices[0]);
    s->removeCapPoint(vertices[0]);
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
    double pi = M_PI;
    //    double pi = 3.14159265359;
    const Obj<Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    int dim = m->getDimension();
    //if (dim != 2) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    double coords_a[nDim], coords_b[nDim], coords_c[nDim], coords_d[nDim];
    PetscMemcpy(coords_a, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(coords_b, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(coords_c, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(coords_d, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    if (corner_angle(dim, coords_b, coords_a, coords_d) + 
        corner_angle(dim, coords_b, coords_c, coords_d) >= pi) return PETSC_FALSE;
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_2D_21BoundFlip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    s->removeBasePoint(cells[1]);
    s->removeBasePoint(cells[0]);
    s->removeCapPoint(vertices[3]);
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
    s->removeCapPoint(vertices[0]);
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

  void Surgery_3D_23Flip_Setup(Obj<Mesh> m, Mesh::point_type a, Mesh::point_type b, Mesh::point_type c, Mesh::point_type * cells, Mesh::point_type * vertices) {
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
    if (join->size() != 2) {
      throw Exception("Incoherent mesh in 23 flip");
    }
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
    //    PetscPrintf(m->comm(), "2-3 Flip: %d %d, %d %d %d %d %d\n", cells[0], cells[1], vertices[0], vertices[1], vertices[2], vertices[3], vertices[4]);
    return;

  }

  PetscTruth Surgery_3D_23Flip_Possible (Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices){
    //The 2-3 flip is only possible if the line between a and d lies inside the volume; this occurs when, for example,
    //volume(abde + bcde + cade) <== volume(abcd + abce)
    
    int dim = m->getDimension();
    if (dim != 3) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    //kill this thing cheaply if possible:
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    if (m->getSieve()->nJoin1(line)->size() != 0) return PETSC_FALSE;
    const Obj<Mesh::real_section_type> coordinates = m->getRealSection("coordinates");
    double a_coords[nDim], b_coords[nDim], c_coords[nDim], d_coords[nDim], e_coords[nDim];
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    PetscMemcpy(e_coords, coordinates->restrictPoint(vertices[4]), dim*sizeof(double));
    double volume_1 = tetrahedron_volume(dim, a_coords, b_coords, d_coords, e_coords);
    double volume_2 =  tetrahedron_volume(dim, b_coords, c_coords, d_coords, e_coords);
    double volume_3 =  tetrahedron_volume(dim, c_coords, a_coords, d_coords, e_coords);
    double now_volume = tetrahedron_volume(dim, a_coords, b_coords, c_coords, d_coords)
      + tetrahedron_volume(dim, a_coords, b_coords, c_coords, e_coords);
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

    //check to make sure that there is currently not a join between the two endpoints of this;
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    if (s->nJoin1(line)->size() != 0) {
      throw Exception("line already exists for some reason!");
    }  
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

    if (m->debug()) {
    
    //debug time
    int cone1_size = s->cone(cells[0])->size();
    int cone2_size = s->cone(cells[1])->size();
    int cone3_size = s->cone(newCell)->size();
    if (cone1_size != 4 || cone2_size != 4 || cone3_size != 4) {
      PetscPrintf(m->comm(), "involved vertices %d, %d, %d, %d, %d\n", vertices[0], vertices[1], vertices[2], vertices[3], vertices[4]);
      PetscPrintf(m->comm(), "cone sizes %d, %d, %d\n", cone1_size, cone2_size, cone3_size);
      throw Exception("Flip sizes screwed up");
    }
    Obj<Mesh::sieve_type::supportSet> lens;
    line->clear();
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    lens = s->nJoin1(line);
    int lens_size = lens->size();
    if (lens_size != 3) {
      PetscPrintf(m->comm(), "Resulting Lens Size of line %d %d: %d\n", vertices[3], vertices[4], lens_size);
      Mesh::sieve_type::supportSet::iterator ls_iter = lens->begin();
      Mesh::sieve_type::supportSet::iterator ls_iter_end = lens->end();
      for (; ls_iter != ls_iter_end; ls_iter++) {
        PetscPrintf(m->comm(), "%d: ", *ls_iter);
        Obj<Mesh::sieve_type::coneSequence> error_cone = s->cone(*ls_iter);
	Mesh::sieve_type::coneSequence::iterator ec_iter = error_cone->begin();
	Mesh::sieve_type::coneSequence::iterator ec_iter_end = error_cone->end();
        while (ec_iter != ec_iter_end) {
	  PetscPrintf(m->comm(), "%d ", *ec_iter);
          ec_iter++;
        }
        PetscPrintf(m->comm(), "\n");
      }
      PetscPrintf(m->comm(), "\n");
      throw Exception("Flip interior line screwed up");
    }
    //check interior edges
    line->clear();
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    line->insert(vertices[1]);
    lens = s->nJoin1(line);
    if (lens->size() != 2) {
      PetscPrintf(m->comm(), "face: %d %d %d has %d sides\n", vertices[3], vertices[4], vertices[1], lens->size());
      throw Exception("2-3 flip: bad interior face");
    }
    line->clear();
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    line->insert(vertices[2]);
    lens = s->nJoin1(line);
    if (lens->size() != 2) {
      PetscPrintf(m->comm(), "face: %d %d %d has %d sides\n", vertices[3], vertices[4], vertices[2], lens->size());
      throw Exception("2-3 flip: bad interior face");
    }
    line->clear();
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    line->insert(vertices[0]);
    lens = s->nJoin1(line);
    if (lens->size() != 2) {
      PetscPrintf(m->comm(), "face: %d %d %d has %d sides\n", vertices[3], vertices[4], vertices[0], lens->size());
      throw Exception("2-3 flip: bad interior face");
    }
    line->clear();
    line->insert(vertices[3]);
    line->insert(vertices[0]);
    line->insert(vertices[1]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[3], vertices[0], vertices[1], s->nJoin1(line)->size());
      throw Exception("Bad face Created in 2-3 flip");
    }
    line->clear();
    line->insert(vertices[3]);
    line->insert(vertices[1]);
    line->insert(vertices[2]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[3], vertices[1], vertices[2], s->nJoin1(line)->size());
      throw Exception("Bad face Created in 2-3 flip");
    }
    line->clear();
    line->insert(vertices[3]);
    line->insert(vertices[2]);
    line->insert(vertices[0]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[3], vertices[2], vertices[0], s->nJoin1(line)->size());
      throw Exception("Bad face Created in 2-3 flip");
    }    line->clear();
    line->insert(vertices[4]);
    line->insert(vertices[0]);
    line->insert(vertices[1]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[4], vertices[0], vertices[1], s->nJoin1(line)->size());
      throw Exception("Bad face Created in 2-3 flip");
    }
    line->clear();
    line->insert(vertices[4]);
    line->insert(vertices[1]);
    line->insert(vertices[2]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[4], vertices[1], vertices[2], s->nJoin1(line)->size());
      throw Exception("Bad face Created in 2-3 flip");
    }
    line->clear();
    line->insert(vertices[4]);
    line->insert(vertices[2]);
    line->insert(vertices[0]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[0], vertices[4], vertices[2], s->nJoin1(line)->size());
      throw Exception("Bad face Created in 2-3 flip");
    }
    }
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
    if (corners->size() != 4) throw Exception("bad element: 3-2");
    Mesh::sieve_type::coneSequence::iterator c_iter = corners->begin();
    Mesh::sieve_type::coneSequence::iterator c_iter_end = corners->end();
    bool first = true;
    Mesh::point_type p;
    while (c_iter != c_iter_end) {
      p = *c_iter;
      if (p != a && p != b) {
        if (first) {
          vertices[2] = p;
          first = false;
        } else {
          vertices[3] = p;
        }
      }
      c_iter++;
    }
    corners = s->cone(cells[1]);
    if (corners->size() != 4) throw Exception("bad element: 3-2");
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
    //    PetscPrintf(m->comm(), "3-2 Flip: %d %d %d, %d %d %d %d %d\n", cells[0], cells[1], cells[2], vertices[0], vertices[1], vertices[2], vertices[3], vertices[4]);
  }
  PetscTruth Surgery_3D_32Flip_Possible (Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices){
    //The 3-2 flip is only possible if each new subvolume will have volume less than the previous three combined (convex)
    int dim = m->getDimension();
    if (dim != 3) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    const Obj<Mesh::real_section_type> coordinates = m->getRealSection("coordinates");
    double a_coords[nDim], b_coords[nDim], c_coords[nDim], d_coords[nDim], e_coords[nDim];
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

    Obj<Mesh::sieve_type::supportSet> line;

    if (m->debug()) {

    line = new Mesh::sieve_type::supportSet();    
    line->clear();
    line->insert(vertices[2]);
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    Obj<Mesh::sieve_type::supportSet> doublet = s->nJoin1(line);
    if (doublet->size() != 0) {
      PetscPrintf(m->comm(), "doublet size: %d\n", doublet->size());
      throw Exception("face already occupied in 3-2 flip");
    }

    }
    
    s->addArrow(vertices[0], cells[0]);
    s->addArrow(vertices[2], cells[0]);
    s->addArrow(vertices[3], cells[0]);
    s->addArrow(vertices[4], cells[0]);

    s->addArrow(vertices[1], cells[1]);
    s->addArrow(vertices[2], cells[1]);
    s->addArrow(vertices[3], cells[1]);
    s->addArrow(vertices[4], cells[1]);
    if (m->debug()) {
    //error checking;
    int cone1_size = s->cone(cells[0])->size();
    int cone2_size = s->cone(cells[1])->size();
    if (cone1_size != 4 || cone2_size != 4) {
      PetscPrintf(m->comm(), "Cone Sizes, %d, %d\n", cone1_size, cone2_size);
      throw Exception("Screwed up 3-2 flip");
    }
    line->clear();
    line->insert(vertices[2]);
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    if (s->nJoin1(line)->size() != 2) {
      PetscPrintf(m->comm(), "Lens Size: %d\n", s->nJoin1(line)->size());
      throw Exception("Bad Lens Created in 3-2 flip");
    }
    //check the consistency of the resulting suspension exterior faces;
    line->clear();
    line->insert(vertices[0]);
    line->insert(vertices[2]);
    line->insert(vertices[3]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[0], vertices[2], vertices[3], s->nJoin1(line)->size());
      throw Exception("Bad Lens Created in 3-2 flip");
    }
    line->clear();
    line->insert(vertices[0]);
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[0], vertices[3], vertices[4], s->nJoin1(line)->size());
      throw Exception("Bad Lens Created in 3-2 flip");
    }
    line->clear();
    line->insert(vertices[0]);
    line->insert(vertices[4]);
    line->insert(vertices[2]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[0], vertices[4], vertices[2], s->nJoin1(line)->size());
      throw Exception("Bad Lens Created in 3-2 flip");
    }
    line->clear();
    line->insert(vertices[1]);
    line->insert(vertices[2]);
    line->insert(vertices[3]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[1], vertices[2], vertices[3], s->nJoin1(line)->size());
      throw Exception("Bad Lens Created in 3-2 flip");
    }
    line->clear();
    line->insert(vertices[1]);
    line->insert(vertices[3]);
    line->insert(vertices[4]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[1], vertices[3], vertices[4], s->nJoin1(line)->size());
      throw Exception("Bad Lens Created in 3-2 flip");
    }
    line->clear();
    line->insert(vertices[1]);
    line->insert(vertices[4]);
    line->insert(vertices[2]);
    if (s->nJoin1(line)->size() > 2) {
      PetscPrintf(m->comm(), "Exterior doublet (%d %d %d) Size: %d\n", vertices[1], vertices[4], vertices[2], s->nJoin1(line)->size());
      throw Exception("Bad Lens Created in 3-2 flip");
    }
    }
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



//just like the 4-1 except one of the 4 is missing. (make the one of the four missing the one involving b, c, d)

/*     _e_    exterior        
    _-' | '-_
 _-'    |    '-_
b-------a-------c     b-------a-------c
 \      |      /       \      |      /
  \    1|     /         \     |     /
   \    |    /           \    1    /
    \ 2 e 3 /      ->     \   |   /
     \  |  /               \  |  /
      \ | /                 \ | /
       \|/                   \|/
        d                     d
*/

  void Surgery_3D_31BoundFlip_Setup(Obj<Mesh> m, Mesh::point_type e, Mesh::point_type * cells, Mesh::point_type * vertices) {
    vertices[4] = e;
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::sieve_type::supportSequence> cell_points = s->support(e);
    if (cell_points->size() != 3) throw Exception("wrong flip: 3-1 Boundary flip");
    Mesh::sieve_type::supportSequence::iterator cp_iter = cell_points->begin();
    cells[0] = *cp_iter;
    cp_iter++;
    cells[1] = *cp_iter;
    cp_iter++;
    cells[2] = *cp_iter;
    Obj<Mesh::sieve_type::coneSet> vertex_points = s->support(cell_points);
    if (vertex_points->size() != 5) throw Exception("Erroneous Flip: 3-1 Boundary flip");
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    Obj<Mesh::sieve_type::supportSet> link;
    Mesh::sieve_type::coneSet::iterator vp_iter = vertex_points->begin();
    Mesh::sieve_type::coneSet::iterator vp_iter_end = vertex_points->end();
    while (vp_iter != vp_iter_end) {
      if (*vp_iter != e) {
        line->clear();
        line->insert(e);
        line->insert(*vp_iter);
        link = s->nJoin1(line);
        if (link->size() == 3) {
          vertices[0] = *vp_iter;
        } else {
	  Mesh::sieve_type::supportSet::iterator l_end = link->end();
          bool link_has_1 = (link->find(cells[0]) != l_end);
	  bool link_has_2 = (link->find(cells[1]) != l_end);
          bool link_has_3 = (link->find(cells[3]) != l_end);
	  if (link_has_1 && link_has_2) {
            vertices[1] = *vp_iter;
          } else if (link_has_2 && link_has_3) {
            vertices[2] = *vp_iter;
	  } else if (link_has_3 && link_has_1) {
            vertices[3] = *vp_iter;
          } else {
	    throw Exception("Flip Error: Boundflip 3-1");
          }
        }
      }
      vp_iter++;
    }
    return;
  }

  PetscTruth Surgery_3D_31BoundFlip_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices) {
    int dim = m->getDimension();
    if (dim != 3) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    const Obj<Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
    double a_coords[nDim], b_coords[nDim], c_coords[nDim], d_coords[nDim], e_coords[nDim];
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    PetscMemcpy(e_coords, coordinates->restrictPoint(vertices[4]), dim*sizeof(double));
    double new_volume = tetrahedron_volume(dim, a_coords, b_coords, c_coords, d_coords);
    double cap_volume = tetrahedron_volume(dim, a_coords, b_coords, d_coords, e_coords);
    double old_volume = tetrahedron_volume(dim, a_coords, b_coords, d_coords, e_coords) + tetrahedron_volume(dim, b_coords, c_coords, d_coords, e_coords) + tetrahedron_volume(dim, c_coords, a_coords, d_coords, e_coords);
    //CASE: e is concave, removing e increases the volume;
    if (cap_volume - old_volume > 0 && new_volume < cap_volume) return PETSC_FALSE;
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_3D_31BoundFlip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    s->removeBasePoint(cells[0]);
    s->removeBasePoint(cells[1]);
    s->removeBasePoint(cells[2]);
    
    s->addArrow(vertices[0], cells[0]);
    s->addArrow(vertices[1], cells[0]);
    s->addArrow(vertices[2], cells[0]);
    s->addArrow(vertices[3], cells[0]);
    return maxIndex;
  }

  /*

  3D extension of the 2D 2-2 boundflip -- we've removed all internal subvolumes in our way on an edge; save for two

     a              a
    /|\            /|\
   / | \          / | \
  /  |  \        /  1  \
 /_-'|e-_\      /_-'e'-_\
c  1 | 2  d -> c---------d
 \   |   /      \   |   /
  \  |  /        \  2  /
   \ | /          \ | /
    \|/            \|/
     b              b

a, b, c, and d are on the boundary in question

  */

  void Surgery_3D_22BoundFlip_Setup(Obj<Mesh> m, Mesh::point_type a, Mesh::point_type b, Mesh::point_type e, Mesh::point_type * cells, Mesh::point_type * vertices) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    vertices[0] = a;
    vertices[1] = b;
    vertices[4] = e;
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    Obj<Mesh::sieve_type::supportSet> lens = s->nJoin1(line);
    if (lens->size() != 2) throw Exception("Wrong Flip: Boundary 2-2");
    Mesh::sieve_type::supportSet::iterator l_iter = lens->begin();
    cells[0] = *l_iter;
    l_iter++;
    cells[1] = *l_iter;
    Obj<Mesh::sieve_type::coneSet> corners = s->cone(lens);
    if (corners->size() != 5) throw Exception("Bad Corners on 2-2 Boundflip");
    Mesh::sieve_type::coneSet::iterator c_iter = corners->begin();
    Mesh::sieve_type::coneSet::iterator c_iter_end = corners->end();
    int index = 2;
    while (c_iter != c_iter_end) {
      if (*c_iter != a && *c_iter != b && *c_iter != e) {
        vertices[index] = *c_iter;
        index++;
      }
      c_iter++;
    }
    return;
  }

  PetscTruth Surgery_3D_22BoundFlip_Possible(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices) {

    //criterion for success:  both new volumes are less than the old volume. (they don't overlap)
    int dim = m->getDimension();
    const double pi = M_PI;
    if (dim != 3) throw ALE::Exception("Wrong dimension");
    const int nDim = 3;
    //    double pi = 3.141592653589793238;
    const Obj<Mesh::real_section_type> coordinates = m->getRealSection("coordinates");
    double a_coords[nDim], b_coords[nDim], c_coords[nDim], d_coords[nDim], e_coords[nDim];
    PetscMemcpy(a_coords, coordinates->restrictPoint(vertices[0]), dim*sizeof(double));
    PetscMemcpy(b_coords, coordinates->restrictPoint(vertices[1]), dim*sizeof(double));
    PetscMemcpy(c_coords, coordinates->restrictPoint(vertices[2]), dim*sizeof(double));
    PetscMemcpy(d_coords, coordinates->restrictPoint(vertices[3]), dim*sizeof(double));
    PetscMemcpy(e_coords, coordinates->restrictPoint(vertices[4]), dim*sizeof(double));
    //NAH! only consider surface angles adding to form the curvature... (which should work)... these are convex or a saddle (saddle = invalid)
    //double old_volume = tetrahedron_volume(dim, a_coords, b_coords, c_coords, e_coords) +
    //                    tetrahedron_volume(dim, a_coords, b_coords, d_coords, e_coords);
    //double volume_1 = tetrahedron_volume(dim, a_coords, c_coords, d_coords, e_coords);
    //double volume_2 = tetrahedron_volume(dim, b_coords, c_coords, d_coords, e_coords);
    //if (volume_1 < old_volume && volume_2 < old_volume) return PETSC_TRUE;
    double curvature_1 = corner_angle(dim, a_coords, c_coords, b_coords) + corner_angle(dim, a_coords, d_coords, b_coords) +
      corner_angle(dim, a_coords, c_coords, e_coords) + corner_angle(dim, a_coords, d_coords, e_coords);
    if (curvature_1 >= 2*pi) return PETSC_FALSE;
    double curvature_2 = corner_angle(dim, b_coords, c_coords, a_coords) + corner_angle(dim, b_coords, d_coords, a_coords) +
      corner_angle(dim, b_coords, c_coords, e_coords) + corner_angle(dim, b_coords, d_coords, e_coords);
    if (curvature_2 >= 2*pi) return PETSC_FALSE;
    return PETSC_TRUE;
  }

  Mesh::point_type Surgery_3D_22BoundFlip(Obj<Mesh> m, Mesh::point_type * cells, Mesh::point_type * vertices, Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    s->removeBasePoint(cells[0]);
    s->removeBasePoint(cells[1]);
    
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

//remove a 1D point
  int Surgery_1D_Remove_Vertex(Obj<Mesh> m, Mesh::point_type vertex, ALE::Mesh::point_type maxIndex) {
    Obj<Mesh::sieve_type> s = m->getSieve();
    Mesh::sieve_type::supportSet edges_to_remove;
    Obj<Mesh::sieve_type::supportSequence> v_support = s->support(vertex);
    Mesh::sieve_type::supportSequence::iterator vs_iter = v_support->begin();
    Mesh::sieve_type::supportSequence::iterator vs_iter_end = v_support->end();
    //find a reasonable neighbor to contract to
    Mesh::point_type neighbor = vertex;
    Obj<Mesh::sieve_type::coneSet> neighbors = s->cone(v_support);
    if (neighbors->size() <= 1) {
      //leave this vertex alone
    } else {
      Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
      Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end(); 
      while (n_iter != n_iter_end && neighbor == vertex) {
        neighbor = *n_iter;
        n_iter++;
      }
      //next, contract all edges to this neighbor
      edges_to_remove.clear();
      while (vs_iter != vs_iter_end) {
        s->addArrow(neighbor, *vs_iter);
        if(s->cone(*vs_iter)->size() < 3) edges_to_remove.insert(*vs_iter);
        vs_iter++;
      }
    //next, remove the vertex and the edges that should be removed
      s->removeBasePoint(vertex);
      s->removeCapPoint(vertex);
      Mesh::sieve_type::supportSet::iterator etr_iter = edges_to_remove.begin();
      Mesh::sieve_type::supportSet::iterator etr_iter_end = edges_to_remove.end();
      while (etr_iter != etr_iter_end) {
        s->removeBasePoint(*etr_iter);
        s->removeCapPoint(*etr_iter);
        etr_iter++;
      }
    }
    return maxIndex;
  }


//remove a 2D point

  int Surgery_2D_Remove_Vertex(Obj<Mesh> m, Mesh::point_type vertex, Obj<Mesh> bound_m, ALE::Mesh::point_type maxIndex) {
    //get potential ears
    //    int dim = m->getDimension();  //limits some of the operations we can do to 2D-centric and nD-centric
    Obj<Mesh::sieve_type> s = m->getSieve();
    //Obj<Mesh::label_type> boundary = m->getLabel("marker"); //this could be erroneously marked, unfortunately; it's used for other things.
    bool on_boundary = (bound_m->getSieve()->hasPoint(vertex));
    //if (on_boundary) PetscPrintf(m->comm(), "Boundary node.");
    Mesh::point_type cells[4];
    Mesh::point_type vertices[5];
    int order[9];
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

      if (neighbor_size == 3 && !on_boundary) {
         
  	//PetscPrintf(m->comm(), "flip: 3-1\n");      
        Surgery_2D_31Flip_Setup(m, vertex, cells, vertices);
        if (Surgery_2D_31Flip_Possible(m, cells, vertices)) {
        Surgery_2D_31Flip(m, cells, vertices, maxIndex);
        }
	/*
      } else if (neighbor_size == 3 && on_boundary) {
  
  	//PetscPrintf(m->comm(), "2-1\n");
        Surgery_2D_21BoundFlip_Setup(m, vertex, cells, vertices);
        Surgery_2D_21BoundFlip(m, cells, vertices, maxIndex);
	*/
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
          if (s->nJoin1(line)->size() != 2 || bound_m->getSieve()->nJoin1(line)->size() != 0) {
            //do nothing
          } else {
            //2->2 flip
            //PetscPrintf(m->comm(), "2-2 attempt\n");
            Surgery_2D_22Flip_Setup(m, vertex, *n_iter, cells, vertices, order); 
            if (Surgery_2D_22Flip_Possible(m, cells, vertices, order)) {
              Surgery_2D_22Flip(m, cells, vertices, order, maxIndex);  //in 2D there are no cases where we have to take the maxIndex into account
              changed_neighbors = true;
            }
          }
        }
        n_iter++;
      }
      //SEEMS TO BE CAUSING PROBLEMS
      /*
      if (n_iter == n_iter_end && neighbor_size == 4 && !on_boundary) { //this is the ONLY safe time to use this
        Surgery_2D_LineContract_Direction(m, vertex, cells, vertices);
        if (Surgery_2D_LineContract_Possible(m, cells, vertices)) {
          Surgery_2D_LineContract(m, cells, vertices, maxIndex);
        }
      }
      */
      if (n_iter == n_iter_end && on_boundary) { //last-ditch, forced in edges exist.
  	//PetscPrintf(m->comm(), "flip: Contract\n"); 
	Mesh::point_type other_line_end = *neighbors->begin();  //if it's an isolated vertex just contract along a single line
	Obj<Mesh::sieve_type::coneSet> bound_neighbors = bound_m->getSieve()->cone(bound_m->getSieve()->support(vertex));
	Mesh::sieve_type::coneSet::iterator bn_iter = bound_neighbors->begin();
	Mesh::sieve_type::coneSet::iterator bn_iter_end = bound_neighbors->end();
        while (bn_iter != bn_iter_end) {
          if (*bn_iter != vertex) other_line_end = *bn_iter; //find a line to contract along
          bn_iter++;
        }
        Surgery_2D_LineContract_Setup(m, vertex, other_line_end, cells, vertices);
        if (Surgery_2D_LineContract_Possible(m, cells, vertices)) {
          Surgery_2D_LineContract(m, cells, vertices, maxIndex);
        }
      } //last-ditch boundary contract: find another vertex on the boundary in the neighbor set and contract to it.  NO GUARANTEES

      if (!changed_neighbors) {
        //we're done if the local topology hasn't changed
        remove_finished = true;
      } else {
        neighbors = s->cone(s->support(vertex));
      }
    }
    //PetscPrintf(m->comm(), "removed %d\n", vertex);
    if (bound_m)if (!s->hasPoint(vertex)) Surgery_1D_Remove_Vertex(bound_m, vertex, maxIndex);
    return maxIndex;
  }

//remove a 3D point


  Mesh::point_type Surgery_3D_Remove_Vertex(Obj<Mesh> m, Mesh::point_type vertex, ALE::Mesh::point_type maxIndex) {
    //get potential ears
    Mesh::point_type cur_maxIndex = maxIndex;
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::label_type> boundary = m->getLabel("marker"); 
    //this could be erroneously marked, unfortunately; it's used for other things.
    //bool on_boundary = (m->getValue(boundary, vertex) == 1);
    //if (on_boundary) PetscPrintf(m->comm(), "Boundary node.");
    Mesh::point_type cells[4];
    Mesh::point_type vertices[7];
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    Obj<Mesh::sieve_type::supportSet> lens;

  //go through neighbors seeing if they are legitimate two-ears; if so then flip and repeat.
    //    bool remove_finished = false;
    //Algorithm: Do 2-3 flips until the cardinality of the neighbors of n on the link is 3, then do a 3-2 flip to remove it
    //new idea: enqueue the link vertex we want to work on next; go until the queue is empty.
    //NEW IDEA: NESTED QUEUES
    Obj<Mesh::sieve_type::supportSet> neighbor_queue = new Mesh::sieve_type::supportSet();
    Obj<Mesh::sieve_type::supportSet> link_neighbor_queue = new Mesh::sieve_type::supportSet();

    //add the neighbors to the neighbor queue.
    Obj<Mesh::sieve_type::coneSet> neighbors = s->cone(s->support(vertex));
    Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
    Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
    //loop doing 2-3 and 3-2 flips until there are no more legitimate flips; this will involve flipping on a given neighbor fan; reducing
    //its cardinality until we can put a cap on it, or it is concave.
    while (n_iter != n_iter_end) {
      bool reset_neighbors = false;
      if (*n_iter != vertex) {
        Mesh::point_type cur_link_vertex = *n_iter;
        
        line->clear();
        //REMOVE THIS LATER BECAUSE IT'S GROSSLY INEFFICIENT
        //      PetscPrintf(m->comm(), "# neighbors: %d\n", s->cone(s->support(vertex))->size());
        //      PetscPrintf(m->comm(), "processing the line: %d, %d\n", vertex, cur_link_vertex);
        line->insert(cur_link_vertex);
        line->insert(vertex);
        lens = s->nJoin1(line);
  
        //add the cone of the lens that isn't vertex or cur_link_vertex
        Obj<Mesh::sieve_type::coneSet> lens_vertices = s->cone(lens); 
 
        Mesh::sieve_type::coneSet::iterator lv_iter = lens_vertices->begin();
        Mesh::sieve_type::coneSet::iterator lv_iter_end = lens_vertices->end();
        while (lv_iter != lv_iter_end) {
          bool reset_lens = false;
    	  Mesh::point_type cur_link_neighbor_vertex = *lv_iter;
            if (*lv_iter != vertex && *lv_iter != cur_link_vertex) {
            //REMOVE LATER BECAUSE THIS IS GROSSLY INEFFICIENT
  	    //        PetscPrintf(m->comm(), "#link neighbors: %d\n", s->cone(s->nJoin1(line))->size() - 2);
  	    //        PetscPrintf(m->comm(), "Processing the link line: %d, %d\n", cur_link_vertex, cur_link_neighbor_vertex);        
            //do 2-3 flips until you can't anymore.
            Surgery_3D_23Flip_Setup(m, vertex, cur_link_vertex, cur_link_neighbor_vertex, cells, vertices);
            if (Surgery_3D_23Flip_Possible(m, cells, vertices)) {
  	    //          PetscPrintf(m->comm(), "Flipping.\n");
              cur_maxIndex = Surgery_3D_23Flip(m, cells, vertices, cur_maxIndex);
              //reget the neighbors
              lens = s->nJoin1(line);
              lens_vertices = s->cone(lens);
              lv_iter = lens_vertices->begin();
              lv_iter_end = lens_vertices->end();
              reset_lens = true;
	    }
          }
	    if (!reset_lens)lv_iter++;
        }
        //cap the 3-ear if possible; otherwise go on;
        lens = s->nJoin1(line);
        lens_vertices = s->cone(lens);
        if (lens_vertices->size() == 5) {
          Surgery_3D_32Flip_Setup(m, vertex, cur_link_vertex, cells, vertices);
          if (Surgery_3D_32Flip_Possible(m, cells, vertices)) {
  	  //          PetscPrintf(m->comm(), "Flipping.\n");
            cur_maxIndex = Surgery_3D_32Flip(m, cells, vertices, cur_maxIndex);
            neighbors = s->cone(s->support(vertex));
            n_iter = neighbors->begin();
            n_iter_end = neighbors->end();
            reset_neighbors = true;
            }
          }
        } 
      if (!reset_neighbors) n_iter++;
    } //end of our link stuffing task; if we can now simply pop the point out, do it.
    Obj<Mesh::sieve_type::supportSequence> neighbor_cells = s->support(vertex);  
    PetscPrintf(m->comm(), "Done with link decimation; # surrounding cells: %d\n", neighbor_cells->size());
    if (neighbor_cells->size() == 4) {
      Surgery_3D_41Flip_Setup(m, vertex, cells, vertices);
      if (Surgery_3D_41Flip_Possible(m, cells, vertices)) {
        cur_maxIndex = Surgery_3D_41Flip(m, cells, vertices, cur_maxIndex);
      }
    }
    return cur_maxIndex;
  }

  Mesh::point_type Surgery_Remove_Vertex(Obj<Mesh> m, Mesh::point_type vertex, Obj<Mesh> bound_m, Mesh::point_type maxIndex, int effectiveDimension = -1) {
      Mesh::point_type cur_maxIndex = maxIndex;
      int dim;
      if (effectiveDimension != -1) {
        dim = effectiveDimension;
      } else {
        dim = m->getDimension();
      }
      if (dim == 2) {
        cur_maxIndex = Surgery_2D_Remove_Vertex(m, vertex, bound_m, cur_maxIndex);
      } else if (dim == 3) {
        cur_maxIndex = Surgery_3D_Remove_Vertex(m, vertex, cur_maxIndex);  //doesn't currently work; messy.
      }
      return cur_maxIndex;
  }

  Mesh::point_type Surgery_Remove_VertexSet(Obj<Mesh> m, Obj<Mesh::sieve_type::supportSet> vertices, Obj<Mesh> b_m, Mesh::point_type maxIndex, int effectiveDimension = -1) {
    Mesh::point_type cur_maxIndex = maxIndex;
    Mesh::sieve_type::supportSet::iterator v_iter = vertices->begin();
    Mesh::sieve_type::supportSet::iterator v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      cur_maxIndex = Surgery_Remove_Vertex(m, *v_iter, b_m, cur_maxIndex, effectiveDimension);
      v_iter++;
    }
    return cur_maxIndex;
  }

  Mesh::point_type Surgery_Remove_AllButVertexSet(Obj<Mesh> m, Obj<Mesh::sieve_type::supportSet> kept_vertices, Obj<Mesh> b_m, Mesh::point_type maxIndex, int effectiveDimension = -1) {
    Obj<Mesh::label_sequence> vertices = m->depthStratum(0);
    Mesh::label_sequence::iterator v_iter = vertices->begin();
    Mesh::label_sequence::iterator v_iter_end = vertices->end();
    Mesh::sieve_type::supportSet::iterator k_iter_end = kept_vertices->end();
    Mesh::point_type cur_maxIndex = maxIndex;
    while (v_iter != v_iter_end) {
      if (kept_vertices->find(*v_iter) == kept_vertices->end()) {
        cur_maxIndex = Surgery_Remove_Vertex(m, *v_iter, b_m, maxIndex, effectiveDimension);        
      }
      v_iter++;
    }
    m->stratify();
    //PetscPrintf(m->comm(), "%d vertices left\n", m->depthStratum(0)->size());
    return cur_maxIndex;
  }

  void Surgery_2D_Improve_Mesh(Obj<Mesh> m, Obj<Mesh> bound_m = PETSC_NULL) {
    //we SHOULD be traversing; starting from the BEST triangle in the mesh and moving outwards as we know we can improve angles.
    if (m->depth() != 1) throw Exception("uninterpolated meshes only");
    //improve the edges of a vertex and traverse outwards to its neighbors; do until all vertices are done.  
    //DO NOT do edges that are in the boundary.
    Mesh::point_type cells_points[2], vertices_points[4];
    int order[6];
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::label_sequence> vertices = m->depthStratum(0);
    ALE::Mesh::label_sequence::iterator v_iter = vertices->begin();
    ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
    Obj<Mesh::sieve_type::supportSet> line = new Mesh::sieve_type::supportSet();
    while (v_iter != v_iter_end) {  //do the DUMBEST THING: give every mesh vertex a chance to have its link improved; potentially flipping each edge twice.
      //potential future change: traverse outwards from a reasonable triangle improving its bad neighbors.
      Obj<Mesh::sieve_type::coneSet> neighbors = s->cone(s->support(*v_iter));
      Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
      Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
      while (n_iter != n_iter_end) {
        line->clear();
        line->insert(*n_iter);
        line->insert(*v_iter);
        if (bound_m) {
          //check if this line is in the boundary
          if (bound_m->getSieve()->nJoin1(line)->size() == 0) { //this line is NOT in the boundary mesh and therefore cool
            //now just check
            if (s->nJoin1(line)->size() == 2) {
              Surgery_2D_22Flip_Setup(m, *n_iter, *v_iter, cells_points, vertices_points, order);
              if (Surgery_2D_22Flip_Possible(m, cells_points, vertices_points, order)) 
		if (Surgery_2D_22Flip_Preferable(m, cells_points, vertices_points, order)) {
                  Surgery_2D_22Flip(m, cells_points, vertices_points, order, 0);
                }
            }
          }
        } else if (s->nJoin1(line)->size() == 2) { //quick error check; njoin1 of this thing in the mesh-to-be-improved should be of cardinality 2
          if (s->nJoin1(line)->size() == 2) {
            Surgery_2D_22Flip_Setup(m, *n_iter, *v_iter, cells_points, vertices_points, order);
            if (Surgery_2D_22Flip_Possible(m, cells_points, vertices_points, order)) 
              if (Surgery_2D_22Flip_Preferable(m, cells_points, vertices_points, order)) {
                Surgery_2D_22Flip(m, cells_points, vertices_points, order, 0);
              }
          }
        }
        n_iter++;
      }
      v_iter++;
    }
  }

  void Surgery_3D_Improve_Mesh(Obj<Mesh> m, Obj<Mesh> bound_m) {
    //can be done much better than removal, but without removal its pointless
    //WHAT TO WRITE:  for each doublet check if the average curvatures of its corners would be improved if it were flipped to a 3-lens
    throw Exception("3D Mesh Improvement not implemented.");
  }

  Obj<Mesh> Surgery_2D_Coarsen_Mesh(Obj<Mesh> m, Obj<Mesh::sieve_type::supportSet> kept_vertices, Obj<Mesh> b_m = PETSC_NULL) {
    //uninterpolate a new copy of the mesh
    int dim = m->getDimension();
    int depth = m->depth();
    Obj<Mesh> new_mesh = new Mesh(m->comm(), dim, m->debug());
    Obj<Mesh::sieve_type> new_sieve = new Mesh::sieve_type(m->comm(), m->debug());
    new_mesh->setSieve(new_sieve);
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh::label_sequence> vertices = m->depthStratum(0);
    Mesh::label_sequence::iterator v_iter = vertices->begin();
    Mesh::label_sequence::iterator v_iter_end = vertices->end();
    while (v_iter != v_iter_end) {
      Obj<Mesh::sieve_type::supportArray> cells = s->nSupport(*v_iter, depth);
      Mesh::sieve_type::coneArray::iterator c_iter = cells->begin();
      Mesh::sieve_type::coneArray::iterator c_iter_end = cells->end();
      while (c_iter != c_iter_end) {
        new_sieve->addArrow(*v_iter, *c_iter);
        c_iter++;
      }
      v_iter++;
    }
    //coarsen the mesh
    new_mesh->setRealSection("coordinates", m->getRealSection("coordinates"));
    new_mesh->stratify(); 
    Surgery_Remove_AllButVertexSet(new_mesh, kept_vertices, b_m, 0, 2);
    //improve the mesh
   Surgery_2D_Improve_Mesh(new_mesh, b_m);
    return new_mesh;
  }

  Obj<Mesh> Surgery_1D_Coarsen_Mesh(Obj<Mesh> m, Obj<Mesh::sieve_type::supportSet> kept_vertices) {
    //simpler case; just remove all vertices not in kept_vertices through contraction to a neighbor
    Obj<Mesh::label_sequence> vertices = m->depthStratum(0);
    Mesh::sieve_type::supportSet vertices_to_remove;
    Obj<Mesh::sieve_type> s = m->getSieve();
    Obj<Mesh> new_mesh = new Mesh(m->comm(), m->getDimension(), m->debug());
    Obj<Mesh::sieve_type> new_sieve = new Mesh::sieve_type(m->comm(), m->debug());
    new_mesh->setSieve(new_sieve);
    Mesh::sieve_type::supportSet edges_to_remove;
    Mesh::label_sequence::iterator v_iter = vertices->begin();
    Mesh::label_sequence::iterator v_iter_end = vertices->end();
    Mesh::sieve_type::supportSet::iterator kv_end = kept_vertices->end();
    while (v_iter != v_iter_end) {
      //copy over the old mesh
      ALE::Obj<ALE::Mesh::sieve_type::supportSequence> lines = s->support(*v_iter);
      ALE::Mesh::sieve_type::supportSequence::iterator l_iter = lines->begin();
      ALE::Mesh::sieve_type::supportSequence::iterator l_iter_end = lines->end();
      while (l_iter != l_iter_end) {
        new_sieve->addArrow(*v_iter, *l_iter);
        l_iter++;
      }
      if (kept_vertices->find(*v_iter) == kv_end) { //invert the kept set
        vertices_to_remove.insert(*v_iter);
      }
      v_iter++;
    }
    Mesh::sieve_type::supportSet::iterator vtr_iter = vertices_to_remove.begin();
    Mesh::sieve_type::supportSet::iterator vtr_iter_end = vertices_to_remove.end();
    while (vtr_iter != vtr_iter_end) {
	Obj<Mesh::sieve_type::supportSequence> v_support = new_sieve->support(*vtr_iter);
	Mesh::sieve_type::supportSequence::iterator vs_iter = v_support->begin();
	Mesh::sieve_type::supportSequence::iterator vs_iter_end = v_support->end();
        if (v_support->size() != 2) {
          //leave this vertex alone
        } else {
	  Mesh::point_type involved_edges[2];
	  Mesh::point_type involved_vertices[2];
	  int involved_index = 0;
	  while (vs_iter != vs_iter_end) {
	    involved_edges[involved_index] = *vs_iter;
	    Obj<Mesh::sieve_type::coneSequence> s_cone = new_sieve->cone(*vs_iter);
	    Mesh::sieve_type::coneSequence::iterator sc_iter = s_cone->begin();
	    Mesh::sieve_type::coneSequence::iterator sc_iter_end = s_cone->end();
	    if (s_cone->size() != 2) throw Exception("Bad edge in 1D Coarsen");
	    while (sc_iter != sc_iter_end) {
	      if (*sc_iter != *vtr_iter) involved_vertices[involved_index] = *sc_iter;
	      sc_iter++;
	    }
	    involved_index++;
	    vs_iter++;
	  }
	  new_sieve->removeBasePoint(involved_edges[0]);
	  new_sieve->removeCapPoint(*vtr_iter);
	  new_sieve->addArrow(involved_vertices[0], involved_edges[1]);
	  
	    /*
	      Mesh::sieve_type::coneSet::iterator n_iter = neighbors->begin();
	      Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end(); 
	      while (n_iter != n_iter_end && neighbor == *vtr_iter) {
	      neighbor = *n_iter;
	      n_iter++;
	      }
	      //next, contract all edges to this neighbor
	      edges_to_remove.clear();
	      while (vs_iter != vs_iter_end) {
	      new_sieve->addArrow(neighbor, *vs_iter);
	      if(new_sieve->cone(*vs_iter)->size() < 3) edges_to_remove.insert(*vs_iter);
	      vs_iter++;
	      }
	      //next, remove the vertex and the edges that should be removed
	      new_sieve->removeBasePoint(*vtr_iter);
	      new_sieve->removeCapPoint(*vtr_iter);
	      Mesh::sieve_type::supportSet::iterator etr_iter = edges_to_remove.begin();
	      Mesh::sieve_type::supportSet::iterator etr_iter_end = edges_to_remove.end();
	      while (etr_iter != etr_iter_end) {
	      new_sieve->removeBasePoint(*etr_iter);
	      new_sieve->removeCapPoint(*etr_iter);
	      etr_iter++;
	      }
	    */
        }
	vtr_iter++;
    }
    new_mesh->stratify();
    new_mesh->setRealSection("coordinates", m->getRealSection("coordinates"));
    return new_mesh;
  }
}

PetscTruth Surgery_2D_CheckConsistency(Obj<Mesh> m) {

  //debugging function for all of this in order to allow 
  return PETSC_FALSE;
}
