#ifndef included_ALE_CoSieve_hh
#define included_ALE_CoSieve_hh

#ifndef  included_ALE_Sieve_hh
#include <sieve/Sieve.hh>
#endif

namespace ALE {
  // A Topology is a collection of Sieves
  //   Each Sieve has a label, which we call a \emph{patch}
  //   The collection itself we call a \emph{sheaf}
  //   The main operation we provide in Topology is the creation of a \emph{label}
  //     A label is a bidirectional mapping of Sieve points to integers, implemented with a Sifter
  template<typename Patch_, typename Sieve_, typename Alloc_ = malloc_allocator<typename Sieve_::point_type> >
  class Topology : public ALE::ParallelObject {
  public:
    typedef Patch_                                                patch_type;
    typedef Sieve_                                                sieve_type;
    typedef Alloc_                                                alloc_type;
    typedef typename alloc_type::template rebind<int>::other      int_alloc_type;
    typedef typename sieve_type::point_type                       point_type;
    typedef typename alloc_type::template rebind<std::pair<const patch_type, Obj<sieve_type> > >::other       pair1_alloc_type;
    typedef typename std::map<patch_type, Obj<sieve_type>, std::less<patch_type>, pair1_alloc_type>           sheaf_type;
    typedef typename ALE::Sifter<int, point_type, int>                                                        patch_label_type;
    typedef typename alloc_type::template rebind<patch_label_type>::other                                     patch_label_alloc_type;
    typedef typename patch_label_alloc_type::pointer                                                          patch_label_ptr;
    typedef typename alloc_type::template rebind<std::pair<const patch_type, Obj<patch_label_type> > >::other pair2_alloc_type;
    typedef typename std::map<patch_type, Obj<patch_label_type>, std::less<patch_type>, pair2_alloc_type>     label_type;
    typedef typename alloc_type::template rebind<std::pair<const patch_type, int> >::other                    pair3_alloc_type;
    typedef typename std::map<patch_type, int, std::less<patch_type>, pair3_alloc_type>                       max_label_type;
    typedef typename alloc_type::template rebind<std::pair<const std::string, label_type> >::other            pair4_alloc_type;
    typedef typename std::map<const std::string, label_type, std::less<const std::string>, pair4_alloc_type>  labels_type;
    typedef typename patch_label_type::supportSequence                       label_sequence;
    typedef typename std::set<point_type, std::less<point_type>, alloc_type> point_set_type;
    typedef typename alloc_type::template rebind<point_set_type>::other      point_set_alloc_type;
    typedef typename point_set_alloc_type::pointer                           point_set_ptr;
    typedef typename ALE::Sifter<int,point_type,point_type>                  send_overlap_type;
    typedef typename alloc_type::template rebind<send_overlap_type>::other   send_overlap_alloc_type;
    typedef typename send_overlap_alloc_type::pointer                        send_overlap_ptr;
    typedef typename ALE::Sifter<point_type,int,point_type>                  recv_overlap_type;
    typedef typename alloc_type::template rebind<recv_overlap_type>::other   recv_overlap_alloc_type;
    typedef typename recv_overlap_alloc_type::pointer                        recv_overlap_ptr;
  protected:
    sheaf_type     _sheaf;
    labels_type    _labels;
    int            _maxHeight;
    max_label_type _maxHeights;
    int            _maxDepth;
    max_label_type _maxDepths;
    bool           _calculatedOverlap;
    Obj<send_overlap_type> _sendOverlap;
    Obj<recv_overlap_type> _recvOverlap;
    Obj<send_overlap_type> _distSendOverlap;
    Obj<recv_overlap_type> _distRecvOverlap;
    // Work space
    Obj<point_set_type>    _modifiedPoints;
    alloc_type             _allocator;
    int_alloc_type         _int_allocator;
  public:
    Topology(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _maxHeight(-1), _maxDepth(-1), _calculatedOverlap(false) {
      send_overlap_ptr pSendOverlap = send_overlap_alloc_type(this->_allocator).allocate(1);
      send_overlap_alloc_type(this->_allocator).construct(pSendOverlap, send_overlap_type(this->comm(), this->debug()));
      this->_sendOverlap = Obj<send_overlap_type>(pSendOverlap, sizeof(send_overlap_type));
      ///this->_sendOverlap    = new send_overlap_type(this->comm(), this->debug());
      recv_overlap_ptr pRecvOverlap = recv_overlap_alloc_type(this->_allocator).allocate(1);
      recv_overlap_alloc_type(this->_allocator).construct(pRecvOverlap, recv_overlap_type(this->comm(), this->debug()));
      this->_recvOverlap = Obj<recv_overlap_type>(pRecvOverlap, sizeof(recv_overlap_type));
      ///this->_recvOverlap    = new recv_overlap_type(this->comm(), this->debug());
      point_set_ptr pModPoints = point_set_alloc_type(this->_allocator).allocate(1);
      point_set_alloc_type(this->_allocator).construct(pModPoints, point_set_type());
      this->_modifiedPoints = Obj<point_set_type>(pModPoints, sizeof(point_set_type));
      ///this->_modifiedPoints = new point_set_type();
    };
    virtual ~Topology() {};
  public: // Verifiers
    void checkPatch(const patch_type& patch) {
      if (this->_sheaf.find(patch) == this->_sheaf.end()) {
        ostringstream msg;
        msg << "Invalid topology patch: " << patch << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
    void checkLabel(const std::string& name, const patch_type& patch) {
      this->checkPatch(patch);
      if ((this->_labels.find(name) == this->_labels.end()) || (this->_labels[name].find(patch) == this->_labels[name].end())) {
        ostringstream msg;
        msg << "Invalid label name: " << name << " for patch " << patch << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
    bool hasPatch(const patch_type& patch) {
      if (this->_sheaf.find(patch) != this->_sheaf.end()) {
        return true;
      }
      return false;
    };
    bool hasLabel(const std::string& name, const patch_type& patch) {
      if ((this->_labels.find(name) != this->_labels.end()) && (this->_labels[name].find(patch) != this->_labels[name].end())) {
        return true;
      }
      return false;
    };
  public: // Accessors
    const Obj<sieve_type>& getPatch(const patch_type& patch) {
      this->checkPatch(patch);
      return this->_sheaf[patch];
    };
    void setPatch(const patch_type& patch, const Obj<sieve_type>& sieve) {
      this->_sheaf[patch] = sieve;
    };
    int getValue (const Obj<patch_label_type>& label, const point_type& point, const int defValue = 0) {
      const Obj<typename patch_label_type::coneSequence>& cone = label->cone(point);

      if (cone->size() == 0) return defValue;
      return *cone->begin();
    };
    template<typename InputPoints>
    int getMaxValue (const Obj<patch_label_type>& label, const Obj<InputPoints>& points, const int defValue = 0) {
      int maxValue = defValue;

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        maxValue = std::max(maxValue, this->getValue(label, *p_iter, defValue));
      }
      return maxValue;
    }
    void setValue(const Obj<patch_label_type>& label, const point_type& point, const int value) {
      label->setCone(value, point);
    };
    const Obj<patch_label_type>& createLabel(const patch_type& patch, const std::string& name) {
      this->checkPatch(patch);
      ///patch_label_ptr pLabel = patch_label_alloc_type(this->_allocator).allocate(1);
      ///patch_label_alloc_type(this->_allocator).construct(pLabel, patch_label_type(this->comm(), this->debug()));
      ///this->_labels[name][patch] = Obj<patch_label_type>(pLabel, sizeof(patch_label_type));
      this->_labels[name][patch] = new patch_label_type(this->comm(), this->debug());
      return this->_labels[name][patch];
    };
    const Obj<patch_label_type>& getLabel(const patch_type& patch, const std::string& name) {
      this->checkLabel(name, patch);
      return this->_labels[name][patch];
    };
    const Obj<label_sequence>& getLabelStratum(const patch_type& patch, const std::string& name, int value) {
      this->checkLabel(name, patch);
      return this->_labels[name][patch]->support(value);
    };
    const sheaf_type& getPatches() {
      return this->_sheaf;
    };
    const labels_type& getLabels() {
      return this->_labels;
    };
    void clear() {
      this->_sheaf.clear();
      this->_labels.clear();
      this->_maxHeight = -1;
      this->_maxHeights.clear();
      this->_maxDepth = -1;
      this->_maxDepths.clear();
    };
    const Obj<send_overlap_type>& getSendOverlap() const {return this->_sendOverlap;};
    void setSendOverlap(const Obj<send_overlap_type>& overlap) {this->_sendOverlap = overlap;};
    const Obj<recv_overlap_type>& getRecvOverlap() const {return this->_recvOverlap;};
    void setRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_recvOverlap = overlap;};
    const Obj<send_overlap_type>& getDistSendOverlap() const {return this->_distSendOverlap;};
    void setDistSendOverlap(const Obj<send_overlap_type>& overlap) {this->_distSendOverlap = overlap;};
    const Obj<recv_overlap_type>& getDistRecvOverlap() const {return this->_distRecvOverlap;};
    void setDistRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_distRecvOverlap = overlap;};
  public: // Stratification
    template<class InputPoints>
    void computeHeight(const Obj<patch_label_type>& height, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points, int& maxHeight) {
      this->_modifiedPoints->clear();

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        // Compute the max height of the points in the support of p, and add 1
        int h0 = this->getValue(height, *p_iter, -1);
        int h1 = this->getMaxValue(height, sieve->support(*p_iter), -1) + 1;

        if(h1 != h0) {
          this->setValue(height, *p_iter, h1);
          if (h1 > maxHeight) maxHeight = h1;
          this->_modifiedPoints->insert(*p_iter);
        }
      }
      // FIX: We would like to avoid the copy here with cone()
      if(this->_modifiedPoints->size() > 0) {
        this->computeHeight(height, sieve, sieve->cone(this->_modifiedPoints), maxHeight);
      }
    }
    void computeHeights() {
      const std::string name("height");

      this->_maxHeight = -1;
      for(typename sheaf_type::iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
        const Obj<patch_label_type>& label = this->createLabel(s_iter->first, name);

        this->_maxHeights[s_iter->first] = -1;
        this->computeHeight(label, s_iter->second, s_iter->second->leaves(), this->_maxHeights[s_iter->first]);
        if (this->_maxHeights[s_iter->first] > this->_maxHeight) this->_maxHeight = this->_maxHeights[s_iter->first];
      }
    }
    int height() const {return this->_maxHeight;};
    int height(const patch_type& patch) {
      this->checkPatch(patch);
      return this->_maxHeights[patch];
    }
    int height(const patch_type& patch, const point_type& point) {
      return this->getValue(this->_labels["height"][patch], point, -1);
    }
    const Obj<label_sequence>& heightStratum(const patch_type& patch, int height) {
      return this->getLabelStratum(patch, "height", height);
    }
    template<class InputPoints>
    void computeDepth(const Obj<patch_label_type>& depth, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points, int& maxDepth) {
      this->_modifiedPoints->clear();

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        // Compute the max depth of the points in the cone of p, and add 1
        int d0 = this->getValue(depth, *p_iter, -1);
        int d1 = this->getMaxValue(depth, sieve->cone(*p_iter), -1) + 1;

        if(d1 != d0) {
          this->setValue(depth, *p_iter, d1);
          if (d1 > maxDepth) maxDepth = d1;
          this->_modifiedPoints->insert(*p_iter);
        }
      }
      // FIX: We would like to avoid the copy here with support()
      if(this->_modifiedPoints->size() > 0) {
        this->computeDepth(depth, sieve, sieve->support(this->_modifiedPoints), maxDepth);
      }
    }
    void computeDepths() {
      const std::string name("depth");

      this->_maxDepth = -1;
      for(typename sheaf_type::iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
        const Obj<patch_label_type>& label = this->createLabel(s_iter->first, name);

        this->_maxDepths[s_iter->first] = -1;
        this->computeDepth(label, s_iter->second, s_iter->second->roots(), this->_maxDepths[s_iter->first]);
        if (this->_maxDepths[s_iter->first] > this->_maxDepth) this->_maxDepth = this->_maxDepths[s_iter->first];
      }
    }
    int depth() const {return this->_maxDepth;};
    int depth(const patch_type& patch) {
      this->checkPatch(patch);
      return this->_maxDepths[patch];
    }
    int depth(const patch_type& patch, const point_type& point) {
      return this->getValue(this->_labels["depth"][patch], point, -1);
    }
    const Obj<label_sequence>& depthStratum(const patch_type& patch, int depth) {
      return this->getLabelStratum(patch, "depth", depth);
    }
#undef __FUNCT__
#define __FUNCT__ "Topology::stratify"
    void stratify() {
      ALE_LOG_EVENT_BEGIN;
      this->computeHeights();
      this->computeDepths();
      ALE_LOG_EVENT_END;
    }
  public: // Viewers
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a Topology\n");
      } else {
        PetscPrintf(comm, "viewing Topology '%s'\n", name.c_str());
      }
      PetscPrintf(comm, "  maximum height %d maximum depth %d\n", this->height(), this->depth());
      for(typename sheaf_type::const_iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
        ostringstream txt;

        txt << "Patch " << s_iter->first;
        s_iter->second->view(txt.str().c_str(), comm);
        PetscPrintf(comm, "  maximum height %d maximum depth %d\n", this->height(s_iter->first), this->depth(s_iter->first));
      }
      for(typename labels_type::const_iterator l_iter = this->_labels.begin(); l_iter != this->_labels.end(); ++l_iter) {
        PetscPrintf(comm, "  label %s constructed\n", l_iter->first.c_str());
      }
    }
  public:
    void constructOverlap(const patch_type& patch) {
      if (this->_calculatedOverlap) return;
      if (this->hasPatch(patch)) {
        this->constructOverlap(this->getPatch(patch)->base(), this->_sendOverlap, this->_recvOverlap);
        this->constructOverlap(this->getPatch(patch)->cap(), this->_sendOverlap, this->_recvOverlap);
      }
      if (this->debug()) {
        this->_sendOverlap->view("Send overlap");
        this->_recvOverlap->view("Receive overlap");
      }
      this->_calculatedOverlap = true;
    }
    template<typename Sequence>
    void constructOverlap(const Obj<Sequence>& points, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
      point_type *sendBuf = this->_allocator.allocate(points->size());
      for(unsigned int i = 0; i < points->size(); ++i) {this->_allocator.construct(sendBuf+i, point_type());}
      ///point_type *sendBuf = new point_type[points->size()];
      int         size    = 0;
      for(typename Sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
        sendBuf[size++] = *l_iter;
      }
      int *sizes   = this->_int_allocator.allocate(this->commSize()+1);   // The number of points coming from each process
      int *offsets = this->_int_allocator.allocate(this->commSize()+1); // Prefix sums for sizes
      int *oldOffs = this->_int_allocator.allocate(this->commSize()+1); // Temporary storage
      for(int i = 0; i < this->commSize()+1; ++i) {
        this->_int_allocator.construct(sizes+i,   0);
        this->_int_allocator.construct(offsets+i, 0);
        this->_int_allocator.construct(oldOffs+i, 0);
      }
      ///int *sizes   = new int[this->commSize()];
      ///int *offsets = new int[this->commSize()+1];
      ///int *oldOffs = new int[this->commSize()+1];
      point_type *remotePoints = NULL;            // The points from each process
      int        *remoteRanks  = NULL;            // The rank and number of overlap points of each process that overlaps another

      // Change to Allgather() for the correct binning algorithm
      MPI_Gather(&size, 1, MPI_INT, sizes, 1, MPI_INT, 0, this->comm());
      if (this->commRank() == 0) {
        offsets[0] = 0;
        for(int p = 1; p <= this->commSize(); p++) {
          offsets[p] = offsets[p-1] + sizes[p-1];
        }
        remotePoints = this->_allocator.allocate(offsets[this->commSize()]);
        for(int i = 0; i < offsets[this->commSize()]; ++i) {this->_allocator.construct(remotePoints+i, point_type());}
        ///remotePoints = new point_type[offsets[this->commSize()]];
      }
      MPI_Gatherv(sendBuf, size, MPI_INT, remotePoints, sizes, offsets, MPI_INT, 0, this->comm());
      std::map<int, std::map<int, std::set<point_type> > > overlapInfo; // Maps (p,q) to their set of overlap points

      if (this->commRank() == 0) {
        for(int p = 0; p < this->commSize(); p++) {
          std::sort(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]]);
        }
        for(int p = 0; p <= this->commSize(); p++) {
          oldOffs[p] = offsets[p];
        }
        for(int p = 0; p < this->commSize(); p++) {
          for(int q = p+1; q < this->commSize(); q++) {
            std::set_intersection(&remotePoints[oldOffs[p]], &remotePoints[oldOffs[p+1]],
                                  &remotePoints[oldOffs[q]], &remotePoints[oldOffs[q+1]],
                                  std::insert_iterator<std::set<point_type> >(overlapInfo[p][q], overlapInfo[p][q].begin()));
            overlapInfo[q][p] = overlapInfo[p][q];
          }
          sizes[p]     = overlapInfo[p].size()*2;
          offsets[p+1] = offsets[p] + sizes[p];
        }
        remoteRanks = this->_int_allocator.allocate(offsets[this->commSize()]);
        for(int i = 0; i < offsets[this->commSize()]; ++i) {this->_int_allocator.construct(remoteRanks+i, 0);}
        ///remoteRanks = new int[offsets[this->commSize()]];
        int       k = 0;
        for(int p = 0; p < this->commSize(); p++) {
          for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
            remoteRanks[k*2]   = r_iter->first;
            remoteRanks[k*2+1] = r_iter->second.size();
            k++;
          }
        }
      }
      int numOverlaps;                          // The number of processes overlapping this process
      MPI_Scatter(sizes, 1, MPI_INT, &numOverlaps, 1, MPI_INT, 0, this->comm());
      int *overlapRanks = this->_int_allocator.allocate(numOverlaps);
      for(int i = 0; i < numOverlaps; ++i) {this->_int_allocator.construct(overlapRanks+i, 0);}
      ///int *overlapRanks = new int[numOverlaps]; // The rank and overlap size for each overlapping process
      MPI_Scatterv(remoteRanks, sizes, offsets, MPI_INT, overlapRanks, numOverlaps, MPI_INT, 0, this->comm());
      point_type *sendPoints = NULL;            // The points to send to each process
      if (this->commRank() == 0) {
        for(int p = 0, k = 0; p < this->commSize(); p++) {
          sizes[p] = 0;
          for(int r = 0; r < (int) overlapInfo[p].size(); r++) {
            sizes[p] += remoteRanks[k*2+1];
            k++;
          }
          offsets[p+1] = offsets[p] + sizes[p];
        }
        sendPoints = this->_allocator.allocate(offsets[this->commSize()]);
        for(int i = 0; i < offsets[this->commSize()]; ++i) {this->_allocator.construct(sendPoints+i, point_type());}
        ///sendPoints = new point_type[offsets[this->commSize()]];
        for(int p = 0, k = 0; p < this->commSize(); p++) {
          for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
            int rank = r_iter->first;
            for(typename std::set<point_type>::iterator p_iter = (overlapInfo[p][rank]).begin(); p_iter != (overlapInfo[p][rank]).end(); ++p_iter) {
              sendPoints[k++] = *p_iter;
            }
          }
        }
      }
      int numOverlapPoints = 0;
      for(int r = 0; r < numOverlaps/2; r++) {
        numOverlapPoints += overlapRanks[r*2+1];
      }
      point_type *overlapPoints = this->_allocator.allocate(numOverlapPoints);
      for(int i = 0; i < numOverlapPoints; ++i) {this->_allocator.construct(overlapPoints+i, point_type());}
      ///point_type *overlapPoints = new point_type[numOverlapPoints];
      MPI_Scatterv(sendPoints, sizes, offsets, MPI_INT, overlapPoints, numOverlapPoints, MPI_INT, 0, this->comm());

      for(int r = 0, k = 0; r < numOverlaps/2; r++) {
        int rank = overlapRanks[r*2];

        for(int p = 0; p < overlapRanks[r*2+1]; p++) {
          point_type point = overlapPoints[k++];

          sendOverlap->addArrow(point, rank, point);
          recvOverlap->addArrow(rank, point, point);
        }
      }

      delete [] overlapPoints;
      delete [] overlapRanks;
      delete [] sizes;
      delete [] offsets;
      delete [] oldOffs;
      if (this->commRank() == 0) {
        delete [] remoteRanks;
        delete [] remotePoints;
        delete [] sendPoints;
      }
    }
  };

  // An Overlap is a Sifter describing the overlap of two Sieves
  //   Each arrow is local point ---(remote point)---> remote rank right now
  //     For XSifter, this should change to (local patch, local point) ---> (remote rank, remote patch, remote point)
}

#endif
