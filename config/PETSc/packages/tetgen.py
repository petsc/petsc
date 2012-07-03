import PETSc.package

structDecl = '''
#include <stdio.h> // Needed for FILE, NULL
class tetgenio {

  public:
    enum {FILENAMESIZE = 1024};
    enum {INPUTLINESIZE = 1024};

    typedef struct {
      int *vertexlist;
      int numberofvertices;
    } polygon;

    static void init(polygon* p) {
      p->vertexlist = (int *) NULL;
      p->numberofvertices = 0;
    }

    typedef struct {
      polygon *polygonlist;
      int numberofpolygons;
      double *holelist;
      int numberofholes;
    } facet;

    static void init(facet* f) {
      f->polygonlist = (polygon *) NULL;
      f->numberofpolygons = 0;
      f->holelist = (double *) NULL;
      f->numberofholes = 0;
    }

    typedef struct {
      int fmark1, fmark2;
      double transmat[4][4];
      int numberofpointpairs;
      int *pointpairlist;
    } pbcgroup;

  public:

    int firstnumber; 
    int mesh_dim;
    double *pointlist;
    double *pointattributelist;
    double *addpointlist;
    int *pointmarkerlist;
    int numberofpoints;
    int numberofpointattributes;
    int numberofaddpoints;
    int *tetrahedronlist;
    double *tetrahedronattributelist;
    double *tetrahedronvolumelist;
    int *neighborlist;
    int numberoftetrahedra;
    int numberofcorners;
    int numberoftetrahedronattributes;
    facet *facetlist;
    int *facetmarkerlist;
    int numberoffacets;
    double *holelist;
    int numberofholes;
    double *regionlist;
    int numberofregions;
    double *facetconstraintlist;
    int numberoffacetconstraints;
    double *segmentconstraintlist;
    int numberofsegmentconstraints;
    double *nodeconstraintlist;
    int numberofnodeconstraints;
    pbcgroup *pbcgrouplist;
    int numberofpbcgroups;
    int *trifacelist;
    int *trifacemarkerlist;
    int numberoftrifaces;
    int *edgelist;
    int *edgemarkerlist;
    int numberofedges;

  public:
    void initialize();
    void deinitialize();
    bool load_node_call(FILE* infile, int markers, char* nodefilename);
    bool load_node(char* filename);
    bool load_addnodes(char* filename);
    bool load_pbc(char* filename);
    bool load_var(char* filename);
    bool load_poly(char* filename);
    bool load_off(char* filename);
    bool load_ply(char* filename);
    bool load_stl(char* filename);
    bool load_medit(char* filename);
    bool load_plc(char* filename, int object);
    bool load_tetmesh(char* filename);
    void save_nodes(char* filename);
    void save_elements(char* filename);
    void save_faces(char* filename);
    void save_edges(char* filename);
    void save_neighbors(char* filename);
    void save_poly(char* filename);
    char *readline(char* string, FILE* infile, int *linenumber);
    char *findnextfield(char* string);
    char *readnumberline(char* string, FILE* infile, char* infilename);
    char *findnextnumber(char* string);
    tetgenio() {initialize();}
    ~tetgenio() {deinitialize();}
};
'''

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://tetgen.berlios.de/files/tetgen1.4.3.tar.gz']
    self.functions = ['tetrahedralize']
    self.functionsCxx = [1, structDecl+'void tetrahedralize(char *switches, tetgenio *in, tetgenio *out, tetgenio *addin = NULL, tetgenio *bgmin = NULL);', 'tetrahedralize((char *) "", NULL, NULL)']
    self.includes  = ['tetgen.h']
    self.liblist   = [['libtet.a']]
    self.cxx       = 1
    self.needsMath = 1
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
    self.make            = framework.require('config.programs', self)
    self.deps            = []
    return

  def Install(self):
    import os, sys
    import config.base
    import fileinput

    libDir         = os.path.join(self.installDir, 'lib')
    includeDir     = os.path.join(self.installDir, 'include')
    makeinc        = os.path.join(self.packageDir, 'make.inc')
    configheader   = os.path.join(self.packageDir, 'configureheader.h')

    # This make.inc stuff is completely unnecessary for compiling TetGen. It is
    # just here for comparing different PETSC_ARCH's
    self.setCompilers.pushLanguage('C++')
    g = open(makeinc,'w')
    g.write('SHELL            = '+self.programs.SHELL+'\n')
    g.write('CP               = '+self.programs.cp+'\n')
    g.write('RM               = '+self.programs.RM+'\n')
    g.write('MKDIR            = '+self.programs.mkdir+'\n')
    g.write('OMAKE            = '+self.make.make+' '+self.make.flags+'\n')

    g.write('CLINKER          = '+self.setCompilers.getLinker()+'\n')
    g.write('AR               = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS          = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX    = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB           = '+self.setCompilers.RANLIB+'\n')
    g.write('SL_LINKER_SUFFIX = '+self.setCompilers.sharedLibraryExt+'\n')

    g.write('TETGEN_ROOT      = '+self.packageDir+'\n')
    g.write('PREFIX           = '+self.installDir+'\n')
    g.write('LIBDIR           = '+libDir+'\n')
    g.write('INSTALL_LIB_DIR  = '+libDir+'\n')
    g.write('TETGENLIB        = $(LIBDIR)/libtet.$(AR_LIB_SUFFIX)\n')
    g.write('SHLIB            = libtet\n')

    cflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
    cflags += ' '+self.headers.toString('.')
    cflags += ' -fPIC'
    predcflags = '-O0 -fPIC'    # Need to compile without optimization

    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS         = '+cflags+'\n')
    g.write('PREDCXXFLAGS   = '+predcflags+'\n')
    self.setCompilers.popLanguage()
    g.close()

    # I'd rather have this than to completely fork TetGen
    new_file = open(os.path.join(self.packageDir, 'tetgen_def.h'),'w')
    old_file = open(os.path.join(self.packageDir, 'tetgen.h'))
    for line in old_file:
      new_file.write(line.replace('// #define TETLIBRARY', '#define TETLIBRARY'))
    new_file.close()
    old_file.close()

    # Now compile & install
    if self.installNeeded('make.inc'):
      self.framework.outputHeader(configheader)
      try:
        self.logPrintBox('Compiling & installing TetGen; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make CXX="'+ self.setCompilers.getCompiler() + '" CXXFLAGS="' + cflags + '" PREDCXXFLAGS="' + predcflags + '" tetlib && mv tetgen_def.h tetgen.h && cp *.a ' + libDir + ' && rm *.a *.o', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on TetGen: '+str(e))
      output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cp -f '+os.path.join(self.packageDir, 'tetgen.h')+' '+includeDir, timeout=5, log = self.framework.log)
      self.postInstall(output1+err1+output2+err2,'make.inc')

    return self.installDir
