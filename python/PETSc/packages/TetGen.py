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

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['hg://www.mcs.anl.gov/petsc/tetgen-dev','ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/TetGen.tar.gz']
    self.functions = ['tetrahedralize']
    self.functionsCxx = [1, structDecl+'void tetrahedralize(char *switches, tetgenio *in, tetgenio *out);', 'tetrahedralize("", NULL, NULL)']
    self.includes  = ['tetgen.h']
    self.liblist   = [['libtetgen.a']]
    self.cxx       = 1
    self.needsMath = 1
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def InstallOld(self):
    import os, sys
    tetgenDir = self.getDir()
    installDir = os.path.join(tetgenDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # We could make a check of the md5 of the current configure framework
    self.logPrintBox('Configuring and compiling TetGen; this may take several minutes')
    try:
      import cPickle
      import logger
      # Split Graphs into its own repository
      oldDir = os.getcwd()
      os.chdir(tetgenDir)
      oldLog = logger.Logger.defaultLog
      logger.Logger.defaultLog = file(os.path.join(tetgenDir, 'build.log'), 'w')
      mod  = self.getModule(tetgenDir, 'make')
      make = mod.Make(configureParent = cPickle.loads(cPickle.dumps(self.framework)),module = mod)
      make.prefix = installDir
      make.framework.argDB['with-petsc'] = 1
      make.builder.argDB['ignoreCompileOutput'] = 1
      make.run()
      del sys.modules['make']
      logger.Logger.defaultLog = oldLog
      os.chdir(oldDir)
    except RuntimeError, e:
      raise RuntimeError('Error running configure on TetGen: '+str(e))
    self.framework.actions.addArgument('TetGen', 'Install', 'Installed TetGen into '+installDir)
    return tetgenDir

  def Install(self):
    import os, sys
    # Get the ParMetis directories
    tetgenDir      = self.getDir()
    installDir     = os.path.join(tetgenDir, self.arch.arch)
    libDir         = os.path.join(installDir, 'lib')
    includeDir     = os.path.join(installDir, 'include')
    makeinc        = os.path.join(tetgenDir, 'make.inc')
    installmakeinc = os.path.join(installDir, 'make.inc')
    configheader   = os.path.join(tetgenDir, 'configureheader.h')

    # Configure ParMetis 
    if os.path.isfile(makeinc):
      os.unlink(makeinc)
    g = open(makeinc,'w')
    g.write('SHELL          = '+self.programs.SHELL+'\n')
    g.write('CP             = '+self.programs.cp+'\n')
    g.write('RM             = '+self.programs.RM+'\n')
    g.write('MKDIR          = '+self.programs.mkdir+'\n')

    g.write('AR             = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS        = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX  = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB         = '+self.setCompilers.RANLIB+'\n')

    g.write('TETGEN_ROOT    = '+tetgenDir+'\n')
    g.write('PREFIX         = '+installDir+'\n')
    g.write('LIBDIR         = '+libDir+'\n')
    g.write('TETGENLIB      = $(LIBDIR)/libtetgen.$(AR_LIB_SUFFIX)\n')
    
    self.setCompilers.pushLanguage('C')
    cflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
    cflags += ' '+self.headers.toString('.')
        
    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS         = '+cflags)
    self.setCompilers.popLanguage()
    g.close()

    # Now compile & install
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isdir(libDir):
      os.mkdir(libDir)
    if not os.path.isdir(includeDir):
      os.mkdir(includeDir)
    
    if not os.path.isfile(installmakeinc) or not (self.getChecksum(installmakeinc) == self.getChecksum(makeinc)):
      self.framework.log.write('Have to rebuild TetGen, make.inc != '+installmakeinc+'\n')
      self.framework.outputHeader(configheader)
      try:
        import config.base
        self.logPrintBox('Compiling & installing TetGen; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+tetgenDir+'; make clean; make tetlib; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on TetGen: '+str(e))
    else:
      self.framework.log.write('Did not need to compile downloaded TetGen\n')
    output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(tetgenDir, 'tetgen.h')+' '+includeDir, timeout=5, log = self.framework.log)[0]
    output  = config.base.Configure.executeShellCommand('cp -f '+makeinc+' '+installDir, timeout=5, log = self.framework.log)[0]
    self.framework.actions.addArgument('TetGen', 'Install', 'Installed TetGen into '+installDir)

    return self.getDir()
