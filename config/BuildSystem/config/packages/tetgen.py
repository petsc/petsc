import config.package

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

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download     = ['http://www.tetgen.org/1.5/src/tetgen1.5.1.tar.gz',
                         'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/tetgen1.5.1.tar.gz']
    self.functions    = ['tetrahedralize']
    self.functionsCxx = [1, structDecl+'void tetrahedralize(char *switches, tetgenio *in, tetgenio *out, tetgenio *addin = NULL, tetgenio *bgmin = NULL);', 'tetrahedralize((char *) "", NULL, NULL)']
    self.includes     = ['tetgen.h']
    self.liblist      = [['libtet.a']]
    self.cxx          = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.languages       = framework.require('PETSc.options.languages',   self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.mathlib]
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
    g.write('OMAKE            = '+self.make.make+' '+self.make.noprintdirflag+'\n')

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

    cflags = self.removeWarningFlags(self.setCompilers.getCompilerFlags())
    cflags += ' '+self.headers.toString('.')
    cflags += ' -fPIC'
    cflags += ' -DTETLIBRARY'
    predcflags = '-O0 -fPIC'    # Need to compile without optimization

    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS         = '+cflags+'\n')
    g.write('PREDCXXFLAGS   = '+predcflags+'\n')
    g.close()

    # Now compile & install
    if self.installNeeded('make.inc'):
      self.framework.outputHeader(configheader)
      try:
        self.logPrintBox('Compiling & installing TetGen; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.log)
        output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make CXX="'+ self.setCompilers.getCompiler() + '" CXXFLAGS="' + cflags + '" PREDCXXFLAGS="' + predcflags + '" tetlib && '+self.installSudo+'cp *.a ' + libDir + ' && rm *.a *.o', timeout=2500, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on TetGen: '+str(e))
      output2,err2,ret2  = config.package.Package.executeShellCommand(self.installSudo+'cp -f '+os.path.join(self.packageDir, 'tetgen.h')+' '+includeDir, timeout=60, log = self.log)
      self.postInstall(output1+err1+output2+err2,'make.inc')

    self.setCompilers.popLanguage()
    return self.installDir

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    include = '#include <tetgen.h>'
    body = \
'''
  char args[] = "";
  tetgenio in,out;
  tetrahedralize(args, &in, &out);
'''
    self.pushLanguage('Cxx')
    oldFlags = self.compilers.CXXPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CXXPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    if not self.checkCompile(include,body):
      self.compilers.CXXPPFLAGS += ' -DTETLIBRARY'
      if self.checkCompile(include,body):
        self.addDefine('HAVE_TETGEN_TETLIBRARY_NEEDED',1)
      else:
        raise RuntimeError('Unable to compile with TetGen')
    if not self.checkLink(include,body):
      raise RuntimeError('Unable to link TetGen')
    self.compilers.CXXPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    self.popLanguage()
