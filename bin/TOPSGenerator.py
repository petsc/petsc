#!/usr/bin/env python
"""
   Generates the code needed to connect an application code 
to the TOPS Solver Components
"""

from TOPSInstaller import *


if __name__ == '__main__':
        args = []
        title = "TOPS Code Generator"
        message1 = """SciDAC TOPS Code Generator"""
        message2 = """The DOE Mathematics SciDAC TOPS ISIC develops software for the
scalable solution of optimizations, eigenvalues and algebraic systems. More
information on TOPS may be found at http://www.tops-scidac.org. \n\n
This tool allows you to define the class of algebraic problem you are solving
and generates the appropriate "glue" code needed to use the TOPS Solver Components"""

	result = buttonbox(message=message1, title=title, choices = ["Cancel", "Continue"],fontSize = 20,message2=message2)
        if result == "Cancel": sys.exit()

	dim = int(buttonbox(message="Dimension of the grid?", title=title, choices = ["1", "2", "3"],fontSize = 20))

        lin = buttonbox(message="Type of algebraic problem?", title=title, choices = ["linear", "nonlinear"],fontSize = 20)
        if lin == "nonlinear":
          jac = buttonbox(message="Will you provide analytic Jacobian?", title=title, choices = ["No", "Yes"],fontSize = 20,message2="Otherwise it will be computed via finite differencing")
        
        ig = buttonbox(message="Will you provide an initial guess?", title=title, choices = ["No","Yes"],fontSize = 20)

        dof = int(enterbox("Number of degrees of freedom per grid point?",title,argDefaultText="1"))

        grid = buttonbox(message="Type of grid?", title=title, choices = ["logically rectangular", "unstructured"],fontSize = 20)
        if grid == "logically rectangular":
          uns = "S"
          staggered = buttonbox(message="Are you using a staggered grid?", title=title, choices = ["No","Yes"],fontSize = 20)
        else:
          uns = "Uns"
 
        app = enterbox("Name of application?",title)

        # generate the sidl
        import os.path
        import sys
        if not os.path.isdir(app): os.mkdir(app)
        f = file(os.path.join(app,app+'.sidl'), 'w')
        f.write('package '+app+' version 0.0.0 {\n')
        f.write('  class System implements-all TOPS.System.System,gov.cca.Component, gov.cca.ports.GoPort,\n')
        f.write('TOPS.System.Initialize.Once, TOPS.System.Initialize.EverySolve,\n')
        if lin == 'nonlinear':
          f.write('TOPS.System.Compute.Residual')
          if jac == 'yes':
            f.write(',TOPS.System.Compute.Jacobian')
        else: 
          f.write('TOPS.System.Compute.RightHandSide,')
          f.write('TOPS.System.Compute.Matrix')
        if ig == 'yes':
          f.write(',TOPS.System.Compute.InitialGuess')
        f.write('{}\n}\n')
        f.close()

        f = file(os.path.join(app,'makefile'),'w')
        f.write('EXNAME      = '+app+'\n')
        f.write('SIDL        = ${EXNAME}.sidl\n')
        f.write('TLIBNAME    = lib${EXNAME}\n')
        f.write('SIDLEXCLUDE = -e TOPS[\.a-zA-Z_]* -e gov.cca[\._a-zA-Z_]*\n')
        f.write('TOPSCLIENT_LIB = -L${PETSC_LIB_DIR} ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -ltopsclient-c++\n')
        f.write('DIRS        = server/c++\n')
        f.write('include ${PETSC_DIR}/src/tops/makefile.rules\n')
        f.close()

        result = buttonbox(message="Ready to generate code", title=title, choices = ["Continue"],fontSize = 20,message2="This may take several minutes.")

        import commands
        (status,out) = commands.getstatusoutput('cd '+app+';make server/c++/obj/makefile')
        if status:
          result = buttonbox(message="SIDL code generation failed", title=title, choices = ["Ok"],fontSize = 20,message2=out)


        # Add the common code 
        f = file(os.path.join(app,'server','c++',app+'_System_Impl.cc'),'r')
        text = f.read()
        f.close()

        bscode = 'this->solver = (TOPS::'+uns+'tructured::Solver)solver;'
        text = text.replace('begin('+app+'.System.setSolver)','begin('+app+'.System.setSolver)\n'+bscode)
 
        if dof > 1:
          bscode = '  this->solver.setBlockSize('+str(dof)+');'
          text = text.replace('begin('+app+'.System.initializeOnce)','begin('+app+'.System.initializeOnce)\n'+bscode)

        bscode = '''TOPS::'''+uns+'''tructured::Solver solver = this->solver;'''
        if grid == 'logically rectangular':
          if dof > 1: sct = 1
          else: sct = 0
          bscode = bscode + ''' 
            int xs = x.lower('''+str(sct)+''');      
            int xm = x.length('''+str(sct)+''') - 1;'''
          if dim > 1:
            bscode = bscode + '''
            int ys = x.lower('''+str(sct+1)+''');
            int ym = x.length('''+str(sct+1)+''') - 1;\n'''
          if dim > 2:
            bscode = bscode + '''
            int zs = x.lower('''+str(sct+2)+''');
            int zm = x.length('''+str(sct+2)+''') - 1;\n'''
          if dim > 2:
            bscode = bscode + '''for (int k=zs; k<zs+km; k++) {\n'''
          if dim > 1:
            bscode = bscode + '''for (int j=ys; j<ys+ym; j++) {\n'''
          bscode = bscode + '''
            for (int i=xs; i<xs+xm; i++) {
            }'''
          if dim > 2:
            bscode = bscode + '''}\n'''
          if dim > 1:
            bscode = bscode + '''}\n'''
        if lin == 'nonlinear':
          text = text.replace('begin('+app+'.System.computeResidual)','begin('+app+'.System.computeResidual)\n'+bscode)
          if jac == 'Yes':
            text = text.replace('begin('+app+'.System.Jacobian)','begin('+app+'.System.Jacobian)\n'+bscode)
        else:
          text = text.replace('begin('+app+'.System.computeRightHandSide)','begin('+app+'.System.computeRightHandSide)\n'+bscode)
          text = text.replace('begin('+app+'.System.computeMatrix)','begin('+app+'.System.computeMatrix)\n'+bscode)
        if ig:
          text = text.replace('begin('+app+'.System.initalGuess)','begin('+app+'.System.initialGuess)\n'+bscode)

        bscode = ''' 
          myServices = services;
          gov::cca::TypeMap tm = services.createTypeMap();
          if(tm._is_nil()) {
             fprintf(stderr, "Error:: %s:%d: gov::cca::TypeMap is nil\\n",__FILE__, __LINE__);
             exit(1);
          }
          gov::cca::Port p = self;      //  Babel required casting
          if(p._is_nil()) {
            fprintf(stderr, "Error:: %s:%d: Error casting self to gov::cca::Port \\n",__FILE__, __LINE__);
            exit(1);
          }
  
          myServices.addProvidesPort(p,"TOPS.System","TOPS.System", tm);
          myServices.addProvidesPort(p,"TOPS.System.Initialize.Once","TOPS.System.Initialize.Once", tm);
          myServices.addProvidesPort(p,"TOPS.System.Initialize.EverySolve","TOPS.System.Initialize.EverySolve", tm);

          // GoPort (instead of main)
          myServices.addProvidesPort(p,"DoSolve","gov.cca.ports.GoPort",myServices.createTypeMap());

          myServices.registerUsesPort("TOPS.'''+uns+'''tructured.Solver","TOPS.'''+uns+'''tructured.Solver", tm);'''
        if ig:
          bscode = bscode + '''myServices.addProvidesPort(p,"TOPS.System.Compute.InitialGuess","TOPS.System.Compute.InitialGuess", tm);'''
        if lin == 'nonlinear':
          bscode = bscode + '''myServices.addProvidesPort(p,"TOPS.System.Compute.Residual","TOPS.System.Compute.Residual", tm);'''
          if jac == 'Yes':
            bscode = bscode + '''myServices.addProvidesPort(p,"TOPS.System.Compute.Jacobian","TOPS.System.Compute.Jacobian", tm);'''
        else:
          bscode = bscode + '''myServices.addProvidesPort(p,"TOPS.System.Compute.RightHandSize","TOPS.System.Compute.RightHandSide", tm);'''
          bscode = bscode + '''myServices.addProvidesPort(p,"TOPS.System.Compute.Matrix","TOPS.System.Compute.Matrix", tm);'''
        text = text.replace('begin('+app+'.System.setServices)','begin('+app+'.System.setServices)\n'+bscode)


        bscode = '''  int argc = 1; 
          char *argv[1];
          argv[0] = (char*) malloc(10*sizeof(char));
          strcpy(argv[0],"'''+app+'''");

          this->solver = myServices.getPort("TOPS.'''+uns+'''tructured.Solver");
          this->solver.Initialize(sidl::array<std::string>::create1d(argc,(const char**)argv));
          this->solver.solve();
          myServices.releasePort("TOPS.'''+uns+'''tructuredSolver");'''
        text = text.replace('begin('+app+'.System.go)','begin('+app+'.System.go)\n'+bscode)

        f = file(os.path.join(app,'server','c++',app+'_System_Impl.cc'),'w')
        f.write(text)
        f.close()

        f = file(os.path.join(app,'server','c++',app+'_System_Impl.hh'),'r')
        text = f.read()
        f.close()

        bscode = '''#include "TOPS.hh"'''
        text = text.replace('begin('+app+'.System._includes)','begin('+app+'.System._includes)\n'+bscode)

        bscode = '''    TOPS::'''+uns+'''tructured::Solver solver;
          gov::cca::Services myServices;'''
        text = text.replace('begin('+app+'.System._implementation)','begin('+app+'.System._implementation)\n'+bscode)

        f = file(os.path.join(app,'server','c++',app+'_System_Impl.hh'),'w')
        f.write(text)
        f.close()

        text = '''#!ccaffeine bootstrap file. 
          # ------- don't change anything ABOVE this line.-------------
          path set '''+os.path.join('@PETSC_LIB_DIR@','cca')+'''
          repository get-global TOPS.StructuredSolver
          repository get-global '''+app+'''.System
          instantiate TOPS.StructuredSolver solver
          instantiate '''+app+'''.System system
          connect solver TOPS.System.Initialize.Once system TOPS.System.Initialize.Once
          connect solver TOPS.System.Initialize.EverySolve system TOPS.System.Initialize.EverySolve
          connect system TOPS.Structured.Solver solver TOPS.Structured.Solver
          parameter solver tops_options options "-snes_monitor -ksp_monitor"\n'''
        if ig:
          text = text + '''connect solver TOPS.System.Compute.InitialGuess system TOPS.System.Compute.InitialGuess\n'''
        if lin == 'nonlinear''':
          text = text + '''connect solver TOPS.System.Compute.Residual system TOPS.System.Compute.Residual\n'''
        text = text + '''go system DoSolve
          quit'''
        f = file(os.path.join(app,app+'_rc.in'),'w')
        f.write(text)
        f.close()

        result = buttonbox(message="Code generated",title=title,choices = ["Ok"],fontSize = 20,message2="Now edit "+app+"/server/c++/"+app+"_System_Impl.cc")
