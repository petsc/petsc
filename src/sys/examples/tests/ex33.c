
/*T
      description: test of subtests and test parsing generates lots of small tests
T*/
 
int main(int argc, char **argv) {return 0;}

/*TEST
   test: 
     # Test loop separation
     suffix: 1
     command: echo ${args}
     nsize: {{1,2}}
     args: -a b -c {{d "e,f" g}} -h {{"i,j" "k"}} -l {{foo bar}separate output}

   test: # Simple subtest
     suffix: 2
     command: echo ${args}

     args: -a b -c {{d "e,f" g}} 

     test:
       args:  -my_arg cg -my_arg2 {{p q}}

     test:
       args: -foo {{"s,t" "u,v"}} -rtol {{1.e-4 1.e-5}}

   test: # subtests with suffix to generate separate files
     suffix: 3
     command: echo ${args}

     args: -a b -c {{d "e,f" g}}

     test:
       suffix: cg
       args:  -my_arg cg

     test:
       suffix: gmres
       args:  -my_arg gmres -rtol {{1.e-4 1.e-5}}

   test: # Everything separate test generation added
     suffix: 4
     command: echo ${args}

     args: -a b -c {{d "e,f" g}}

     test:
       suffix: cg
       args:  -my_arg cg 

     test:
       suffix: gmres
       args: -rtol {{1.e-4 1.e-5}separate output}
TEST*/
