/*$Id: PETScRun.java,v 1.1 2000/10/06 19:23:15 bsmith Exp bsmith $*/
/*
     Compiles and runs a PETSc program
*/

import java.lang.Thread;
import java.io.*;

/*
    This is the class that this file implements (must always be the same as
  the filename.
*/
public class PETScRun {
    Runtime rtime = null;

    /*
     The constructor for the class.
     for using the AMS
    */
    public PETScRun() {/*--------------------------------------------------------*/
      rtime = Runtime.getRuntime()  ;
    }

  
    public String Make(String machine,String petsc_arch,String petsc_dir,String dir,String example) {/*-----*/
        Process make = null;
        try {
System.out.println("petscrsh make "+machine+" "+petsc_arch+" "+petsc_dir+" "+dir+" "+example);
          make = rtime.exec("petscrsh make "+machine+" "+petsc_arch+" "+petsc_dir+" "+dir+" "+example);
        } catch (java.io.IOException e) {return "error";}
        InputStreamReader stream = new InputStreamReader(make.getInputStream());

        String output = new String("");
        char[] errors = new char[1024];
        int    fd     = 0;
        while (true) {
            try {
              fd = stream.read(errors);
            } catch (java.io.IOException ex) {break;}
            if (fd == -1) break;
            output = output+(new String(errors));
        }
        System.out.println(output); 
        return "hi";
    }

    public String Run(String machine,int np,String petsc_arch,String petsc_dir,String dir,String example) {/*-----*/
        return "hi" ;
    }

    /*
        The main() program. Creates an amsoptions object and calls the start()
      method on it.
    */

    public static void main(String s[]) {  /*-------------------------------------*/
      PETScRun prun = new PETScRun();
      String result = prun.Make("fire","solaris","/sandbox/bsmith/petsc","src/vec/examples/tutorials","ex1");
    }

}




