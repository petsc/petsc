/*$Id: PETScRun.java,v 1.2 2000/10/25 18:44:28 bsmith Exp bsmith $*/
/*
     Compiles and runs a PETSc program
*/

import java.lang.Thread;
import java.io.*;
import java.net.*;

public class PETScRun
{
  public PETScRun() {/*--------------------------------------------------------*/
    ;
  }
 
  public void start() throws java.io.IOException
{
    Socket sock = new Socket("fire",2000);
         InputStreamReader stream = new InputStreamReader(sock.getInputStream());

        String output = new String("");
        char[] errors = new char[1024];
        int    fd     = 0;
        while (true) {
        System.out.println("reading"); 
            try {
              fd = stream.read(errors);
            } catch (java.io.IOException ex) {System.out.println("except");break;}
            if (fd == -1) break;
            output = output+(new String(errors));
        }
        System.out.println(output); 
    }

    public static void main(String s[]) {  /*-------------------------------------*/
      PETScRun prun = new PETScRun();
      try {
        prun.start();
      } catch (java.io.IOException ex) {;}
    }
}




