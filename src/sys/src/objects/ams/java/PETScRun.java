/*$Id: PETScRun.java,v 1.3 2000/10/25 19:12:08 bsmith Exp bsmith $*/
/*
     Compiles and runs a PETSc program
*/

import java.lang.Thread;
import java.io.*;
import java.net.*;
import java.util.*;

public class PETScRun
{
  public PETScRun() {/*--------------------------------------------------------*/
    ;
  }
 
  public void start() throws java.io.IOException
  {
    Socket sock = new Socket("cclogin1",2000);
    sock.setSoLinger(true,5);

    /* construct properties to send to server */
    Properties   properties = new Properties();
    properties.setProperty("PETSC_ARCH","solaris");
    properties.setProperty("DIRECTORY","src/vec/examples/tutorials");
    properties.setProperty("EXAMPLE","ex1");
    properties.setProperty("NUMBERPROCESSORS","1");
    properties.setProperty("COMMAND","mpirun");

    (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

    /* get output and print to screen */
    InputStreamReader stream = new InputStreamReader(sock.getInputStream());
    char[] results = new char[128];
    int    fd     = 0;

    while (true) {
      fd = stream.read(results);
      if (fd == -1) {break;}
      System.out.println(new String(results));
    }
  }

  public static void main(String s[])
  {
    PETScRun prun = new PETScRun();
    try {
      prun.start();
    } catch (java.io.IOException ex) {;}
  }
}




