/*$Id: PETScRun.java,v 1.4 2000/10/26 18:01:10 bsmith Exp bsmith $*/
/*
     Compiles and runs a PETSc program
*/

import java.lang.Thread;
import java.io.*;
import java.net.*;
import java.util.*;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class PETScRun extends java.applet.Applet
{
  static final int MACHINE = 0,DIRECTORY = 1,MAXNP = 2,EXAMPLES = 3;
  Hashtable        systems[];

  JPanel      tpanel;
    Choice    arch;
    Choice    dir;
    Choice    example;
    Choice    np;

  JTextArea   opanel;

  public void init() {

    try {
      Socket sock = new Socket(this.getParameter("server"),2000);
      sock.setSoLinger(true,5);

      /* construct properties to send to server */
      Properties   properties = new Properties();
      properties.setProperty("Loadsystems","Yes");
      (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

      ObjectInputStream os = new ObjectInputStream(sock.getInputStream());
      systems            = new Hashtable[4];
      systems[MACHINE]   = (Hashtable) os.readObject();
      systems[DIRECTORY] = (Hashtable) os.readObject();
      systems[MAXNP]     = (Hashtable) os.readObject();
      systems[EXAMPLES]  = (Hashtable) os.readObject();
      sock.close();
    } catch (java.io.IOException ex) {;}
      catch (ClassNotFoundException ex) {    System.out.println("no class");}
 
    this.setLayout(new FlowLayout());

    tpanel = new JPanel(new GridLayout(2,4));
    this.add(tpanel, BorderLayout.NORTH);
      arch = new Choice();
      arch.add("solaris");
      arch.add("linux");
      tpanel.add(arch);
        
      dir = new Choice();
      dir.add("src/vec/examples/tutorials");
      dir.add("src/snes/examples/tutorials");
      tpanel.add(dir);

      example = new Choice();
      example.add("ex1");
      example.add("ex1f");
      tpanel.add(example);

      np = new Choice();
      np.add("1");
      np.add("2");
      tpanel.add(np);

      JButton rbutton = new JButton("Run");
      tpanel.add(rbutton);
      rbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          runprogram("mpirun");
        }
      }); 

      JButton mbutton = new JButton("Make");
      tpanel.add(mbutton);
      mbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          runprogram("make");
        }
      }); 

    opanel = new JTextArea(30,60);
    this.add(new JScrollPane(opanel), BorderLayout.NORTH); 
    opanel.setLineWrap(true);
    opanel.setWrapStyleWord(true);
  }

  public void stop() {
    System.out.println("Called stop");
  }

  public void runprogram(String what)
  {

    try {
      Socket sock = new Socket(this.getParameter("server"),2000);
      sock.setSoLinger(true,5);
      InputStream sstream = sock.getInputStream();

      /* construct properties to send to server */
      Properties   properties = new Properties();
      properties.setProperty("PETSC_ARCH",arch.getSelectedItem());
      properties.setProperty("DIRECTORY",dir.getSelectedItem());
      properties.setProperty("EXAMPLE",example.getSelectedItem());
      properties.setProperty("NUMBERPROCESSORS",np.getSelectedItem());
      properties.setProperty("COMMAND",what);

      (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

      /* get output and print to screen */
      InputStreamReader stream = new InputStreamReader(sstream);
      char[] results = new char[128];
      int    fd      = 0,cnt = 0;

      opanel.setText(null);
      while (true) {
        fd  = stream.read(results);
        if (fd == -1) {break;}
        opanel.append(new String(results,0,fd));
	cnt += fd;
      }
    } catch (java.io.IOException ex) {;}
  }
}




