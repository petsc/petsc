/*$Id: PETScRun.java,v 1.10 2000/11/13 19:18:07 bsmith Exp bsmith $*/
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
import java.awt.print.*;

public class PETScRun extends java.applet.Applet implements Pageable, Printable
{
  static final int MACHINE = 0,DIRECTORY = 1,MAXNP = 2,EXAMPLES = 3,EXAMPLESHELP = 4;
  Hashtable        systems[];
  Socket csock = null;

  java.applet.AppletContext appletcontext;
  java.applet.Applet applet;

  JPanel     tpanel;
  JTextField options,help;

  JCheckBox toptions;
  JCheckBox tbopt;
  JCheckBox snoop;

    Choice    arch;
    Choice    dir;
    Choice    example;
    Choice    np;

  JTextArea   opanel,epanel;

  boolean isviewsource = false;

  public void init() {
    appletcontext = getAppletContext();
    applet = this;
    System.out.println("Codebase"+this.getDocumentBase());

    try {
      System.out.println("parameter"+this.getParameter("server"));
      Socket sock = null;
      try {
        sock = new Socket(this.getParameter("server"),2000);
      } catch(java.net.ConnectException oops) {
        appletcontext.showDocument(new URL("http://www.mcs.anl.gov/petsc/noserver.html"));
        return;
      }
      sock.setSoLinger(true,5);

      /* construct properties to send to server */
      Properties   properties = new Properties();
      properties.setProperty("Loadsystems","Yes");
      (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

      ObjectInputStream os = new ObjectInputStream(sock.getInputStream());
      systems                = new Hashtable[5];
      systems[MACHINE]       = (Hashtable) os.readObject();
      systems[DIRECTORY]     = (Hashtable) os.readObject();
      systems[MAXNP]         = (Hashtable) os.readObject();
      systems[EXAMPLES]      = (Hashtable) os.readObject();
      systems[EXAMPLESHELP]  = (Hashtable) os.readObject();
      sock.close();
    } catch (java.io.IOException ex) {ex.printStackTrace(); System.out.println("no systems");}
      catch (ClassNotFoundException ex) {    System.out.println("no class");}
 
    this.setLayout(new FlowLayout());

    tpanel = new JPanel(new GridLayout(3,4));
    this.add(tpanel, BorderLayout.NORTH);
      
    help = new JTextField(50);
    this.add(help, BorderLayout.NORTH);

    JPanel grid = new JPanel(new GridBagLayout());
    this.add(grid,BorderLayout.NORTH);
    options = new JTextField(50);
    grid.add(new JLabel("Options"));
    grid.add(options);


      arch = new Choice();
      arch.addItemListener(new ItemListener() {
                            public void itemStateChanged(ItemEvent e) {
                              Choice choice = (Choice) e.getItemSelectable();
                              setnp(np,choice.getSelectedItem());}
	                  });
      Enumeration keys = systems[MAXNP].keys();
      while (keys.hasMoreElements()) {
        arch.add((String)keys.nextElement());
      }
      tpanel.add(arch);
        
      dir = new Choice();
      String pexampled = this.getParameter("EXAMPLEDIRECTORY");
      if (pexampled == null) {
        dir.addItemListener(new ItemListener() {
                            public void itemStateChanged(ItemEvent e) {
                              Choice choice = (Choice) e.getItemSelectable();
                              setexamples(example,choice.getSelectedItem());}
	                  });
        keys = systems[EXAMPLES].keys();
        while (keys.hasMoreElements()) {
          dir.add((String)keys.nextElement());
        }
      } else { /* .html file indicates which example to run */
        dir.add(pexampled);
      }
      tpanel.add(dir);

      example = new Choice();
      String pexample = this.getParameter("EXAMPLE");
      if (pexample == null) {
        example.addItemListener(new ItemListener() {
                            public void itemStateChanged(ItemEvent e) {
                              Choice choice = (Choice) e.getItemSelectable();
                              String directory = dir.getSelectedItem();
                              ArrayList ehelp = (ArrayList)systems[EXAMPLESHELP].get(directory);
                              ArrayList ex = (ArrayList)systems[EXAMPLES].get(directory);
                              int index = ex.indexOf(choice.getSelectedItem());
                              help.setText((String)ehelp.get(index));}
	                  });
        setexamples(example,(String)systems[EXAMPLES].keys().nextElement());
      } else {  /* .html file indicates which example to run */
        example.add(pexample);
      }
      tpanel.add(example);

      String phelp = this.getParameter("HELP"); /* .html file indicates help message */
      if (phelp != null) {
        this.help.setText(phelp);
      }

      np = new Choice();
      setnp(np,(String)systems[MAXNP].keys().nextElement());
      tpanel.add(np);

      JButton mbutton = new JButton("Make");
      tpanel.add(mbutton);
      mbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          runprogram("make");
        }
      }); 

      JButton tbutton = new JButton("Test");
      tpanel.add(tbutton);
      tbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          runprogram("maketest");
        }
      }); 

      JButton rbutton = new JButton("Run");
      tpanel.add(rbutton);
      rbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          runprogram("mpirun");
        }
      }); 

      JButton sbutton = new JButton("View source");
      tpanel.add(sbutton);
      sbutton.addActionListener(new ActionListener(){
        public void actionPerformed(ActionEvent e) { 
          isviewsource = true;
          runprogram("makehtml");
        }
      }); 

      toptions = new JCheckBox("Set options graphically");
      tpanel.add(toptions);

      snoop = new JCheckBox("Snoop on running program");
      tpanel.add(snoop);

      tbopt = new JCheckBox("Compile debug version");
      tpanel.add(tbopt);


    epanel = new JTextArea(4,60);
    this.add(new JScrollPane(epanel), BorderLayout.NORTH); 
    epanel.setLineWrap(true);
    epanel.setWrapStyleWord(true);

    opanel = new JTextArea(20,60);
    this.add(new JScrollPane(opanel), BorderLayout.NORTH); 
    opanel.setLineWrap(true);
    opanel.setWrapStyleWord(true);
  }

  /*
      Fills in the examples pull down menu based on the directory selected
  */
  public void setexamples(Choice example,String dir) {
    ArrayList ex = (ArrayList)systems[EXAMPLES].get(dir);
    Iterator its = ex.iterator();
    example.removeAll();
    while (its.hasNext()) {
      example.add((String)its.next());
    }
    ArrayList ehelp = (ArrayList)systems[EXAMPLESHELP].get(dir);
    System.out.println("ehelp"+ehelp+(String)ehelp.get(0));
    this.help.setText((String)ehelp.get(0));
  }

  /*
     Fills in the number processors pull down menu based on the arch selected
  */
  public void setnp(Choice np,String arch){
    int onp = Integer.parseInt((String)systems[MAXNP].get(arch)),i;
    np.removeAll();
    for (i=1; i<=onp;i++) {
      np.add(""+i);
    }
  }

  public void stop() {
    System.out.println("Called stop");
  }

  public void runprogram(String what)
  {
    try {
      csock = new Socket(this.getParameter("server"),2000);
      final Socket sock = csock;
      sock.setSoLinger(true,5);
      final InputStream sstream = sock.getInputStream();

      /* construct properties to send to server */
      final Properties   properties = new Properties();
      properties.setProperty("PETSC_ARCH",arch.getSelectedItem());
      properties.setProperty("DIRECTORY",dir.getSelectedItem());
      properties.setProperty("EXAMPLE",example.getSelectedItem());
      properties.setProperty("NUMBERPROCESSORS",np.getSelectedItem());
      properties.setProperty("COMMAND",what);
      if (tbopt.isSelected()) {
        properties.setProperty("BOPT","g");
      } else {
        properties.setProperty("BOPT","O");
      }
      String soptions = options.getText();
      if (toptions.isSelected() && what.equals("mpirun")) {
	URL urlb = this.getDocumentBase();
        soptions += " -ams_publish_options";
	try {
	  final URL url = new URL(""+urlb+"PETScOptions.html");  
          System.out.println("loading url"+url);
          (new Thread() {
            public void run() {
              System.out.println("open new");
              appletcontext.showDocument(url,"AMSOptions");
              System.out.println("done open new");
            }
	  }).start();
        } catch (MalformedURLException ex) {System.out.println("bad luck");;} 
      } 
      if (snoop.isSelected() && what.equals("mpirun")) {
	URL urlb = this.getDocumentBase();
        soptions += " -ams_publish_objects";
	try {
	  final URL url = new URL(""+urlb+"PETScView.html");  
          System.out.println("loading url"+url);
          (new Thread() {
            public void run() {
              System.out.println("open new");
              appletcontext.showDocument(url,"PETScView");
              System.out.println("done open new");
            }
	  }).start();
        } catch (MalformedURLException ex) {System.out.println("bad luck");;} 
      } 
      properties.setProperty("OPTIONS",soptions);
      System.out.println("getting back");

      /* get output and print to screen */
      (new Thread() {
        public void run() {
          try {
            (new ObjectOutputStream(sock.getOutputStream())).writeObject(properties);

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
          opanel.append("----DONE-----");
          donerun();
        }
      }).start();

      /* get stderr and popup */
      (new Thread() {
        public void run() {
          try {
            Socket nsock = new Socket(applet.getParameter("server"),2002);
            nsock.setSoLinger(true,5);
            InputStreamReader stream = new InputStreamReader(nsock.getInputStream());
            char[] results = new char[128];
            int    fd      = 0;

            epanel.setText(null);
            while (true) {
              fd  = stream.read(results);
              if (fd == -1) {break;}
              System.out.println("PETScRun:got stderr output"+results);
              epanel.append(new String(results,0,fd));
            }
            stream.close();
            nsock.close();
          } catch (java.io.IOException ex) {System.out.println("PETScRun: error getting stderr");}
          System.out.println("PETScRun: finished checking for stderr");
        }
      }).start();

    } catch (java.io.IOException ex) {;}
  }

  PageFormat format;

  public void donerun() 
  {
    System.out.println("done run");
    PrinterJob job = PrinterJob.getPrinterJob();
    format = job.pageDialog(job.defaultPage());
    job.setPageable(this);
    job.printDialog();
    try {
      job.print();
    } catch (java.awt.print.PrinterException e) {System.out.println("problem printing");;}

    if (isviewsource) {
      String ex = example.getSelectedItem();
      URL    urlb = applet.getDocumentBase();
  
      try {
        String s = null;
        if (ex.endsWith("f90") || ex.endsWith("f")) s = "_F";
        else s = "_c";
        final URL url = new URL(""+urlb+"../"+dir.getSelectedItem()+"/"+ex+s+".html");  
        System.out.println("showing"+url);
        appletcontext.showDocument(url,"Source");
      } catch (MalformedURLException oops) { System.out.println("bad:showing"+urlb);;} 
    }
    isviewsource = false;
  }

  public int        getNumberOfPages()         {return 1;}
  public PageFormat getPageFormat(int pagenum) {return format;}
  public Printable  getPrintable(int pagenum)  {return this;}

  public int print(Graphics g,PageFormat format,int pagenum) {
    Graphics2D g2 = (Graphics2D) g;
    g2.translate(format.getImageableX(),format.getImageableY());
        System.out.println("about to paint");
    this.paint(g2);
        System.out.println("done painting");
    return 1;
  }
}




