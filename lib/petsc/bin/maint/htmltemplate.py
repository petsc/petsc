import time
def getHeader(title):
  """ Generalized header"""
  firstpart="""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<title>
"""+title+"""
</title>
<style type="text/css">
div.main {
  max-width: 1300px;
  background: white;
  margin-left: auto;
  margin-right: auto;
  padding: 20px;
  padding-top: 0;
  background: #FBFBFB;
  /* border: 5px solid #CCCCCC;
  border-radius: 10px; */
}
/*table, th, td {
   border: 1px solid black;
}*/
table {
   border: 1px solid black;
}
table, th, td {
   vertical-align: top;
}
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
table {
  /*border: 1px solid black;
  border-radius: 10px;*/
  padding: 3px;
  margin-top: 0;
}
td a:link, td a:visited, td a:focus, td a:active {
  font-weight: bold;
  text-decoration: underline;
  color: black;
  border: 1px solid black;
}
td a:hover {
  font-weight: bold;
  text-decoration: underline;
  color: black;
}
td.border {
  border-top: 2px solid #EEE;
}
th.gray {
  background: #FFFFFF;
  border-top: 10px solid #999;
  padding: 0px;
  padding-top: 0px;
  padding-bottom: 0px;
  font-size: 1.1em;
  font-weight: bold;
  text-align: center;
}
th {
  border-top: 1px solid #000;
  padding: 10px;
  padding-top: 5px;
  padding-bottom: 5px;
  font-size: 1.1em;
  font-weight: bold;
  text-align: center;
}
td.desc {
  max-width: 650px;
  padding: 2px;
  font-size: 0.9em;
}
td.green {
  text-align: center;
  vertical-align: middle;
  padding: 2px;
  background: #01DF01;
  min-width: 50px;
}
td.yellow {
  text-align: center;
  vertical-align: middle;
  padding: 2px;
  background: #F4FA58;
  min-width: 50px;
}
td.red {
  text-align: center;
  vertical-align: middle;
  padding: 2px;
  background: #FE2E2E;
  min-width: 50px;
}
</style>
</head>
<body><div class="main"> 



<center><span style=\"font-size:1.3em; font-weight: bold;\">
"""+title+"""
</span><br />
Last update: """+ time.strftime('%a, %d %b %Y %H:%M:%S %z') +""" </center>\n\n
</span></center><br>\n\n
<center><span style=\"font-size:1.3em; font-weight: bold;\">\n

"""
  return firstpart
