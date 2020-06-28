#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Izaak Neutelings (2017)
import re
from math import sqrt, pow, log
from array import array
from copy import copy, deepcopy
from ROOT import TH1D, TH2D
from TauFW.Plotter.plot.strings import *
from TauFW.Plotter.plot.utils import LOG


class Variable(object):
  """
  Variable class to:
     - hold all relevant information of a variable that will be plotted;
         var name, filename friendly name, LaTeX friendly title, binning, ...
     - allow for variable binning into TH1
     - allow for contextual binning, i.e. depending on channel/selection/...
     - easy string conversions: filename, LaTeX, ...
     - analysis-specific operations: applying variations, ...
  """
  
  def __init__(self, name, *args, **kwargs):
    strings              = [a for a in args if isinstance(a,str) ]
    self.name            = name
    self.name0           = name # back up for addoverflowToLastBinInName
    self.title           = strings[0] if strings else self.name
    self.filename        = makefilename(self.name.replace('/',"_"))
    self.title           = kwargs.get('title',           self.title    ) # for plot axes
    self.filename        = kwargs.get('filename',        self.filename ) # for file
    self.filename        = self.filename.replace('$NAME',self.name).replace('$VAR',self.name) #.replace('$FILE',self.filename)
    self.tag             = kwargs.get('tag',             ""            )
    self.units           = kwargs.get('units',           True          ) # for plot axes
    self.latex           = kwargs.get('latex',           True          ) # for plot axes
    self.nbins           = None
    self.min             = None
    self.max             = None
    self.bins            = None
    self.cut             = kwargs.get('cut',             ""            )
    self.weight          = kwargs.get('weight',          ""            )
    self.weight_data     = kwargs.get('weight_data',     ""            )
    self.setbins(*args)
    self.dividebybinsize = kwargs.get('dividebybinsize', self.hasvariablebinning() )
    self.data            = kwargs.get('data',            True          ) # also draw data
    self.flag            = kwargs.get('flag',            ""            ) # flag, e.g. 'up', 'down', ...
    self.binlabels       = kwargs.get('labels',          [ ]           )
    self.ymin            = kwargs.get('ymin',            None          )
    self.ymax            = kwargs.get('ymax',            None          )
    self.rmin            = kwargs.get('rmin',            None          )
    self.rmax            = kwargs.get('rmax',            None          )
    self.logx            = kwargs.get('logx',            False         )
    self.logy            = kwargs.get('logy',            False         )
    self.ymargin         = kwargs.get('ymargin',         1.80 if self.logy else 1.16 )
    self.logyrange       = kwargs.get('logyrange',       None          )
    self.position        = kwargs.get('position',        ""            ) # legend position
    self.ncolumns        = kwargs.get('ncolumns',        1             ) # legend position
    #self.plot            = kwargs.get('plots',           True          )
    self.only            = kwargs.get('only',            [ ]           )
    self.veto            = kwargs.get('veto',            [ ]           )
    self.blindcuts       = kwargs.get('blind',           ""            )
    self.addoverflow     = kwargs.get('addoverflow',     False         ) # add overflow to last bin
    if self.ymargin<1:
      self.ymargin += 1.
    if self.latex:
      self.title = makeLatex(self.title,units=self.units)
      if 'ctitle' in kwargs:
        for ckey, title in kwargs['ctitle'].iteritems():
          kwargs['ctitle'][ckey] = makeLatex(title)
    if self.only:
      if not isList(self.only): self.only = [ self.only ]
    if self.veto:
      if not isList(self.veto): self.veto = [ self.veto ]
    if self.binlabels and len(self.binlabels)<self.nbins:
      LOG.warning("Variable.init: len(binlabels)=%d < %d=nbins"%(len(self.binlabels),self.nbins))
    if self.addoverflowToLastBin:
      self.addoverflowToLastBinInName()
    self.contexttitle    = getcontextFromDict(kwargs, None,           key='ctitle'                ) # context-dependent title
    self.contextbinning  = getcontextFromDict(kwargs, args,           key='cbinning',  regex=True ) # context-dependent binning
    self.contextposition = getcontextFromDict(kwargs, self.position,  key='cposition', regex=True ) # context-dependent position
    self.contextblind    = getcontextFromDict(kwargs, self.blindcuts, key='cblind',    regex=True ) # context-dependent blind limits
    self.contextymargin  = getcontextFromDict(kwargs, self.ymargin,   key='cymargin',  regex=True ) # context-dependent ymargin
    self.contextcut      = getcontextFromDict(kwargs, self.cut,       key='ccut',      regex=True ) # context-dependent cuts
    self.contextweight   = getcontextFromDict(kwargs, self.weight,    key='cweight',   regex=True ) # context-dependent cuts
  
  @property
  def var(self): return self.name
  @var.setter
  def var(self,value): self.name = value
      
  @property
  def xmin(self): return self.min
  @xmin.setter
  def xmin(self,value): self.xmin = value
  
  @property
  def xmax(self): return self.max
  @xmax.setter
  def xmax(self,value): self.xmax = value
  
  def __str__(self):
    """Returns string representation of Variable object."""
    return self.name
  
  def __repr__(self):
    """Returns string representation of Variable object."""
    #return '<%s.%s("%s","%s",%s,%s,%s)>'%(self.__class__.__module__,self.__class__.__name__,self.name,self.title,self.nbins,self.xmin,self.xmax)
    return '<%s("%s","%s",%s,%s,%s) at %s>'%(self.__class__.__name__,self.name,self.title,self.nbins,self.xmin,self.xmax,hex(id(self)))
  
  def __iter__(self):
    """Start iteration over variable information."""
    for i in [self.name,self.nbins,self.min,self.max]:
      yield i
  
  def __gt__(self,ovar):
    """Order alphabetically."""
    return self.filename > ovar.filename
  
  def clone(self,*args,**kwargs):
    """Shallow copy."""
    if not args:
      args = self.nbins,self.min,self.max
    newdict = self.__dict__.copy()
    if 'filename' in kwargs:
      kwargs['filename'] = kwargs['filename'].replace('$FILE',self.filename)
    for key in kwargs:
      if key in newdict:
        newdict.pop(key)
    newvar = Variable(self.name,*args,**kwargs)
    newvar.__dict__.update(newdict)
    return newvar
  
  def issame(self,ovar,**kwargs):
    """Compare Variable objects."""
    return self.name==ovar.name and self.getbins()==ovar.getbins()
  
  def printbinning(self,filename=False):
    """Print the variable name with the binning."""
    if filename:
      return '%s(%s,%s,%s)'%(self.filename,self.nbins,self.xmin,self.xmax)
    else:
      return '%s(%s,%s,%s)'%(self.name,self.nbins,self.xmin,self.xmax)
  
  def setbins(self,*args):
    """Set binning: (N,min,max), or bins if it is set"""
    numbers         = [a for a in args if isNumber(a) ]
    bins            = [a for a in args if isList(a)   ]
    if len(numbers)==3:
      self.nbins    = numbers[0]
      self.min      = numbers[1]
      self.max      = numbers[2]
      self.bins     = None
    elif len(bins)>0:
      bins          = list(bins[0])
      self.nbins    = len(bins)-1
      self.min      = bins[0]
      self.max      = bins[-1]
      self.bins     = bins
    else:
      print error('Variable: bad arguments "%s" for binning!'%(args,))
      exit(1)
  
  def getbins(self,full=False):
    """Get binning: (N,xmin,xmax), or bins if it is set"""
    if self.hasvariablebinning():
      return self.bins
    elif full:
      return [self.min+i*(self.max-self.min)/self.nbins for i in xrange(self.nbins+1)]
    else:
      return (self.nbins,self.min,self.max)
  
  def hasvariablebinning(self):
    """True if bins is set."""
    return self.bins!=None
  
  def hasintbinning(self):
    """True if binning is integer."""
    width = (self.max-self.min)/self.nbins
    return self.bins==None and int(self.min)==self.min and int(self.max)==self.max and width==1
  
  def ispartof(self, *searchterms, **kwargs):
    """Check if all labels are in the variable's name, title."""
    searchterms = [l for l in searchterms if l!='']
    if not searchterms: return False
    found       = True
    regex       = kwargs.get('regex',     False )
    exlcusive   = kwargs.get('exclusive', True  )
    for searchterm in searchterms:
      if not regex:
        searchterm = re.sub(r"([^\.])\*",r"\1.*",searchterm) # replace * with .*
      if exlcusive:
        for varlabel in [self.name,self.title]:
          matches  = re.findall(searchterm,varlabel)
          if matches:
            break # try next searchterm, or return True
        else:
          return False # none of the labels contain the searchterm
      else: # inclusive
        for varlabel in [self.name,self.title]:
          matches  = re.findall(searchterm,varlabel)
          if matches:
            return True # one of the searchterm has been found
    return exlcusive
  
  def changecontext(self,*args,**kwargs):
    """Change the contextual title, binning or position for a set of arguments, if it is available"""
    if self.contexttitle:
      title = self.contexttitle.getcontext(*args)
      if title!=None:
        self.title = title
    if self.contextbinning:
      binning = self.contextbinning.getcontext(*args)
      if isinstance(binning,list):
        binning = (binning,)
      if binning!=None:
        self.setbins(*binning)
      if self.addoverflowToLastBin:
        self.addoverflowToLastBinInName() # in case the last bin changed
      self.dividebybinsize = kwargs.get('dividebybinsize',self.hasvariablebinning())
    if self.contextposition:
      position = self.contextposition.getcontext(*args)
      if position!=None:
        self.position = position
    if self.contextymargin:
      ymargin = self.contextymargin.getcontext(*args)
      if ymargin!=None:
        self.ymargin = ymargin
    if self.contextcut:
      cut = self.contextcut.getcontext(*args)
      if cut!=None:
        self.cut = cut
    if self.contextweight:
      weight = self.contextweight.getcontext(*args)
      if weight!=None:
        self.weight = weight
  
  def plotfor(self,*strings,**kwargs):
    """Check is selection is vetoed for this variable."""
    verbosity = LOG.getverbosity(kwargs)
    strings   = list(strings)
    for i, string in enumerate(strings):
      if string.__class__.__name__=='Selection':
        string     = string.selection
        strings[i] = string
      for searchterm in self.veto:
        if re.search(searchterm,string):
          LOG.verbose('Variable.plotfor: Regex match of string "%s" to "%s"'%(string,searchterm),verbosity,level=2)
          return False
    if len(self.only)==0:
      return True
    for i, string in enumerate(strings):
      for searchterm in self.only:
        if re.search(searchterm,string):
          LOG.verbose('Variable.plotfor: Regex match of string "%s" to "%s"'%(string,searchterm),verbosity,level=2)
          return True
    return False
  
  def unwrap(self):
    return (self.name,self.nbins,self.min,self.max)
  
  def getnametitle(self,name=None,title=None,tag=None):
    """Help function to create name and title."""
    if tag and tag[0]!='_':
      tag   = '_'+tag
    if name==None:
      name  = self.filename+tag
    if title==None:
      title = self.title
    name = name.replace('(','').replace(')','').replace('[','').replace(']','').replace(',','-').replace('.','p')
    return name, title
  
  def createTH1(self,name=None,title=None,**kwargs):
    """Create a 1D histogram."""
    tag     = kwargs.get('tag',     ""          )
    poisson = kwargs.get('poisson', False       )
    sumw2   = kwargs.get('sumw2',   not poisson )
    xtitle  = kwargs.get('xtitle',  self.title  )
    #TH1Class = TH1D # TH1I if self.hasintbinning() else TH1D
    name, title = self.getnametitle(name,title,tag)
    if self.hasvariablebinning():
      hist = TH1D(name,title,self.nbins,array('d',list(self.bins)))
    else:
      hist = TH1D(name,title,self.nbins,self.min,self.max)
    if poisson:
      hist.SetBinErrorOption(TH1D.kPoisson)
    elif sumw2:
      hist.Sumw2()
    hist.GetXaxis().SetTitle(xtitle)
    #hist.SetDirectory(0)
    return hist
  
  def createTH2(self,yvariable,name=None,title=None,**kwargs):
    """Create a 2D histogram."""
    tag     = kwargs.get('tag',     ""          )
    poisson = kwargs.get('poisson', False       )
    sumw2   = kwargs.get('sumw2',   not poisson )
    xtitle  = kwargs.get('xtitle',  self.title  )
    name, title = self.getnametitle(name,title,tag)
    if self.hasvariablebinning() and yvariable.hasvariablebinning():
      hist = TH2D(name,title,self.nbins,array('d',list(self.bins)),yvariable.nbins,array('d',list(yvariable.bins)))
    elif self.hasvariablebinning():
      hist = TH2D(name,title,self.nbins,array('d',list(self.bins)),yvariable.nbins,yvariable.min,yvariable.max)
    else:
      hist = TH2D(name,title,self.nbins,self.min,self.max,yvariable.nbins,yvariable.min,yvariable.max)
    if poisson:
      hist.SetBinErrorOption(TH2D.kPoisson)
    elif sumw2:
      hist.Sumw2()
    hist.GetXaxis().SetTitle(xtitle)
    return hist
  
  def drawcommand(self,name=None,tag="",bins=False):
    """Create variable expression for the Tree.Draw method."""
    histname, title = self.getnametitle(name,None,tag)
    if bins:
      cmd = "%s >> %s(%d,%s,%s)"%(self.name,histname,self.nbins,self.min,self.max)
    else:
      cmd = "%s >> %s"%(self.name,histname)
    return cmd
  
  def drawcommand2D(self,yvariable,name=None,tag="",bins=False):
    """Create variable expression for the Tree.Draw method."""
    histname, title = self.getnametitle(name,None,tag)
    if bins:
      cmd = "%s:%s >> %s(%d,%s,%s,%d,%s,%s)"%(yvariable.name,self.name,histname,self.nbins,self.min,self.max,yvariable.nbins,yvariable.min,yvariable.max)
    else:
      cmd = "%s:%s >> %s"%(yvariable.name,self.name,histname)
    return cmd
  
  def shift(self,jshift,**kwargs):
    """Create new variable with a shift tag added to its name."""
    if len(jshift)>0 and jshift[0]!='_':
      jshift = '_'+jshift
    newname               = shift(self.name,jshift,**kwargs)
    newvariable           = deepcopy(self)
    newvariable.name      = newname
    if not kwargs.get('keepFilename',False) and self.name != newname:
      newvariable.filename += jshift
    return newvariable
  
  def shiftname(self,jshift,**kwargs):
    return shift(self.name,jshift,**kwargs)
  
  def blind(self,bmin,bmax):
    """Return selection string that blinds some window (bmin,bmax),
    making sure the cuts match the bin edges of some (N,xmin,xmax) binning."""
    blindcut = ""
    if bmax<=bmin:
     LOG.ERROR('Variable.blind: "%s" has window a = %s <= %s = b !'%(self.name0,bmin,bmax))
    xlow, xhigh = bmin, bmax
    nbins, xmin, xmax = self.nbins, self.min, self.max
    if self.hasvariablebinning():
      bins = self.bins
      for xval in bins:
        if xval>bmin: break
        xlow = xval
      for xval in reversed(bins):
        if xval<bmax: break
        xhigh = xval
    else:
      binwidth   = float(xmax-xmin)/nbins
      if xmin<bmin<xmax:
        bin, rem = divmod(bmin-xmin,binwidth)
        xlow     = bin*binwidth
      if xmin<bmax<xmax:
        bin, rem = divmod(bmax-xmin,binwidth)
        if rem>0:
          bin   += 1
        xhigh    = bin*binwidth
    blindcut = "(%s<%s || %s<%s)"%(self.name,xlow,xhigh,self.name)
    LOG.verbose('Variable.blind: blindcut = "%s" for a (%s,%s) window and (%s,%s,%s) binning'%(blindcut,bmin,bmax,nbins,xmin,xmax),verbosity,level=2) 
    return blindcut
  
  def addoverflow(self,**kwargs):
    """Modify variable name in order to add the overflow to the last bin."""
    verbose = kwargs.get('verbose', False)
    if self.hasvariablebinning():
      width     = self.bins[-1]-self.bins[-2]
      threshold = self.bins[-2] + 0.90*width
    else:
      width     = (self.max-self.min)/float(self.nbins)
      threshold = self.max - 0.90*width
    self.name   = "min(%s,%s)"%(self.name0,threshold)
    if verbose:
      print "Variable.addoverflow: '%s' -> '%s' for binning '%s'"%(self.name0,self.name,self.getbins())
    return self.name
 
var = Variable # short name


def wrapvariable(*args,**kwargs):
  """Help function to wrap variable arguments into a Variable object."""
  if len(args)==4 or len(args)==5:
    return Variable(args) # (var,nbins,xmin,xmax)
  elif len(args)==1 and isinstance(args[0],Variable):
    return args[0]
  LOG.warning('wrapvariable: Could not unwrap arguments "%s" to a Variable object. Returning None.'%args)
  return None
  

def unwrapvariablebinning(*args,**kwargs):
  """Help function to unwrap variable arguments to return variable name, number of bins,
  minumum and maximum x axis value."""
  if len(args)==4:
    return args # (var,nbins,xmin,xmax)
  elif len(args)==1 and isintance(args[0],Variable):
    return args[0].unwrap()
  LOG.warning('unwrapvariablebinning: Could not unwrap arguments "%s" to a Variable object. Returning None.'%args)
  return None
  
