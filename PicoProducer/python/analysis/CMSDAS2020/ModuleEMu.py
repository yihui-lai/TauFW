# Author: Izaak Neutelings (May 2020)
# Description: Simple module to pre-select emu events
from ROOT import TFile, TTree, TH1D
from ROOT import Math
import math
import numpy as np
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from TauFW.PicoProducer.analysis.utils import LeptonPair, idIso, matchtaujet, deltaPhi

# Inspired by 'Object' class from NanoAODTools.
# Convenient to do so to be able to add MET as 4-momentum to other physics objects using p4()
class Met(Object):
  def __init__(self,event,prefix,index=None):
    self.eta = 0.0
    self.mass = 0.0
    Object.__init__(self,event,prefix,index)


class ModuleEMu(Module):
  
  def __init__(self,fname,**kwargs):
    self.outfile = TFile(fname,'RECREATE')
    self.default_float = -999.0
    self.default_int = -999
    self.dtype      = kwargs.get('dtype', 'data')
    self.ismc       = self.dtype=='mc'
    self.isdata     = self.dtype=='data'
  
  def beginJob(self):
    """Prepare output analysis tree and cutflow histogram."""
    
    # CUTFLOW HISTOGRAM
    self.cutflow           = TH1D('cutflow','cutflow',25,0,25)
    self.cut_none          = 0
    self.cut_trig          = 1
    self.cut_muon          = 2
    self.cut_muon_veto     = 3
    self.cut_electron      = 4
    self.cut_electron_veto = 5
    self.cut_pair          = 6
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_none,           "no cut"        )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_trig,           "trigger"       )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_muon,           "muon"          )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_muon_veto,      "muon     veto" )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_electron,            "electron"           )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_electron_veto,  "electron veto" )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_pair,           "pair"          )
    
    # TREE
    self.tree        = TTree('tree','tree')
    self.pt_1        = np.zeros(1,dtype='f')
    self.eta_1       = np.zeros(1,dtype='f')
    self.q_1         = np.zeros(1,dtype='i')
    self.id_1        = np.zeros(1,dtype='?')
    self.iso_1       = np.zeros(1,dtype='f')
    self.genmatch_1  = np.zeros(1,dtype='f')
    self.decayMode_1 = np.zeros(1,dtype='i')
    self.pt_2        = np.zeros(1,dtype='f')
    self.eta_2       = np.zeros(1,dtype='f')
    self.q_2         = np.zeros(1,dtype='i')
    self.id_2        = np.zeros(1,dtype='i')
    self.anti_e_2    = np.zeros(1,dtype='i')
    self.anti_mu_2   = np.zeros(1,dtype='i')
    self.iso_2       = np.zeros(1,dtype='f')
    self.genmatch_2  = np.zeros(1,dtype='f')
    self.decayMode_2 = np.zeros(1,dtype='i')

    self.m_vis       = np.zeros(1,dtype='f')
    self.genWeight   = np.zeros(1,dtype='f')
    self.jetnumber = np.zeros(1,dtype='i')
    self.jet_leading_pt = np.zeros(1,dtype='f')
    self.jet_leading_eta = np.zeros(1,dtype='f')
    self.jet_sub_pt = np.zeros(1,dtype='f')
    self.jet_sub_eta = np.zeros(1,dtype='f')

    self.bjetnumber = np.zeros(1,dtype='i')
    self.bjet_leading_pt = np.zeros(1,dtype='f')
    self.bjet_leading_eta = np.zeros(1,dtype='f')
    self.bjet_sub_pt = np.zeros(1,dtype='f')
    self.bjet_sub_eta = np.zeros(1,dtype='f')
    
    self.metpt = np.zeros(1,dtype='f')
    self.metphi = np.zeros(1,dtype='f')
    self.metsumEt = np.zeros(1,dtype='f')

    self.puppimetpt = np.zeros(1,dtype='f')
    self.puppimetphi = np.zeros(1,dtype='f')
    self.puppimetsumEt = np.zeros(1,dtype='f')

    self.z_vis_pt = np.zeros(1,dtype='f')
    self.z_full_pt = np.zeros(1,dtype='f')
    self.dphi = np.zeros(1,dtype='f')
    self.mt = np.zeros(1,dtype='f')
    self.miss_Dzeta = np.zeros(1,dtype='f')
    self.vis_Dzeta = np.zeros(1,dtype='f')
    self.deltaR_mu_e  = np.zeros(1,dtype='f')
    self.pu_rho         = np.zeros(1,dtype='f')
    self.npvs           = np.zeros(1,dtype='i')
    self.gennpus = np.zeros(1,dtype='i')

    self.tree.Branch('pt_1',         self.pt_1,        'pt_1/F'       )
    self.tree.Branch('eta_1',        self.eta_1,       'eta_1/F'      )
    self.tree.Branch('q_1',          self.q_1,         'q_1/I'        )
    self.tree.Branch('id_1',         self.id_1,        'id_1/O'       )
    self.tree.Branch('iso_1',        self.iso_1,       'iso_1/F'      )
    self.tree.Branch('genmatch_1',   self.genmatch_1,  'genmatch_1/F' )
    self.tree.Branch('decayMode_1',  self.decayMode_1, 'decayMode_1/I')
    self.tree.Branch('pt_2',         self.pt_2,  'pt_2/F'             )
    self.tree.Branch('eta_2',        self.eta_2, 'eta_2/F'            )
    self.tree.Branch('q_2',          self.q_2,   'q_2/I'              )
    self.tree.Branch('id_2',         self.id_2,  'id_2/I'             )
    self.tree.Branch('anti_e_2',     self.anti_e_2,   'anti_e_2/I'    )
    self.tree.Branch('anti_mu_2',    self.anti_mu_2,  'anti_mu_2/I'   )
    self.tree.Branch('iso_2',        self.iso_2, 'iso_2/F'            )
    self.tree.Branch('genmatch_2',   self.genmatch_2,  'genmatch_2/F' )
    self.tree.Branch('decayMode_2',  self.decayMode_2, 'decayMode_2/I')
    self.tree.Branch('m_vis',        self.m_vis, 'm_vis/F'            )
    self.tree.Branch('genWeight',    self.genWeight,   'genWeight/F'  )
    self.tree.Branch('jetnumber',          self.jetnumber,         'jetnumber/I'        )
    self.tree.Branch('jet_leading_pt',         self.jet_leading_pt,        'jet_leading_pt/F'       )
    self.tree.Branch('jet_leading_eta',         self.jet_leading_eta,        'jet_leading_eta/F'       )
    self.tree.Branch('jet_sub_pt',         self.jet_sub_pt,        'jet_sub_pt/F'       )
    self.tree.Branch('jet_sub_eta',         self.jet_sub_eta,        'jet_sub_eta/F'       )

    self.tree.Branch('bjetnumber',          self.bjetnumber,         'bjetnumber/I'        )
    self.tree.Branch('bjet_leading_pt',         self.bjet_leading_pt,        'bjet_leading_pt/F'       )
    self.tree.Branch('bjet_leading_eta',         self.bjet_leading_eta,        'bjet_leading_eta/F'       )
    self.tree.Branch('bjet_sub_pt',         self.bjet_sub_pt,        'bjet_sub_pt/F'       )
    self.tree.Branch('bjet_sub_eta',         self.bjet_sub_eta,        'bjet_sub_eta/F'       )
    self.tree.Branch('metpt',         self.metpt,  'metpt/F'             )
    self.tree.Branch('metphi',         self.metphi,  'metphi/F'             )
    self.tree.Branch('metsumEt',         self.metsumEt,  'metsumEt/F'             )

    self.tree.Branch('puppimetpt',         self.puppimetpt,  'puppimetpt/F'             )
    self.tree.Branch('puppimetphi',         self.puppimetphi,  'puppimetphi/F'             )
    self.tree.Branch('puppimetsumEt',         self.puppimetsumEt,  'puppimetsumEt/F'             )
    self.tree.Branch('z_vis_pt',         self.z_vis_pt,  'z_vis_pt/F'             )
    self.tree.Branch('z_full_pt',         self.z_full_pt,  'z_full_pt/F'             )
    self.tree.Branch('dphi',         self.dphi,  'dphi/F'             )
    self.tree.Branch('mt',         self.mt,  'mt/F'             )
    self.tree.Branch('miss_Dzeta',         self.miss_Dzeta,  'miss_Dzeta/F'             )
    self.tree.Branch('vis_Dzeta',         self.vis_Dzeta,  'vis_Dzeta/F'             )
    self.tree.Branch('deltaR_mu_e',         self.deltaR_mu_e,  'deltaR_mu_e/F'             )
    self.tree.Branch('pu_rho',         self.pu_rho,  'pu_rho/F'             )
    self.tree.Branch('npvs',         self.npvs,  'npvs/I'             )
    self.tree.Branch('gennpus',         self.gennpus,  'gennpus/I'             )

  def endJob(self):
    """Wrap up after running on all events and files"""
    self.outfile.Write()
    self.outfile.Close()
  
  def analyze(self, event):
    """Process event, return True (pass, go to next module) or False (fail, go to next event)."""
    
    # NO CUT
    self.cutflow.Fill(self.cut_none)
    
    # TRIGGER
    if not event.HLT_IsoMu24 or event.HLT_IsoMu27: return False
    self.cutflow.Fill(self.cut_trig)
    
    ##### MUON #######################################
    muons = [ ]
    veto_muons = [ ]
    for muon in Collection(event,'Muon'):
      good_muon = muon.mediumId and muon.pfRelIso04_all < 0.5 and abs(muon.eta) < 2.4
# and muon.dz<0.2 and muon.dxy<0.045
      signal_muon = good_muon and muon.pt > 25.0 
      veto_muon   = good_muon and muon.pt > 15.0 and muon.pt <= 25.0 
      if signal_muon:
        muons.append(muon)
      if veto_muon: 
        veto_muons.append(muon)
    if len(muons)==0: return False
    self.cutflow.Fill(self.cut_muon)
    if len(veto_muons) > 1: return False
    self.cutflow.Fill(self.cut_muon_veto)
    

    ##### ELECTRON ###################################
    electrons = [ ]
    veto_electrons = [ ]
    for electron in Collection(event,'Electron'):
      good_electron = electron.mvaFall17V2noIso_WPL and electron.pfRelIso03_all < 0.5 and abs(electron.eta) < 2.3 
#and electron.convVeto and electron.lostHits<=1 and electron.dz<0.2 and electron.dxy<0.045
      signal_electron = good_electron and electron.pt > 15.0
      veto_electron   = good_electron and electron.pt > 10.0 and electron.pt <= 15.0
      if signal_electron:
        electrons.append(electron)
      if veto_electron:
        veto_electrons.append(electron)
    if len(electrons) == 0: return False
    self.cutflow.Fill(self.cut_electron)
    if len(veto_electrons) > 1: return False
    self.cutflow.Fill(self.cut_electron_veto)


    ##### eMu PAIR #################################
    dileps = [ ]
    for electron in electrons:
      for muon in muons:
        if muon.DeltaR(electron)<0.5: continue
        ltau = LeptonPair(electron,electron.pfRelIso03_all,muon,muon.pfRelIso04_all)
        dileps.append(ltau)
    if len(dileps)==0:
      return False
    electron, muon = max(dileps).pair
    #electron.tlv   = electron.p4()
    #muon.tlv       = muon.p4()
    self.cutflow.Fill(self.cut_pair)
    


    # SELECT Jets
    jets = []
    bjets = []
    self.jet_leading_pt[0]=-99
    self.jet_leading_eta[0]=-99
    self.jet_sub_pt[0]=-99
    self.jet_sub_eta[0]=-99
    self.bjet_leading_pt[0]=-99
    self.bjet_leading_eta[0]=-99
    self.bjet_sub_pt[0]=-99
    self.bjet_sub_eta[0]=-99
    for jet in Collection(event,'Jet'):
      veto_jet = jet.pt > 20.0 and abs(jet.eta) < 4.7 and jet.puId>0 and jet.jetId>1
      if veto_jet and jet.DeltaR(muon)>=0.5 and jet.DeltaR(electron)>=0.5:
        veto_basic_jet = jet.pt > 30.0 
        if veto_basic_jet:
          jets.append(jet)
        veto_bjet = jet.btagDeepFlavB > 0.2770 and abs(jet.eta) < 2.5
        if veto_bjet:
          bjets.append(jet)
    self.jetnumber[0]=len(jets)
    self.bjetnumber[0]=len(bjets)
    if self.jetnumber[0]>0:
      jet = max(jets,key=lambda p: p.pt)
      self.jet_leading_pt[0]=jet.pt
      self.jet_leading_eta[0]=jet.eta
    if self.jetnumber[0]>1:
      new_jets = set(jets)
      new_jets.remove(max(jets,key=lambda p: p.pt)) 
      jet = max(new_jets,key=lambda p: p.pt)
      self.jet_sub_pt[0]=jet.pt
      self.jet_sub_eta[0]=jet.eta
    if self.bjetnumber[0]>0:
      bjet = max(bjets,key=lambda p: p.pt)
      self.bjet_leading_pt[0]=bjet.pt
      self.bjet_leading_eta[0]=bjet.eta
    if self.bjetnumber[0]>1:
      new_bjets = set(bjets)
      new_bjets.remove(max(bjets,key=lambda p: p.pt))
      bjet = max(new_bjets,key=lambda p: p.pt)
      self.bjet_sub_pt[0]=bjet.pt
      self.bjet_sub_eta[0]=bjet.eta

    # CHOOSE MET definition
    # TODO section 4: compare the PuppiMET and (PF-based) MET in terms of mean, resolution and data/expectation agreement of their own distributions and of related quantities
    # and choose one of them for the refinement of Z to tautau selection.
    puppimet = Met(event, 'PuppiMET')
    met = Met(event, 'MET')

    self.metpt[0] = met.pt
    self.metphi[0] = met.phi
    self.metsumEt[0] = met.sumEt
    self.puppimetpt[0] = puppimet.pt
    self.puppimetphi[0] = puppimet.phi
    self.puppimetsumEt[0] = puppimet.sumEt
    
    # SAVE VARIABLES
    self.pt_1[0]        = muon.pt
    self.eta_1[0]       = muon.eta
    self.q_1[0]         = muon.charge
    self.id_1[0]        = muon.mediumId
    self.iso_1[0]       = muon.pfRelIso04_all # keep in mind: the SMALLER the value, the more the muon is isolated
    self.pt_2[0]        = electron.pt
    self.eta_2[0]       = electron.eta
    self.q_2[0]         = electron.charge
    self.iso_2[0]       = electron.pfRelIso03_all # keep in mind: the HIGHER the value of the discriminator, the more the tau is isolated
    self.m_vis[0]       = (muon.p4()+electron.p4()).M()

######### add high level variables #########
    self.z_vis_pt[0]    = (muon.p4()+electron.p4()).Pt()
    self.z_full_pt[0]   = (puppimet.p4()+muon.p4()+electron.p4()).Pt()
    self.dphi[0]        = deltaPhi(muon.phi , puppimet.phi )
    self.mt[0]          = math.sqrt( 2*muon.pt*puppimet.sumEt*(1 - math.cos(self.dphi[0])) ) 
    bisec = (muon.phi + electron.phi)/2.0
    self.miss_Dzeta[0]  = puppimet.pt * math.cos(bisec - puppimet.phi)
    self.vis_Dzeta[0]   = muon.pt*math.cos(bisec - muon.phi) + electron.pt*math.cos(bisec - muon.phi)
    self.deltaR_mu_e[0]  = muon.DeltaR(electron)
    self.pu_rho[0]         = event.fixedGridRhoFastjetAll
    self.npvs[0]           = event.PV_npvs



    if self.ismc:
      self.genmatch_1[0]  = muon.genPartFlav # in case of muons: 1 == prompt muon, 15 == muon from tau decay, also other values available for muons from jets
      self.genmatch_2[0]  = electron.genPartFlav #Flavour of genParticle for MC matching to status==1 electrons or photons: 1 = prompt electron (including gamma*->mu mu), 15 = electron from prompt tau, 22 = prompt photon (likely conversion), 5 = electron from b, 4 = electron from c, 3 = electron from light or unknown, 0 = unmatched
      self.genWeight[0]   = event.genWeight
      self.gennpus[0]        = event.Pileup_nPU
    self.tree.Fill()
    
    return True
