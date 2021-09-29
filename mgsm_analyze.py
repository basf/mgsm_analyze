#!/usr/bin/env python
#
# This file is part of mgsm_analyze
#
# Copyright (C) BASF SE 2021
#
# mgsm_analyze is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mgsm_analyze is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mgsm_analyze. If not, see <https://www.gnu.org/licenses/>.

from __future__ import division, print_function, unicode_literals
import os
import numpy as np
from pymatgen.io.babel import BabelMolAdaptor
from scipy.signal import argrelextrema
from tools import *
import openbabel as ob
import subprocess
from time import gmtime, strftime


from platform import python_version
if python_version() < "3.0.0":
    class FileNotFoundError(OSError):
        pass


class AnalyzeMGSM(object):
    
    """
    Class which analyzes the result from a single-ended molecular growing string calculation 
    (link to Prof. Zimmerman's git: https://github.com/ZimmermanGroup/molecularGSM)
    """

    def __init__(self, path):

        self.symbols = []
        self.path = path
        self.ts_id = None
        
        # main input/output files of MGSM are read in
        self.paradata = read_paragsm(path=self.path)
        self.initial, self.symbols = read_initial(path=self.path)
        
        self.node_structures, self.energies = read_stringfile(path=self.path) 
        # note: the self.energies are relative energies 
        # w.r.t the energy of the first structure!
        
        self.eref = read_eref(path=self.path) # in kcal/mol
        self.PEH = None # node ID's of educt, TS and product structures 

        if self.eref != None and self.energies != None:
            # determine the absolute energies of each node on the string (in kcal/mol)
            self.energies_abs_kcal = [float(elem) + float(self.eref) for elem in self.energies] # in kcal/mol
        else:
            self.energies_abs_kcal = None

        if self.paradata != None:
            # obtain the TS-status, the number of imaginary frequencies and the node ID of the TS from the mGSM output file
            self.tsstatus, self.imagfreqs, self.ts_id_paragsm = self.analyze_paragsmdata(self.paradata)
        else:
            self.tsstatus = "wall-time exc."
            self.imagfreqs = None
            self.ts_id_paragsm = None

        # call main function "analyseReactionPath()", which performs a detailed analysis of the nodes on the stringfile
        self.ts_bond_change, self.rule_change_after_ts, self.rule_change_before_ts = self.analyseReactionPath()

        # write out xyz files of educt, ts and product structures
        write_e_ts_p_xyz(PEH=self.PEH, path=self.path, node_structures=self.node_structures)

        # read input reaction rule used for single-ended mGSM run
        self.initial_rule = read_rules(path=self.path)

        # derive the absolute and relative TS energies from the mGSM stringfiles
        if self.ts_id != None and self.energies_abs_kcal != None:
            # case 1: a TS has been found
            self.TSenergy_abs_kcal = self.energies_abs_kcal[int(self.ts_id)]
            self.TSenergy_rel_kcal = round(self.energies[int(self.PEH[1])]-self.energies[int(self.PEH[0])],2)
        else:
            # case 2: no TS has been found
            self.TSenergy_abs_kcal = None
            self.TSenergy_rel_kcal = None

        if self.PEH != None:
            # a reaction path has been obtained with at least a reactant and product structure (no TS required, can be also
            # a "flatland" path)
            self.Penergy_rel_kcal = round(self.energies[int(self.PEH[2])]-self.energies[int(self.PEH[0])],2)
            self.educt_analysis(path=self.path)
            self.product_analysis(path=self.path)
            self.educt_inchi, self.ts_inchi, self.product_inchi, self.inchi = generate_SDFs_INCHIs(path=self.path, rule=self.ts_bond_change, symbols=self.symbols)
        else:
            # the mGSM calculation probably failed and no product structure has been obtained
            self.Penergy_rel_kcal = None
            self.educt_inchi = None
            self.ts_inchi = None
            self.product_inchi = None
            self.inchi = None

        # collect all results in dictionary self.TSinfo
        self.TSinfo = self.collect_results()

        # write out mgsm-analyze.log file
        self.write_log_file()

    
    def educt_analysis(self, path):

        """ 
        Analyse educt node structure: check, if the educt node structure contains of several molecules 
        and write out the xyz files of the educt structure(s).
        """

        inchi_list = []

        educt_list = molecule_separation(mol=self.node_structures[self.PEH[0]], symbols=self.symbols)

        for i, educt in enumerate(educt_list):
            if os.path.isfile(str(path)+"/educt_"+str(i)+".sdf"):
                inchi_cmd = ['obabel', '-isdf', str(path), '/educt_'+str(i)+'.sdf', '-xX', 'FixedH', 'DoNotAddH',
                             '--separate', '-oinchi']
                try:
                    process = subprocess.run(inchi_cmd, stdout=subprocess.PIPE, universal_newlines=True)
                    tmp = process.stdout.split()[0]
                    if len(process.stdout) > 1:
                        print("educt contains of several molecules???")
                    inchi_list.append(tmp)
                except IndexError:
                    continue

            mol = self.node_structures[self.PEH[0]]
            out = open(self.path+"/educt_"+str(i)+".xyz", 'w')
            out.write(str(len(educt)) + "\n\n")

            for id in educt:
                atom = mol.GetAtom(id)
                out.write(ob.OBElementTable().GetSymbol(atom.GetAtomicNum()) + " " + str(atom.GetX()) + " " +
                          str(atom.GetY()) + " " + str(atom.GetZ()) + "\n")
            out.close()

        return

    def product_analysis(self, path):

        """ 
        Analyse product node structure: check, if the product node structure contains of several molecules 
        and write out the xyz files of the (separated) product structure(s).
        """

        self.product_list = molecule_separation(mol=self.node_structures[self.PEH[2]], symbols=self.symbols)

        for i, prod in enumerate(self.product_list):
            # prod: atom ID's of separated molecules in product
            # get product complex (might consist of several molecules)
            mol = self.node_structures[self.PEH[2]]
            # create xyz files of separated products!!!
            out = open(self.path+"/prod_"+str(i)+".xyz", 'w')
            out.write(str(len(prod)) + "\n\n")
            for id in prod:
                atom = mol.GetAtom(id)
                out.write(ob.OBElementTable().GetSymbol(atom.GetAtomicNum()) + " " + str(atom.GetX()) + " " +
                          str(atom.GetY()) + " " + str(atom.GetZ()) + "\n")
            out.close()

        return 0


    def collect_results(self):

        """ Collect all results and add the information to a dictionary called TMPdic """

        if self.tsstatus in ["-XTS-", "-TS-", "-FL-", "-diss", "-add_node-"]:
            
            TMPdic = {"ts_energy_abs": self.TSenergy_abs_kcal, "ts_energy_string": self.TSenergy_rel_kcal,
                      "ts_status": self.tsstatus, "imag_freqs": self.imagfreqs,
                      "p_energy_string": self.Penergy_rel_kcal}

            if self.tsstatus == "-XTS-" or self.tsstatus == "-TS-":
                TMPdic["ts_path"] = self.path + "/ts"
            else:
                TMPdic["ts"] = None

            TMPdic["final_rule"] = self.ts_bond_change
            TMPdic["rule_change_before_ts"] = self.rule_change_before_ts
            TMPdic["rule_change_after_ts"] = self.rule_change_after_ts
            TMPdic["product"] = self.node_structures[self.PEH[2]]
            TMPdic["product_list"] = self.product_list

            # correct OpenBabel molecule object of product structure by distance-based connectivity: 
            # the node_structure itself will be overwritten!

            TMPdic["product_inchi"] = self.product_inchi
            TMPdic["product_node"] = self.PEH[2]
            TMPdic["educt"] = self.node_structures[self.PEH[0]]
            TMPdic["nnodes"] = self.PEH[2]
            TMPdic["initial"] = self.initial
            TMPdic["initial_rule"] = self.initial_rule
            if self.energies == None:
                TMPdic["nnodes"] = 0
            else:
                TMPdic["nnodes"] = len(self.energies)
            TMPdic["rule_found"] = None
        else:
            TMPdic = {"ts_energy_abs": None, "ts_energy_string": None, 
                       "ts_status": self.tsstatus, "imag_freqs": None, "p_energy_string": None}
            if self.node_structures != None:
                TMPdic["product"] = self.node_structures[-1]
                TMPdic["initial_rule"] = self.initial_rule
                TMPdic["final_rule"] = []
                TMPdic["rule_change_after_ts"] = None
                TMPdic["rule_change_before_ts"] = None

        return TMPdic

    def write_log_file(self):

        """ Write a log file with most important output from mGSM analysis """

        log = open(self.path+"/mgsm_analyze.log", "w")
        log.write("Time: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n\n")

        log.write("#########################\n")
        log.write("#  Analyze MGSM output  #\n")
        log.write("#########################\n\n")
        log.write("TS status: " + str(self.tsstatus) + "\n\n")
        log.write("estimated Ea (from stringfile): " + str(self.TSenergy_rel_kcal) + " kcal/mol\n\n")
        log.write("Number of imaginary frequencies: " + str(self.imagfreqs) + "(approximated)\n")
        log.write("Node ID's of educt, TS, product: " + str(self.PEH) + "\n\n")
        log.write("Initial rule:\t\t" + str(", ".join(self.initial_rule)) + "\n")
        log.write("Final rule after MGSM:\t" + str(", ".join(self.ts_bond_change)) + "\n\n")

        if self.PEH != None:
            log.write("Rule change before TS?:\t" + str(self.TSinfo["rule_change_before_ts"]) + "\n")
            log.write("Rule change after TS?:\t" + str(self.TSinfo["rule_change_after_ts"]) + "\n")
            log.write("Product inchi:\t" + str(self.TSinfo["product_inchi"]) +"\n")
            log.write("Reaction inchi:\t" + str(self.inchi) + "\n")

        log.close()


    def check_connectivity(self, obm):
        
        """ Compare the openbabel connectivity with the distance-based connectivity and adds missing bonds 
        (directly to OpenBabel molecule object) if needed. """
        
        edgesOB = []
        for bond in ob.OBMolBondIter(obm):
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.GetIdx() > atom2.GetIdx() and (atom2.GetIdx(), atom1.GetIdx()) not in edgesOB:
                edgesOB.append((atom2.GetIdx(), atom1.GetIdx()))
            elif atom2.GetIdx() > atom1.GetIdx() and (atom1.GetIdx(), atom2.GetIdx()) not in edgesOB:
                edgesOB.append((atom1.GetIdx(), atom2.GetIdx()))
        edgesT = sorted(connectivity(mol=obm, symbols=self.symbols))
        diff_T_OB = set(edgesT) - set(edgesOB)  
        
        # we have to add these connectivities to OBmol
        for item in diff_T_OB:
            obm.AddBond(item[0], item[1], 1)


    def analyze_paragsmdata(self, paragsmdata):
        
        """ Analyse data in main mGSM output file ("paragsm0000" file)"""

        ts_id_paragsm = None    # ID of the node which corresponds to the TS (according to mGSM)
        imagfreqs = None    # number of imaginary frequencies at TS
        tsstatus = None     # the status of the mGSM calculation

        for line in paragsmdata:
            tmp = line.split()
            if "exiting" in line:
                if " cannot add node, exiting" in line:
                    tsstatus = "-add_node-"
                elif " Too many SCF failures" in line:
                    tsstatus = "-SCF_failures-"
                else:
                    tsstatus = "-UNKOWN failure-"
            elif len(tmp) == 4:
                if tmp[0] == "found" and tmp[3][:10] == "eigenvalue":
                    imagfreqs = int(tmp[1])
            elif len(tmp) >= 20:
                if "opt_iters" in line:
                    tsstatus = tmp[19]
                    ts_id_paragsm = tmp[18]
            elif "-exit early-" in line:
                # not in line opt_iters, additional check
                tsstatus = "-exit early-"
            elif "opt_i:" in line:
                if " Too many failed SCF's":
                    print(" opt_i: Exiting! Too many failed SCF's ")
                else:
                    tsstatus = "-UNKOWN failure-"

        return tsstatus, imagfreqs, ts_id_paragsm


    def analyseReactionPath(self):
        
        """ Analyse mGSM reaction path according to different courses"""

        ts_bonds = []

        # check status of MGSM run (as provided in paragsm output file)
        # other stati than the ones mentioned in the list, are not analyzed here, because we assume a failure of the 
        # reaction path optimization
        if self.tsstatus in ["-XTS-", "-TS-", "-FL-", "-diss", "-add_node-"]: 

            # generating List of ob.OBMol objects
            node_structures_ob = []
            # self.node_structures contains the XYZ structures of the nodes
            for mol in self.node_structures:
               node_structures_ob.append(BabelMolAdaptor(mol).openbabel_mol)

            energies_np = np.array(self.energies)
            # Determine global energy maximum
            globalmax_e = energies_np.max()
            globalmax_id = self.energies.index(globalmax_e) 

            # for -diss or -FL- paths, we use the global energy maximum as energy reference!
            # This means that for ascending paths, the product energy is chosen and for descending paths the reactant energy...
            
            # Find local maxima/minima on the reaction path
            localmax_id = list(argrelextrema(energies_np, np.greater)[0])
            localmin_id = list(argrelextrema(energies_np, np.less)[0])

            # Find maximum of local maxima
            localmax_e = []
            for item in localmax_id:
                localmax_e.append(round(float(energies_np[item]), 6))

    	    ###################################################################################
            # A: the highest energy of the reaction path is either the first or the last node #
            # This could mean that:                                                           #
            #       A.1 The reaction path is ascending/descending OR                          #
            #       A.2 The reaction paths goes down in energy and then overcomes a barrier   #
            ###################################################################################

            if globalmax_id == 0 or globalmax_id == len(energies_np)-1:

                if len(localmax_id) == 0:
                    
                    ##############################################
                    # Case A.1: descending/ascending path, no TS #
                    ##############################################

                    print("descending or ascending path")
                    if self.tsstatus == "-add_node-" or self.tsstatus == "-diss":
                        # change tsstatus to "-FL-": then also no TS exists and imagfreqs and TSenergy are not defined.
                        self.tsstatus = "-FL-"
                        self.imagfreqs = None
                        self.TSenergy_abs_kcal = None
                        self.TSenergy_rel_kcal = None
                else:
                    
                    ##############################################
                    # Case A.2: at least 1 local maximum exist(s)! # --> probably a TS has been found!
                    ##############################################

                    # it might be that several local maxima exist. Therefore we need to loop over all of them and 
                    # identify the first local maximum which corresponds to a change in the connectivity

                    for i in range(len(localmax_e)):
                        emax = localmax_e[i]
                        self.ts_id = self.energies.index(round(float(emax), 6))
                        
                        # if local max is no "reaction" but only conformational change --> take next local max!
                        conn_firstnode = connectivity(mol=self.node_structures[0], symbols=self.symbols)

                        # check if self.ts_id+2 exists
                        if self.ts_id+2 <= len(self.energies)-1:
                            if self.energies[self.ts_id+2] < self.energies[self.ts_id+1]:
                                bonds_ts_tmp = connectivity(mol=self.node_structures[self.ts_id + 2], symbols=self.symbols)
                            else:
                                bonds_ts_tmp = connectivity(mol=self.node_structures[self.ts_id + 1], symbols=self.symbols)
                        else:
                            bonds_ts_tmp = connectivity(mol=self.node_structures[self.ts_id + 1], symbols=self.symbols)

                        ts_change = compare_connectivity(conn_firstnode, bonds_ts_tmp)

                        if ts_change == []: # no reaction occured
                            self.ts_id = None  
                            continue
                        else:
                            break

                    ###################################################################################################
                    # check, if the TS-node ID identified above fits to the one suggested by mGSM (from paragsm file):
                    ###################################################################################################

                    if self.ts_id_paragsm != None and self.ts_id != None:
                        if int(self.ts_id) != int(self.ts_id_paragsm): 
                            print("\nWARNING: Paragsm TS ID and ID based on string analysis do not match")


            ###########################################################
            # B: at least 1 maximum exists which could be a TS        #
            ###########################################################    
            
            else:
                
                # it might be that several local maxima exist. Therefore we need to loop over all of them and 
                # identify the first local maximum which corresponds to a change in the connectivity
                
                for i in range(len(localmax_e)):
                    emax = localmax_e[i]
                    self.ts_id = self.energies.index(round(float(emax), 6))
                    
                    # if local max is no "reaction" but only conformational change --> take next local max!
                    conn_firstnode = connectivity(mol=self.node_structures[0], symbols=self.symbols)
                    
                    # check if self.ts_id+2 exists
                    if self.ts_id+2 <= len(self.energies)-1:
                        if self.energies[self.ts_id+2] < self.energies[self.ts_id+1]:
                            bonds_ts_tmp = connectivity(mol=self.node_structures[self.ts_id + 2], symbols=self.symbols)
                        else:
                            bonds_ts_tmp = connectivity(mol=self.node_structures[self.ts_id + 1], symbols=self.symbols)
                    else:
                        bonds_ts_tmp = connectivity(mol=self.node_structures[self.ts_id + 1], symbols=self.symbols)

                    ts_change = compare_connectivity(conn_firstnode, bonds_ts_tmp)

                    if ts_change == []: # and len(localmax_e) > 1:  # no reaction occured, search for next maximum
                        self.ts_id = None
                        continue
                    else:
                        break

                ###################################################################################################
                # check, if the TS-node ID identified above fits to the one suggested by mGSM (from paragsm file):
                ###################################################################################################
                if self.ts_id_paragsm != None and self.ts_id != None:
                    if int(self.ts_id) != int(self.ts_id_paragsm): 
                        print("\nWARNING: Paragsm TS ID and ID based on string analysis do not match")


            #################################################################
            # NEXT step: find local minima next to ts -> educt and product  #
            #################################################################

            # initialize educt and product ID's to first and last node of the reaction path
            e_id = 0
            p_id = len(self.energies)-1

            # For barrier-less reactions ("-FL-"), one can set the ts_id to the educt or product ID
            if self.ts_id == None:
                if globalmax_id == 0:
                    self.ts_id = e_id
                elif globalmax_id == len(energies_np)-1:
                    self.ts_id = p_id
            
            # Loop over all local minima
            for item in localmin_id:
                if self.ts_id != None:
                    if item > self.ts_id and item < p_id:
                        p_id = item
                        # take first local minimum structure AFTER TS structure
                        break
                    elif item < self.ts_id and item > e_id:
                        e_id = item
                        # take first local minimum structure BEFORE TS structure

            # Two things need to be checked
            # 1: whether initial and first node structure have the same connectivity
            # 2: whether the first node structure (= partially optimized initial input structure) 
            #    and the newly assigned "educt" structure have the same connectivity.

            #################################################################################
            # Case 1: check for connectivity changes in preoptimization of MGSM             #
            # then the connectivity would already change from initial educts to first node! #
            #################################################################################
            # 
            # We decided to take this part out, because often in the initial structure (=precomplex)
            # some atom-distances are close and are mistakenly identified as bonded.
            # Thus, an additional connectivity is recognized, but it is not TRUE!!! 

            if e_id != 0:

                #################################################################
                # Case 2: compare connectivity of first node of stringfile      #
                # and the node which has been detected as "educt" structure     #
                #################################################################

                conn_firstnode = connectivity(node_structures_ob[0], symbols=self.symbols)
                conn_educt = connectivity(node_structures_ob[e_id], symbols=self.symbols)
                topo_diff = compare_connectivity(conn_firstnode, conn_educt)

                # if a connectivity change has been detected, the "educt" structure is no real educt structure, because
                # its connectivity differs from the first node structure and thus a reaction has happened w/o any barrier (-FL-)
                if topo_diff != []:
                    p_id = e_id
                    e_id = 0
                    self.ts_id = None
                    print("\nWARNING: FL detected\n")
                    # change ts-status to FL!
                    self.tsstatus = "-FL-"
                    self.imagfreqs = None
                    self.TSenergy_abs_kcal = None
                    self.TSenergy_rel_kcal = None
                    ts_bonds = topo_diff

            self.PEH = [e_id, self.ts_id, p_id]

        # if self.tsstatus NOT in ["-XTS-", "-TS-", "-FL-", "-diss", "-add_node-"], thus mGSM probably failed...
        else: 
            self.PEH = None

        # Next step: detect actual transformation from educt -> product

        rule_change_after_ts = False
        rule_change_before_ts = False
        last_change = None

        if self.PEH != None:
            
            # DEVELOPMENTAL part:
            # The idea is to find out if connectivity changes happen between the TS and the (identified) product structure.
            # This is a non-trivial task, because the bonding situation between educt and product is not well-defined due to
            # the reaction which takes place. However, with this code we try to detect additional connectivity changes which 
            # ONLY happen after the TS connectivity change. 

            conn_educt = connectivity(mol=self.node_structures[self.PEH[0]], symbols=self.symbols)

            # loop over all nodes between educt and product (including them)
            educt_nodeid = self.PEH[0]
            ts_nodeid = self.PEH[1]
            product_nodeid = self.PEH[2]

            for i in range(educt_nodeid, product_nodeid+1):
                
                conn_node = connectivity(mol=self.node_structures[i], symbols=self.symbols)
                # compare connectivities of educt and current node structure (in loop)
                topo_change = compare_connectivity(conn_educt, conn_node)

                if i == educt_nodeid:
                    last_change = topo_change
                else:
                    if ts_nodeid != None: # TS exists
                        # We define that a "rule_change_after_ts" exists, if we detect a connectivity change between the
                        # structure which has a node id of the TSnode+2 and the product node.
                        if topo_change != last_change and i < ts_nodeid - 1: 
                            diff = list(set(topo_change) - set(last_change))
                            if diff != []:
                                rule_change_before_ts = True
                                print("\nWARNING: rule_change_before_ts\n")
                                break
                        if topo_change != last_change and i > ts_nodeid + 1: 
                            diff = list(set(topo_change) - set(last_change))
                            if diff != []:
                                rule_change_after_ts = True
                                print("\nWARNING: rule_change_after_ts\n")
                                break
                        last_change = topo_change

            conn_product = connectivity(mol=self.node_structures[self.PEH[2]], symbols=self.symbols)
            # END of DEVELOPMENTAL part

            # find connectivity changes between educt and product structures:
            ts_bonds = compare_connectivity(conn_educt, conn_product)
        
        else: 
            ts_bonds = []
            rule_change_after_ts = None
            rule_change_before_ts = None

        # write out actual rule (bond changes) to "final_rule.txt"
        out = open(self.path+"/final_rule.txt", "w")
        out.write(" ".join(ts_bonds)+"\n")
        out.close()

        if self.tsstatus == "-FL-" or self.tsstatus == "-diss":
            self.ts_id = None
            self.TSenergy_abs_kcal = None
            self.TSenergy_rel_kcal = None

        return ts_bonds, rule_change_after_ts, rule_change_before_ts


def main():

    """
    Analyze MGSM output and print out results.
    """

    analyze = AnalyzeMGSM(path=os.getcwd()) 

    print("\n\n#########################")
    print("#  Analyze MGSM output  #")
    print("#########################\n")
    print("PATH:", analyze.path, "\n")
    print("TS status:", analyze.tsstatus, "\n")
    print("Estimated Ea (from stringfile):", analyze.TSenergy_rel_kcal, " kcal/mol\n")
    print("Number of imaginary frequencies:", analyze.imagfreqs, "(approximated)\n")
    print("Node ID's of educt, TS, product:", analyze.PEH, "\n")
    print("Initial rule:\t\t", ", ".join(analyze.initial_rule))
    print("Final rule after MGSM:\t", ", ".join(analyze.ts_bond_change), "\n")
    if analyze.PEH != None:
        print("Rule change before TS?:\t", analyze.TSinfo["rule_change_before_ts"])
        print("Rule change after TS?:\t", analyze.TSinfo["rule_change_after_ts"])
        print("Product inchi:\t", analyze.TSinfo["product_inchi"])
        print("Reaction inchi:\t", analyze.inchi)

if __name__ == "__main__":

    main() 
