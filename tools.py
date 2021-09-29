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
from copy import deepcopy
import numpy as np
import networkx as nx
import os
from pymatgen.io.babel import BabelMolAdaptor
import openbabel as ob
from subprocess import PIPE, Popen
import subprocess
try:
    from pymatgen import IMolecule, Molecule
    from pymatgen.io.babel import BabelMolAdaptor
    import openbabel as ob
except:
    pass


def generate_SDFs_INCHIs(path=None, rule=None, symbols=None):

    """
    Generate SDF's and inchi's of educt, TS and product structures
    """
    cwd = os.getcwd()
    os.chdir(path)

    obconversion = ob.OBConversion()
    obconversion.SetInFormat("sdf")
    educt_ob = ob.OBMol()

    if os.path.isfile("educt.xyz"):
        ed = readXYZmulti("educt.xyz")
        educt_ob = BabelMolAdaptor(ed[0]).openbabel_mol
        os.system("obabel -ixyz educt.xyz -osdf > educt.sdf 2> /dev/null ")
        # determine connectivity based on distance criterion
        final_conn = connectivity(mol=educt_ob, symbols=symbols)
        # adapt original openbabel SDF by modifying the connectivity according to our distance-based connectivity!
        modify_SDF(ini="educt.sdf", conn=final_conn, output="educt_corr")

    educt_mol = ob.OBMol()
    obconversion.ReadFile(educt_mol, "educt_corr.sdf")

    educt_conn = connectivity(mol=educt_mol, symbols=symbols)

    #Generate an SDF file for the TS structure based on the SDF of the educt node structure and the bond formation parts
    #occuring during the reaction
    adds = []

    final_conn = deepcopy(educt_conn)

    for r in rule:
        if "+" in r:
            # collect all bond formation rules (="adds")
            val1 = int(r.split("+")[0])
            val2 = int(r.split("+")[1])
            if val1 > val2 and (val2, val1) not in adds:
                adds.append((val2, val1))
            elif val2 > val1 and (val1, val2) not in adds:
                adds.append((val1, val2))
    for add in adds:
        if add in educt_conn:
            print("WARNING: TS bond formation already there...: ", add)
        else:
            final_conn.append(add)

    modify_SDF(ini="educt_corr.sdf", conn=final_conn, output="TS")
    #####################

    # Note that we decided to use system calls for obabel transformation here, even if this could also be done directly over
    # OB objects/functions, to be sure that we obtain the same results as in our direct execution of obabel.
    # This can be done better! 
    os.system("obabel -ixyz product.xyz -osdf > product.sdf 2> /dev/null ")

    educt_inchi = "None"
    product_inchi = "None"
    ts_inchi = "None"
    product_sdfs = None

    try:
        ts_inchi = get_inchi("TS.sdf")
        educt_inchi = get_inchi("educt.sdf")
        product_inchi = get_inchi("product.sdf")
        product_sdfs = get_separated_sdfs(sdf="product.sdf")

        # write separated product structures
        for i, sdf in enumerate(product_sdfs):
            out = open("prod_"+str(i)+".sdf", "w")
            out.write("".join(sdf))
            out.close()

    except IndexError:
        pass

    # write out inchi.txt file with the inchi's of the educt, ts and product structures 
    inchi_out = open("inchi.txt", "w")
    inchi_out.write(educt_inchi + "\n")
    inchi_out.write(ts_inchi + "\n")
    inchi_out.write(product_inchi)
    inchi_out.close()
    reaction_inchi = educt_inchi + " -> " + ts_inchi + " -> " + product_inchi

    os.chdir(cwd)

    return educt_inchi, ts_inchi, product_inchi, reaction_inchi

def read_paragsm(path=None):

    """ Read main mGSM outputfile (paragsm0000 file) and return content"""
    
    status = []
    parafile = 'paragsm0000'
    paragsm = os.path.join(path, "scratch", parafile)
    paradata = None
    try:
        with open(paragsm, "r") as gsmfile:
            paradata = gsmfile.readlines()
        gsmfile.close()
    except: 
        status.append("No " + parafile + " found!")

    return paradata

def read_stringfile(path=None):

    """ Read mGSM stringfile and extract energies and stuctures for each node """
    
    stringfile = os.path.join(path, "stringfile.xyz0000")
    # number 0000 is fixed doe to fixed format of submitting script
    if os.path.isfile(stringfile):
        node_structures = readXYZmulti(stringfile)
        # node stuctures is list of open babel objects
        energies = []
        for mol in node_structures:
            energies.append(float(mol.GetTitle()))
    else:
        node_structures = None
        energies = None

    return node_structures, energies


def read_eref(path=None):

    """ Read reference energy (kcal/mol) from firstnode file """
    
    firstnodefile = os.path.join(path, "scratch", "firstnode.xyz0000")
    if os.path.isfile(firstnodefile):
        firstnode_structures = readXYZmulti(firstnodefile)
        energies = []
        for mol in firstnode_structures:
            energies.append(float(mol.GetTitle()))
        eref = energies[-1]
    else:
        print(firstnodefile +" not found!")
        eref = None

    return eref


def read_initial(path):

    """ Read the initial xyz structure for the mGSM calculation"""
    
    initialfile = os.path.join(path, "scratch", "initial0000.xyz")
    symbols = []

    if os.path.isfile(initialfile):
        initialxyz = readXYZmulti(initialfile)
        mol_ob = BabelMolAdaptor(initialxyz[0]).openbabel_mol
        for atom in ob.OBMolAtomIter(mol_ob):
            symbols.append(ob.OBElementTable().GetSymbol(atom.GetAtomicNum()))
    else:
        print(initialfile +" not found!")

    return initialxyz[0], symbols


def read_rules(path=None):

    """ Read the original reaction rules that have been used as input for mGSM """
    
    rulefile = os.path.join(path, "scratch", "ISOMERS0000")

    try:
        with open(rulefile, "r") as file:
            ruledata = file.readlines()
        file.close()
    except FileNotFoundError():
        print("No " + rulefile + " found!")
        raise FileNotFoundError()
    except IOError():
        print("No " + rulefile + " found!")
        raise IOError()

    bond_change = []
    for line in ruledata:
        if len(line.split()) < 3:
            continue
        elif line.split()[0] == "ADD" or line.split()[0] == "BREAK":
            tmp1 = int(line.split()[1])
            tmp2 = int(line.split()[2])
            if tmp1 > tmp2 and line.split()[0] == "ADD":
                bond_change.append(str(tmp2)+"+"+str(tmp1))
            elif tmp1 <= tmp2 and line.split()[0] == "ADD":
                bond_change.append(str(tmp1)+"+"+str(tmp2))
            elif tmp1 > tmp2 and line.split()[0] == "BREAK":
                bond_change.append(str(tmp2)+"-"+str(tmp1))
            elif tmp1 <= tmp2 and line.split()[0] == "BREAK":
                bond_change.append(str(tmp1)+"-"+str(tmp2))
                
    return sorted(bond_change)


def compare_connectivity(bondlist1, bondlist2): 
    
    """ Compare two connectivity lists and return the differences between the two lists """

    diff1 = set(bondlist1) - set(bondlist2) 
    diff2 = set(bondlist2) - set(bondlist1)

    diff = []
    for item in diff1:
        diff.append(str(item[0])+"-"+str(item[1]))
    for item2 in diff2:
        diff.append(str(item2[0])+"+"+str(item2[1]))

    return sorted(diff)

def read_SDF(file):

    """ Read in the connectivity from an SDF file and return the connectivity as a list """

    conn = []
    inp = open(file, "r")
    inpdata = inp.readlines()
    start = int(inpdata[3].split()[0]) + 4
    end = start + int(inpdata[3].split()[1])
    for i in range(start, end):
        val1 = int(inpdata[i].split()[0])
        val2 = int(inpdata[i].split()[1])
        if val1 > val2 and (val2, val1) not in conn:
            conn.append((val2, val1))
        elif val2 > val1 and (val1, val2) not in conn:
            conn.append((val1, val2))

    return conn

def connectivity(mol, symbols, no_H = False):

    """ Determine the connectivity of a molecule based on distance criteria """

    mol_ob = BabelMolAdaptor(mol).openbabel_mol
    edges = []
    for i, atom1 in enumerate(ob.OBMolAtomIter(mol_ob)):
        for j, atom2 in enumerate(ob.OBMolAtomIter(mol_ob)):
            if i != j:
                try:
                    symbol1 = symbols[i]
                    symbol2 = symbols[j]
                    if no_H == True and (symbol1 == "H" or symbol2 == "H"):
                        continue
                    coord1 = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
                    coord2 = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])
                    dist = np.linalg.norm(coord1-coord2)

                    # Get radius which serves as a threshold to define whether two atoms are bonded or not
                    radius = kov_rad(symbol1, symbol2)

                    if (dist < radius and dist > 0.0):
                        if atom1.GetIdx() > atom2.GetIdx() and (atom2.GetIdx(), atom1.GetIdx()) not in edges:
                            edges.append((atom2.GetIdx(), atom1.GetIdx()))
                        elif atom2.GetIdx() > atom1.GetIdx() and (atom1.GetIdx(), atom2.GetIdx()) not in edges:
                            edges.append((atom1.GetIdx(), atom2.GetIdx()))
                except IndexError:
                    continue
    return edges

def modify_SDF(charge=None, ini=None, conn=None, output=None):

    """ Modify an SDF file (given as "ini") based on a provided connectivity list ("conn") and 
    write out a new SDF file named according to the output variable """

    inp = open(ini, "r")
    inpdata = inp.readlines()
    end = int(inpdata[3].split()[0]) + 4
    newSDF = open(output + ".sdf", "w")
    newSDF.write(inpdata[0])
    newSDF.write(inpdata[1] + "\n")
    newSDF.write("%3s%3i%s\n" % (inpdata[3].split()[0], len(conn), "  0  0  1  0  0  0  0  0999 V2000"))
    try:
        for i in range(4, end):
            newSDF.write(inpdata[i])
        for j in range(len(conn)):
            newSDF.write("%3i%3i%s" % (conn[j][0], conn[j][1], "  1  0  0  0  0\n"))
    except IndexError:
        pass
    if charge != None:
        # Note: we are currently just attaching the overall charge to the first atom!
        # This is INCORRECT, but helps us to store the charge information in the SDF.
        # One would need to identify the local charges to improve on this.
        newSDF.write("M  CHG  1  1  "+str(charge)+"\n")
        newSDF.write("M  END\n")
    newSDF.write("$$$$")
    newSDF.close()


def kov_rad(symbol1, symbol2):

    """ Determine a "covalent radius" (distance between two atoms), which serves as a threshold for identifying
    which atoms are bonded and which are not """

    rad = {"H": 0.32, "He": 0.32,
           "Li": 0.76, "Be": 0.45, "B": 0.82, "C": 0.77, "N": 0.71, "O": 0.73, "F": 0.71, "Ne": 0.69,
           "Na": 1.02, "Mg": 0.72, "Al": 0.535, "Si": 1.11, "P": 1.06, "S": 1.02, "Cl": 0.99, "Ar": 0.97,
           "K": 1.38, "Ca": 1.00, "Sc": 1.44, "Ti": 1.36, "V": 1.25, "Cr": 1.27, "Mn": 1.39, "Fe": 1.25,
           "Co": 1.26, "Ni": 1.21, "Cu": 1.38, "Zn": 1.31, "Ga": 1.26, "Ge": 1.22, "As": 1.21, "Se": 1.16,
           "Br": 1.14, "Kr": 1.10,
           "Rb": 1.66, "Sr": 1.32, "Y": 1.62, "Zr": 1.48, "Nb": 1.37, "Mo": 1.45, "Tc": 1.31, "Ru": 1.26,
           "Rh": 1.35, "Pd": 1.31, "Ag": 1.53, "Cd": 1.48, "In": 1.44, "Sn": 1.41, "Sb": 1.38, "Te": 1.35,
           "I": 1.33, "Xe": 1.30,
           "Cs": 2.25, "Ba": 1.98, "La": 1.69, "Hf": 1.50, "Ta": 1.38, "W": 1.46, "Re": 1.59, "Os": 1.28,
           "Ir": 1.37, "Pt": 1.38, "Au": 1.36, "Hg": 1.49, "Tl": 1.48, "Pb": 1.46, "Bi": 1.46, "Po": 1.40,
           "At": 1.45, "Rn": 1.45}

    # Radii of elements found at http://de.wikipedia.org/wiki/Kovalenter_Radius
    # Ionic radii for alkaline and alkaline earth metals + Al found in ref: https://doi.org/10.1107/S0567739476001551

    rad_1 = rad[symbol1]
    rad_2 = rad[symbol2]

    rad_out = 1.3*(rad_1+rad_2)

    return rad_out

def get_inchi(sdf):

    """ Create an InChI with fixed hydrogens (!) based on an SDF """
    # Note: a non-standard InChI is generated (due to the usage of the two additional flags)

    inchi_cmd = ['obabel', '-isdf', str(sdf), '-xX', 'FixedH', 'DoNotAddH', '-oinchi']
    process = subprocess.run(inchi_cmd, stdout=subprocess.PIPE, universal_newlines=True)
    inchi = process.stdout.split()[0]
    return inchi

def get_separated_sdfs(sdf):

    """ Generate several SDF's from one SDF file, if disconnected molecules exist. """ 
    # Note: if one takes the product.sdf here with XYZ coordinates, the split of the sdf's happens according to the
    # coordinates and NOT according to the topology!!!

    sdf_cmd = "obabel -isdf " + str(sdf) + " -xX FixedH DoNotAddH --separate -osdf 2> /dev/null"
    sdf_out = Popen(sdf_cmd, stdout=PIPE, stderr=PIPE, shell=True).stdout.readlines()

    start = 0
    separated_sdfs = []
    
    for i, line in enumerate(sdf_out):
        #line = line.decode("utf-8")
        if line == "$$$$\n":
            end = i
            mol = [ sdf.decode("utf-8") for sdf in sdf_out[start:end]]
            #mol = [sdf for sdf in sdf_out[start:end]]
            start = end+1
            separated_sdfs.append(mol)

    return separated_sdfs

def molecule_separation(mol=None, symbols=None):
    
    """
    Check whether the product object contains one or more separated molecules. 
    It returns the list of the atom IDs belonging to one molecule.
    """
    natoms = mol.NumAtoms()
    prod_ob = BabelMolAdaptor(mol).openbabel_mol
    atomid_list = []
    edges = connectivity(mol=prod_ob, symbols=symbols)

    if len(edges) != 0:  # otherwise it is just an atom!
        G = nx.Graph()
        G.add_edges_from(edges)
        if not nx.is_connected(G):
            sub_graphs = nx.connected_component_subgraphs(G)
            graphs = [list(sg.nodes()) for sg in sub_graphs]
            atomid_list = sorted(graphs)
        else:
            atomid_list = [sorted(list(G.nodes()))]

    # add single atoms!
    flat_list = [item for sublist in atomid_list for item in sublist]
    if len(flat_list) != natoms:
        for j in range(1, natoms + 1):
            if j not in flat_list:
                atomid_list.append([j])

    return atomid_list

def OBMOLsplit(mol):

    """ Split an openbabel molecule object into disconnected molecules 
    (if there are any, otherwise return the complete molecule) """

    mols = []
    for m in mol.Separate():
        mols.append(m)
    return mols


def write_e_ts_p_xyz(PEH=None, path=None, node_structures=None):

    """ Write out files with the XYZ coordinates of the educt, TS and product structures """

    if PEH != None:
        educt = node_structures[PEH[0]]
        if PEH[1] != None:
            ts = node_structures[PEH[1]]

        product = node_structures[PEH[2]]

        out = open(path+"/product.xyz", "w")
        natoms = product.NumAtoms()
        out.write(str(natoms)+"\n\n")
        for atom in ob.OBMolAtomIter(product):
            out.write(ob.OBElementTable().GetSymbol(atom.GetAtomicNum()) + " " + str(atom.GetX()) + " " +
                        str(atom.GetY()) + " " + str(atom.GetZ()) + "\n")
        out.close()

        if PEH[1] != None:
            out2 = open(path+"/ts.xyz", "w")
            out2.write(str(natoms)+"\n")
            out2.write("0.00 \n")
            #out2.write(str(ts_energy)+"\n")
            ts_coords = []
            for atom in ob.OBMolAtomIter(ts):
                out2.write(ob.OBElementTable().GetSymbol(atom.GetAtomicNum()) + " " + str(atom.GetX()) + " " +
                            str(atom.GetY()) + " " + str(atom.GetZ()) + "\n")
                ts_coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
            out2.close()


        out3 = open(path+"/educt.xyz", "w")
        out3.write(str(natoms)+"\n\n")
        for atom in ob.OBMolAtomIter(educt):
            out3.write(ob.OBElementTable().GetSymbol(atom.GetAtomicNum()) + " " + str(atom.GetX()) + " " +
                        str(atom.GetY()) + " " + str(atom.GetZ()) + "\n")
        out3.close()


def readXYZmulti(datei):

    """
    Function which reads in XYZ files and returns openbabel molecule objects
    :param datei: file containing XYZ structure(s)
    :return: OBMol object(s) 
    """
    obconversion = ob.OBConversion()
    obconversion.SetInFormat("xyz")
    mol = ob.OBMol()
    mols = []

    control = obconversion.ReadFile(mol, str(datei))
    while control:
        mols.append(mol)
        mol = ob.OBMol()
        control = obconversion.Read(mol)

    return mols

