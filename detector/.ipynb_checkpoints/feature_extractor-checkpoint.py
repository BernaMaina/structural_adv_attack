#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
import angr
import statistics
import os, csv, sys
import signal, gc, re


class timeout(Exception):
    """Custom exception to handle timeout errors."""
    pass

def signal_handler(signum, frame):
    """Raises a TimeoutException when a timeout signal is received."""
    raise timeout("Timed out!")

def angr_extract_feature(fname, trans, cal, ctl):

    """
    Extracts graphical properties and instruction counts from the binary's CFG.

    Args:
    - fname (str): Path to the binary file.
    - trans (list): List of transfer-related instructions.
    - cal (list): List of calculation-related instructions.
    - ctl (list): List of control-related instructions.

    Returns:
    - List: A list of extracted features including graph properties and instruction-based features.
    """

    p = angr.Project(fname, load_options={'auto_load_libs': False})
    
    # Generate a static CFG
    cfg = p.analyses.CFGFast()
    # Generate a dynamic CFG
    #cfg = p.analyses.CFGEmulated(keep_state=True)

    G = cfg.graph
    G_undirected = G.to_undirected() # cfg graphs are directed and not undirected

    # ---------------------------------------------------------

    # nodes & edges
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()

    # ---------------------------------------------------------
    
    # degree
    # Cal degree
    idegree = {d[0]:d[1] for d in G.in_degree()}
    odegree = {d[0]:d[1] for d in G.out_degree()}

    # normalized
    norm_in_degree = {_:idegree[_]/sum(idegree.values()) for _ in idegree}
    norm_out_degree = {_:odegree[_]/sum(odegree.values()) for _ in odegree}

    # mean
    in_degree = np.mean([_ for _ in norm_in_degree.values()])
    out_degree = np.mean([_ for _ in norm_out_degree.values()])

    # ---------------------------------------------------------

    # density
    density = nx.density(G)

    # closeness_centrality
    closeness_centrality = np.mean(list(nx.closeness_centrality(G).values()))

    # betweeness_centrality
    betweeness_centrality = np.mean(list(nx.betweenness_centrality(G).values()))

    # connected_components
    connected_components = nx.number_connected_components(G_undirected)

    # ---------------------------------------------------------
    
    # shortest_path
    short_path = dict(nx.all_pairs_shortest_path(G))
    short_path_value = {}

    for i in short_path:
        temp = {}
        for j in short_path[i]:
            if i != j:
                temp[j] = len(short_path[i][j])
        short_path_value[i] = temp

    # ---------------------------------------------------------

    # diameter and radius
    sp = []
    for _ in short_path_value:
        sp.extend([i for i in short_path_value[_].values()])

    diameter = max(sp)
    radius = min(sp)

    # ---------------------------------------------------------
    instr_list = {'trans': 0, 'cal': 0, 'ctl': 0}

    regex = re.compile('\t\w+\t')
    instruction = []
    block_num = nodes
    func_size = []

    for n in G.nodes(data=True):
        try:
            block_split = p.factory.block(n[0].function_address).capstone.insns
            func_size.append(len(block_split))
            for __ in block_split:
                instruction.append(regex.findall(str(__))[0][1:-1])
        except Exception as e:
            pass
            print(e)


    for _ in instruction:
        if _ in trans:
            instr_list['trans'] += 1
        elif _ in cal:
            instr_list['cal'] += 1
        elif _ in ctl:
            instr_list['ctl'] += 1

    # total instruction count
    total_trans = instr_list['trans']
    total_cal = instr_list['cal']
    total_ctl = instr_list['ctl']

    # Avg. instruction count
    for _ in instr_list.keys():
        instr_list[_] /= block_num
    avg_trans = instr_list['trans']
    avg_cal = instr_list['cal']
    avg_ctl = instr_list['ctl']
    # ---------------------------------------------------------

    avg_block = edges / block_num
    avg_block_size = statistics.mean(func_size)

    # return
    return [nodes, edges, out_degree, in_degree, density, closeness_centrality, betweeness_centrality, connected_components, diameter, radius,
            total_trans, total_cal, total_ctl, avg_trans, avg_cal, avg_ctl, avg_block, avg_block_size]

def read_label():

    """
    Reads labels, thresholds, and architectures from the dataset CSV.

    Returns:
    - tuple: Contains dictionaries for labels, thresholds, and architectures.
    """
    label_dict = {'BenignWare':0, 'Mirai':1, 'Tsunami':2, 'Hajime':3, 'Dofloo':4, 'Bashlite':5, 'Xorddos':6, 'Android':7, 'Pnscan':8, 'Unknown':9}
    Arch_dict = {'armel': 1, 'x86el': 2, 'mipsel': 3, 'sparceb': 4, 'x86_64el': 5, 'ppceb': 6, 'mipseb': 7, 'mips64eb': 8, 'ppcel': 9, 'armeb': 10, 'm68keb': 11, 'unknown': 12}
    label = {}
    threshold = {}
    Arch = {}
    with open('metadata.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        for row in rows:
            threshold[row[0]] = row[2]
            label[row[0]] = label_dict[row[1]]
            Arch[row[0]] = Arch_dict[row[3]]
    print('---- finish read label ----\n')

    return label, threshold, Arch

def feature_csv(label, threshold, feature_vec, Arch):

    """
    Writes the extracted features to a CSV file.

    Args:
    - label (dict): Dictionary of labels.
    - threshold (dict): Dictionary of thresholds.
    - feature_vec (dict): Dictionary of feature vectors.
    - Arch (dict): Dictionary of architectures.
    """

    column_names = ['filename', 'label', 'arch', 'nodes', 'edges', 'out_degree', 'in_degree', 
                    'density', 'closeness_centrality', 'betweeness_centrality', 
                    'connected_components', 'diameter', 'radius', 'total_trans', 
                    'total_cal', 'total_ctl', 'avg_trans', 'avg_cal', 'avg_ctl', 
                    'avg_block', 'avg_block_size']
    
    # Check if the file exists and is empty
    file_exists = os.path.isfile('./csv_outputs/train_data.csv')
    
    with open('./csv_outputs/train_data.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header only if the file is new or empty
        if not file_exists or os.stat('./csv_outputs/train_data.csv').st_size == 0:
            writer.writerow(column_names)
        
        for i in feature_vec:
            if threshold[i] != False:
                writer.writerow([i, label[i], Arch[i]] + feature_vec[i])
            if sum(feature_vec[i]) == 0:
                print(i, 'error')


def main():

    """
    Main function to extract features from ELF files and write them to a CSV file.
    """
    sys.setrecursionlimit(100000000)
    Timeout = 600
    
    #elf_type = ['benignware']
    elf_type = ['mal']

    feature_vec = {}
    count = 0

    # read label
    csv.field_size_limit(sys.maxsize)
    label, threshold, Arch = read_label()

    # instruction category
    trans = ['mov', 'movabs', 'push', 'pop', 'lb', 'lbu', 'lh', 'lw', 'sb', 'sh', 'sw', 'ldc', 'lea', 'restore', 
             'lswi', 'sts', 'usp', 'srs', 'pea', 'lui', 'lhu', 'rdcycle', 'rdtime', 'rdinstret']

    
    cal = ['add', 'sub', 'inc', 'xor', 'sar', 'addi', 'addiu', 'addu', 'and', 'ldr', 'andi', 'nor', 'or', 'ori', 'subu', 'auipc',
           'xori', 'div', 'divu', 'mfhi', 'mflo', 'mthi', 'mtlo', 'mult', 'multu', 'sll', 'sllv', 'sra', 'srav', 'srl', 'adc',
           'srlv', 'bic', 'xnor', 'not', 'eor', 'asr', 'fabs', 'abs', 'mac', 'neg', 'cmp', 'test', 'slti', 'slt', 'sltu', 'srai', 
           'sltui', 'sltiu', 'cmn', 'fcmp', 'dcbi', 'tas', 'btst', 'cbw', 'cwde', 'cdqe', 'cdq', 'slli', 'srli',  'sbb']

    
    ctl = ['jmp', 'jz', 'jnz', 'jne', 'je', 'call', 'jr', 'beq', 'bge', 'bgeu', 'bgez', 'bgezal', 'bgtz', 'blez', 'blt', 'bltu', 'bltz',
           'bltzal', 'bne', 'break', 'j', 'jal', 'jalr', 'mfc0', 'mtc0', 'syscall', 'leave', 'hvc', 'svc', 'hlt', 'arpl', 'sys', 'ti', 
           'trap', 'ret', 'retn', 'bl', 'bicc', 'bclr', 'bsrf', 'rte', 'wait', 'fwait', 'wfe', 'ecall', 'ebreak', 'jb', 'jbe']
    
    for Type in elf_type:
        ELF_path = '/home/bernard/dataset/' + Type
        for dirPath, dirNames, fileNames in os.walk(ELF_path):
            dirNames.sort()
            #fileNames.sort()
            print('Extracting CFG from ', dirPath)
            for f in fileNames:
                try:
                    signal.signal(signal.SIGALRM, signal_handler)
                    signal.alarm(Timeout)

                    #file_dir = f[:2]
                    #fname = ELF_path + '/' + file_dir + '/' + f
                    fname = ELF_path + '/' + f
                    feature_vec[f] = angr_extract_feature(fname, trans, cal, ctl)
                    signal.alarm(0)

                    if f in feature_vec.keys():
                        count += 1

#                     if count % 1 == 0:
                    print('extract %d files' % count)

                    # feature output
                    feature_csv(label, threshold, feature_vec, Arch)
                    #del feature_vec
                    gc.collect()
                    feature_vec = {}

                    # log
                    logfile = open('train_set.txt', 'w')
                    logfile.write('extracted %d files\n' % count)
                    logfile.close()

                except timeout:
                    print('Timed out!')
                    signal.alarm(0)
                    del feature_vec
                    gc.collect()
                    feature_vec = {}
                    continue
                except KeyboardInterrupt:
                    print('KeyboardInterrupt!')
                    signal.alarm(0)
                    del feature_vec
                    gc.collect()
                    feature_vec = {}
                    continue
                except Exception as e:
                    print(f, ' Unknown error occured!')
                    print(e)
                    signal.alarm(0)
                    del feature_vec
                    gc.collect()
                    feature_vec = {}
                    continue            


print('finish!')

if __name__ == '__main__':
    main()

