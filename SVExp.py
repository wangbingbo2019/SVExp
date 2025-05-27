import sys
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
from pandas.core.frame import DataFrame
import scanpy as sc
import scipy as sp
from scipy.spatial.distance import euclidean
from scipy.stats import norm, entropy, mannwhitneyu, spearmanr, pearsonr, nbinom
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score, silhouette_samples, mutual_info_score, adjusted_rand_score
import matplotlib.image as mpimg
from matplotlib import rcParams
import seaborn as sns
import anndata as ad
import h5py
import json
from PIL import Image

def cal_distance(x, y):
    l = len(x)
    pbar = tqdm(total=l)
    d_mtx = np.zeros([l,l], dtype=np.float32)
    for i in range(l):
        for j in range(i,l):
            d_mtx[i][j] = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
            d_mtx[j][i] = d_mtx[i][j]
        pbar.update(1)
    for i in range(l):
        d_mtx[i][i] = 999999999
    return d_mtx
    
def neg_log_likelihood(params, data):
    n, p = params
    return -np.sum(nbinom.logpmf(data,n,p))

def data_generate(species, df_count, loc, df_domain, target_domain, target_correlation, l_range, noise_level, simulate_nums, sim_dir, LR_path='./LR_data/human-', use_real_names=False):
    if species not in ['mouse', 'human']:
        raise ValueError('species must be one of the following: '+str(['mouse', 'human']))
    
    if species == 'mouse':
        f = open('./mouse_human_homologs.txt','r')
        m2h = []
        for i in f:
            m2h.append(i[:-1].split('\t'))
        f.close()
        
        df_value = df_count.values.T.tolist()
        df_tmp=[]
        re_col=[]
        for i in range(len(df_count.columns)):
            for j in range(len(m2h)):
                if m2h[j][0] == df_count.columns[i]:
                    if m2h[j][1] not in re_col:
                        re_col.append(m2h[j][1])
                        df_tmp.append(df_value[i])
                    break
        df_count = DataFrame(df_tmp,index=re_col,columns=df_count.index).T
    
    geneInter = pd.read_csv(LR_path + 'interaction_input_CellChatDB.csv.gz', index_col=0, compression='gzip')
    comp = pd.read_csv(LR_path + 'complex_input_CellChatDB.csv', header=0, index_col=0)
    
    geneInter = geneInter.sort_values('annotation')
    ligand = geneInter.ligand.values
    receptor = geneInter.receptor.values
    
    

    loc1=loc.loc[df_count.index,'x'].values
    loc2=loc.loc[df_count.index,'y'].values
    
    print('calculating neighbours graph...')
    method='knn'
    threshold=6
    d_mtx = cal_distance(loc1, loc2)
    
    if method == 'knn':
        l = len(loc1)
        neighbours = []
        for i in range(l):
            ind, _ = zip(*sorted(enumerate(d_mtx[i]), key = lambda x : x[1]))
            neighbours.append(list(ind)[:threshold])
    elif method == 'co':
        l = len(loc1)
        neighbours = []
        for i in range(l):
            neighbours.append([])
            for j in range(l):
                if d_mtx[i][j] < threshold and j != i:
                    neighbours[i].append(j)
    
    with open(sim_dir+'neighbours.txt','w') as f:
        for i in range(len(neighbours)):
            f.write(df_count.index[i]+' ')
            for j in neighbours[i]:
                f.write(df_count.index[j]+' ')
            f.write('\n')
    
    domain_labels = np.array(df_domain.loc[:,'label'].tolist())
    domain_labels = np.array([str(x) for x in domain_labels])
    domain_label_set = list(set(domain_labels))
    
    total_count_per_gene = df_count.sum().sort_values()
    target_g = total_count_per_gene.index[-1]
    init = (1, 0.5)
    bounds = ((1e-6, None), (1e-6, 1 - 1e-6))

    data = [int(x) for x in df_count.loc[:,target_g]]
    res = minimize(neg_log_likelihood, init, args=(data,), bounds=bounds)
    n, p = res.x
    
    LRs=[]
    LR_names=[]
    have={}
    sim_gene=sum(simulate_nums)

    if use_real_names:
        for i in range(len(ligand)):
            if sim_gene == 0:
                break
            if ligand[i] in comp.index or receptor[i] in comp.index or ligand[i] in receptor or receptor[i] in ligand:
                continue
            flag = False
            for j in range(len(ligand)):
                if ligand[i] == ligand[j] and receptor[j] in comp.index:
                    flag = True
                    break
                if receptor[i] == receptor[j] and ligand[j] in comp.index:
                    flag = True
                    break
            if flag:
                continue
            if ligand[i] not in have and receptor[i] not in have and ligand[i] != receptor[i]:
                have[ligand[i]]=1
                for j in range(len(ligand)):
                    if ligand[i] == ligand[j]:
                        have[receptor[j]]=1
                
                have[receptor[i]]=1
                for j in range(len(receptor)):
                    if receptor[i] == receptor[j]:
                        have[ligand[j]]=1
                LRs.append(ligand[i])
                LRs.append(receptor[i])
                LR_names.append(geneInter.index[i])
                sim_gene-=1

    while sim_gene != 0:
        LRs.append('sim_l'+str(sim_gene))
        LRs.append('sim_r'+str(sim_gene))
        LR_names.append('sim_l'+str(sim_gene)+'_''sim_r'+str(sim_gene))
        sim_gene-=1

    neighbours = []
    with open(sim_dir+'neighbours.txt','r') as f:
        for line in f:
            s = line.split(' ')[:-1]
            neighbours.append(s[1:])

    neighbours_ind = {}
    for i in range(len(neighbours)):
        neighbours_ind[i] = []
        for j in range(len(df_domain.index)):
            if df_domain.index[j] in neighbours[i]:
                neighbours_ind[i].append(j)

    pbar = tqdm(total=len(noise_level)*len(l_range)*sum(simulate_nums))
    sim_id = 0
    for l1 in range(len(noise_level)):
        for l2 in range(len(l_range)):
            sim_data = pd.DataFrame(index=df_count.index, columns=LRs)
            ground_truth = pd.DataFrame(index=LR_names,columns=['spatial variable'])
            sim_ind = 0
            for i in range(len(simulate_nums)):
                now_nums = simulate_nums[i]
                while now_nums > 0:
                    simulated_g1 = nbinom.rvs(n, p, size=len(data))
                    simulated_g2 = nbinom.rvs(n, p, size=len(data))
                    
                    if target_correlation[i] == 0:
                        inds = [ind for ind, x in enumerate(domain_labels) if x in target_domain[i]]
                        tmp_exp = [simulated_g1[g] for g in inds]
                        mea = np.mean(tmp_exp)
                        b = sorted(tmp_exp)[-int(len(tmp_exp)*0.1)]
                        for ind in inds:
                            avg_exp = 0
                            for nn in neighbours_ind[ind]:
                                avg_exp += simulated_g1[nn]
                            avg_exp /= len(neighbours_ind[ind])
                            
                            rand_select = np.random.uniform(0, 1)
                            if rand_select > noise_level[l1]:
                                simulated_g2[ind] = np.random.uniform(-l_range[l2][1],-l_range[l2][0]) * (avg_exp - mea) + b + np.random.normal(0,0.5)
                            simulated_g2[ind] = max(simulated_g2[ind], 0)
                        ground_truth.loc[LR_names[sim_ind//2],'spatial variable'] = 'negative-linear'
                    elif target_correlation[i] == 1:
                        inds = [ind for ind, x in enumerate(domain_labels) if x in target_domain[i]]
                        for ind in inds:
                            avg_exp = 0
                            for nn in neighbours_ind[ind]:
                                avg_exp += simulated_g1[nn]
                            avg_exp /= len(neighbours_ind[ind])
                            
                            rand_select = np.random.uniform(0, 1)
                            if rand_select > noise_level[l1]:
                                simulated_g2[ind] = np.random.uniform(l_range[l2][0],l_range[l2][1]) * avg_exp + np.random.normal(0,0.5)
                        ground_truth.loc[LR_names[sim_ind//2],'spatial variable'] = 'linear'
                    elif target_correlation[i] == 2:
                        inds = [ind for ind, x in enumerate(domain_labels) if x in target_domain[i]]
                        tmp_exp = [simulated_g1[g] for g in inds]
                        ma = max(tmp_exp)
                        mi = min(tmp_exp)
                        medi = np.median(tmp_exp)
                        mea = np.mean(tmp_exp)
                        b = sorted(tmp_exp)[-int(len(tmp_exp)*0.1)]
                        b2 = sorted(tmp_exp)[int(len(tmp_exp)*0.1)]
                        for ind in inds:
                            avg_exp = 0
                            for nn in neighbours_ind[ind]:
                                avg_exp += simulated_g1[nn]
                            avg_exp /= len(neighbours_ind[ind])
                            
                            rand_select = np.random.uniform(0, 1)
                            if rand_select > noise_level[l1]:
                                simulated_g2[ind] = -(b/medi**2) * (avg_exp - medi)**2 + b + np.random.normal(0,b2)
                            simulated_g2[ind] = max(simulated_g2[ind], 0)
                        ground_truth.loc[LR_names[sim_ind//2],'spatial variable'] = 'non-linear'
                    elif target_correlation[i] == 3:
                        inds = [ind for ind, x in enumerate(domain_labels) if x in target_domain[i]]
                        tmp_exp = [simulated_g1[g] for g in inds]
                        ma = max(tmp_exp)
                        mi = min(tmp_exp)
                        medi = np.median(tmp_exp)
                        mea = np.mean(tmp_exp)
                        b = sorted(tmp_exp)[-int(len(tmp_exp)*0.1)]
                        b2 = sorted(tmp_exp)[int(len(tmp_exp)*0.1)]
                        for ind in inds:
                            avg_exp = 0
                            for nn in neighbours_ind[ind]:
                                avg_exp += simulated_g1[nn]
                            avg_exp /= len(neighbours_ind[ind])
                            
                            rand_select = np.random.uniform(0, 1)
                            if rand_select > noise_level[l1]:
                                flag = int(np.random.uniform(0,3))
                                if flag == 0:
                                    simulated_g2[ind] = (-(b/medi**2) * (avg_exp - medi)**2 + b) + np.random.normal(0,b2)
                                if flag == 1:
                                    simulated_g2[ind] = np.random.uniform(-l_range[l2][1],-l_range[l2][0]) * (avg_exp - mea) + b + np.random.normal(0,0.5)
                                if flag == 2:
                                    simulated_g2[ind] = np.random.uniform(l_range[l2][0],l_range[l2][1]) * avg_exp + np.random.normal(0,0.5)
                            simulated_g2[ind] = max(simulated_g2[ind], 0)
                        ground_truth.loc[LR_names[sim_ind//2],'spatial variable'] = 'mixed'
                    else:
                        raise ValueError('target_correlation must be a list consists of 0, 1, 2 and 3')
                
                    sim_data[LRs[sim_ind]] = simulated_g1
                    sim_data[LRs[sim_ind+1]] = simulated_g2
                    now_nums -= 1
                    sim_ind += 2
                    pbar.update(1)

            sim_data.to_csv(sim_dir+'count.csv', sep='\t')
            ground_truth.to_csv(sim_dir+'ground_truth.csv', sep='\t')
            loc.to_csv(sim_dir+'loc.csv', sep='\t')
            df_domain.to_excel(sim_dir+'domain.xlsx')
            sim_id += 1


def validate(lrs, clusters, roi_with_c, domains, ground_truth, out_dir):
    truth = ground_truth.loc[:, 'spatial variable']
    pred = []
    for i in ground_truth.index:
        if not i in clusters.index:
            pred.append(-1)
        else:
            pred.append(clusters.loc[i, 'pattern'])
    ARI = adjusted_rand_score(truth, pred)
    
    si_score = {roi_with_c[i]:[] for i in roi_with_c.keys()}
    
    cluster2domain_count = {i:{roi_with_c[j]:0 for j in roi_with_c.keys()} for i in set(clusters.loc[:, 'pattern'])}
    cluster2domain = {}
    for lr in clusters.index:
        cluster2domain_count[clusters.loc[lr, 'pattern']][roi_with_c[ground_truth.loc[lr, 'spatial variable']]] += 1
    for cluster in cluster2domain_count.keys():
        for domain in cluster2domain_count[cluster].keys():
            if not cluster in cluster2domain or cluster2domain_count[cluster][domain] > max_match:
                cluster2domain[cluster] = domain
                max_match = cluster2domain_count[cluster][domain]
                
    for i in range(len(lrs)):
        cluster = clusters.loc[lrs.index[i], 'pattern']
        silhouette_labels = [x if x==cluster2domain[cluster] else 'others' for x in domains]
        data = []
        for j in range(len(lrs.columns)):
            data.append([lrs.iloc[i][j]])
        score = silhouette_score(np.array(data),silhouette_labels)
        si_score[cluster2domain[cluster]].append(score)
        
    print('ARI:'+str(ARI))
    with open(out_dir+'si_score.txt', 'w') as f:
        json.dump(si_score, f)
    print(si_score)


def plot_validation(si_score, out_dir):
    with open(out_dir+'si_score.txt', 'r') as f:
        si_score = json.load(f)
    print(si_score)
    for i in si_score.keys():
        plt.cla()
        plt.clf()
        data = pd.DataFrame({'x':['']*len(si_score[i]), 'y':si_score[i]})
        sns.violinplot(x='x', y='y', data=data, width=.3, color='r')
        ax = plt.gca()
        #ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticks([-1,-0.5,0,0.5,1])
        ax.set_ylabel('')
        ax.set_yticklabels([-1,-0.5,0,0.5,1], size=20)
        ax.set_xlabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(out_dir+str(i)+'.pdf')










