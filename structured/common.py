import pickle

import mdp
import numpy as np
from numpy import diag, dot, log, transpose as tr, outer
from numpy.random import normal
import scipy
import scipy.linalg
from scipy.linalg import calc_lwork


# Code copied from scipy, simplified and for real matrices only
# matrix inversion
a = scipy.zeros((1,1), dtype='d')
getrf,getri = scipy.linalg.lapack.get_lapack_funcs(('getrf','getri'),(a,))
def invfun(a, overwrite_a=0):
    """ inv(a, overwrite_a=0) -> a_inv

    Return inverse of square matrix a.
    """
    a1 = scipy.asarray_chkfinite(a)
    overwrite_a = overwrite_a or (a1 is not a and not hasattr(a,'__array__'))
    #XXX: C ATLAS versions of getrf/i have rowmajor=1, this could be
    #     exploited for further optimization. But it will be probably
    #     a mess. So, a good testing site is required before trying
    #     to do that.
    if getrf.module_name[:7]=='clapack'!=getri.module_name[:7]:
        # ATLAS 3.2.1 has getrf but not getri.
        lu,piv,info = getrf(tr(a1),
                            rowmajor=0,overwrite_a=overwrite_a)
        lu = tr(lu)
    else:
        lu,piv,info = getrf(a1,overwrite_a=overwrite_a)
    if info==0:
        if getri.module_name[:7] == 'flapack':
            lwork = calc_lwork.getri(getri.prefix,a1.shape[0])
            lwork = lwork[1]
            # XXX: the following line fixes curious SEGFAULT when
            # benchmarking 500x500 matrix inverse. This seems to
            # be a bug in LAPACK ?getri routine because if lwork is
            # minimal (when using lwork[0] instead of lwork[1]) then
            # all tests pass. Further investigation is required if
            # more such SEGFAULTs occur.
            lwork = int(1.01*lwork)
            inv_a,info = getri(lu,piv,
                               lwork=lwork,overwrite_lu=1)
        else: # clapack
            inv_a,info = getri(lu,piv,overwrite_lu=1)
    if info>0: raise np.linalg.LinAlgError, "singular matrix"
    if info<0: raise ValueError,\
       'illegal value in %-th argument of internal getrf|getri'%(-info)
    return inv_a


def log0(x):
    # this is the "entropy" log, i.e. log(x) if x!=0, 0 if x=0
    return np.choose(np.greater(x,0.), (0., log(x)))


def mult_diag(d, mtx, left=True):
    # this is better than dot if dim>10
    if left:
        #return tr(d*tr(mtx))
        return (d*mtx.T).T
    else:
        return d*mtx

a = scipy.zeros((1,1), dtype='d')
gemm, = scipy.linalg.get_blas_funcs(('gemm',), (a,a))

# ##### file management

def pickle_to(obj, fname):
    fid = open(fname, 'w')
    pickle.dump(obj, fid, 1)
    fid.close()

def pickle_from(fname):
    fid = open(fname, 'r')
    obj = pickle.load(fid)
    fid.close()
    return obj

# ##### data transformation

def bimult(w, c, s):
    dy, dc, dsc = w.shape
    tlen = c.shape[0]
    # slower, at least for rel. small dc and ds
    #y = sum([outer(s[:,j]*c[:,i], w[:,i,j])
    #         for i in range(dc) for j in range(ds)])
    y = np.zeros((tlen, dy), 'd')
    for i in range(dc):
        for j in range(dsc):
            y += outer(s[:,i,j]*c[:,i], w[:,i,j])
    return y

# ##### initialization

def random_weights_orth(dy, dc, dsc):
    tmp = mdp.utils.random_rot(dy)
    w = np.zeros((dy, dc, dsc), 'd')
    for i in range(dc):
        w[:,i,:] = tmp[:,i*dsc:(i+1)*dsc]
    return w


def random_weights_gamma(dy, dc, dsc, gamma):
    return normal(scale=np.sqrt(1./gamma), size=(dy,dc,dsc))


def random_weights_beal(dy, dc, dsc):
    g1 = normal(loc=2., scale=1., size=(dy,dc,dsc))
    g2 = normal(loc=-2., scale=1., size=(dy,dc,dsc))
    mix = np.random.randint(2, size=(dy,dc,dsc))
    return np.choose(mix, (g1, g2))


def _high_corrs(smn, thr):
    T, dc = smn.shape
    ncorrs = np.zeros((dc, dc), 'd')
    corrs = np.zeros((dc, dc), 'd')
    for i in range(dc):
        for j in range(dc):
            both = np.logical_and(abs(smn[:,i])>thr, abs(smn[:,j])>thr)
            tmp = np.take(smn[:,i], both.nonzero())[0]**2. * \
                  np.take(smn[:,j], both.nonzero())[0]**2.
            ncorrs[i,j] = len(tmp)
            #corrs[i,j] = sum(tmp)/len(tmp)
            corrs[i,j] = sum(tmp)
    return corrs, ncorrs


def _high_varcond(x, thr):
    # high conditional variance (GSM style)
    T, dc = x.shape
    varcond = np.zeros((dc, dc), dtype='d')
    for i in range(dc):
        large = abs(x[:,i])>thr
        for j in range(dc):
            varcond[j,i] = np.take(x[:,j], large.nonzero()).std()
    # symmetrize
    varcond = 0.5*(varcond+varcond.T)
    return varcond


def _aggregate_greedy(corrs, dc, dsc):
    eliminate = corrs.copy()
    perm = []
    for i in range(dc):
        idx0 = np.argmax(diag(eliminate)>0.)
        perm.append(idx0)
        eliminate[:,idx0] = 0.
        for j in range(1, dsc):
            idx = np.argmax(eliminate[idx0,idx0+1:])+idx0+1
            perm.append(idx)
            eliminate[:,idx] = 0.
    return perm


def _aggregate(corrs, dc, dsc):
    dim = corrs.shape[0]

    # create a list of correlation pairs, sorted by max. correlation
    npairs = dim*(dim-1)/2
    corrs_list = np.zeros((npairs,), dtype='d')
    # the first element in the pair is always smaller
    pairs = np.zeros((npairs,2), dtype='i')
    k = 0
    for i in range(dim):
        for j in range(i+1, dim):
            corrs_list[k] = corrs[i,j]
            pairs[k,:] = [i,j]
            k += 1

    idx = corrs_list.argsort()
    corrs_list.sort()
    pairs = np.take(pairs, idx, axis=0)

    perm_dict = {}
    tot = 0
    for i in range(npairs):
        el0, el1 = pairs[-i-1,:]
        ed0 = perm_dict.get(el0)
        ed1 = perm_dict.get(el1)
        
        # case 1: new pair, add new graph
        if ed0 is None and ed1 is None:
            edge = {el0:None, el1:None}
            perm_dict[el0] = perm_dict[el1] = edge
            tot += 2
        # one of the elements is already in graph, expand
        else:
            # case 1: both elements have graphs, merge
            if ed0 is not None and ed1 is not None:
                if el0 in ed1: continue
                if len(ed0)+len(ed1)>dsc: continue
                
                ed0.update(ed1)
                for a in ed1:
                    perm_dict[a] = ed0
            # case 2: only one element has graph, extend
            else:
                edge = max(perm_dict.get(el0), perm_dict.get(el1))
                if len(edge)+1>dsc: continue
                edge.update([(el0, None), (el1, None)])
                perm_dict[el0] = perm_dict[el1] = edge
                tot += 1
        
        #print el0, el1
        #for el,neigh in perm_dict.items():
        #    print el, neigh.keys()
        #print

        if tot==dc*dsc: break

    eliminated = np.zeros((dim,), dtype='i')
    perm = []
    for k in perm_dict.keys():
        if not eliminated[k]:
            for el in perm_dict[k]:
                perm.append(el)
                eliminated[el]=1
        
    return perm


def aggregate_weights(filters, x, dc, dsc, thr=1.):
    dy = filters.shape[0]
    corrs = _high_varcond(x, thr)
    perm = _aggregate(corrs, dc, dsc)

    w2 = np.zeros((dy, dc, dsc), 'd')
    k = 0
    for i in range(dc):
        for j in range(dsc):
            w2[:,i,j] = filters[:,perm[k]]
            k += 1

    return w2


def initial_ica_sfa_weights(y, dc, dsc, g='pow3', approach='symm', thr=1.):
    if len(y.shape)==2:
        y = y.copy()
        y.resize((1, y.shape[0], y.shape[1]))
    nseq, T, dy = y.shape

    def seqgen():
        for i in range(nseq):
            yield np.squeeze(y[i,:,:])
    
    flow = mdp.Flow([mdp.nodes.WhiteningNode(output_dim=dc*dsc),
                     mdp.nodes.FastICANode(whitened=1, approach=approach,
                                           dtype='d', g=g, max_it=5000)],
                    verbose=1)
    flow.train([seqgen(), seqgen()])
    flow[0].v = np.real_if_close(flow[0].v)
    flow[1].filters = np.real_if_close(flow[1].filters)
    smn = flow.execute(seqgen())

    tmp = tr(dot(mdp.numx_linalg.pinv(flow[1].filters), flow[0].get_recmatrix()))
    return aggregate_weights(tmp, smn, dc, dsc, thr=1.)


#-------- evaluations

# code to generate permutations by Ulrich Hoffmann
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/190465
def xcombinations(items, n):
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in xcombinations(items[:i]+items[i+1:],n-1):
                yield [items[i]]+cc


def xpermutations(items):
    return xcombinations(items, len(items))


def xpartial_permutations(items_list):
    if len(items_list)==0: yield []
    else:
        for p1 in xpermutations(items_list[0]):
            for p2 in xpartial_permutations(items_list[1:]):
                yield p1 + p2


def xsytle_permutations(dc, dsc):
    items_list = [range(i*dc, (i+1)*dc) for i in range(dsc)]
    for p in xpartial_permutations(items_list):
        yield [p[i] for j in range(dc) for i in range(j, dc*dsc, dc)]


def volume(V):
    return np.sqrt(mdp.numx_linalg.det(dot(tr(V), V)))


def _get_basis(idx, w, ww, i):
    if len(idx)>1:
        basis = scipy.linalg.orth(w)
    elif len(idx)==0:
        basis = 0.
    else:
        basis = w/ww[i,idx[0]]
    return basis


def evaluate_style_subspace(w, w2, threshold=0.01):
    dy, dc, dsc = w.shape

    # get thresholds for a weight being zero
    ww = np.sqrt((w*w).sum(axis=0))
    tsh = ww.max()*threshold
    w2w2 = np.sqrt((w2*w2).sum(axis=0))
    tsh2 = w2w2.max()*threshold
    
    vols = []
    # 0 for equality, 1 if dim(w)>dim(w2), 2 if dim(w2)>dim(w)
    dimrel = []
    for i in range(dc):
        # indices of weights in content i considered more than 0
        idx = np.where(ww[i,:]>tsh)[0]
        idx2 = np.where(w2w2[i,:]>tsh2)[0]
        wsub, w2sub = w[:,i,idx], w2[:,i,idx2]

        # very special case: both subspaces are 0
        if len(idx)==0 and len(idx2)==0:
            dimrel.append(0)
            vols.append(0.)
            continue

        # get a basis of the subspace
        basis = _get_basis(idx, wsub, ww, i)
        basis2 = _get_basis(idx2, w2sub, w2w2, i)

        # project it on the other subspace
        if len(idx)>len(idx2):
            dimrel.append(1)
            proj = dot(basis, dot(tr(basis), basis2))
        elif len(idx)<len(idx2):
            dimrel.append(2)
            proj = dot(basis2, dot(tr(basis2), basis))
        else:
            dimrel.append(0)
            proj = dot(basis, dot(tr(basis), basis2))
        
        vols.append(1.-volume(proj))
    return vols, dimrel

def evaluate_style_subspace_perms(w, w2):
    dy, dc, ds = w2.shape
    take = np.take

    mindiff = np.inf
    bestc = None
    bestvols = None
    for permc in xpermutations(range(dc)):
        wp = take(w2, permc, axis=1)
        vols, dimrel = evaluate_style_subspace(w, wp)
        diff = sum(np.array(vols,'d')**2.)
        if diff<mindiff or bestc==None:
            mindiff = diff
            bestc = permc
            bestvols = vols

    return bestvols, dimrel, bestc

def evaluate_subspace(w, w2):
    dy, dc, dsc = w.shape
    a = np.reshape(w, (dy, 1, dc*dsc))
    a2 = np.reshape(w2, (dy, 1, dc*dsc))
    return evaluate_style_subspace(a, a2)[0]

def evaluate_style_subspace_corrs(w, w2, c, c2):
    dy, dc, ds = w2.shape

    corrs = np.zeros((dc, dc), 'd')
    for i in range(dc):
        ci = c[:,i]-c[:,i].mean()
        for j in range(dc):
            cj = c2[:,j]-c2[:,j].mean()
            corrs[i,j] = (ci*cj).mean()
    
    perm = np.zeros((dc,), dtype='i')
    for i in range(dc):
        idx = corrs.argmax()
        idx1 = idx / dc
        idx2 = idx % dc
        perm[idx1] = idx2
        corrs[idx1,:] = -1.
        corrs[:,idx2] = -1.
    
    wp = np.take(w2, perm, axis=1)
    vols, dimrel = evaluate_style_subspace(w, wp)
    
    return vols, dimrel, perm
