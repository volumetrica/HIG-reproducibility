"""
HIG real-image predictive evaluation pipeline
=============================================

Purpose
-------
1. Load real images of arbitrary H x W size (or user-provided segmentation maps).
2. Build a fixed 4-connected segmentation from each image when a label map is not supplied.
3. Construct the intrinsic HIG and extract descriptor sets for:
   - segmentation-size baseline
   - RAG
   - regional contact multigraph
   - HIG without cuts
   - binary-incidence HIG
   - full HIG
4. Evaluate the representations with repeated nested cross-validation using
   regularized logistic regression.
5. Optionally evaluate matched image variants inspired by Fig. 14 of
   Mendes Forte, Passat & Kenmochi (2026): original, quasi-closing,
   quasi-opening, 'as contrasted as possible', 'as flat as possible'.

Important
---------
The Fig.-14 operators themselves are NOT reimplemented here. If exact variants
are desired, generate them with the corresponding topological-tree/connected-
operator implementation and list them in metadata.csv with the same sample_id.
The pipeline then measures HIG descriptor stability and predictive transfer.
"""

from __future__ import annotations

import os
import math
import json
import time
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

OUT = -1
DIR4 = [(-1,0),(1,0),(0,-1),(0,1)]
STRUCT8 = np.ones((3,3), dtype=np.uint8)


# -----------------------------------------------------------------------------
# I/O and segmentation
# -----------------------------------------------------------------------------

def read_grayscale_uint8(path: str) -> np.ndarray:
    """Read any common image format and convert deterministically to 8-bit gray."""
    with Image.open(path) as im:
        if im.mode not in ("L", "I;16", "I", "F"):
            im = im.convert("L")
            arr = np.asarray(im, dtype=np.uint8)
            return arr
        arr = np.asarray(im)

    if arr.dtype == np.uint8:
        return arr
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.nanmin(arr[finite])); hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (255.0 * (arr - lo) / (hi - lo)).clip(0,255)
    return arr.astype(np.uint8)


def read_label_map(path: str) -> np.ndarray:
    """Read a segmentation map. RGB/RGBA colors are converted to integer labels."""
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 2:
        # Preserve arbitrary integer values but compact them for memory.
        _, inv = np.unique(arr, return_inverse=True)
        return inv.reshape(arr.shape).astype(np.int32)
    if arr.ndim == 3:
        # Ignore alpha for label identity if present.
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        flat = arr.reshape(-1, arr.shape[2])
        _, inv = np.unique(flat, axis=0, return_inverse=True)
        return inv.reshape(arr.shape[:2]).astype(np.int32)
    raise ValueError(f"Unsupported segmentation-map shape: {arr.shape}")


def quantize_uniform(gray: np.ndarray, levels: int = 8) -> np.ndarray:
    if levels < 2:
        raise ValueError("levels must be >= 2")
    q = (gray.astype(np.uint16) * levels) // 256
    q[q == levels] = levels - 1
    return q.astype(np.int32)


def quantize_multiotsu(gray: np.ndarray, levels: int = 5) -> np.ndarray:
    """Optional image-adaptive quantization. Requires scikit-image."""
    from skimage.filters import threshold_multiotsu
    vals = np.unique(gray)
    if len(vals) <= 1:
        return np.zeros_like(gray, dtype=np.int32)
    classes = min(levels, len(vals))
    if classes <= 1:
        return np.zeros_like(gray, dtype=np.int32)
    thresholds = threshold_multiotsu(gray, classes=classes)
    return np.digitize(gray, bins=thresholds).astype(np.int32)


def segment_image(path: str,
                  segmentation_path: Optional[str] = None,
                  mode: str = "uniform",
                  levels: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gray_uint8, categorical_segmentation)."""
    gray = read_grayscale_uint8(path)
    if segmentation_path and str(segmentation_path).strip() and os.path.exists(str(segmentation_path)):
        seg = read_label_map(str(segmentation_path))
        if seg.shape != gray.shape:
            raise ValueError(
                f"Image/segmentation size mismatch for {path}: {gray.shape} vs {seg.shape}"
            )
        return gray, seg
    if mode == "uniform":
        seg = quantize_uniform(gray, levels)
    elif mode == "multiotsu":
        seg = quantize_multiotsu(gray, levels)
    else:
        raise ValueError("mode must be 'uniform' or 'multiotsu'")
    return gray, seg


# -----------------------------------------------------------------------------
# Intrinsic HIG engine (sparse, arbitrary H x W)
# -----------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int64)
        self.r = np.zeros(n, dtype=np.uint8)
    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def label_4_regions(seg: np.ndarray):
    """Connected equal-label regions under 4-adjacency."""
    from skimage.measure import label
    # background=-1 ensures label value 0 is treated like any other label.
    rid = label(seg.astype(np.int64), connectivity=1, background=-1).astype(np.int64) - 1
    nreg = int(rid.max()) + 1 if rid.size else 0
    slices = ndi.find_objects((rid + 1).astype(np.int32), max_label=nreg)
    sizes = np.bincount(rid.ravel(), minlength=nreg).astype(np.int64)
    values = ndi.minimum(seg, labels=rid, index=np.arange(nreg)) if nreg else np.array([])
    return rid, nreg, slices, sizes, np.asarray(values)


def build_frontier(region_id: np.ndarray):
    """
    Build compactified frontier graph F.
    Vertices are primal grid vertices encoded as integers.
    Edges store region pairs and adjacent pixel ids.
    """
    h, w = region_id.shape
    gv_w = w + 1
    v1=[]; v2=[]; pa=[]; pb=[]; c1=[]; c2=[]
    vertex_to_edges = defaultdict(list)
    pixpair_to_edge = {}

    def add_edge(a_v, b_v, r1, r2, p1, p2):
        idx = len(v1)
        rr = (int(r1), int(r2))
        if rr[0] > rr[1]: rr = (rr[1], rr[0])
        v1.append(int(a_v)); v2.append(int(b_v)); pa.append(rr[0]); pb.append(rr[1])
        c1.append(int(p1)); c2.append(int(p2))
        vertex_to_edges[int(a_v)].append(idx); vertex_to_edges[int(b_v)].append(idx)
        if p2 >= 0:
            key = (int(min(p1,p2)), int(max(p1,p2)))
            pixpair_to_edge[key] = idx

    # Frame: top/bottom
    for c in range(w):
        p_top = c
        r = int(region_id[0,c])
        add_edge(c, c+1, OUT, r, p_top, -1)
        p_bot = (h-1)*w + c
        a = h*gv_w + c
        add_edge(a, a+1, OUT, int(region_id[h-1,c]), p_bot, -1)
    # Frame: left/right
    for r0 in range(h):
        p_left = r0*w
        a = r0*gv_w
        add_edge(a, a+gv_w, OUT, int(region_id[r0,0]), p_left, -1)
        p_right = r0*w + (w-1)
        a = r0*gv_w + w
        add_edge(a, a+gv_w, OUT, int(region_id[r0,w-1]), p_right, -1)

    # Interior horizontal boundaries: between rows r-1 and r
    diff = region_id[:-1,:] != region_id[1:,:]
    rr, cc = np.nonzero(diff)
    for r0,c0 in zip(rr.tolist(), cc.tolist()):
        rline = r0 + 1
        a = rline*gv_w + c0
        p_up = r0*w + c0
        p_dn = (r0+1)*w + c0
        add_edge(a, a+1, int(region_id[r0,c0]), int(region_id[r0+1,c0]), p_up, p_dn)

    # Interior vertical boundaries: between columns c-1 and c
    diff = region_id[:,:-1] != region_id[:,1:]
    rr, cc = np.nonzero(diff)
    for r0,c0 in zip(rr.tolist(), cc.tolist()):
        cline = c0 + 1
        a = r0*gv_w + cline
        p_l = r0*w + c0
        p_r = r0*w + c0 + 1
        add_edge(a, a+gv_w, int(region_id[r0,c0]), int(region_id[r0,c0+1]), p_l, p_r)

    F = {
        'v1': np.asarray(v1, dtype=np.int64),
        'v2': np.asarray(v2, dtype=np.int64),
        'pa': np.asarray(pa, dtype=np.int64),
        'pb': np.asarray(pb, dtype=np.int64),
        'c1': np.asarray(c1, dtype=np.int64),
        'c2': np.asarray(c2, dtype=np.int64),
        'vertex_to_edges': vertex_to_edges,
        'pixpair_to_edge': pixpair_to_edge,
        'shape': (h,w),
    }
    return F


def frontier_components(F):
    n = len(F['v1'])
    comp_of = -np.ones(n, dtype=np.int64)
    comps=[]
    vte=F['vertex_to_edges']
    for start in range(n):
        if comp_of[start] >= 0: continue
        cid=len(comps); comp_of[start]=cid; st=[start]; edges=[]
        while st:
            e=st.pop(); edges.append(e)
            for v in (int(F['v1'][e]), int(F['v2'][e])):
                for f in vte[v]:
                    if comp_of[f] < 0:
                        comp_of[f]=cid; st.append(f)
        comps.append(edges)
    return comps, comp_of


def continuation_classes(F):
    vte=F['vertex_to_edges']
    junctions={v for v,es in vte.items() if len(es)>=3}
    n=len(F['v1']); class_of=-np.ones(n,dtype=np.int64); classes=[]
    for start in range(n):
        if class_of[start]>=0: continue
        cid=len(classes); class_of[start]=cid
        pair=(int(F['pa'][start]),int(F['pb'][start]))
        st=[start]; support=[]
        while st:
            e=st.pop(); support.append(e)
            for v in (int(F['v1'][e]),int(F['v2'][e])):
                if v in junctions: continue
                for f in vte[v]:
                    if class_of[f] < 0 and (int(F['pa'][f]),int(F['pb'][f]))==pair:
                        class_of[f]=cid; st.append(f)
        endpoints=[]
        for e in support:
            a,b=int(F['v1'][e]),int(F['v2'][e])
            if a in junctions: endpoints.append(a)
            if b in junctions: endpoints.append(b)
        if len(endpoints)==0:
            typ='enc0'
        elif len(endpoints)==2:
            typ='enc1' if endpoints[0]==endpoints[1] else 'seg'
        else:
            raise RuntimeError(f"Invalid frontier-continuation class with {len(endpoints)} terminal occurrences")
        classes.append({'id':cid,'type':typ,'edges':support,'pair':pair})
    return classes, junctions


def _boundary_component_ids_for_hole(hole_mask, region_mask, r0, c0, W, F, comp_of):
    """Return frontier-component ids on the 4-interface between a hole and its carrier."""
    cids=set(); pmap=F['pixpair_to_edge']
    # vertical pixel adjacencies (row neighbors)
    A = hole_mask[:-1,:] & region_mask[1:,:]
    rr,cc=np.nonzero(A)
    for r,c in zip(rr.tolist(),cc.tolist()):
        p1=(r0+r)*W+(c0+c); p2=(r0+r+1)*W+(c0+c)
        e=pmap.get((min(p1,p2),max(p1,p2)))
        if e is not None: cids.add(int(comp_of[e]))
    A = region_mask[:-1,:] & hole_mask[1:,:]
    rr,cc=np.nonzero(A)
    for r,c in zip(rr.tolist(),cc.tolist()):
        p1=(r0+r)*W+(c0+c); p2=(r0+r+1)*W+(c0+c)
        e=pmap.get((min(p1,p2),max(p1,p2)))
        if e is not None: cids.add(int(comp_of[e]))
    # horizontal pixel adjacencies (column neighbors)
    A = hole_mask[:,:-1] & region_mask[:,1:]
    rr,cc=np.nonzero(A)
    for r,c in zip(rr.tolist(),cc.tolist()):
        p1=(r0+r)*W+(c0+c); p2=(r0+r)*W+(c0+c+1)
        e=pmap.get((min(p1,p2),max(p1,p2)))
        if e is not None: cids.add(int(comp_of[e]))
    A = region_mask[:,:-1] & hole_mask[:,1:]
    rr,cc=np.nonzero(A)
    for r,c in zip(rr.tolist(),cc.tolist()):
        p1=(r0+r)*W+(c0+c); p2=(r0+r)*W+(c0+c+1)
        e=pmap.get((min(p1,p2),max(p1,p2)))
        if e is not None: cids.add(int(comp_of[e]))
    return cids


def identify_cuts_local(region_id, region_slices, F, f_components, comp_of):
    """
    Identify cuts through the regional 4-(8) hole correspondence.
    Hole searches are restricted to each region's bounding box.
    """
    h,w=region_id.shape
    frame_free=[]
    for cid,edges in enumerate(f_components):
        ee=np.asarray(edges,dtype=np.int64)
        if not np.any((F['pa'][ee]==OUT)|(F['pb'][ee]==OUT)):
            frame_free.append(cid)

    cut_map={}; hole_counts=np.zeros(len(region_slices),dtype=np.int64)
    for rid,sl in enumerate(region_slices):
        if sl is None: continue
        rs,cs=sl
        # Exact bbox of the connected region. Holes cannot touch bbox boundary.
        sub = region_id[rs,cs]
        rm = (sub==rid)
        comp,nc=ndi.label(~rm, structure=STRUCT8)
        if nc==0: continue
        border=np.concatenate([comp[0,:],comp[-1,:],comp[:,0],comp[:,-1]])
        external=set(np.unique(border).tolist()); external.discard(0)
        hole_labels=[x for x in range(1,nc+1) if x not in external]
        hole_counts[rid]=len(hole_labels)
        r0=rs.start or 0; c0=cs.start or 0
        for hl in hole_labels:
            hm=(comp==hl)
            cids=_boundary_component_ids_for_hole(hm,rm,r0,c0,w,F,comp_of)
            if len(cids)!=1:
                raise RuntimeError(
                    f"Cut-hole interface check failed for region {rid}: component ids={sorted(cids)}"
                )
            cid=next(iter(cids))
            if cid in cut_map and cut_map[cid]!=rid:
                raise RuntimeError("A frame-free frontier component received two distinct carrier regions")
            cut_map[cid]=rid

    if set(frame_free)!=set(cut_map):
        missing=set(frame_free)-set(cut_map); extra=set(cut_map)-set(frame_free)
        raise RuntimeError(f"Cut-hole bijection check failed. missing={len(missing)}, extra={len(extra)}")

    cuts=[{'id':i,'component':cid,'carrier':cut_map[cid],'edges':f_components[cid]}
          for i,cid in enumerate(frame_free)]
    return cuts,hole_counts


def build_sparse_incidence(F, continuation, junctions, cuts, nregions):
    seg=[g for g in continuation if g['type']=='seg']
    enc1=[g for g in continuation if g['type']=='enc1']
    enc0=[g for g in continuation if g['type']=='enc0']
    V1=seg+enc1+enc0+cuts
    V1_types=(['seg']*len(seg)+['enc1']*len(enc1)+['enc0']*len(enc0)+['cut']*len(cuts))
    junc_list=sorted(junctions)
    n0=len(junc_list)+len(enc0); n1=len(V1); n2=nregions+1
    jrow={v:i for i,v in enumerate(junc_list)}
    encrow={g['id']:len(junc_list)+i for i,g in enumerate(enc0)}

    r01=[];c01=[];d01=[]; r12=[];c12=[];d12=[]
    for j,g in enumerate(V1):
        bc=defaultdict(int)
        for e in g['edges']:
            a,b=int(F['v1'][e]),int(F['v2'][e])
            if a in junctions: bc[a]+=1
            if b in junctions: bc[b]+=1
        for v,m in bc.items():
            r01.append(jrow[v]); c01.append(j); d01.append(m)
        if g.get('type')=='enc0' or (g in enc0):
            r01.append(encrow[g['id']]); c01.append(j); d01.append(2)
        if 'carrier' in g:  # cut
            r12.append(j); c12.append(int(g['carrier'])); d12.append(2)
        else:
            a,b=g['pair']
            if a==OUT:
                ca=nregions
            else:
                ca=a
            if b==OUT:
                cb=nregions
            else:
                cb=b
            r12 += [j,j]; c12 += [ca,cb]; d12 += [1,1]
    M01=coo_matrix((d01,(r01,c01)),shape=(n0,n1),dtype=np.int32).tocsr()
    M12=coo_matrix((d12,(r12,c12)),shape=(n1,n2),dtype=np.int32).tocsr()
    return {
        'seg':seg,'enc1':enc1,'enc0':enc0,'cuts':cuts,'V1':V1,
        'V1_types':V1_types,'junctions':junc_list,
        'nV0':n0,'nV1':n1,'nV2':n2,'M01':M01,'M12':M12
    }


def graph_components_from_incidence(M01: csr_matrix, M12: csr_matrix):
    n0,n1=M01.shape; _,n2=M12.shape
    uf=UnionFind(n0+n1+n2); off1=n0; off2=n0+n1
    rr,cc=M01.nonzero()
    for a,b in zip(rr.tolist(),cc.tolist()): uf.union(a,off1+b)
    rr,cc=M12.nonzero()
    for a,b in zip(rr.tolist(),cc.tolist()): uf.union(off1+a,off2+b)
    roots={uf.find(i) for i in range(n0+n1+n2)}
    return len(roots)


def rag_and_contact_stats(continuation, nregions):
    pair_mult=Counter()
    for g in continuation:
        a,b=g['pair']
        if a==OUT or b==OUT: continue
        pair_mult[(a,b)] += 1
    rag_edges=len(pair_mult); contact_edges=sum(pair_mult.values())
    deg=np.zeros(nregions,dtype=np.int64)
    for a,b in pair_mult:
        deg[a]+=1; deg[b]+=1
    if nregions>1:
        rag_density=2*rag_edges/(nregions*(nregions-1))
    else:
        rag_density=0.0
    vals=np.array(list(pair_mult.values()),dtype=float)
    return {
        'rag_v':nregions,'rag_e':rag_edges,'rag_density':rag_density,
        'rag_cycle_rank':rag_edges-nregions+1 if nregions else 0,
        'rag_deg_mean':float(deg.mean()) if nregions else 0.0,
        'rag_deg_std':float(deg.std()) if nregions else 0.0,
        'rag_deg_max':int(deg.max()) if nregions else 0,
        'contact_edges':int(contact_edges),
        'parallel_excess':int(contact_edges-rag_edges),
        'contact_mult_mean':float(vals.mean()) if len(vals) else 0.0,
        'contact_mult_max':float(vals.max()) if len(vals) else 0.0,
    }


def compute_hig_descriptors(seg: np.ndarray, validate_euler: bool=True) -> Dict[str,Any]:
    """Construct the intrinsic HIG and return publication-oriented descriptors."""
    t0=time.perf_counter()
    h,w=seg.shape; pixels=h*w
    rid,nregions,slices,sizes,region_values=label_4_regions(seg)
    F=build_frontier(rid)
    fcomps,comp_of=frontier_components(F)
    continuation,junctions=continuation_classes(F)
    cuts,hole_counts=identify_cuts_local(rid,slices,F,fcomps,comp_of)
    H=build_sparse_incidence(F,continuation,junctions,cuts,nregions)
    M01,M12=H['M01'],H['M12']
    n0,n1,n2=H['nV0'],H['nV1'],H['nV2']
    chi_cell=n2-n1+n0
    if validate_euler and chi_cell!=2:
        raise AssertionError(f"Cellular Euler identity failed: {chi_cell}")
    nnz01,nnz12=int(M01.nnz),int(M12.nnz)
    m01,m12=int(M01.sum()),int(M12.sum())
    Vg=n0+n1+n2; Eg=m01+m12; chi_graph=Vg-Eg
    comps=graph_components_from_incidence(M01,M12)
    beta_graph=Eg-Vg+comps

    # No-cut HIG: cuts are last columns/rows by construction.
    ngeom=len(H['seg'])+len(H['enc1'])+len(H['enc0'])
    M01g=M01[:,:ngeom]; M12g=M12[:ngeom,:]
    E_no_cut=int(M01g.sum()+M12g.sum())
    V_no_cut=n0+ngeom+n2

    # Binary full HIG
    E_binary=nnz01+nnz12
    chi_binary=Vg-E_binary

    # Cubical model sizes for arbitrary h,w.
    K0=(h+1)*(w+1); K1=h*(w+1)+(h+1)*w; K2=h*w
    cubical_cells=K0+K1+K2
    # Same normalization convention used in the synthetic notebook.
    cubical_incidence_support=max(1,4*K1)

    rstats=rag_and_contact_stats(continuation,nregions)
    internal_frontier=int(np.sum((F['pa']!=OUT)&(F['pb']!=OUT)))
    frame_frontier=len(F['v1'])-internal_frontier
    used_labels=int(len(np.unique(seg)))

    holes=hole_counts.astype(float)
    desc={
        'height':h,'width':w,'pixels':pixels,'log_pixels':float(np.log1p(pixels)),
        'aspect_ratio':float(w/h if h else 0),'labels_used':used_labels,
        'frontier_internal':internal_frontier,'frontier_frame':frame_frontier,
        'frontier_per_pixel':float(internal_frontier/max(1,pixels)),
        'V2_img':nregions,'V2':n2,
        'V1_seg':len(H['seg']),'V1_enc1':len(H['enc1']),'V1_enc0':len(H['enc0']),
        'V1_enc':len(H['enc1'])+len(H['enc0']),'V1_cut':len(H['cuts']),'V1':n1,
        'V0_junc':len(H['junctions']),'V0_enc':len(H['enc0']),'V0':n0,
        'chi_cell':chi_cell,
        'nnz_M01':nnz01,'nnz_M12':nnz12,'m01':m01,'m12':m12,
        'd01':float(nnz01/max(1,n0*n1)),'d12':float(nnz12/max(1,n1*n2)),
        'graph_V':Vg,'graph_E':Eg,'chi_graph':chi_graph,
        'graph_components':comps,'beta1_graph':beta_graph,
        'binary_graph_E':E_binary,'binary_chi_graph':chi_binary,
        'no_cut_graph_V':V_no_cut,'no_cut_graph_E':E_no_cut,
        'no_cut_chi_graph':V_no_cut-E_no_cut,
        'multiplicity_excess':Eg-E_binary,
        'holes_mean':float(holes.mean()) if len(holes) else 0.0,
        'holes_max':float(holes.max()) if len(holes) else 0.0,
        'holes_var':float(holes.var()) if len(holes) else 0.0,
        'holes_prop_positive':float(np.mean(holes>0)) if len(holes) else 0.0,
        'V1_seg_per_region':float(len(H['seg'])/max(1,nregions)),
        'V1_cut_per_region':float(len(H['cuts'])/max(1,nregions)),
        'V0_junc_per_region':float(len(H['junctions'])/max(1,nregions)),
        'rho_HIG':float(Vg/max(1,cubical_cells)),
        'rho_inc':float((nnz01+nnz12)/cubical_incidence_support),
        'construction_seconds':float(time.perf_counter()-t0),
    }
    desc.update(rstats)
    return desc


# -----------------------------------------------------------------------------
# Representation feature sets
# -----------------------------------------------------------------------------

FEATURE_SETS = {
    'Segmentation-size baseline': [
        'log_pixels','aspect_ratio','labels_used','V2_img',
        'frontier_internal','frontier_per_pixel'
    ],
    'RAG': [
        'log_pixels','V2_img','rag_e','rag_density','rag_cycle_rank',
        'rag_deg_mean','rag_deg_std','rag_deg_max'
    ],
    'Regional contact multigraph': [
        'log_pixels','V2_img','rag_e','rag_density','rag_cycle_rank',
        'contact_edges','parallel_excess','contact_mult_mean','contact_mult_max'
    ],
    'HIG without cuts': [
        'log_pixels','V2_img','V1_seg','V1_enc1','V1_enc0','V0_junc','V0_enc',
        'V1_seg_per_region','V0_junc_per_region','no_cut_graph_E','no_cut_chi_graph',
        'nnz_M01','d01'
    ],
    'Binary-incidence HIG': [
        'log_pixels','V2_img','V1_seg','V1_enc1','V1_enc0','V1_cut','V0_junc','V0_enc',
        'V1_seg_per_region','V1_cut_per_region','V0_junc_per_region',
        'holes_mean','holes_max','holes_var','holes_prop_positive',
        'nnz_M01','nnz_M12','d01','d12','binary_graph_E','binary_chi_graph'
    ],
    'Full HIG': [
        'log_pixels','V2_img','V1_seg','V1_enc1','V1_enc0','V1_cut','V0_junc','V0_enc',
        'V1_seg_per_region','V1_cut_per_region','V0_junc_per_region',
        'holes_mean','holes_max','holes_var','holes_prop_positive',
        'nnz_M01','nnz_M12','m01','m12','d01','d12',
        'graph_E','chi_graph','beta1_graph','multiplicity_excess'
    ],
}


def extract_dataset(metadata: pd.DataFrame,
                    segmentation_mode: str='uniform',
                    levels: int=8,
                    validate_euler: bool=True,
                    progress: bool=True) -> pd.DataFrame:
    """
    metadata required columns: sample_id, image_path, label.
    optional: group, variant, segmentation_path.
    """
    required={'sample_id','image_path','label'}
    missing=required-set(metadata.columns)
    if missing: raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    rows=[]
    iterator=metadata.itertuples(index=False)
    if progress: iterator=tqdm(list(iterator), desc='HIG descriptors')
    for row in iterator:
        d=row._asdict()
        segpath=d.get('segmentation_path',None)
        if isinstance(segpath,float) and np.isnan(segpath): segpath=None
        _,seg=segment_image(str(d['image_path']), segpath, segmentation_mode, levels)
        feat=compute_hig_descriptors(seg, validate_euler=validate_euler)
        base={k:d.get(k) for k in metadata.columns}
        base.update(feat); rows.append(base)
    return pd.DataFrame(rows)


def build_metadata_from_class_folders(root: str,
                                      extensions=('.png','.jpg','.jpeg','.tif','.tiff','.bmp')) -> pd.DataFrame:
    """root/class_name/image.ext -> binary or multiclass metadata."""
    root=Path(root); rows=[]
    classes=[p for p in sorted(root.iterdir()) if p.is_dir()]
    for cls in classes:
        for p in sorted(cls.rglob('*')):
            if p.suffix.lower() in extensions:
                rows.append({
                    'sample_id':p.stem,'image_path':str(p),'label':cls.name,
                    'group':p.stem,'variant':'original','segmentation_path':''
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Predictive evaluation
# -----------------------------------------------------------------------------

def _binary_metrics(y, p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=(p>=0.5).astype(int)
    auc=roc_auc_score(y,p); ap=average_precision_score(y,p); ba=balanced_accuracy_score(y,pred)
    cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    sens=tp/max(1,tp+fn); spec=tn/max(1,tn+fp)
    return {'ROC-AUC':auc,'PR-AUC':ap,'Balanced accuracy':ba,'Sensitivity':sens,'Specificity':spec}


def _encode_binary_labels(s: pd.Series):
    vals=list(pd.unique(s))
    if len(vals)!=2:
        raise ValueError(f"This publication evaluation currently expects exactly two classes; found {vals}")
    mapping={vals[0]:0,vals[1]:1}
    return s.map(mapping).astype(int).to_numpy(), mapping


def _outer_split_list(y, groups, k, repeats, seed):
    splits=[]
    idx=np.arange(len(y))
    for rep in range(repeats):
        if groups is None:
            cv=StratifiedKFold(n_splits=k,shuffle=True,random_state=seed+rep)
            rep_splits=list(cv.split(idx,y))
        else:
            cv=StratifiedGroupKFold(n_splits=k,shuffle=True,random_state=seed+rep)
            rep_splits=list(cv.split(idx,y,groups))
        splits.append(rep_splits)
    return splits


def _inner_cv(y_train, groups_train, k, seed):
    if groups_train is None:
        return StratifiedKFold(n_splits=k,shuffle=True,random_state=seed)
    return StratifiedGroupKFold(n_splits=k,shuffle=True,random_state=seed)


def evaluate_representations(feature_df: pd.DataFrame,
                             variant: str='original',
                             group_col: Optional[str]='group',
                             k: int=5,
                             repeats: int=5,
                             inner_k: int=3,
                             C_grid=(0.01,0.1,1.0,10.0,100.0),
                             seed: int=20260927,
                             class_weight=None):
    """Repeated nested CV with identical outer folds for every representation."""
    df=feature_df.copy()
    if 'variant' in df.columns and variant is not None:
        df=df[df['variant'].fillna('original')==variant].copy()
    if df['sample_id'].duplicated().any():
        raise ValueError("Primary predictive evaluation needs one row per sample_id. Use variants separately.")
    y,label_mapping=_encode_binary_labels(df['label'])
    groups=None
    if group_col and group_col in df.columns and df[group_col].notna().any():
        groups=df[group_col].fillna(df['sample_id']).astype(str).to_numpy()
    splits=_outer_split_list(y,groups,k,repeats,seed)

    pred_rows=[]; repeat_rows=[]
    for rep_name,cols in FEATURE_SETS.items():
        X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(float)
        for rep,rep_splits in enumerate(splits):
            p=np.zeros(len(df),dtype=float); fold_id=np.full(len(df),-1,dtype=int)
            for fold,(tr,te) in enumerate(rep_splits):
                gtr=groups[tr] if groups is not None else None
                inner=_inner_cv(y[tr],gtr,inner_k,seed+1000*rep+fold)
                pipe=Pipeline([
                    ('scale',StandardScaler()),
                    ('clf',LogisticRegression(max_iter=5000,solver='lbfgs',class_weight=class_weight))
                ])
                search=GridSearchCV(pipe,{'clf__C':list(C_grid)},scoring='roc_auc',cv=inner,n_jobs=-1)
                if gtr is None:
                    search.fit(X[tr],y[tr])
                else:
                    search.fit(X[tr],y[tr],groups=gtr)
                p[te]=search.predict_proba(X[te])[:,1]; fold_id[te]=fold
            met=_binary_metrics(y,p); met.update({'Representation':rep_name,'Repeat':rep})
            repeat_rows.append(met)
            for i in range(len(df)):
                pred_rows.append({
                    'Representation':rep_name,'Repeat':rep,'Fold':int(fold_id[i]),
                    'sample_id':df.iloc[i]['sample_id'],'y':int(y[i]),'probability':float(p[i])
                })
    repeat_df=pd.DataFrame(repeat_rows); pred_df=pd.DataFrame(pred_rows)
    summary=repeat_df.groupby('Representation').agg({
        'ROC-AUC':['mean','std'],'PR-AUC':['mean','std'],'Balanced accuracy':['mean','std'],
        'Sensitivity':['mean','std'],'Specificity':['mean','std']
    })
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    return summary,repeat_df,pred_df,label_mapping


def _aggregate_oof(pred_df, representation):
    p=pred_df[pred_df['Representation']==representation]
    return p.groupby('sample_id',as_index=False).agg(y=('y','first'),probability=('probability','mean'))


def bootstrap_metric_ci(pred_df: pd.DataFrame,
                       representation: str,
                       n_boot: int=2000,
                       seed: int=20260927):
    a=_aggregate_oof(pred_df,representation)
    y=a.y.to_numpy(int); p=a.probability.to_numpy(float)
    rng=np.random.default_rng(seed); vals=[]
    n=len(a)
    for _ in range(n_boot):
        idx=rng.integers(0,n,n)
        if len(np.unique(y[idx]))<2: continue
        vals.append(_binary_metrics(y[idx],p[idx]))
    out={}
    point=_binary_metrics(y,p)
    for key in point:
        arr=np.array([v[key] for v in vals])
        out[key]=(point[key],float(np.percentile(arr,2.5)),float(np.percentile(arr,97.5)))
    return out


def paired_auc_differences(pred_df: pd.DataFrame,
                           reference: str='Full HIG',
                           n_boot: int=2000,
                           seed: int=20260927):
    ref=_aggregate_oof(pred_df,reference).set_index('sample_id')
    rng=np.random.default_rng(seed); rows=[]
    for rep in pred_df['Representation'].unique():
        if rep==reference: continue
        alt=_aggregate_oof(pred_df,rep).set_index('sample_id')
        ids=ref.index.intersection(alt.index)
        y=ref.loc[ids,'y'].to_numpy(int); pr=ref.loc[ids,'probability'].to_numpy(float); pa=alt.loc[ids,'probability'].to_numpy(float)
        point=roc_auc_score(y,pr)-roc_auc_score(y,pa); vals=[]; n=len(ids)
        for _ in range(n_boot):
            idx=rng.integers(0,n,n)
            if len(np.unique(y[idx]))<2: continue
            vals.append(roc_auc_score(y[idx],pr[idx])-roc_auc_score(y[idx],pa[idx]))
        rows.append({'Reference':reference,'Comparator':rep,'Delta ROC-AUC':point,
                     'CI low':np.percentile(vals,2.5),'CI high':np.percentile(vals,97.5)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Fig.-14-style matched-view analysis
# -----------------------------------------------------------------------------

def fig14_descriptor_drift(feature_df: pd.DataFrame,
                           original='original',
                           feature_cols: Optional[List[str]]=None):
    """Relative L1 drift of matched HIG descriptors versus the original image."""
    if feature_cols is None:
        feature_cols=FEATURE_SETS['Full HIG']
    if 'variant' not in feature_df.columns:
        raise ValueError("metadata must contain a 'variant' column")
    base=feature_df[feature_df['variant']==original].set_index('sample_id')
    rows=[]
    for var in sorted(feature_df['variant'].dropna().unique()):
        if var==original: continue
        cur=feature_df[feature_df['variant']==var].set_index('sample_id')
        ids=base.index.intersection(cur.index)
        for sid in ids:
            z0=base.loc[sid,feature_cols].to_numpy(float); z1=cur.loc[sid,feature_cols].to_numpy(float)
            den=np.abs(z0).sum()+1e-12; d=np.abs(z1-z0).sum()/den
            rows.append({'sample_id':sid,'variant':var,'D_feat':float(d)})
    detail=pd.DataFrame(rows)
    summary=detail.groupby('variant')['D_feat'].agg(['mean','std','median','count']).reset_index() if len(detail) else pd.DataFrame()
    return summary,detail


def evaluate_original_to_variants(feature_df: pd.DataFrame,
                                  representation='Full HIG',
                                  original='original',
                                  group_col: Optional[str]='group',
                                  k=5,repeats=5,inner_k=3,
                                  C_grid=(0.01,0.1,1,10,100),
                                  seed=20260927):
    """
    Train/tune only on original images in each outer training fold and evaluate
    the held-out sample IDs on every matched variant. This avoids treating
    Fig.-14-style variants as independent samples.
    """
    cols=FEATURE_SETS[representation]
    orig=feature_df[feature_df['variant']==original].copy()
    if orig['sample_id'].duplicated().any(): raise ValueError('Duplicate original sample_id')
    y,mapping=_encode_binary_labels(orig['label'])
    groups=None
    if group_col and group_col in orig.columns and orig[group_col].notna().any():
        groups=orig[group_col].fillna(orig['sample_id']).astype(str).to_numpy()
    splits=_outer_split_list(y,groups,k,repeats,seed)
    variants=sorted(feature_df['variant'].dropna().unique())
    byvar={v:feature_df[feature_df['variant']==v].set_index('sample_id') for v in variants}
    Xo=orig[cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
    all_rows=[]
    for rep,rep_splits in enumerate(splits):
        probs={v:np.full(len(orig),np.nan) for v in variants}
        for fold,(tr,te) in enumerate(rep_splits):
            gtr=groups[tr] if groups is not None else None
            inner=_inner_cv(y[tr],gtr,inner_k,seed+1000*rep+fold)
            pipe=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=5000,solver='lbfgs'))])
            search=GridSearchCV(pipe,{'clf__C':list(C_grid)},scoring='roc_auc',cv=inner,n_jobs=-1)
            if gtr is None: search.fit(Xo[tr],y[tr])
            else: search.fit(Xo[tr],y[tr],groups=gtr)
            test_ids=orig.iloc[te]['sample_id'].tolist()
            for v in variants:
                tab=byvar[v]
                if not all(sid in tab.index for sid in test_ids): continue
                Xv=tab.loc[test_ids,cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
                probs[v][te]=search.predict_proba(Xv)[:,1]
        for v,p in probs.items():
            ok=np.isfinite(p)
            if ok.sum()==len(orig):
                met=_binary_metrics(y,p); met.update({'Variant':v,'Repeat':rep,'Representation':representation})
                all_rows.append(met)
    detail=pd.DataFrame(all_rows)
    summary=detail.groupby('Variant').agg({'ROC-AUC':['mean','std'],'PR-AUC':['mean','std'],'Balanced accuracy':['mean','std']})
    summary.columns=['_'.join(c) for c in summary.columns]; summary=summary.reset_index()
    return summary,detail,mapping


# -----------------------------------------------------------------------------
# Publication exports
# -----------------------------------------------------------------------------

def export_publication_tables(outdir: str,
                              feature_df: pd.DataFrame,
                              summary: Optional[pd.DataFrame]=None,
                              pred_df: Optional[pd.DataFrame]=None,
                              drift_summary: Optional[pd.DataFrame]=None,
                              transfer_summary: Optional[pd.DataFrame]=None):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    feature_df.to_csv(out/'hig_features.csv',index=False)
    if summary is not None: summary.to_csv(out/'predictive_performance.csv',index=False)
    if pred_df is not None:
        pred_df.to_csv(out/'oof_predictions.csv',index=False)
        paired_auc_differences(pred_df).to_csv(out/'paired_auc_differences.csv',index=False)
    if drift_summary is not None: drift_summary.to_csv(out/'fig14_descriptor_drift.csv',index=False)
    if transfer_summary is not None: transfer_summary.to_csv(out/'fig14_predictive_transfer.csv',index=False)

    if summary is not None:
        lines=[r"\begin{table}[ht]",r"\centering",
               r"\caption{Predictive performance of progressively richer segmentation representations. Values are mean $\pm$ standard deviation across repeated outer cross-validation runs.}",
               r"\label{tab:representation-performance}",r"\begin{tabular}{lccc}",r"\hline",
               r"Representation & ROC--AUC & PR--AUC & Balanced accuracy \\",r"\hline"]
        for _,r in summary.iterrows():
            lines.append(f"{r['Representation']} & {r['ROC-AUC_mean']:.3f} $\\pm$ {r['ROC-AUC_std']:.3f} & {r['PR-AUC_mean']:.3f} $\\pm$ {r['PR-AUC_std']:.3f} & {r['Balanced accuracy_mean']:.3f} $\\pm$ {r['Balanced accuracy_std']:.3f} \\")
        lines += [r"\hline",r"\end{tabular}",r"\end{table}"]
        (out/'table_predictive_performance.tex').write_text('\n'.join(lines),encoding='utf-8')

    if drift_summary is not None and len(drift_summary):
        lines=[r"\begin{table}[ht]",r"\centering",
               r"\caption{HIG descriptor drift for matched topology-oriented image variants relative to the original image.}",
               r"\label{tab:fig14-drift}",r"\begin{tabular}{lccc}",r"\hline",
               r"Variant & Mean $D_{\mathrm{feat}}$ & Std. dev. & Matched images \\",r"\hline"]
        for _,r in drift_summary.iterrows():
            lines.append(f"{r['variant']} & {r['mean']:.4f} & {r['std']:.4f} & {int(r['count'])} \\")
        lines += [r"\hline",r"\end{tabular}",r"\end{table}"]
        (out/'table_fig14_descriptor_drift.tex').write_text('\n'.join(lines),encoding='utf-8')

    if transfer_summary is not None and len(transfer_summary):
        lines=[r"\begin{table}[ht]",r"\centering",
               r"\caption{Predictive transfer from original images to matched topology-oriented variants. Models are trained and tuned only on original-image training folds.}",
               r"\label{tab:fig14-transfer}",r"\begin{tabular}{lccc}",r"\hline",
               r"Variant & ROC--AUC & PR--AUC & Balanced accuracy \\",r"\hline"]
        for _,r in transfer_summary.iterrows():
            lines.append(f"{r['Variant']} & {r['ROC-AUC_mean']:.3f} $\\pm$ {r['ROC-AUC_std']:.3f} & {r['PR-AUC_mean']:.3f} $\\pm$ {r['PR-AUC_std']:.3f} & {r['Balanced accuracy_mean']:.3f} $\\pm$ {r['Balanced accuracy_std']:.3f} \\")
        lines += [r"\hline",r"\end{tabular}",r"\end{table}"]
        (out/'table_fig14_predictive_transfer.tex').write_text('\n'.join(lines),encoding='utf-8')

    return str(out)
