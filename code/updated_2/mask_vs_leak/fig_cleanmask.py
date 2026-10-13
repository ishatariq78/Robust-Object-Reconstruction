# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
The clean-mask experiment, as a picture.

Pot painted into the PHOTOS only, never into the scan dots, on tables the fine-tune never saw.
Three columns: guidance off, the mask the pipeline builds today, and a verified-clean mask with
the pot removed by a depth test.

Rows are ordered by how much the clean mask helped, best first — so if there is an effect it is
at the top and cannot be hidden by ordering. Every panel carries its own Leak% so the picture
and the number can be checked against each other.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
OUT=HERE/"outputs"/"clean_test"; CL=REPO/"updated_2/mask_vs_leak/img_clutter"
sys.path.insert(0,str(REPO/"updated_2"))
from metrics import score_paths
from fig_improvement import render, flat, paint_by_distance, font, GTC, INTRC, INK, MUTED

man=json.loads((OUT/"manifest.json").read_text())
rows=[]
for n in sorted(man):
    lab=CL/f"{n}__label.npz"
    if not lab.exists(): continue
    v=[]
    for arm in ("off__s0","sam2__s0","oracle__s0"):
        p=man[n].get(arm)
        v.append(float(score_paths(p,label_npz=str(lab))["leak_pct"])
                 if p and Path(p).exists() else None)
    if None in v: continue
    rows.append((n,v,v[1]-v[2]))          # clean mask minus SAM2 mask
rows.sort(key=lambda r:-r[2])
rows=rows[:5]

cell,labw,hdr,gap=260,200,100,8
cols=["truth + the pot","guidance OFF","mask as built today","verified CLEAN mask"]
W=labw+len(cols)*(cell+gap); H=hdr+len(rows)*(cell+36)
sh=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(sh)
d.text((16,16),"Pot in the photos, but not in the mask",fill=INK,font=font(26,True))
d.text((16,52),"red = surface built ON the pot.  14 tables the model never saw; the 5 where the "
               "clean mask helped most.",fill=MUTED,font=font(15))
for i,c in enumerate(cols):
    x=labw+i*(cell+gap); tw=d.textlength(c,font=font(15,True))
    d.text((x+(cell-tw)/2,hdr-24),c,fill=MUTED,font=font(15,True))
for r,(n,v,_g) in enumerate(rows):
    z=np.load(CL/f"{n}__label.npz")
    tgt=trimesh.Trimesh(vertices=z["target_vertices"],faces=z["target_faces"],process=False)
    intr=trimesh.Trimesh(vertices=z["distractor_vertices"],faces=z["distractor_faces"],process=False)
    y=hdr+r*(cell+36)
    allv=np.vstack([tgt.vertices,intr.vertices])
    centre=(allv.min(0)+allv.max(0))/2
    radius=float(np.linalg.norm(allv.max(0)-allv.min(0)))*0.5
    thr=0.03*float(np.max(tgt.extents))
    ip,_=trimesh.sample.sample_surface(intr,40000,seed=5)
    d.text((16,y+cell//2-8),n.split("__")[1][:18],fill=INK,font=font(16,True))
    sh.paste(render([flat(tgt,GTC),flat(intr,INTRC)],centre,radius,(cell,cell)),(labw,y))
    for i,arm in enumerate(("off__s0","sam2__s0","oracle__s0")):
        p=man[n].get(arm)
        if not (p and Path(p).exists()): continue
        m=trimesh.load(p,force="mesh")
        if not len(getattr(m,"faces",[])): continue
        x=labw+(i+1)*(cell+gap)
        sh.paste(render(paint_by_distance(m,ip,thr,True),centre,radius,(cell,cell)),(x,y))
        txt=f"{v[i]:.2f}% on the pot"
        col=(44,127,86) if i==2 and v[2]<v[0] else MUTED
        tw=d.textlength(txt,font=font(14,i==2))
        d.text((x+(cell-tw)/2,y+cell+4),txt,fill=col,font=font(14,i==2))
    print(f"  {n.split('__')[1]:22} off {v[0]:.2f} -> sam2 {v[1]:.2f} -> clean {v[2]:.2f}")
sh.save(OUT/"fig_cleanmask.jpg",quality=94)
print(f"wrote {OUT/'fig_cleanmask.jpg'}")
