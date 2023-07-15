import json 
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.set_num_threads(16)
from transformers import pipeline


annotation_folder = "PATH_TO_ANNOT/annotations{}".format("_v2")
# load Ego4D classes for verbs and nouns

def get_vnmap(tax):
    # map cluster of verb/nouns to a map(python dict)
    # e.g. {"foo_(bar)"} --> {"foo" : 0, "bar" : 0}
    vmap = {}
    for idxv, v in enumerate(tax["verbs"]):
        v = v.replace("_(", "*").replace("/", "*").replace(")", "").replace(",", "*")
        for idxvv, vv in enumerate(v.split("*")):
            if vv == "":
                continue 
            if vv[0] == "_":
                vv = vv[1:]
            vv = vv.replace("_", " ")
            if vv in vmap:
                if vmap[vv] != idxv and idxvv == 0:
                    pass
                else:
                    continue 
            vmap[vv] = idxv

    nmap = {}
    for idxn, n in enumerate(tax["nouns"]): # n = a_(b/c)
        n = n.replace("_(", "*").replace("/", "*").replace(")", "").replace(",", "*")
        for idxnn, nn in enumerate(n.split("*")): # nn = a or b or c
            if nn == "":
                continue 
            if nn[0] == "_":
                nn = nn[1:]
            nn= nn.replace("_", " ")
            if nn in nmap:
                if nmap[nn] != idxn and idxnn == 0:
                    nmap[nn] = idxn 
                else:
                    continue 
            else:
                nmap[nn] = idxn

    return vmap, nmap

with open("{}/fho_lta_taxonomy.json".format(annotation_folder), "r") as f:
    tax = json.load(f)
vmap, nmap = get_vnmap(tax)


top_k = 50
top_p = 0.5
print("hyperparams: top_k {} top_p {}".format(top_k, top_p))

# %%
# language model inference interface
generator = pipeline('text-generation', model="gpt2", device="cuda") 
# %%
prompt_full = """
Narrations: A person is rolling out bricks with a hammer. A person is standing on top of a brick wall. A person is holding onto a piece of wood in the dirt. A person is using a brick to make bricks. A person is standing next to some bricks and a brick wall. A person is using a brick to make a brick. A person is working on some bricks in the dirt. A person is working on a brick factory in india.
Actions: turn mold, move mold, clean mold, put container, cut cement, mold cement, adjust mold, fill cement, scrape cement, put cement, move mold, turn mold, remove mold, clean mold, put mold, cut cement, mold cement, adjust mold, put cement, scrape cement, put cement, move mold, turn mold, remove mold, move mold, clean mold, put mold, cut cement.


Narrations: A person is making bricks with his hands. A person is working on bricks in a factory. A person is holding bricks in front of a brick making machine. A person is holding a box of bricks with the word "sms" on it. A person is making bricks with the words sun and moon. A person is digging up a dirt pile with a shovel. A person is using bricks to build a house. A person is digging up some dirt with a shovel.
Actions: move mold, turn mold, remove mold, put mold, inspect brick, dip cement, take mold, dip cement, take water, take water, pour cement, take water, pour cement, take water, wash mold, pour mold, clean mold, clean mold, take water, pour water, take water, wash mold, clean water, take water, wash mold, clean mold, wash mold, clean mold.


Narrations: A person is using bricks to build a house in india. A person is using a brick to make bricks. A person is making bricks with a shovel. A person is laying down some bricks in a dirt field. A person is riding a bike down the road. A person is placing bricks into a large pile of bricks. A person is using a brick to make bricks. A person is standing in front of a brick wall.
Actions: mold cement, put cement, scrape cement, put cement, move mold, remove mold, turn mold, move mold, clean mold, put container, cut cement, mold cement, adjust mold, fill cement, scrape cement, put cement, move mold, turn mold, remove mold, clean mold, put mold, cut cement, mold cement, adjust mold, put cement, scrape cement, put cement, move mold.


Narrations: A person is using a shovel to dig into a rock. A person is laying on top of a pile of bricks. A person is using a brick to build a gun. A person is using bricks to build a wall in india. A person is placing bricks into a pile of bricks. A person is working on a brick making machine. A person is holding a piece of wood while standing in a dirt area. A person is digging a hole in the dirt.
Actions: remove cement, apply sand, put sand, carry mold, turn mold, move mold, put sand, turn mold, put mold, touch floor, mold cement, carry cement, remove cement, throw cement, put sand, turn mold, turn mold, put mold, take clay, put cement, throw cement, touch sand, turn mold, put sand, put mold, turn mold, clean floor, take cement.


Narrations: A person is making bricks out of mud. A person is holding bricks on a brick making machine. A person is holding bricks that say sri lanka in india. A person is sitting on the ground with a shovel in the mud. A person is working on a brick with a shovel. A person is making a rock out of mud. A person is working on a piece of dirt. A person is using bricks in the dirt.
Actions: turn mold, remove brick, hit mold, pack sand, put mold, cut cement, mold cement, scrape cement, touch sand, mix cement, move mold, turn mold, remove mold, touch brick, pour sand, put mold, move floor, cut cement, scrape cement, put cement, move mold, turn mold, remove mold, touch brick, pour mold, put mold, cut cement, mold cement.


Narrations: A person is holding a bat on a dirt field. A person is working on a brick wall with some clay. A person is using bricks to build a house in india. A person is using a brick mold to make bricks. A person is making bricks with a shovel. A person is laying down some bricks on a dirt floor. A person is standing on a dirt road with some bricks on the ground. A person is using a brick to build bricks.
Actions: clean mold, cut cement, mold cement, put cement, scrape cement, put cement, move mold, remove mold, turn mold, move mold, clean mold, put container, cut cement, mold cement, adjust mold, fill cement, scrape cement, put cement, move mold, turn mold, remove mold, clean mold, put mold, cut cement, mold cement, adjust mold, put cement, scrape cement.


Narrations: A person is laying down some bricks on the ground. A person is holding a brick while making bricks in a brick factory. A person is making bricks on a dirt road. A person is digging in the dirt near a pile of dirt. A person is standing in front of some bricks on a dirt road. A person is holding bricks with the words "sun" on them. A person is making bricks with numbers on them. A person is making bricks with their hands in a field.
Actions: adjust mold, put cement, scrape cement, put cement, move mold, turn mold, stick brick, lift mold, hit mold, pack sand, pour sand, put brick, move floor, cut cement, mold cement, adjust mold, put cement, clean cement, put cement, move mold, turn brick, remove mold, pour sand, put mold, cut cement, mold cement, adjust mold, put cement.


Narrations: A person is using bricks to build a wall. A person is making bricks with bricks in front of them. A person is holding a brick that has the word sams written on it. A person is using bricks to build a house in india. A person is sitting on the ground near some brick blocks. A person is working on the ground with a brick. A person is working with a shovel in a dirt field. A person is digging a large rock out of the ground.
Actions: move mold, turn mold, remove mold, touch brick, pour sand, put mold, move floor, cut cement, scrape cement, put cement, move mold, turn mold, remove mold, touch brick, pour mold, put mold, cut cement, mold cement, adjust mold, scrape cement, put cement, wipe sand, hit cement, move brick, turn brick, remove brick, hit brick, clean brick.


Narrations: A person is placing bricks in a pile of bricks. A person is working on a brick wall in india. A person is making bricks in the dirt. A person is making bricks on a brick making machine. A person is making bricks on a dirt road. A person is working with a shovel on a dirt surface. A person is placing bricks on the ground. A person is working with bricks in the dirt. A person is working with a large brick. A person is making bricks with his hands. A person is cutting out the letters of a brick. A person is making bricks with the word sms on them.
Actions: remove mold, pour sand, put mold, move mold, cut cement, mold clay, pull mold, put clay, remove clay, throw clay, move mold, turn mold, 

"""

def get_parsed(vn, vmap, nmap):
    vfind = []
    nfind = []
    while vn.startswith(" "):
        vn = vn[1:]
    for v in vmap:
        if vn.startswith(v + " "):
            vfind.append(v)
    # "turn on " not "turn"
    vfind = [sorted(vfind, key=lambda x : -len(x))[0]] if len(vfind) else vfind

    remove_prep = vn
    prep = ["in", "under", "on", "with", "from", "to", "out of", "into", "onto"]
    for _ in prep:
        remove_prep = remove_prep.split(" {} ".format(_))[0]
    for n in nmap:
        # noun not at strating pos., not equal to verb
        if (" " + n + " " in remove_prep  or remove_prep.endswith(" " + n)) and n not in vfind:
            nfind.append(n)

    nfind_filtered = []
    for n in nfind:
        remove = False 
        for nn in nfind:
            if n in nn and n != nn:
                remove = True 
                break 
        if not remove:
            nfind_filtered.append(n)
    nfind = nfind_filtered

    if "nut" in nfind:
        nfind = ["nut"]

    if len(nfind) > 1:
        # touch cover of the book, car floor, mix a and b
        nfind = sorted(nfind, key=lambda x : -remove_prep.find(x))[:1]

    return vfind, nfind

pred = {"verb": [], "noun": []}
cntt = 0
nfail = 0
while cntt < 5:
    gen = generator(prompt_full, max_new_tokens=80, num_return_sequences=5, pad_token_id=50256,
        top_k=top_k,
        top_p=top_p,
    )
    for x in gen:
        pred_this = []
        try:
            for vn in x["generated_text"][len(prompt_full):].split(".")[0].split(", "):
                # single action
                vfind, nfind = get_parsed(vn, vmap, nmap) 
                if len(vfind) and len(nfind):
                    pred_this.append([vmap[vfind[0]], nmap[nfind[0]]])

            if len(pred_this) == 0:
                raise Exception
            if len(pred_this) < 20:
                pred_this += [pred_this[-1]] * (20 - len(pred_this)) 
        except Exception as e:
            continue 

        pred_this = pred_this[:20]

        duplicate = False 
        for prev_v, prev_n in zip(pred["verb"], pred["noun"]):
            if (np.array(prev_v) == np.array([x[0] for x in pred_this])).all() and (np.array(prev_n) == np.array([x[1] for x in pred_this])).all():
                duplicate = True
                break 
        if duplicate:
            continue # of while

        cntt += 1
        pred["verb"].append([x[0] for x in pred_this][:20])
        pred["noun"].append([x[1] for x in pred_this][:20])

        if cntt == 5:
            break # of while

    nfail += 1
    if nfail > 10:
        break # of while

if nfail > 10:
    if len(pred["verb"]):
        while len(pred["verb"]) < 5:
            pred["verb"].append(pred["verb"][-1])
            pred["noun"].append(pred["noun"][-1])
    else:
        print("Failed to generate using the following prompt: " + prompt_full)