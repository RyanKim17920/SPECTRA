"""Per-entry citation validator: resolves each identifier and compares the claimed
title / first author / year / venue / volume / pages against the registry record."""
import json, re, subprocess, sys, difflib

def get(url):
    try:
        return subprocess.run(["curl","-sL","--max-time","25",url],
                              capture_output=True,text=True).stdout
    except Exception: return ""

def crossref(doi):
    r=get(f"https://api.crossref.org/works/{doi}")
    try: m=json.loads(r)["message"]
    except Exception: return None
    return dict(title=(m.get("title") or [""])[0],
                author=(m.get("author") or [{}])[0].get("family",""),
                year=str(((m.get("issued",{}).get("date-parts") or [[""]])[0])[0]),
                venue=(m.get("container-title") or [""])[0],
                volume=str(m.get("volume","")), pages=str(m.get("page","")))
def arxiv(aid):
    r=get(f"https://export.arxiv.org/api/query?id_list={aid}")
    t=re.findall(r"<title>(.*?)</title>",r,re.S)
    if len(t)<2: return None
    au=re.findall(r"<name>(.*?)</name>",r)
    pub=re.search(r"<published>(\d{4})",r)
    return dict(title=" ".join(t[1].split()),author=au[0].split()[-1] if au else "",
                year=pub.group(1) if pub else "",venue="arXiv",volume="",pages="")
def httpok(u):
    c=subprocess.run(["curl","-sL","-o","/dev/null","-w","%{http_code}","--max-time","25",u],
                     capture_output=True,text=True).stdout.strip()
    return c

def sim(a,b):
    n=lambda x: re.sub(r"[^a-z0-9 ]"," ",x.lower()).split()
    return difflib.SequenceMatcher(None," ".join(n(a))," ".join(n(b))).ratio()

ENTRIES=json.load(open("scripts/citations.json"))
rows=[]
for e in ENTRIES:
    rec=None; how=""
    if e.get("doi"): rec,how=crossref(e["doi"]),"crossref:"+e["doi"]
    elif e.get("arxiv"): rec,how=arxiv(e["arxiv"]),"arxiv:"+e["arxiv"]
    checks=[]
    if rec is None:
        if e.get("url"):
            code=httpok(e["url"]); how="url:"+e["url"]
            checks.append(("resolves", "PASS" if code=="200" else "FAIL", code))
        else:
            checks.append(("resolve","FAIL","no identifier"))
    else:
        checks.append(("resolves","PASS","found"))
        s=sim(e["title"],rec["title"])
        checks.append(("title","PASS" if s>0.82 else "CHECK",f"{s:.2f} :: {rec['title'][:58]}"))
        checks.append(("author","PASS" if e["author"].lower()==rec["author"].lower() else "CHECK",
                       f"claim {e['author']} / reg {rec['author']}"))
        checks.append(("year","PASS" if e["year"]==rec["year"] else "CHECK",
                       f"claim {e['year']} / reg {rec['year']}"))
        if e.get("venue") and rec["venue"]:
            vs=sim(e["venue"],rec["venue"])
            checks.append(("venue","PASS" if vs>0.55 else "CHECK",f"{vs:.2f} :: {rec['venue'][:40]}"))
        if e.get("volume") and rec["volume"]:
            checks.append(("volume","PASS" if e["volume"]==rec["volume"] else "CHECK",
                           f"claim {e['volume']} / reg {rec['volume']}"))
        if e.get("pages") and rec["pages"]:
            norm=lambda p:p.replace("--","-").replace("–","-")
            checks.append(("pages","PASS" if norm(e["pages"])==norm(rec["pages"]) else "CHECK",
                           f"claim {e['pages']} / reg {rec['pages']}"))
    if e.get("url") and rec is not None:
        checks.append(("url",("PASS" if httpok(e["url"])=="200" else "FAIL"),e["url"]))
    rows.append((e["key"],how,checks))

bad=0
for k,how,checks in rows:
    st="PASS" if all(c[1]=="PASS" for c in checks) else ("FAIL" if any(c[1]=="FAIL" for c in checks) else "CHECK")
    bad+= st!="PASS"
    print(f"\n[{st}] {k}   ({how})")
    for name,res,det in checks:
        if res!="PASS": print(f"    {res:5s} {name:8s} {det}")
print(f"\n{len(rows)} entries, {len(rows)-bad} clean, {bad} needing review")
