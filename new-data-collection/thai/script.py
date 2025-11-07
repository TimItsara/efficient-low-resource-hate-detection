# prep_thai.py
import argparse, os, math, random
import pandas as pd
from pathlib import Path

def normalize_df(df):
    # try to find text column
    text_cols = [c for c in df.columns if c.lower() in {"text","tweet","content","message"}]
    if not text_cols:
        # fallback: first string-like column
        text_cols = [c for c in df.columns if df[c].dtype==object]
    text_col = text_cols[0]
    # try to find label column
    lbl_candidates = [c for c in df.columns if "hate" in c.lower() or c.lower() in {"label","labels","y"}]
    if not lbl_candidates:
        raise ValueError("Could not find a label column (looked for 'hate', 'label', 'labels', 'y').")
    lbl_col = lbl_candidates[0]

    def to_bin(x):
        s = str(x).strip().lower()
        if s in {"1","true","hatespeech","hate","toxic","hs"}: return 1
        if s in {"0","false","nonhatespeech","not_hate","non-hate","nonhate","clean"}: return 0
        # try numeric
        try:
            v = int(float(s))
            return 1 if v==1 else 0
        except: 
            return None

    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": df[lbl_col].map(to_bin)
    }).dropna(subset=["label"]).reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    # basic cleaning: drop very short / duplicates
    out = out[out["text"].str.strip().str.len() >= 3]
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return out

def stratified_split(df, n_dev=500, n_test=2000, seed=42):
    rnd = random.Random(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    parts = []
    for lbl in [0,1]:
        sub = df[df.label==lbl]
        parts.append(("dev", sub.iloc[:math.ceil(n_dev*len(sub)/len(df))]))
        parts.append(("test", sub.iloc[math.ceil(n_dev*len(sub)/len(df)):math.ceil((n_dev+n_test)*len(sub)/len(df))]))
        parts.append(("rest", sub.iloc[math.ceil((n_dev+n_test)*len(sub)/len(df)):]))
    dev = pd.concat([p[1] for p in parts if p[0]=="dev"]).sample(frac=1.0, random_state=seed)
    test = pd.concat([p[1] for p in parts if p[0]=="test"]).sample(frac=1.0, random_state=seed)
    rest = pd.concat([p[1] for p in parts if p[0]=="rest"]).sample(frac=1.0, random_state=seed)
    # if a class vanished in dev/test because of extreme imbalance, fallback to simple split
    for dname, d in [("dev",dev),("test",test)]:
        if d["label"].nunique()<2 and rest["label"].nunique()==2:
            need = 1
            add = rest.groupby("label").head(need)
            d = pd.concat([d, add]).drop_duplicates().sample(frac=1.0, random_state=seed)
            rest = rest.drop(add.index)
            if dname=="dev": dev=d
            else: test=d
    return dev, test, rest.rename(columns={"text":"text","label":"label"})

def sample_fewshot(train_pool, N, seed):
    rs = seed
    # try stratified if both classes present and N>=2
    if train_pool["label"].nunique()==2 and N>=2:
        k0 = max(1, N//2)
        k1 = N - k0
        s0 = train_pool[train_pool.label==0].sample(n=min(k0, len(train_pool[train_pool.label==0])), random_state=rs)
        s1 = train_pool[train_pool.label==1].sample(n=min(k1, len(train_pool[train_pool.label==1])), random_state=rs)
        samp = pd.concat([s0,s1]).sample(frac=1.0, random_state=rs)
    else:
        samp = train_pool.sample(n=min(N, len(train_pool)), random_state=rs)
    # ensure at least one hateful item if available
    if 1 in train_pool.label.values and (samp.label==1).sum()==0:
        pos = train_pool[train_pool.label==1].sample(n=1, random_state=rs)
        samp = pd.concat([samp.iloc[:-1], pos]).sample(frac=1.0, random_state=rs)
    return samp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to HateThaiSent.csv")
    ap.add_argument("--out_dir", required=True, help="Output dataset folder")
    ap.add_argument("--dev", type=int, default=500)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--Ns", nargs="+", type=int, default=[20,200,2000])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1,11)))
    ap.add_argument("--split_seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(args.input)
    df = normalize_df(df_raw)

    # splits
    dev, test, train_pool = stratified_split(df, n_dev=args.dev, n_test=args.test, seed=args.split_seed)
    dev.to_csv(os.path.join(args.out_dir, "dev.csv"), index=False)
    test.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    train_pool.to_csv(os.path.join(args.out_dir, "train_pool.csv"), index=False)

    # few-shot
    for N in args.Ns:
        for sd in args.seeds:
            samp = sample_fewshot(train_pool, N, sd)
            out_dir = Path(args.out_dir) / "train"
            out_dir.mkdir(parents=True, exist_ok=True)
            samp.to_csv(out_dir / f"train_{N}_rs{sd}.csv", index=False)
    # summary
    print("Saved:", args.out_dir)
    print("Counts:", {"dev":len(dev), "test":len(test), "train_pool":len(train_pool)})
    print("Hate rates:", {
        "dev": round(dev.label.mean(),3),
        "test": round(test.label.mean(),3),
        "train_pool": round(train_pool.label.mean(),3),
    })

if __name__ == "__main__":
    main()


# python script.py \
#   --input ./HateThaiSent.csv \
#   --out_dir ./datasets/thai_hatesent \
#   --dev 500 --test 2000 \
#   --Ns 20 50 100 200 500 1000 2000 \
#   --seeds 1 2 3 4 5 6 7 8 9 10
