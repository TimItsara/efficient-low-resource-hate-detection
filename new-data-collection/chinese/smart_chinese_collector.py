#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Chinese collector that respects Reddit's rate limits
Automatically waits when hitting 429 errors and continues collection
"""

import argparse
import json
import time
import sys
import urllib.request
import urllib.parse
import urllib.error
import random
import re
from datetime import datetime

UA = "COMP8240-Student-DataCollection/1.0"
CN_RE = re.compile(r"[\u4e00-\u9fff]")

def http_get_smart(url, base_sleep=3.0, wait_on_429=600):
    """Smart HTTP get that waits appropriately on 429"""
    url_safe = urllib.parse.quote(url.encode('utf-8'), safe=':/?#[]@!$&\'()*+,;=')
    
    attempt = 0
    max_attempts = 20
    
    while attempt < max_attempts:
        try:
            req = urllib.request.Request(url_safe, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read().decode("utf-8", errors="ignore")
            time.sleep(base_sleep + random.random())
            return json.loads(data)
            
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Hit rate limit - wait 10 minutes
                print(f"\n⚠️  Rate limit hit (429). Waiting {wait_on_429//60} minutes...", file=sys.stderr)
                print(f"   Time: {datetime.now().strftime('%H:%M:%S')}", file=sys.stderr)
                
                # Wait with countdown
                for remaining in range(wait_on_429, 0, -30):
                    mins, secs = divmod(remaining, 60)
                    print(f"   Resuming in {mins}m {secs}s...", end='\r', file=sys.stderr)
                    time.sleep(30)
                
                print(f"\n   Resuming collection...                    ", file=sys.stderr)
                attempt += 1
                continue
            else:
                raise
                
        except Exception as e:
            if attempt < 3:
                time.sleep(base_sleep * 2)
                attempt += 1
                continue
            else:
                raise
    
    return None

def chinese_stats(s):
    if not s: return 0, 0.0
    total = len(s)
    cn = sum(1 for ch in s if '\u4e00' <= ch <= '\u9fff')
    return cn, (cn/total if total else 0.0)

def chinese_enough(s, min_chars, min_ratio):
    cn, ratio = chinese_stats(s)
    return cn >= min_chars and ratio >= min_ratio

def fetch_listing(sub, mode="hot", after=None, base_sleep=3.0):
    base_url = f"https://www.reddit.com/r/{sub}"
    path = f"/{mode}.json"
    params = {"limit": 100}
    if after: params["after"] = after
    
    url = base_url + path + "?" + urllib.parse.urlencode(params)
    return http_get_smart(url, base_sleep=base_sleep)

def fetch_comments(permalink, base_sleep=3.0, limit=50, sample_pct=1.0):
    if random.random() > sample_pct:
        return []
    
    url = f"https://www.reddit.com{permalink}.json?limit=500"
    data = http_get_smart(url, base_sleep=base_sleep)
    
    if not data or len(data) < 2:
        return []
    
    comments = []
    for item in data[1].get("data", {}).get("children", [])[:limit]:
        if item.get("kind") == "t1":
            body = item.get("data", {}).get("body", "").strip()
            if body:
                comments.append({
                    "body": body,
                    "id": item.get("data", {}).get("id"),
                    "created_utc": item.get("data", {}).get("created_utc")
                })
    
    return comments

def main():
    parser = argparse.ArgumentParser(description='Smart Chinese collector with rate limit handling')
    parser.add_argument('--subs', nargs='+', default=['China', 'Sino', 'ChineseLanguage'])
    parser.add_argument('--mode', choices=['hot', 'new', 'top'], default='hot')
    parser.add_argument('--target', type=int, default=4000, help='Target number of items')
    parser.add_argument('--chinese-min-chars', type=int, default=15)
    parser.add_argument('--chinese-min-ratio', type=float, default=0.6)
    parser.add_argument('--sleep', type=float, default=3.5)
    parser.add_argument('--wait-on-429', type=int, default=600, help='Seconds to wait on 429 (default: 10min)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from existing file')
    args = parser.parse_args()
    
    print("="*70)
    print("SMART CHINESE COLLECTOR - Rate Limit Aware")
    print("="*70)
    print(f"Target: {args.target} Chinese items")
    print(f"Strategy: Wait {args.wait_on_429//60}min on rate limits, then continue")
    print(f"Sleep: {args.sleep}s between requests")
    print(f"Subreddits: {', '.join(args.subs)}")
    print("="*70)
    print(f"\nThis will take time but will eventually reach {args.target} items!")
    print(f"Estimated: {(args.target * args.sleep / 60):.0f}-{(args.target * args.sleep / 60 * 2):.0f} minutes (if no 429)")
    print(f"With 429s: Could take several hours")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load existing data if resuming
    kept = 0
    seen_ids = set()
    
    if args.resume:
        try:
            with open(args.out, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    seen_ids.add(item['meta']['id'])
                    kept += 1
            print(f"\n✓ Resuming: Found {kept} existing items")
        except FileNotFoundError:
            print(f"\n⚠️  No existing file, starting fresh")
    
    mode = 'a' if args.resume else 'w'
    
    with open(args.out, mode, encoding='utf-8') as fo:
        for sub_idx, sub in enumerate(args.subs):
            if kept >= args.target:
                break
            
            print(f"\n{'='*70}")
            print(f"[{sub_idx+1}/{len(args.subs)}] Subreddit: r/{sub}")
            print(f"{'='*70}")
            
            page = 0
            after = None
            
            while kept < args.target:
                page += 1
                print(f"\nPage {page} | Progress: {kept}/{args.target} ({kept/args.target*100:.1f}%)")
                
                data = fetch_listing(sub, mode=args.mode, after=after, base_sleep=args.sleep)
                if not data:
                    print(f"  No more data from r/{sub}")
                    break
                
                posts = data.get("data", {}).get("children", [])
                if not posts:
                    break
                
                for post in posts:
                    if kept >= args.target:
                        break
                    
                    p = post.get("data", {})
                    post_id = p.get("id")
                    
                    if post_id in seen_ids:
                        continue
                    
                    # Check post
                    title = (p.get("title") or "").strip()
                    selftext = (p.get("selftext") or "").strip()
                    text = (title + " " + selftext).strip()
                    
                    cn_chars, cn_ratio = chinese_stats(text)
                    
                    if text and chinese_enough(text, args.chinese_min_chars, args.chinese_min_ratio):
                        item = {
                            "text": text,
                            "label": "",
                            "source": "reddit",
                            "meta": {
                                "type": "post",
                                "id": post_id,
                                "subreddit": sub,
                                "created_utc": p.get("created_utc"),
                                "chinese_chars": cn_chars,
                                "chinese_ratio": round(cn_ratio, 3)
                            }
                        }
                        fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                        fo.flush()
                        seen_ids.add(post_id)
                        kept += 1
                        print(f"  ✓ Post {kept}/{args.target}: {cn_ratio*100:.0f}% Chinese")
                    
                    # Check comments
                    if kept < args.target and p.get("permalink"):
                        comments = fetch_comments(p["permalink"], base_sleep=args.sleep, limit=100, sample_pct=0.5)
                        
                        for c in comments:
                            if kept >= args.target:
                                break
                            
                            comment_id = c.get("id")
                            if comment_id in seen_ids:
                                continue
                            
                            body = c.get("body", "").strip()
                            c_cn_chars, c_cn_ratio = chinese_stats(body)
                            
                            if body and chinese_enough(body, args.chinese_min_chars, args.chinese_min_ratio):
                                item = {
                                    "text": body,
                                    "label": "",
                                    "source": "reddit",
                                    "meta": {
                                        "type": "comment",
                                        "id": comment_id,
                                        "parent": post_id,
                                        "subreddit": sub,
                                        "created_utc": c.get("created_utc"),
                                        "chinese_chars": c_cn_chars,
                                        "chinese_ratio": round(c_cn_ratio, 3)
                                    }
                                }
                                fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                                fo.flush()
                                seen_ids.add(comment_id)
                                kept += 1
                                print(f"  ✓ Comment {kept}/{args.target}: {c_cn_ratio*100:.0f}% Chinese")
                
                after = data.get("data", {}).get("after")
                if not after:
                    break
    
    print(f"\n{'='*70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Chinese items: {kept}")
    print(f"Saved to: {args.out}")
    
    if kept < args.target:
        print(f"\n⚠️  Only collected {kept}/{args.target} items")
        print(f"   Reddit may not have enough Chinese content")
        print(f"   Run with --resume to continue later")
    else:
        print(f"\n🎉 SUCCESS! Collected {kept} Chinese items!")

if __name__ == "__main__":
    main()
