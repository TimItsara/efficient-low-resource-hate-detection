# Free Resources to Get 8000 Chinese Language Items - Quick Start Guide

## ✅ What You Have Now

## 🎯 Goal
- **8000 Chinese language items** for hate speech dataset

```bash
# Collect 8000 items from Reddit (takes 20-30 hours)
python smart_chinese_collector.py \
  --subs China Sino ChineseLanguage sino_zh \
  --mode hot \
  --target 8000 \
  --chinese-min-chars 15 \
  --chinese-min-ratio 0.6 \
  --sleep 3.5 \
  --wait-on-429 600 \
  --out chinese_8000.jsonl
```


**Need help?** Check the individual scraper files for more options and documentation.
