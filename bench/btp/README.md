# By the People accuracy check

Outside ground truth for the transcription pipeline: Library of Congress
"By the People" pages, transcribed by one volunteer and accepted by another.
Default set is the Historical Legal Reports campaign (typed government reports,
LCCN 2023446314), picked because LoC chose those reports for poor OCR.

```
python bench/btp/prep.py --n 30                     # sample + stage pages under data/btp-*/
NACA_VLM_URL=http://127.0.0.1:8782 NACA_VLM_DIR=vision_8blora_v2_1 \
  python pipeline/pass4a.py --pages-file data/btp/pages.jsonl
python bench/btp/score_btp.py                       # CER/WER per engine
```

Staged GT has status `btp_reviewed`, so `score.py` and `BENCHMARKS.md` never see it.
Results go to `data/btp/scores*.json*`. The campaign prompt says "NACA report page";
it is run unchanged so the score reflects the production config.
