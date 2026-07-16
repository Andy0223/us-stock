Put the latest Firstrade positions CSV here, then run:

```bash
./scripts/run_firstrade_sync.sh
```

The importer updates `data/holdings.csv` only when it can recognize ticker,
share quantity, and average cost columns. If the official export uses different
column names, the script stops and prints the detected columns instead of
guessing.
