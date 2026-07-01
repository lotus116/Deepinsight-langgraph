"""Validate Gold SQL for AdventureWorks - writes to file"""
import json
import os
from sqlalchemy import create_engine, text
import pandas as pd

def test_gold_sql():
    with open("eval_suite/evaluation_set_adventureworks.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    engine = create_engine(
        os.getenv("DEEPINSIGHT_ADVENTUREWORKS_DB_URI", "mysql+pymysql://root@localhost:3306/adventureworks"),
        pool_pre_ping=True
    )
    
    results = []
    failed = []
    
    for case in cases:
        case_id = case["id"]
        gold_sql = case["gold_sql"]
        
        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(text(gold_sql), conn)
            results.append(f"[OK] {case_id}: {len(df)} rows")
        except Exception as e:
            err_msg = str(e)[:200].replace('\n', ' ')
            failed.append({"id": case_id, "sql": gold_sql[:100], "error": err_msg})
            results.append(f"[FAIL] {case_id}: {err_msg[:80]}")
    
    # Write results to file
    with open("test_gold_sql_adventureworks_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Total: {len(cases)}, Passed: {len(cases) - len(failed)}, Failed: {len(failed)}\n\n")
        for r in results:
            f.write(r + "\n")
        
        if failed:
            f.write("\n\nFailed Gold SQL Details:\n" + "="*60 + "\n")
            for item in failed:
                f.write(f"\n{item['id']}:\n  SQL: {item['sql']}...\n  Error: {item['error']}\n")
    
    print(f"Results written to test_gold_sql_adventureworks_result.txt")
    print(f"Total: {len(cases)}, Passed: {len(cases) - len(failed)}, Failed: {len(failed)}")
    
    if failed:
        print("\nFailed cases:")
        for item in failed:
            print(f"  {item['id']}: {item['error'][:60]}...")

if __name__ == "__main__":
    test_gold_sql()
