"""Run from the project root: python challenge/validate_submission.py FILE CATALOG."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from s2f.challenge import validate_submission

if __name__=="__main__":
    try:
        print(f"VALID: {len(validate_submission(sys.argv[1],sys.argv[2]))} clips")
    except (ValueError,IndexError) as error:
        sys.exit(f"ERROR: {error}")
