"""HTTP smoke test for YOLO detection service."""

import argparse
import json
import sys
import urllib.error
import urllib.request


def check_s1(url):
    """S1: GET /info returns 200 and device == 'cpu'."""
    try:
        with urllib.request.urlopen(f"{url}/info") as resp:
            if resp.status != 200:
                return "FAIL", f"status {resp.status}"
            data = json.loads(resp.read())
            if data.get("device") != "cpu":
                return "FAIL", f"device is {data.get('device')}, not cpu"
            return "PASS", "device is cpu"
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        return "FAIL", str(e)


def check_s2(url):
    """S2: GET / returns 200 and Content-Type starts with text/html."""
    try:
        with urllib.request.urlopen(f"{url}/") as resp:
            if resp.status != 200:
                return "FAIL", f"status {resp.status}"
            ct = resp.headers.get("Content-Type", "")
            if not ct.startswith("text/html"):
                return "FAIL", f"Content-Type is {ct}"
            return "PASS", "html served"
    except (urllib.error.URLError, OSError) as e:
        return "FAIL", str(e)


def check_s3_s7(url, image_path):
    """S3-S7: POST /detect with conf=0.1 and conf=0.9."""
    results = {}
    for conf in [0.1, 0.9]:
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            req = urllib.request.Request(
                f"{url}/detect?conf={conf}",
                data=data,
                headers={"Content-Type": "image/jpeg"},
                method="POST"
            )
            with urllib.request.urlopen(req) as resp:
                if resp.status != 200:
                    results[conf] = ("FAIL", f"status {resp.status}")
                    continue
                body = json.loads(resp.read())
                results[conf] = ("data", body)
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            results[conf] = ("FAIL", str(e))
    return results


def check_s8(url):
    """S8: POST /detect with body 'not an image' returns 400."""
    try:
        req = urllib.request.Request(
            f"{url}/detect?conf=0.1",
            data=b"not an image",
            headers={"Content-Type": "image/jpeg"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            return "FAIL", f"expected 400, got {resp.status}"
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return "PASS", "returns 400"
        return "FAIL", f"expected 400, got {e.code}"
    except urllib.error.URLError as e:
        return "FAIL", str(e)


def check_s9(url):
    """S9: POST /detect with empty body returns 400."""
    try:
        req = urllib.request.Request(
            f"{url}/detect?conf=0.1",
            data=b"",
            headers={"Content-Type": "image/jpeg"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            return "FAIL", f"expected 400, got {resp.status}"
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return "PASS", "returns 400"
        return "FAIL", f"expected 400, got {e.code}"
    except urllib.error.URLError as e:
        return "FAIL", str(e)


def check_s10(url):
    """S10: POST /detect?conf=abc returns 422."""
    try:
        req = urllib.request.Request(
            f"{url}/detect?conf=abc",
            data=b"x",
            headers={"Content-Type": "image/jpeg"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            return "FAIL", f"expected 422, got {resp.status}"
    except urllib.error.HTTPError as e:
        if e.code == 422:
            return "PASS", "returns 422"
        return "FAIL", f"expected 422, got {e.code}"
    except urllib.error.URLError as e:
        return "FAIL", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--image")
    args = parser.parse_args()

    passed = 0
    failed = 0
    skipped = 0

    # S1
    result, desc = check_s1(args.url)
    print(f"{result} S1 GET /info device check ({desc})")
    if result == "PASS":
        passed += 1
    elif result == "FAIL":
        failed += 1

    # S2
    result, desc = check_s2(args.url)
    print(f"{result} S2 GET / serves HTML ({desc})")
    if result == "PASS":
        passed += 1
    elif result == "FAIL":
        failed += 1

    # S3-S7
    if not args.image:
        print("SKIP S3 POST /detect with photo (no image)")
        print("SKIP S4 detections include person (no image)")
        print("SKIP S5 boxes are normalized (no image)")
        print("SKIP S6 confidence is valid (no image)")
        print("SKIP S7 conf=0.9 has <= detections than conf=0.1 (no image)")
        skipped += 5
    else:
        results = check_s3_s7(args.url, args.image)

        conf01 = results.get(0.1)
        conf09 = results.get(0.9)

        # S3
        if conf01 and conf01[0] == "data":
            body = conf01[1]
            required = {"width", "height", "inference_ms", "detections"}
            if required <= set(body.keys()):
                print("PASS S3 POST /detect returns documented shape")
                passed += 1
            else:
                print(f"FAIL S3 POST /detect missing keys {required - set(body.keys())}")
                failed += 1
        else:
            print(f"FAIL S3 POST /detect {conf01[1] if conf01 else 'unknown error'}")
            failed += 1

        # S4
        if conf01 and conf01[0] == "data":
            body = conf01[1]
            has_person = any(d.get("label") == "person" for d in body.get("detections", []))
            if has_person:
                print("PASS S4 detections include person")
                passed += 1
            else:
                print("FAIL S4 no person detected in photo")
                failed += 1
        else:
            print("SKIP S4 detections include person (S3 failed)")
            skipped += 1

        # S5
        if conf01 and conf01[0] == "data":
            body = conf01[1]
            valid = True
            for det in body.get("detections", []):
                box = det.get("box", [])
                if len(box) != 4:
                    valid = False
                    break
                x1, y1, x2, y2 = box
                if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                    valid = False
                    break
                if not (x1 < x2 - 1e-6 and y1 < y2 - 1e-6):
                    valid = False
                    break
            if valid:
                print("PASS S5 boxes are normalized and ordered")
                passed += 1
            else:
                print("FAIL S5 boxes invalid")
                failed += 1
        else:
            print("SKIP S5 boxes normalized (S3 failed)")
            skipped += 1

        # S6
        if conf01 and conf01[0] == "data":
            body = conf01[1]
            valid = True
            for det in body.get("detections", []):
                conf = det.get("conf", -1)
                if not (0.1 <= conf <= 1.0):
                    valid = False
                    break
            if valid:
                print("PASS S6 confidence >= 0.1 and <= 1.0")
                passed += 1
            else:
                print("FAIL S6 confidence out of range")
                failed += 1
        else:
            print("SKIP S6 confidence valid (S3 failed)")
            skipped += 1

        # S7
        if conf01 and conf01[0] == "data" and conf09 and conf09[0] == "data":
            count01 = len(conf01[1].get("detections", []))
            count09 = len(conf09[1].get("detections", []))
            if count09 <= count01:
                print("PASS S7 conf=0.9 has fewer or equal detections")
                passed += 1
            else:
                print(f"FAIL S7 conf=0.9 has {count09} dets, conf=0.1 has {count01}")
                failed += 1
        else:
            print("SKIP S7 comparison (missing S3 data)")
            skipped += 1

    # S8
    result, desc = check_s8(args.url)
    print(f"{result} S8 garbage body returns 400 ({desc})")
    if result == "PASS":
        passed += 1
    elif result == "FAIL":
        failed += 1

    # S9
    result, desc = check_s9(args.url)
    print(f"{result} S9 empty body returns 400 ({desc})")
    if result == "PASS":
        passed += 1
    elif result == "FAIL":
        failed += 1

    # S10
    result, desc = check_s10(args.url)
    print(f"{result} S10 non-numeric conf returns 422 ({desc})")
    if result == "PASS":
        passed += 1
    elif result == "FAIL":
        failed += 1

    print(f"passed {passed}, failed {failed}, skipped {skipped}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
