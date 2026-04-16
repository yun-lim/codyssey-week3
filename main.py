from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple


EPSILON = 1e-9
MODE1_SIZE = 3
BENCHMARK_REPETITIONS = 10
INPUT_ERROR_MSG = "입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요."


Matrix = List[List[float]]


def shape(matrix: Any) -> Tuple[int, int]:
    if not isinstance(matrix, list) or not matrix:
        return 0, 0
    if not all(isinstance(row, list) for row in matrix):
        return 0, 0
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    if not all(len(row) == cols for row in matrix):
        return 0, 0
    return rows, cols


def to_float_matrix(matrix: Any) -> Optional[Matrix]:
    rows, cols = shape(matrix)
    if rows == 0 or cols == 0:
        return None
    out: Matrix = []
    for row in matrix:
        converted: List[float] = []
        for value in row:
            try:
                converted.append(float(value))
            except (TypeError, ValueError):
                return None
        out.append(converted)
    return out


def parse_row(line: str, expected_cols: int) -> Optional[List[float]]:
    tokens = line.strip().split()
    if len(tokens) != expected_cols:
        return None
    row: List[float] = []
    for token in tokens:
        try:
            row.append(float(token))
        except ValueError:
            return None
    return row


def read_matrix_from_user(size: int, title: str) -> Matrix:
    print(f"{title} (각 줄에 {size}개의 숫자를 공백으로 구분)")
    while True:
        matrix: Matrix = []
        for row_idx in range(size):
            line = input(f"{row_idx + 1}/{size}: ")
            row = parse_row(line, size)
            if row is None:
                print(INPUT_ERROR_MSG)
                break
            matrix.append(row)
        if len(matrix) == size:
            return matrix


def normalize_label(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"cross", "+"}:
        return "Cross"
    if value in {"x", "×", "*"}:
        return "X"
    return None


def compare_labels(raw: Any) -> Optional[str]:
    return normalize_label(raw)


def mac(pattern: Matrix, kernel: Matrix) -> float:
    n = len(pattern)
    total = 0.0
    for i in range(n):
        pattern_row = pattern[i]
        kernel_row = kernel[i]
        for j in range(n):
            total += pattern_row[j] * kernel_row[j]
    return total


def classify_scores(score_a: float, score_b: float) -> str:
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    if score_a > score_b:
        return "Cross"
    return "X"


def benchmark_size(size: int) -> Tuple[float, int]:
    pattern = generate_pattern(size, "cross")
    filter_cross = generate_filter(size, "cross")
    filter_x = generate_filter(size, "x")

    start = time.perf_counter()
    for _ in range(BENCHMARK_REPETITIONS):
        mac(pattern, filter_cross)
        mac(pattern, filter_x)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / BENCHMARK_REPETITIONS
    return elapsed_ms, size * size


def benchmark_pair(pattern: Matrix, filter_a: Matrix, filter_b: Matrix, repeats: int = BENCHMARK_REPETITIONS) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        mac(pattern, filter_a)
        mac(pattern, filter_b)
    return (time.perf_counter() - start) * 1000.0 / repeats


def generate_pattern(size: int, shape_type: str) -> Matrix:
    pattern = [[0.0 for _ in range(size)] for _ in range(size)]
    center = size // 2
    if shape_type == "cross":
        for i in range(size):
            pattern[center][i] = 1.0
            pattern[i][center] = 1.0
    elif shape_type == "x":
        for i in range(size):
            pattern[i][i] = 1.0
            pattern[i][size - i - 1] = 1.0
    return pattern


def generate_filter(size: int, shape_type: str) -> Matrix:
    return generate_pattern(size, shape_type)


def print_perf_table(sizes: List[int]) -> None:
    print("\n" + "=" * 52)
    print("성능 분석")
    print("=" * 52)
    print(f"{ '크기':>10} | {'평균(ms)':>12} | {'연산 횟수(N^2)':>16}")
    print("-" * 52)
    for size in sizes:
        avg_ms, operations = benchmark_size(size)
        print(f"{size:>5}x{size:<5} | {avg_ms:>10.6f} | {operations:>16d}")


def parse_size_from_key(key: str) -> Optional[int]:
    match = re.match(r"^size_(\d+)_(?:.+)$", key)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_matrix_field(item: Dict[str, Any]) -> Optional[Matrix]:
    candidate_keys = ("input", "pattern", "data")
    raw = None
    for key in candidate_keys:
        if key in item:
            raw = item[key]
            break
    if raw is None:
        return None
    return to_float_matrix(raw)


def resolve_filter_pair(filters: Any) -> Tuple[Optional[Matrix], Optional[Matrix], Optional[str]]:
    if not isinstance(filters, dict):
        return None, None, "필터 항목이 객체가 아닙니다."

    cross_filter = None
    x_filter = None
    cross_candidate = filters.get("cross")
    x_candidate = filters.get("x")
    if cross_candidate is not None:
        cross_filter = to_float_matrix(cross_candidate)
    if x_candidate is not None:
        x_filter = to_float_matrix(x_candidate)

    if cross_filter is not None and x_filter is not None:
        return cross_filter, x_filter, None

    for key, value in filters.items():
        normalized = compare_labels(key)
        candidate = to_float_matrix(value)
        if candidate is None:
            continue
        if normalized == "Cross" and cross_filter is None:
            cross_filter = candidate
            continue
        if normalized == "X" and x_filter is None:
            x_filter = candidate

    if cross_filter is None and isinstance(filters, dict) and len(filters) > 0:
        items = list(filters.values())
        if len(items) >= 2:
            cross_filter = to_float_matrix(items[0])
            x_filter = to_float_matrix(items[1])

    if cross_filter is None:
        return None, x_filter, "Cross 필터를 찾지 못했습니다."
    if x_filter is None:
        return cross_filter, None, "X 필터를 찾지 못했습니다."
    return cross_filter, x_filter, None


def validate_pair(filters: Matrix, size: int, label: str, failures: List[str]) -> bool:
    rows, cols = shape(filters)
    if rows != size or cols != size:
        failures.append(f"{label} 크기 불일치: {rows}x{cols} != {size}x{size}")
        return False
    return True


def run_mode1() -> None:
    print("\n[1] 모드: 사용자 입력 (3x3)")
    filter_a = read_matrix_from_user(MODE1_SIZE, "필터 A")
    filter_b = read_matrix_from_user(MODE1_SIZE, "필터 B")
    pattern = read_matrix_from_user(MODE1_SIZE, "패턴")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    if abs(score_a - score_b) < EPSILON:
        result = "판정 불가"
    elif score_a > score_b:
        result = "A"
    else:
        result = "B"

    print("\n" + "=" * 52)
    print("[3] MAC 결과")
    print("=" * 52)
    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    elapsed = benchmark_pair(pattern, filter_a, filter_b)
    print(f"연산 시간(평균/{BENCHMARK_REPETITIONS}회): {elapsed:.6f} ms")
    print(f"판정: {result}")

    print_perf_table([MODE1_SIZE])


def run_mode2(path: str = "data.json") -> None:
    print("\n[2] 모드: data.json 분석")
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError:
        print(f"data.json 파일을 찾을 수 없습니다. 경로: {path}")
        return
    except json.JSONDecodeError as e:
        print(f"data.json 파싱 실패: {e}")
        return

    if not isinstance(payload, dict):
        print("data.json 형식이 올바르지 않습니다. 최상위 객체가 객체(딕셔너리)여야 합니다.")
        return

    raw_filters = payload.get("filters")
    raw_patterns = payload.get("patterns")
    if not isinstance(raw_filters, dict):
        print("data.json에 filters 항목이 없거나 형식이 잘못되었습니다.")
        return
    if not isinstance(raw_patterns, dict):
        print("data.json에 patterns 항목이 없거나 형식이 잘못되었습니다.")
        return

    total = len(raw_patterns)
    passed = 0
    failed: List[Tuple[str, str]] = []

    print("\n" + "=" * 52)
    print("패턴 분석")
    print("=" * 52)

    for case_id, item in raw_patterns.items():
        if not isinstance(item, dict):
            failed.append((case_id, "항목이 객체가 아닙니다."))
            continue

        reasons: List[str] = []
        size = parse_size_from_key(case_id)
        if size is None:
            failed.append((case_id, "케이스 키 형식이 size_<N>_<id>가 아닙니다."))
            continue

        filter_block_key = f"size_{size}"
        size_filter_group = raw_filters.get(filter_block_key)
        if size_filter_group is None:
            failed.append((case_id, f"filters에 {filter_block_key}가 없습니다."))
            continue

        filter_a, filter_b, reason = resolve_filter_pair(size_filter_group)
        if reason:
            failed.append((case_id, reason))
            continue
        validate_pair(filter_a, size, "Cross 필터", reasons)
        validate_pair(filter_b, size, "X 필터", reasons)
        if reasons:
            failed.append((case_id, ", ".join(reasons)))
            continue

        pattern = extract_matrix_field(item)
        if pattern is None:
            failed.append((case_id, "input(패턴) 필드가 없거나 숫자형이 아닙니다."))
            continue
        rows, cols = shape(pattern)
        if rows != size or cols != size:
            failed.append((case_id, f"패턴 크기 불일치: {rows}x{cols} != {size}x{size}"))
            continue

        score_cross = mac(pattern, filter_a)
        score_x = mac(pattern, filter_b)
        decision = classify_scores(score_cross, score_x)

        expected = compare_labels(item.get("expected", item.get("label", item.get("answer", item.get("filter")))))

        if expected is None:
            failed.append((case_id, "expected(예상 라벨) 값이 없습니다. (+, x 또는 cross, x 중 하나 필요)"))
            continue

        if decision == expected:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed.append((case_id, f"예상={expected}, 판정={decision}"))

        print(f"[{case_id}] size={size}")
        print(f"  Cross 점수: {score_cross}")
        print(f"  X 점수: {score_x}")
        print(f"  판정: {decision}")
        print(f"  결과: {status}")

    print("\n" + "=" * 52)
    print("결과 리포트")
    print("=" * 52)
    failed_count = len(failed)
    print(f"총 테스트 수: {total}")
    print(f"통과 수: {passed}")
    print(f"실패 수: {failed_count}")

    if failed:
        print("\n실패 케이스")
        for case_id, reason in failed:
            print(f"  - {case_id}: {reason}")
    else:
        print("\n실패 케이스가 없습니다.")
        print("원인:")
        print("  - 라벨 정규화를 통해 Cross/ X만으로 통일 처리했고")
        print("  - 점수 동점 비교에 epsilon(1e-9)를 사용해 부동소수점 오차를 제거했기 때문입니다.")

    print_perf_table([3, 5, 13, 25])


def main() -> None:
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")

    choice = input("선택: ").strip()
    if choice == "1":
        run_mode1()
    elif choice == "2":
        run_mode2()
    else:
        print("1 또는 2를 입력해 주세요.")


if __name__ == "__main__":
    main()
