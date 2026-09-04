import math

from src.evaluation.metrics import (
    precision_recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    mean_average_precision,
    ndcg_at_k,
)


EPS = 1e-6


def assert_close(actual, expected, name):
    assert math.isclose(
        actual,
        expected,
        rel_tol=EPS,
        abs_tol=EPS
    ), (
        f"\n{name} FAILED"
        f"\nExpected: {expected:.6f}"
        f"\nActual:   {actual:.6f}"
    )


def show_metrics(name, docs_true, queries, top_k):
    precision, recall = precision_recall_at_k(
        docs_true,
        queries,
        top_k
    )

    hit_rate = hit_rate_at_k(
        docs_true,
        queries,
        top_k
    )

    mrr = mean_reciprocal_rank(
        docs_true,
        queries
    )

    map_score = mean_average_precision(
        docs_true,
        queries
    )

    ndcg = ndcg_at_k(
        docs_true,
        queries,
        top_k
    )

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(f"Precision@{top_k}: {precision:.6f}")
    print(f"Recall@{top_k}:    {recall:.6f}")
    print(f"HitRate@{top_k}:   {hit_rate:.6f}")
    print(f"MRR:               {mrr:.6f}")
    print(f"MAP:               {map_score:.6f}")
    print(f"NDCG@{top_k}:      {ndcg:.6f}")

    return (
        precision,
        recall,
        hit_rate,
        mrr,
        map_score,
        ndcg
    )


# ============================================================
# TEST 1
# PERFECT RETRIEVAL
# ============================================================

def test_perfect():
    docs_true = [
        [1, 2],
        [3, 4],
        [5]
    ]

    queries = [
        [1, 2],
        [3, 4],
        [5]
    ]

    top_k = 2

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 1 - PERFECT",
        docs_true,
        queries,
        top_k
    )

    # Query 3 chỉ trả 1 document nhưng top_k = 2.
    # Theo implementation của bạn:
    # Precision query 3 = 1 / 2
    #
    # Mean P = (1 + 1 + 0.5) / 3

    expected_precision = (
        1 +
        1 +
        1/2
    ) / 3

    assert_close(
        p,
        expected_precision,
        "Precision"
    )

    assert_close(r, 1.0, "Recall")
    assert_close(hit, 1.0, "HitRate")
    assert_close(mrr, 1.0, "MRR")
    assert_close(map_score, 1.0, "MAP")
    assert_close(ndcg, 1.0, "NDCG")


# ============================================================
# TEST 2
# COMPLETE MISS
# ============================================================

def test_complete_miss():
    docs_true = [
        [10],
        [20],
        [30]
    ]

    queries = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    top_k = 3

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 2 - COMPLETE MISS",
        docs_true,
        queries,
        top_k
    )

    assert_close(p, 0.0, "Precision")
    assert_close(r, 0.0, "Recall")
    assert_close(hit, 0.0, "HitRate")
    assert_close(mrr, 0.0, "MRR")
    assert_close(map_score, 0.0, "MAP")
    assert_close(ndcg, 0.0, "NDCG")


# ============================================================
# TEST 3
# NORMAL MIXED CASE
# ============================================================

def test_normal():
    docs_true = [
        [2, 4],
        [10, 50],
        [100]
    ]

    queries = [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        [5, 6, 7, 8]
    ]

    top_k = 3

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 3 - NORMAL MIXED",
        docs_true,
        queries,
        top_k
    )

    # Precision
    #
    # Q1 = 1/3
    # Q2 = 1/3
    # Q3 = 0

    expected_precision = (
        1/3 +
        1/3 +
        0
    ) / 3

    # Recall
    #
    # Q1 = 1/2
    # Q2 = 1/2
    # Q3 = 0

    expected_recall = (
        1/2 +
        1/2 +
        0
    ) / 3

    # HitRate
    # hit, hit, miss

    expected_hit = 2 / 3

    # MRR
    #
    # Q1 first relevant rank 2 -> 1/2
    # Q2 first relevant rank 1 -> 1
    # Q3 -> 0

    expected_mrr = (
        1/2 +
        1 +
        0
    ) / 3

    # MAP
    #
    # Q1:
    # relevant ranks 2 and 4
    # P@2 = 1/2
    # P@4 = 2/4
    # AP = (1/2 + 2/4)/2 = 0.5
    #
    # Q2:
    # only 10 retrieved, 50 missed
    # P@1 = 1
    # AP = 1/2
    #
    # Q3 = 0

    expected_map = (
        0.5 +
        0.5 +
        0
    ) / 3

    assert_close(
        p,
        expected_precision,
        "Precision"
    )

    assert_close(
        r,
        expected_recall,
        "Recall"
    )

    assert_close(
        hit,
        expected_hit,
        "HitRate"
    )

    assert_close(
        mrr,
        expected_mrr,
        "MRR"
    )

    assert_close(
        map_score,
        expected_map,
        "MAP"
    )

    assert 0 <= ndcg <= 1


# ============================================================
# TEST 4
# RELEVANT AT RANK 1, 2, 3, 4
# ============================================================

def test_different_ranks():
    docs_true = [
        [1],
        [5],
        [9],
        [16]
    ]

    queries = [
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 10],
        [13, 14, 15, 16]
    ]

    top_k = 4

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 4 - DIFFERENT RELEVANT RANKS",
        docs_true,
        queries,
        top_k
    )

    expected_precision = 1 / 4
    expected_recall = 1.0
    expected_hit = 1.0

    expected_mrr = (
        1 +
        1/2 +
        1/3 +
        1/4
    ) / 4

    # Với mỗi query chỉ có 1 relevant document:
    # AP = Reciprocal Rank
    expected_map = expected_mrr

    expected_ndcg = (
        1 +
        1 / math.log2(3) +
        1 / math.log2(4) +
        1 / math.log2(5)
    ) / 4

    assert_close(
        p,
        expected_precision,
        "Precision"
    )

    assert_close(
        r,
        expected_recall,
        "Recall"
    )

    assert_close(
        hit,
        expected_hit,
        "HitRate"
    )

    assert_close(
        mrr,
        expected_mrr,
        "MRR"
    )

    assert_close(
        map_score,
        expected_map,
        "MAP"
    )

    assert_close(
        ndcg,
        expected_ndcg,
        "NDCG"
    )


# ============================================================
# TEST 5
# MULTIPLE RELEVANT DOCUMENTS
# ============================================================

def test_multiple_relevant():
    docs_true = [
        [2, 4, 6],
        [10, 20, 30]
    ]

    queries = [
        [2, 1, 4, 3, 6],
        [10, 50, 20, 60, 30]
    ]

    top_k = 5

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 5 - MULTIPLE RELEVANT",
        docs_true,
        queries,
        top_k
    )

    # 3 relevant / top 5
    assert_close(
        p,
        3/5,
        "Precision"
    )

    assert_close(
        r,
        1.0,
        "Recall"
    )

    assert_close(
        hit,
        1.0,
        "HitRate"
    )

    # First relevant rank 1 for both
    assert_close(
        mrr,
        1.0,
        "MRR"
    )

    # Both have relevant ranks:
    # 1,3,5
    #
    # P@1 = 1
    # P@3 = 2/3
    # P@5 = 3/5

    expected_map = (
        1 +
        2/3 +
        3/5
    ) / 3

    assert_close(
        map_score,
        expected_map,
        "MAP"
    )

    assert 0 < ndcg < 1


# ============================================================
# TEST 6
# PARTIAL RECALL
# ============================================================

def test_partial_recall():
    docs_true = [
        [2, 4, 6, 8]
    ]

    queries = [
        [2, 1, 4, 3]
    ]

    top_k = 4

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 6 - PARTIAL RECALL",
        docs_true,
        queries,
        top_k
    )

    # Relevant retrieved = {2,4}
    #
    # P = 2/4
    # R = 2/4

    assert_close(
        p,
        0.5,
        "Precision"
    )

    assert_close(
        r,
        0.5,
        "Recall"
    )

    assert_close(
        hit,
        1.0,
        "HitRate"
    )

    # First relevant at rank 1
    assert_close(
        mrr,
        1.0,
        "MRR"
    )

    # P@1 = 1
    # P@3 = 2/3
    #
    # Total relevant = 4
    expected_map = (
        1 +
        2/3
    ) / 4

    assert_close(
        map_score,
        expected_map,
        "MAP"
    )

    assert 0 < ndcg < 1


# ============================================================
# TEST 7
# RELEVANT DOCUMENT OUTSIDE TOP-K
# ============================================================

def test_relevant_outside_top_k():
    docs_true = [
        [4],
        [40]
    ]

    queries = [
        [1, 2, 3, 4],
        [10, 20, 30, 40]
    ]

    top_k = 3

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 7 - RELEVANT OUTSIDE TOP-K",
        docs_true,
        queries,
        top_k
    )

    # @3 metrics cannot see rank 4.

    assert_close(
        p,
        0.0,
        "Precision"
    )

    assert_close(
        r,
        0.0,
        "Recall"
    )

    assert_close(
        hit,
        0.0,
        "HitRate"
    )

    assert_close(
        ndcg,
        0.0,
        "NDCG"
    )

    # MRR/MAP use full prediction list.
    assert_close(
        mrr,
        1/4,
        "MRR"
    )

    assert_close(
        map_score,
        1/4,
        "MAP"
    )


# ============================================================
# TEST 8
# TOP K = 1
# ============================================================

def test_top_k_one():
    docs_true = [
        [1],
        [20],
        [5],
        [40]
    ]

    queries = [
        [1, 2],
        [10, 20],
        [5, 6],
        [30, 40]
    ]

    top_k = 1

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 8 - TOP K = 1",
        docs_true,
        queries,
        top_k
    )

    # Hit at first position:
    #
    # Q1 yes
    # Q2 no
    # Q3 yes
    # Q4 no

    expected_at_1 = 2 / 4

    assert_close(
        p,
        expected_at_1,
        "Precision"
    )

    assert_close(
        r,
        expected_at_1,
        "Recall"
    )

    assert_close(
        hit,
        expected_at_1,
        "HitRate"
    )

    assert_close(
        ndcg,
        expected_at_1,
        "NDCG"
    )

    # Full rankings:
    # 1, 2, 1, 2

    expected_mrr = (
        1 +
        1/2 +
        1 +
        1/2
    ) / 4

    assert_close(
        mrr,
        expected_mrr,
        "MRR"
    )

    assert_close(
        map_score,
        expected_mrr,
        "MAP"
    )


# ============================================================
# TEST 9
# TOP K > NUMBER OF RETRIEVED DOCUMENTS
# ============================================================

def test_top_k_greater_than_query_length():
    docs_true = [
        [1, 2]
    ]

    queries = [
        [1, 2]
    ]

    top_k = 5

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 9 - TOP K > PRED LENGTH",
        docs_true,
        queries,
        top_k
    )

    # Theo implementation hiện tại:
    #
    # Precision@5 = 2/5
    #
    # KHÔNG phải 2/2.

    assert_close(
        p,
        2/5,
        "Precision"
    )

    assert_close(
        r,
        1.0,
        "Recall"
    )

    assert_close(
        hit,
        1.0,
        "HitRate"
    )

    assert_close(
        mrr,
        1.0,
        "MRR"
    )

    assert_close(
        map_score,
        1.0,
        "MAP"
    )

    assert_close(
        ndcg,
        1.0,
        "NDCG"
    )


# ============================================================
# TEST 10
# MRR ONLY USES FIRST RELEVANT
# ============================================================

def test_mrr_first_relevant():
    docs_true = [
        [3, 4, 5]
    ]

    queries = [
        [1, 3, 4, 5]
    ]

    top_k = 4

    _, _, _, mrr, map_score, _ = show_metrics(
        "TEST 10 - MRR FIRST RELEVANT ONLY",
        docs_true,
        queries,
        top_k
    )

    # First relevant = 3 at rank 2.

    assert_close(
        mrr,
        1/2,
        "MRR"
    )

    # MAP considers every relevant item:
    #
    # rank 2: P = 1/2
    # rank 3: P = 2/3
    # rank 4: P = 3/4

    expected_map = (
        1/2 +
        2/3 +
        3/4
    ) / 3

    assert_close(
        map_score,
        expected_map,
        "MAP"
    )


# ============================================================
# TEST 11
# SAME PRECISION/RECALL, DIFFERENT RANKING QUALITY
# ============================================================

def test_good_vs_bad_ranking():
    docs_true = [
        [2, 4]
    ]

    good_queries = [
        [2, 4, 1, 3]
    ]

    bad_queries = [
        [1, 3, 2, 4]
    ]

    top_k = 4

    good = show_metrics(
        "TEST 11A - GOOD RANKING",
        docs_true,
        good_queries,
        top_k
    )

    bad = show_metrics(
        "TEST 11B - BAD RANKING",
        docs_true,
        bad_queries,
        top_k
    )

    (
        good_p,
        good_r,
        good_hit,
        good_mrr,
        good_map,
        good_ndcg
    ) = good

    (
        bad_p,
        bad_r,
        bad_hit,
        bad_mrr,
        bad_map,
        bad_ndcg
    ) = bad

    # Both retrieve exactly same relevant documents
    # within Top-4.

    assert_close(
        good_p,
        bad_p,
        "Precision equality"
    )

    assert_close(
        good_r,
        bad_r,
        "Recall equality"
    )

    assert_close(
        good_hit,
        bad_hit,
        "HitRate equality"
    )

    # Ranking-aware metrics must detect difference.

    assert good_mrr > bad_mrr
    assert good_map > bad_map
    assert good_ndcg > bad_ndcg

    assert_close(
        good_mrr,
        1.0,
        "Good MRR"
    )

    assert_close(
        good_map,
        1.0,
        "Good MAP"
    )

    assert_close(
        good_ndcg,
        1.0,
        "Good NDCG"
    )


# ============================================================
# TEST 12
# HIGH HIT RATE BUT LOW RANKING QUALITY
# ============================================================

def test_high_hit_bad_ranking():
    docs_true = [
        [4],
        [8],
        [12]
    ]

    queries = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]

    top_k = 4

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 12 - HIGH HIT RATE, BAD RANKING",
        docs_true,
        queries,
        top_k
    )

    # Relevant always exists in Top-4.

    assert_close(
        hit,
        1.0,
        "HitRate"
    )

    assert_close(
        r,
        1.0,
        "Recall"
    )

    # But always rank 4.

    assert_close(
        mrr,
        1/4,
        "MRR"
    )

    assert_close(
        map_score,
        1/4,
        "MAP"
    )

    expected_ndcg = (
        1 / math.log2(5)
    )

    assert_close(
        ndcg,
        expected_ndcg,
        "NDCG"
    )

    assert_close(
        p,
        1/4,
        "Precision"
    )


# ============================================================
# TEST 13
# DIFFERENT NUMBER OF RELEVANT DOCS PER QUERY
# ============================================================

def test_different_gt_sizes():
    docs_true = [
        [1],
        [10, 20],
        [30, 40, 50, 60]
    ]

    queries = [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        [30, 40, 70, 80]
    ]

    top_k = 4

    p, r, hit, _, _, _ = show_metrics(
        "TEST 13 - DIFFERENT GT SIZES",
        docs_true,
        queries,
        top_k
    )

    # Precision
    #
    # Q1 = 1/4
    # Q2 = 2/4
    # Q3 = 2/4

    expected_precision = (
        1/4 +
        2/4 +
        2/4
    ) / 3

    # Recall
    #
    # Q1 = 1/1
    # Q2 = 2/2
    # Q3 = 2/4

    expected_recall = (
        1 +
        1 +
        1/2
    ) / 3

    assert_close(
        p,
        expected_precision,
        "Precision"
    )

    assert_close(
        r,
        expected_recall,
        "Recall"
    )

    assert_close(
        hit,
        1.0,
        "HitRate"
    )


# ============================================================
# TEST 14
# EMPTY GROUND TRUTH FOR ONE QUERY
# ============================================================

def test_empty_ground_truth():
    docs_true = [
        [],
        [2]
    ]

    queries = [
        [1, 3, 5],
        [2, 4, 6]
    ]

    top_k = 3

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 14 - EMPTY GT FOR ONE QUERY",
        docs_true,
        queries,
        top_k
    )

    # Q1:
    # P = 0
    # R = 0
    #
    # Q2:
    # P = 1/3
    # R = 1

    assert_close(
        p,
        (0 + 1/3) / 2,
        "Precision"
    )

    assert_close(
        r,
        1/2,
        "Recall"
    )

    assert_close(
        hit,
        1/2,
        "HitRate"
    )

    # Q1 = 0
    # Q2 relevant rank 1 = 1

    assert_close(
        mrr,
        1/2,
        "MRR"
    )

    assert_close(
        map_score,
        1/2,
        "MAP"
    )

    assert_close(
        ndcg,
        1/2,
        "NDCG"
    )


# ============================================================
# TEST 15
# EMPTY PREDICTIONS
# ============================================================

def test_empty_predictions():
    docs_true = [
        [1],
        [2]
    ]

    queries = [
        [],
        []
    ]

    top_k = 5

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 15 - EMPTY PREDICTIONS",
        docs_true,
        queries,
        top_k
    )

    assert_close(p, 0.0, "Precision")
    assert_close(r, 0.0, "Recall")
    assert_close(hit, 0.0, "HitRate")
    assert_close(mrr, 0.0, "MRR")
    assert_close(map_score, 0.0, "MAP")
    assert_close(ndcg, 0.0, "NDCG")


# ============================================================
# TEST 16
# LARGE TOP K
# ============================================================

def test_large_top_k():
    docs_true = [
        [2, 4]
    ]

    queries = [
        [1, 2, 3, 4]
    ]

    top_k = 10

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 16 - LARGE TOP K",
        docs_true,
        queries,
        top_k
    )

    # Strict Precision@K:
    #
    # 2 relevant / 10 positions.

    assert_close(
        p,
        2/10,
        "Precision"
    )

    assert_close(
        r,
        1.0,
        "Recall"
    )

    assert_close(
        hit,
        1.0,
        "HitRate"
    )

    # first relevant at rank 2
    assert_close(
        mrr,
        1/2,
        "MRR"
    )

    # relevant rank 2:
    # P = 1/2
    #
    # relevant rank 4:
    # P = 2/4
    #
    # AP = 0.5

    assert_close(
        map_score,
        0.5,
        "MAP"
    )

    assert 0 < ndcg < 1


# ============================================================
# TEST 17
# PERFECT ORDER WITH EXTRA IRRELEVANT DOCS AFTER GT
# ============================================================

def test_perfect_relevant_prefix():
    docs_true = [
        [2, 4, 6]
    ]

    queries = [
        [2, 4, 6, 100, 200]
    ]

    top_k = 5

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 17 - PERFECT RELEVANT PREFIX",
        docs_true,
        queries,
        top_k
    )

    assert_close(
        p,
        3/5,
        "Precision"
    )

    assert_close(
        r,
        1.0,
        "Recall"
    )

    assert_close(
        hit,
        1.0,
        "HitRate"
    )

    assert_close(
        mrr,
        1.0,
        "MRR"
    )

    assert_close(
        map_score,
        1.0,
        "MAP"
    )

    # Relevant docs occupy ideal ranks 1,2,3.
    assert_close(
        ndcg,
        1.0,
        "NDCG"
    )


# ============================================================
# TEST 18
# MAP PENALIZES MISSING RELEVANT DOCS
# ============================================================

def test_map_penalizes_missing_docs():
    docs_true = [
        [2, 4, 6]
    ]

    queries = [
        [2, 1, 4]
    ]

    top_k = 3

    _, r, _, _, map_score, _ = show_metrics(
        "TEST 18 - MAP PENALIZES MISSED GT",
        docs_true,
        queries,
        top_k
    )

    # Relevant:
    #
    # 2 at rank 1 -> P@1 = 1
    # 4 at rank 3 -> P@3 = 2/3
    # 6 missing
    #
    # AP denominator remains 3.

    expected_map = (
        1 +
        2/3
    ) / 3

    assert_close(
        map_score,
        expected_map,
        "MAP"
    )

    assert_close(
        r,
        2/3,
        "Recall"
    )


# ============================================================
# TEST 19
# NDCG RANKING SENSITIVITY
# ============================================================

def test_ndcg_ranking_sensitivity():
    docs_true = [
        [2, 4]
    ]

    best = [
        [2, 4, 1, 3]
    ]

    middle = [
        [2, 1, 4, 3]
    ]

    worst = [
        [1, 3, 2, 4]
    ]

    top_k = 4

    best_score = ndcg_at_k(
        docs_true,
        best,
        top_k
    )

    middle_score = ndcg_at_k(
        docs_true,
        middle,
        top_k
    )

    worst_score = ndcg_at_k(
        docs_true,
        worst,
        top_k
    )

    print("\n" + "=" * 65)
    print("TEST 19 - NDCG RANKING SENSITIVITY")
    print("=" * 65)

    print(f"Best:   {best_score:.6f}")
    print(f"Middle: {middle_score:.6f}")
    print(f"Worst:  {worst_score:.6f}")

    assert_close(
        best_score,
        1.0,
        "Best NDCG"
    )

    assert (
        best_score
        > middle_score
        > worst_score
    )


# ============================================================
# TEST 20
# REALISTIC RETRIEVAL DATASET
# ============================================================

def test_realistic_dataset():
    docs_true = [
        [2, 5],
        [10],
        [30, 31, 32],
        [50],
        [70, 71]
    ]

    queries = [
        [1, 2, 3, 5, 6],
        [10, 11, 12, 13, 14],
        [30, 40, 31, 50, 60],
        [1, 2, 3, 4, 5],
        [100, 70, 101, 71, 102]
    ]

    top_k = 5

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 20 - REALISTIC DATASET",
        docs_true,
        queries,
        top_k
    )

    expected_precision = (
        2/5 +
        1/5 +
        2/5 +
        0 +
        2/5
    ) / 5

    expected_recall = (
        1 +
        1 +
        2/3 +
        0 +
        1
    ) / 5

    expected_hit = 4 / 5

    expected_mrr = (
        1/2 +
        1 +
        1 +
        0 +
        1/2
    ) / 5

    assert_close(
        p,
        expected_precision,
        "Precision"
    )

    assert_close(
        r,
        expected_recall,
        "Recall"
    )

    assert_close(
        hit,
        expected_hit,
        "HitRate"
    )

    assert_close(
        mrr,
        expected_mrr,
        "MRR"
    )

    assert 0 <= map_score <= 1
    assert 0 <= ndcg <= 1


# ============================================================
# TEST 21
# ALL METRICS MUST BE IN [0,1]
# ============================================================

def test_metric_ranges():
    docs_true = [
        [2, 3],
        [10],
        [30, 40]
    ]

    queries = [
        [1, 2, 3, 4],
        [20, 10, 30, 40],
        [30, 50, 60, 40]
    ]

    top_k = 4

    p, r, hit, mrr, map_score, ndcg = show_metrics(
        "TEST 21 - METRIC RANGE",
        docs_true,
        queries,
        top_k
    )

    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= hit <= 1
    assert 0 <= mrr <= 1
    assert 0 <= map_score <= 1
    assert 0 <= ndcg <= 1


# ============================================================
# TEST 22
# INVALID TOP K
# ============================================================

def test_invalid_top_k():
    docs_true = [
        [1]
    ]

    queries = [
        [1, 2, 3]
    ]

    print("\n" + "=" * 65)
    print("TEST 22 - INVALID TOP K")
    print("=" * 65)

    for top_k in [
        0,
        -1,
        -10
    ]:

        functions = [
            precision_recall_at_k,
            hit_rate_at_k,
            ndcg_at_k
        ]

        for func in functions:

            try:
                func(
                    docs_true,
                    queries,
                    top_k
                )

            except ValueError:
                print(
                    f"PASS: {func.__name__}"
                    f"(top_k={top_k})"
                )

            else:
                raise AssertionError(
                    f"{func.__name__} should raise "
                    f"ValueError for top_k={top_k}"
                )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_perfect()
    test_complete_miss()
    test_normal()

    test_different_ranks()
    test_multiple_relevant()
    test_partial_recall()

    test_relevant_outside_top_k()
    test_top_k_one()
    test_top_k_greater_than_query_length()

    test_mrr_first_relevant()
    test_good_vs_bad_ranking()
    test_high_hit_bad_ranking()

    test_different_gt_sizes()
    test_empty_ground_truth()
    test_empty_predictions()

    test_large_top_k()
    test_perfect_relevant_prefix()

    test_map_penalizes_missing_docs()
    test_ndcg_ranking_sensitivity()

    test_realistic_dataset()
    test_metric_ranges()

    test_invalid_top_k()

    print("\n" + "=" * 65)
    print("ALL RETRIEVAL METRIC TESTS PASSED ✅")
    print("=" * 65)