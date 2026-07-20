"""Tests for japanese.py — morphological phrase-boundary detection.

The whole module degrades gracefully when fugashi/UniDic is not installed.  The
fallback behaviour (functions returning None / False) is covered implicitly by the
segments and fcpxml_telop suites, which run with or without fugashi.  These tests
exercise the *active* path and are skipped when fugashi is unavailable.
"""

import pytest

from preprod import japanese as J

pytestmark = pytest.mark.skipif(not J.available(), reason="fugashi/UniDic not installed")


class TestBreakScores:
    def test_returns_phrase_boundaries(self):
        scores = J.break_scores("そういった好みが含まれている")
        assert scores is not None
        # A boundary exists after the particle が (natural 、 spot).
        text = "そういった好みが含まれている"
        ga_pos = text.index("が") + 1
        assert ga_pos in scores

    def test_particle_boundary_scores_higher_than_dangling_adverb(self):
        text = "花とかそういった全然違う見た目"
        scores = J.break_scores(text)
        after_toka = text.index("か") + 1          # 花とか| — ends in particle
        after_sou = text.index("そう") + 2          # 花とかそう| — dangling adverb
        assert scores[after_toka] > scores[after_sou], \
            f"particle boundary should beat dangling adverb: {scores}"

    def test_no_boundary_inside_a_word(self):
        # 'そういった' is そう+いっ+た; the only boundaries are at phrase heads,
        # never inside the verb いった.
        text = "そういった"
        scores = J.break_scores(text) or {}
        # position 3 (inside いった, between いっ and た) must NOT be a boundary
        assert 4 not in scores


class TestIsContinuation:
    def test_word_straddling_boundary_is_continuation(self):
        # フリー | ランス → フリーランス is one word → mid-word
        assert J.is_continuation("フリー", "ランスの") is True

    def test_dangling_adverb_is_continuation(self):
        # …そう | いった → そう is a dangling modifier of いった
        assert J.is_continuation("全然違う花とかそう", "いった見た目") is True

    def test_dependent_word_head_is_continuation(self):
        # next segment starting with a particle attaches backward
        assert J.is_continuation("これは", "がポイント") is True

    def test_clean_phrase_boundary_is_not_continuation(self):
        # particle は ends a phrase, 明日 (noun) starts a new one → clean
        assert J.is_continuation("今日は", "明日も晴れ") is False

    def test_sentence_end_is_not_continuation(self):
        assert J.is_continuation("終わりです", "次の話") is False

    def test_empty_inputs_safe(self):
        assert J.is_continuation("", "あ") is False
        assert J.is_continuation("あ", "") is False


class TestCrossesMorpheme:
    def test_word_straddling_boundary(self):
        # "Google" is one morpheme spanning the 元Goo|gleの boundary.
        assert J.crosses_morpheme("元Goo", "gleの偉そうな") is True

    def test_clean_boundary_does_not_straddle(self):
        assert J.crosses_morpheme("今日は", "明日も晴れ") is False

    def test_sentence_end_does_not_straddle(self):
        assert J.crosses_morpheme("終わりです", "次の話") is False

    def test_empty_inputs_safe(self):
        assert J.crosses_morpheme("", "あ") is False
        assert J.crosses_morpheme("あ", "") is False


class TestThreadSafety:
    """The shared fugashi Tagger is guarded by a lock (T0250).

    Flask runs threaded=True, so concurrent analyze/export requests hit the
    single shared MeCab Tagger.  Without serialization its internal parse buffer
    races, producing exceptions or clobbered (wrong) results.  This test captures
    single-threaded ground truth, then hammers the tagger from many threads and
    asserts every result matches — catching both crashes and silent corruption.
    """

    def test_concurrent_access_matches_single_threaded(self):
        import threading

        texts = [
            "そういった好みが含まれている",
            "花とかそういった全然違う見た目",
            "今日は明日も晴れです",
            "フリーランスの人々が集まる場所",
            "元Googleの偉そうな人がそう言った",
        ]
        # Ground truth, computed serially (uncontended).
        score_truth = {t: J.break_scores(t) for t in texts}
        cross_truth = J.crosses_morpheme("元Goo", "gleの偉そうな")

        errors: list[str] = []
        mismatches: list[str] = []
        N_THREADS = 16
        barrier = threading.Barrier(N_THREADS)

        def worker() -> None:
            try:
                barrier.wait()  # release all threads simultaneously → max contention
                for _ in range(50):
                    for t in texts:
                        if J.break_scores(t) != score_truth[t]:
                            mismatches.append(t)
                    if J.crosses_morpheme("元Goo", "gleの偉そうな") != cross_truth:
                        mismatches.append("cross")
            except Exception as e:  # pragma: no cover — only hit if the race throws
                errors.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors, f"tagger raised under concurrency: {errors[:5]}"
        assert not mismatches, f"corrupted results under concurrency: {sorted(set(mismatches))}"
