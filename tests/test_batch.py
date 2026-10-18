"""Tests for batch run-directory lookup."""

from __future__ import annotations

from pathlib import Path

import pytest

from tautomator.batch import compound_dir_name, compound_id, find_compound_dir


class TestCompoundDirs:
    def test_a_new_molecule_gets_a_directory_named_for_its_id(self, tmp_path: Path) -> None:
        assert find_compound_dir(tmp_path, "perrin_supp5572") == tmp_path / "perrin_supp5572"

    def test_a_prefixed_directory_is_found_at_any_row_number(self, tmp_path: Path) -> None:
        """The row a molecule sits on moves whenever the CSV is regenerated."""
        old = tmp_path / "004_perrin3020"
        old.mkdir()
        assert find_compound_dir(tmp_path, "perrin3020") == old

    def test_the_id_must_match_whole(self, tmp_path: Path) -> None:
        (tmp_path / "022_perrin_supp5572").mkdir()
        (tmp_path / "003_serjeant25500").mkdir()
        assert find_compound_dir(tmp_path, "supp5572") == tmp_path / "supp5572"
        assert find_compound_dir(tmp_path, "serjeant2550") == tmp_path / "serjeant2550"

    def test_the_unprefixed_directory_wins(self, tmp_path: Path) -> None:
        (tmp_path / "004_perrin3020").mkdir()
        (tmp_path / "perrin3020").mkdir()
        assert find_compound_dir(tmp_path, "perrin3020") == tmp_path / "perrin3020"

    def test_two_prefixed_directories_for_one_id_raise(self, tmp_path: Path) -> None:
        (tmp_path / "003_perrin3020").mkdir()
        (tmp_path / "004_perrin3020").mkdir()
        with pytest.raises(ValueError, match="more than one run directory"):
            find_compound_dir(tmp_path, "perrin3020")

    def test_an_unsafe_id_is_sanitised_the_same_way_both_times(self, tmp_path: Path) -> None:
        assert compound_dir_name("a/b c") == "a_b_c"
        (tmp_path / "001_a_b_c").mkdir()
        assert find_compound_dir(tmp_path, "a/b c") == tmp_path / "001_a_b_c"

    @pytest.mark.parametrize(
        "dir_name", ["022_perrin_supp5572", "perrin_supp5572"], ids=["prefixed", "plain"]
    )
    def test_the_id_is_read_back_from_either_naming(self, dir_name: str) -> None:
        assert compound_id(dir_name) == "perrin_supp5572"
