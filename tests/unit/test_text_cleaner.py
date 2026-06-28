from ingestion.cleaners.text_cleaner import TextCleaner


def test_text_cleaner_removes_empty_lines_and_normalizes_whitespace():

    cleaner = TextCleaner()

    result = cleaner.clean(" hello   world \n\n second\tline ")

    assert result == "hello world\nsecond line"
