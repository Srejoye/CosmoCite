from metadata import parse_summary


def test_parse_summary_keeps_people_and_the_journal():
    result = {"uids": ["11"], "11": {"fulljournalname": "Bone", "authors": [{"name": "Smith A", "authtype": "Author"}, {"name": "NASA", "authtype": "CollectiveName"}]}}
    assert parse_summary(result) == {"PMC11": {"authors": ["Smith A"], "journal": "Bone"}}


def test_parse_summary_skips_records_with_errors():
    assert parse_summary({"uids": ["11"], "11": {"error": "cannot get document summary"}}) == {}
