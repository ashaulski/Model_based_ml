from dpd_ml_project.main import run


def test_run_executes_without_error(capsys):
    run()
    captured = capsys.readouterr()
    assert "DPD iteration complete" in captured.out
