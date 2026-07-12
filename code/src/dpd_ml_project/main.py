from dpd_ml_project.orchestrator.dpd_pipeline import run_iteration


def run() -> None:
    """Run one DPD iteration with default config."""
    result = run_iteration()
    print(f"DPD iteration complete: {result.status}")


if __name__ == "__main__":
    run()
